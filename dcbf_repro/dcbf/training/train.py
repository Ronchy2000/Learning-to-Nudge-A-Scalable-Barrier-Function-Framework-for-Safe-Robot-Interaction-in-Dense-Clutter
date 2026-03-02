from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dcbf.data.dataset import DCBFDataset, split_files_by_ratio
from dcbf.models.dcbf_net import DCBFNet
from dcbf.training.losses import LossConfig, compute_dcbf_losses
from dcbf.utils.io import load_yaml
from dcbf.utils.logging import CSVLogger, JSONLLogger, create_tb_writer
from dcbf.utils.seeding import set_seed


def resolve_data_files(train_glob: str, val_glob: Optional[str] = None) -> Tuple[List[str], List[str]]:
    train_files = sorted(glob.glob(train_glob))
    if len(train_files) == 0:
        raise FileNotFoundError(f"No files matched train_glob={train_glob}")
    if val_glob:
        val_files = sorted(glob.glob(val_glob))
        if len(val_files) == 0:
            raise FileNotFoundError(f"No files matched val_glob={val_glob}")
    else:
        train_files, val_files = split_files_by_ratio(train_files, train_ratio=0.9)
    return train_files, val_files


def build_model(model_cfg: Dict) -> DCBFNet:
    return DCBFNet(
        robot_dim=model_cfg["robot_dim"],
        object_dim=model_cfg["object_dim"],
        history_len=model_cfg["history_len"],
        lstm_hidden=model_cfg["lstm_hidden"],
        lstm_layers=model_cfg["lstm_layers"],
        mlp_hidden=model_cfg["mlp_hidden"],
    )


def run_epoch(
    model: DCBFNet,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    loss_cfg: LossConfig,
    device: str,
    grad_clip: float = 5.0,
    progress_desc: Optional[str] = None,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    agg = {"total": 0.0, "l_s": 0.0, "l_u": 0.0, "l_d": 0.0, "drift_violation_ratio": 0.0}
    b_safe, b_unsafe = [], []
    num_batches = 0

    iterator = tqdm(
        loader,
        leave=False,
        dynamic_ncols=True,
        desc=progress_desc or ("train" if train_mode else "val"),
    )
    for batch in iterator:
        robot_t = batch["robot_t"].to(device)
        robot_tp1 = batch["robot_tp1"].to(device)
        obj_prev = batch["obj_hist_prev"].to(device)
        obj_curr = batch["obj_hist_curr"].to(device)
        safe_label = batch["safe_label"].to(device)

        with torch.set_grad_enabled(train_mode):
            b_t = model(robot_t, obj_prev).squeeze(-1)
            b_tp1 = model(robot_tp1, obj_curr).squeeze(-1)
            losses = compute_dcbf_losses(b_t, b_tp1, safe_label, loss_cfg)
            loss_total = losses["total"]
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

        safe_mask = safe_label > 0.5
        if safe_mask.any():
            b_safe.append(b_tp1[safe_mask].detach().cpu().numpy())
        if (~safe_mask).any():
            b_unsafe.append(b_tp1[~safe_mask].detach().cpu().numpy())

        for key in agg:
            agg[key] += float(losses[key].item())
        num_batches += 1

    if num_batches == 0:
        return {key: 0.0 for key in agg}
    out = {key: value / num_batches for key, value in agg.items()}
    out["b_safe_mean"] = float(np.mean(np.concatenate(b_safe)) if b_safe else 0.0)
    out["b_unsafe_mean"] = float(np.mean(np.concatenate(b_unsafe)) if b_unsafe else 0.0)
    return out


def save_checkpoint(
    path: Path,
    model: DCBFNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float,
    cfg: Dict,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
        "config": cfg,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str, model: DCBFNet, optimizer: Optional[torch.optim.Optimizer] = None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


def train_model(
    cfg: Dict,
    train_files: Optional[Sequence[str]] = None,
    val_files: Optional[Sequence[str]] = None,
    resume: Optional[str] = None,
    out_dir: Optional[str] = None,
    run_name: Optional[str] = None,
    override_epochs: Optional[int] = None,
    override_lr: Optional[float] = None,
) -> Path:
    set_seed(cfg.get("seed", 42))
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    loss_cfg = LossConfig.from_dict(cfg["loss"])
    optim_cfg = cfg["optim"]
    logging_cfg = cfg["logging"]

    if train_files is None or val_files is None:
        train_files, val_files = resolve_data_files(data_cfg["train_glob"], data_cfg.get("val_glob"))
    history_len = model_cfg["history_len"]
    use_global_label = bool(data_cfg.get("use_global_label", False))

    train_ds = DCBFDataset(files=train_files, history_len=history_len, use_global_label=use_global_label)
    val_ds = DCBFDataset(files=val_files, history_len=history_len, use_global_label=use_global_label)
    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        drop_last=False,
    )

    device = optim_cfg.get("device", "cpu")
    model = build_model(model_cfg).to(device)
    lr = override_lr if override_lr is not None else optim_cfg["lr"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=optim_cfg["weight_decay"])

    base_out_dir = Path(out_dir or logging_cfg["out_dir"])
    run_dir = base_out_dir / (run_name or logging_cfg.get("run_name", "dcbf_run"))
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    csv_logger = CSVLogger(run_dir / "metrics.csv")
    jsonl_logger = JSONLLogger(run_dir / "metrics.jsonl")
    tb_writer = create_tb_writer(run_dir / "tb")

    start_epoch = 0
    best_val = float("inf")
    if resume is not None:
        ckpt = load_checkpoint(resume, model, optimizer=optimizer)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", float("inf")))
        print(f"[train] resumed from {resume}, start_epoch={start_epoch}, best_val={best_val:.6f}")

    num_epochs = int(override_epochs if override_epochs is not None else optim_cfg["epochs"])
    print(
        f"[train] start epochs: {num_epochs} (global from {start_epoch} to {start_epoch + num_epochs - 1}), "
        f"train_batches={len(train_loader)}, val_batches={len(val_loader)}"
    )
    for epoch in range(start_epoch, start_epoch + num_epochs):
        local_epoch = epoch - start_epoch + 1
        print(f"[train] epoch {local_epoch}/{num_epochs} (global={epoch}) start")
        train_stats = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            loss_cfg=loss_cfg,
            device=device,
            grad_clip=float(optim_cfg.get("grad_clip", 5.0)),
            progress_desc=f"train e{local_epoch}/{num_epochs}",
        )
        val_stats = run_epoch(
            model,
            val_loader,
            optimizer=None,
            loss_cfg=loss_cfg,
            device=device,
            grad_clip=float(optim_cfg.get("grad_clip", 5.0)),
            progress_desc=f"val   e{local_epoch}/{num_epochs}",
        )
        row = {"epoch": epoch, **{f"train/{k}": v for k, v in train_stats.items()}, **{f"val/{k}": v for k, v in val_stats.items()}}
        csv_logger.log(row)
        jsonl_logger.log(row)
        for key, value in row.items():
            if key == "epoch":
                continue
            tb_writer.add_scalar(key, value, global_step=epoch)

        latest_path = run_dir / "latest.pt"
        save_checkpoint(latest_path, model, optimizer, epoch=epoch, best_val=best_val, cfg=cfg)
        if val_stats["total"] < best_val:
            best_val = val_stats["total"]
            best_path = run_dir / "best.pt"
            save_checkpoint(best_path, model, optimizer, epoch=epoch, best_val=best_val, cfg=cfg)
            print(f"[train] epoch={epoch} new best: {best_val:.6f}")
        if (epoch + 1) % int(optim_cfg.get("save_every", 5)) == 0:
            periodic_path = run_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(periodic_path, model, optimizer, epoch=epoch, best_val=best_val, cfg=cfg)
        print(
            f"[train] epoch {local_epoch}/{num_epochs} (global={epoch}) done | "
            f"train_total={train_stats['total']:.6f} val_total={val_stats['total']:.6f} "
            f"best_val={best_val:.6f} drift_ratio={val_stats['drift_violation_ratio']:.3f}"
        )

    csv_logger.close()
    jsonl_logger.close()
    tb_writer.close()
    return run_dir / "best.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DCBF network with classification + discrete CBF losses.")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint.")
    parser.add_argument("--train_glob", type=str, default=None, help="Override train data glob.")
    parser.add_argument("--val_glob", type=str, default=None, help="Override val data glob.")
    parser.add_argument("--out_dir", type=str, default=None, help="Override output directory.")
    parser.add_argument("--run_name", type=str, default=None, help="Override run name.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml(args.config)
    if args.train_glob is not None:
        cfg["data"]["train_glob"] = args.train_glob
    if args.val_glob is not None:
        cfg["data"]["val_glob"] = args.val_glob
    best_ckpt = train_model(cfg, resume=args.resume, out_dir=args.out_dir, run_name=args.run_name)
    print(f"[train] done. best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
