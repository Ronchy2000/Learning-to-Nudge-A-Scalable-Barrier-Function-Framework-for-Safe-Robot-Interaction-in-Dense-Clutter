# DCBF 全流程命令（collect → train → refine → eval）

## 0) 进入项目

```bash
cd dcbf_repro
```

## 1) 数据采集（4 objects，偏高交互）

```bash
sh scripts/run_collect.sh \
  --num_objects 4 \
  --num_traj 2500 \
  --policy do_nothing \
  --table_half_extent 0.20 \
  --contact_distance 0.10 \
  --tilt_gain 10.0 \
  --tilt_decay 0.0003 \
  --goal_tolerance 0.005 \
  --max_episode_steps 300 \
  --backstep_margin_deg 0.2
```

## 2) 检查采集分布（确认非全 safe）

```bash
LATEST_DATA="$(cat outputs/data/LATEST_RUN)"
python3 -m dcbf.data.collect stats \
  --data_glob "${LATEST_DATA}/train_*.npz" \
  --output_json "${LATEST_DATA}/stats.json"
cat "${LATEST_DATA}/stats.json"
```

## 3) 训练 Ours σ=0.01（Initial）

```bash
RUN_001="ours_sigma_001_$(date +%Y%m%d_%H%M%S)"
sh scripts/run_train.sh \
  --config configs/train.yaml \
  --sigma 0.01 \
  --run_name "${RUN_001}"
CKPT_001_INIT="outputs/train/${RUN_001}/best.pt"
echo "${CKPT_001_INIT}"
```

## 4) Refinement（σ=0.01）

```bash
REFINE_001_TS="$(date +%Y%m%d_%H%M%S)"
sh scripts/run_refine.sh \
  --config configs/refine.yaml \
  --checkpoint "${CKPT_001_INIT}" \
  --dataset_glob "${LATEST_DATA}/train_*.npz" \
  --output_dir "outputs/refine/${REFINE_001_TS}" \
  --run_name "refined_sigma_001_${REFINE_001_TS}"
CKPT_001_REFINED="outputs/refine/${REFINE_001_TS}/refined_sigma_001_${REFINE_001_TS}/best.pt"
echo "${CKPT_001_REFINED}"
```

## 5) 训练 Ours σ=0.02（Initial）

```bash
RUN_002="ours_sigma_002_$(date +%Y%m%d_%H%M%S)"
sh scripts/run_train.sh \
  --config configs/train.yaml \
  --sigma 0.02 \
  --run_name "${RUN_002}"
CKPT_002_INIT="outputs/train/${RUN_002}/best.pt"
echo "${CKPT_002_INIT}"
```

## 6) Refinement（σ=0.02）

```bash
REFINE_002_TS="$(date +%Y%m%d_%H%M%S)"
sh scripts/run_refine.sh \
  --config configs/refine.yaml \
  --checkpoint "${CKPT_002_INIT}" \
  --dataset_glob "${LATEST_DATA}/train_*.npz" \
  --output_dir "outputs/refine/${REFINE_002_TS}" \
  --run_name "refined_sigma_002_${REFINE_002_TS}"
CKPT_002_REFINED="outputs/refine/${REFINE_002_TS}/refined_sigma_002_${REFINE_002_TS}/best.pt"
echo "${CKPT_002_REFINED}"
```

## 7) 论文式评估：Do Nothing / APF / Ours σ=0.01 / Ours σ=0.02

```bash
sh scripts/run_eval.sh \
  --config configs/eval.yaml \
  --methods do_nothing apf ours_sigma_001 ours_sigma_002 \
  --learned_method ours_sigma_001="${CKPT_001_REFINED}" \
  --learned_method ours_sigma_002="${CKPT_002_REFINED}" \
  --num_objects_list 4 10 20 40 \
  --episodes 100
```

## 8) 查看评估输出

```bash
EVAL_DIR="$(cat outputs/eval/LATEST_RUN)"
echo "${EVAL_DIR}"
cat "${EVAL_DIR}/metrics.csv"
ls -lah "${EVAL_DIR}/metrics_plot.png"
```

## 9) 可选：评估 Initial vs Refined（单个 σ）

```bash
sh scripts/run_eval.sh \
  --config configs/eval.yaml \
  --methods do_nothing apf initial_dcbf refined_dcbf \
  --initial_checkpoint "${CKPT_001_INIT}" \
  --refined_checkpoint "${CKPT_001_REFINED}" \
  --num_objects_list 4 10 20 40 \
  --episodes 100
```
