"""验证坐标系、LSTM输入形状、数据分布和 d_thresh 效果"""
import numpy as np
import glob
import torch
from dcbf.models.dcbf_net import DCBFNet

# 1. 验证数据中 robot_t 的坐标是否已经是 object-centric
files = sorted(glob.glob("outputs/data/**/train_*.npz", recursive=True))
d = np.load(files[0])
print("=== 数据字段 ===")
for k in sorted(d.files):
    print(f"  {k}: shape={d[k].shape}, dtype={d[k].dtype}")

robot_t = d["robot_t"]
obj_prev = d["obj_hist_prev"]
print(f"\n=== robot_t (object-centric) ===")
print(f"  shape: {robot_t.shape}")
print(f"  x range: [{robot_t[:,0].min():.4f}, {robot_t[:,0].max():.4f}]")
print(f"  y range: [{robot_t[:,1].min():.4f}, {robot_t[:,1].max():.4f}]")
print(f"  z range: [{robot_t[:,2].min():.4f}, {robot_t[:,2].max():.4f}]")
dist_xy = np.linalg.norm(robot_t[:, :2], axis=1)
print(f"  dist_xy range: [{dist_xy.min():.4f}, {dist_xy.max():.4f}]")

print(f"\n=== obj_hist_prev 锚点验证 ===")
print(f"  shape: {obj_prev.shape}")
print(f"  t=0 xyz mean: {obj_prev[:,0,:3].mean(axis=0)}")
print(f"  t=0 xyz max:  {np.abs(obj_prev[:,0,:3]).max(axis=0)}")
anchor_norm = np.linalg.norm(obj_prev[:, 0, :3], axis=1)
print(f"  t=0 anchor norm: max={anchor_norm.max():.6f}, mean={anchor_norm.mean():.6f}")

# 2. 验证 LSTM input shape
model = DCBFNet()
B = 4
robot_feat = torch.randn(B, 3)
obj_hist = torch.randn(B, 10, 4)
out = model(robot_feat, obj_hist)
print(f"\n=== LSTM input/output 验证 ===")
print(f"  robot_feat: {robot_feat.shape}")
print(f"  obj_hist_feat: {obj_hist.shape} -> (B, T, 4)")
print(f"  output: {out.shape} -> (B, 1)")

# 3. safe/unsafe 分布和 d_thresh 效果
labels = d["label_safe_obj"]
tilts = d["next_tilt_deg"]
n_total = len(labels)
n_safe = int((labels > 0.5).sum())
n_unsafe = int((labels <= 0.5).sum())
print(f"\n=== 标签分布 ===")
print(f"  总计: {n_total}, safe: {n_safe} ({n_safe/n_total*100:.1f}%), unsafe: {n_unsafe} ({n_unsafe/n_total*100:.1f}%)")

# d_thresh 模拟
print(f"\n=== d_thresh filtering 模拟 ===")
for thresh in [0.10, 0.15, 0.20, 0.25]:
    is_unsafe = labels <= 0.5
    has_tilt = tilts > 0.5
    keep = is_unsafe | (dist_xy <= thresh) | has_tilt
    kept = int(keep.sum())
    safe_after = int((labels[keep] > 0.5).sum())
    unsafe_after = int((labels[keep] <= 0.5).sum())
    print(f"  d_thresh={thresh:.2f}: kept={kept} ({kept/n_total*100:.1f}%), safe={safe_after}, unsafe={unsafe_after}, ratio={unsafe_after/max(kept,1)*100:.1f}%")

# 近边界分布
print(f"\n=== 近边界样本分布 ===")
for lo, hi in [(5, 10), (10, 15), (10, 20), (13, 17)]:
    cnt = int(((tilts >= lo) & (tilts <= hi)).sum())
    print(f"  theta in [{lo},{hi}]: {cnt} samples ({cnt/n_total*100:.1f}%)")

print("\n=== 验证完成 ===")
