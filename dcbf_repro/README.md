# DCBF Reproduction (MVP → Full Pipeline)

本工程是论文 **Dense Contact Barrier Functions (DCBF)** 的可复现最小实现，目标是先跑通核心 pipeline，再逐步替换为真实 IsaacLab/Isaac Sim 场景。

> 当前仓库默认使用 `mock` 后端（可离线跑通 collect/train/refine/eval）；`isaaclab` 后端已在 `dcbf/envs/isaaclab_env.py` 留出 TODO 接口，便于你在服务器替换为真实仿真 API。

---

## 1) 安装与目录

```bash
cd dcbf_repro
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果要切换到真实 IsaacLab/Isaac Sim，请额外按官方文档安装对应版本（该部分通常不在 `pip requirements.txt` 内，且与 GPU/驱动强绑定）。

核心结构（与需求对齐）：

- `configs/*.yaml`: 环境、训练、refine、评估配置
- `dcbf/envs`: Panda 平面控制 + clutter cylinders + safety/stall 判据
- `dcbf/data`: 采集、object-centric 数据转换、label
- `dcbf/models`: `LSTM + MLP -> scalar B_i`
- `dcbf/training`: 三项 loss + 训练入口 + checkpoint/tensorboard
- `dcbf/safety`: `B_global = min_i B_i` + 在线采样 safety filter
- `dcbf/refinement`: near-boundary 样本筛选 + 最安全动作 rollout + finetune
- `dcbf/eval`: 指标评估 + 作图
- `scripts`: 一键运行脚本

---

## 2) 快速运行命令

### Step A: 环境自检
```bash
python3 scripts/make_env_check.py --config configs/env.yaml --help
python3 scripts/make_env_check.py --config configs/env.yaml
```
产物：
- `outputs/env_check/env_check_steps.csv`
- `outputs/env_check/summary.json`

### Step B/C: 采集数据（含 nominal/filter/back-stepping）
```bash
sh scripts/run_collect.sh --help
sh scripts/run_collect.sh --num_traj 200 --policy do_nothing --use_filter
python3 -m dcbf.data.collect stats --data_glob "outputs/data/*/train_*.npz"
```
产物：
- `outputs/data/<timestamp>/train_*.npz`, `outputs/data/<timestamp>/val_*.npz`
- `outputs/data/<timestamp>/collect_summary.json`
- `outputs/data/LATEST_RUN`（记录最近一次采集目录）

### 论文同款优先配置（默认配置）
```bash
# 1) 用 4 个物体采集初始数据（对应论文训练设定）
sh scripts/run_collect.sh \
  --config configs/env.yaml \
  --num_objects 4 \
  --num_traj 1200 \
  --policy do_nothing \
  --use_filter

# 2) 训练 Initial DCBF
sh scripts/run_train.sh --config configs/train.yaml

# 3) Refinement（s=4）
sh scripts/run_refine.sh --config configs/refine.yaml

# 4) 评估（4/10/20/40）
sh scripts/run_eval.sh --config configs/eval.yaml
```

### Step D: 训练 Initial DCBF
```bash
sh scripts/run_train.sh --help
sh scripts/run_train.sh --config configs/train.yaml
```
产物：
- `outputs/train/initial_dcbf_<timestamp>/best.pt`
- `outputs/train/initial_dcbf_<timestamp>/metrics.csv`
- `outputs/train/LATEST_RUN`（记录最近一次训练目录）
- `outputs/train/LATEST_CKPT`（记录最近一次训练 best checkpoint）

说明：若服务器上 `tensorboard/protobuf` 版本冲突，训练会自动退化为仅写 `csv/jsonl`（不会中断训练）。

### Step F: Refinement + Finetune
```bash
sh scripts/run_refine.sh --help
sh scripts/run_refine.sh --config configs/refine.yaml
```
产物：
- `outputs/refine/<timestamp>/refined_data/refined_*.npz`
- `outputs/refine/<timestamp>/refined_dcbf_<timestamp>/best.pt`
- `outputs/refine/LATEST_RUN`（记录最近一次 refine 目录）
- `outputs/refine/LATEST_CKPT`（记录最近一次 refine best checkpoint）

### Step G: 评估与作图（N=4/10/20/40）
```bash
sh scripts/run_eval.sh --help
sh scripts/run_eval.sh --config configs/eval.yaml
```
产物：
- `outputs/eval/<timestamp>/metrics.csv`
- `outputs/eval/<timestamp>/episodes.csv`
- `outputs/eval/<timestamp>/metrics_plot.png`（grouped bar chart）
- `outputs/eval/LATEST_RUN`（记录最近一次评估目录）

### Rollout（单方法快速检查）
```bash
sh scripts/run_rollout.sh --help
sh scripts/run_rollout.sh --method do_nothing
sh scripts/run_rollout.sh --method initial_dcbf --checkpoint outputs/train/initial_dcbf/best.pt
```

---

## 3) 与论文公式/模块对应

### 3.1 Object-centric barrier
- 单物体 barrier：$B_i(r_{t+1}^i, O_t^i)$  
- 全局 barrier：$B_{\text{global}} = \min_i B_i$  
对应代码：
- `dcbf/models/dcbf_net.py`
- `dcbf/safety/compose.py`

### 3.2 在线 safety filter（最小扰动）
给定 nominal 动作 $u_{\text{nom}}$：
1. 若 $B_{\text{global}}(\text{next}(u_{\text{nom}})) \ge 0$，直接执行  
2. 否则在 $u_{\text{nom}}$ 周围采样 $K$ 个候选，筛选安全集合  
3. 选取 $\lVert u-u_{\text{nom}} \rVert$ 最小的安全动作  
对应代码：
- `dcbf/safety/filter.py`

### 3.3 三项训练损失
在 `dcbf/training/losses.py` 实现：
- $L_s = mean(ReLU(-B))$（safe 样本应满足 $B \ge 0$）
- $L_u = mean(ReLU(B))$（unsafe 样本应满足 $B < 0$）
- $L_d = mean(ReLU((1-\gamma)B_t - B_{t+1} + \sigma))$（离散不变性约束）
- $L = \eta_s L_s + \eta_u L_u + \eta_d L_d$

### 3.4 Refinement
1. 选 near-boundary：$|B| \le \delta$  
2. 从 snapshot 恢复场景  
3. 每步采样动作并选 $\arg\max_u B_{\text{global}}(\text{next}(u))$ rollout $s$ 步  
4. 追加新数据后 finetune  
对应代码：
- `dcbf/refinement/refine.py`

---

## 4) 数据定义（Step C）

每个对象 i 的样本：
- 输入对：$(r_i^t, O_i^{t-1})$ 与 $(r_i^{t+1}, O_i^t)$
- 标签：`label_safe_obj`（默认）与 `label_safe_global`（可切换）
- snapshot：`snap_ee/snap_goal/snap_object_pos/snap_object_tilt_rad/...`

object-centric 变换 `R(·)` 在：
- `dcbf/utils/geometry.py`

---

## 5) 指标定义（Step G）

每个方法 × 每个 clutter N：
- `success_rate`: 到达目标且无违规
- `violation_rate`: 任意瓶倾角 `>15°`
- `stalling_rate`: 触发 stall 判据
- `avg_episode_steps`: episode 步长均值

---

## 6) 已知简化/偏差（当前 MVP）

1. **默认后端是 mock**：不是 IsaacLab 真实接触动力学；但接口、数据流和训练流程与论文一致。  
2. **object-centric R(·) 当前用平移对齐**：未使用完整姿态旋转对齐。  
3. **label 默认以每物体安全为主**，同时保留 `label_safe_global`，可在训练配置切换。  
4. **snapshot 恢复为轻量状态恢复**（位置/倾角/机器人状态），未包含完整物理引擎内部状态。  
5. **refinement rollout 使用动作采样近似 argmax**（工程可扩展为优化器/CEM）。

---

## 7) 下一步增强建议

1. 在 `isaaclab_env.py` 实现真实 IsaacLab Task/Scene 接口，保留当前观测与动作 schema。  
2. 用真实机器人前向运动学替换 `next_state` 简化积分。  
3. 增强 snapshot（含关节速度/接触状态）以提高 refinement 重放一致性。  
4. 加入更系统的 APF 与更多 baseline。  
5. 扩展为并行环境采集与多 GPU 训练。
