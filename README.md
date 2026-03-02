# DCBF 复现工程

本仓库实现了论文 _Learning to Nudge: A Scalable Barrier Function Framework for Safe Robot Interaction in Dense Clutter_ (ICRA 2026) 中 Dense Contact Barrier Functions (DCBF) 的核心 pipeline。

当前默认使用 mock 物理后端，可在纯 CPU 环境下完整跑通 collect → train → refine → eval 全链路。`dcbf/envs/isaaclab_env.py` 中保留了 IsaacLab 接口的 TODO，方便后续对接真实仿真器。

## 目录结构

```
configs/             环境、训练、refinement、评估配置
dcbf/
  envs/              Panda arm + 圆柱瓶子 mock 环境
  data/              轨迹采集、object-centric 变换、标签
  models/            LSTM + MLP → 单物体 barrier 值 B_i
  training/          三项 loss + 训练循环
  safety/            B_global = min_i B_i + 在线采样 safety filter
  refinement/        near-boundary 筛选 + safest-action rollout + finetune
  eval/              多方法评估 + 结果作图
  utils/             几何变换、IO、日志、seed 管理
scripts/             一键运行脚本
docs/                详细流程文档
```

## 安装

```bash
cd dcbf_repro
pip install -r requirements.txt
```

> 如果使用 conda，激活对应环境后再 pip install。
> IsaacLab/Isaac Sim 需按官方文档额外安装，与本仓库的 pip 依赖无关。

## 快速开始

最简三步：采集 → 训练 → 评估。

```bash
# 1. 采集 1200 条轨迹 (4 objects)
sh scripts/run_collect.sh --num_objects 4 --num_traj 1200

# 2. 训练
sh scripts/run_train.sh

# 3. 评估（含作图）
sh scripts/run_eval.sh
```

每个脚本启动时自动创建带时间戳的输出目录，同时写入 `LATEST_RUN` / `LATEST_CKPT` 指针文件，后续脚本自动读取，不需要手动拼路径。

完整的分步指南（包含 refinement、两个 σ 值对比、全部 6 种方法评估）见 [docs/full_pipeline_commands.md](docs/full_pipeline_commands.md)。

## 与论文的对应关系

### 网络结构

单物体 barrier $B_i(r^{t+1}_i,\, O^t_i)$ 的计算流程：

1. LSTM 编码物体历史 $O^t_i = \{o^{t-T}_i, \ldots, o^t_i\}$，输出末时刻隐状态
2. MLP 编码 object-centric 机器人状态 $r^{t+1}_i$
3. 两路拼接后再接一个 MLP → 标量 barrier 值

全局 barrier：$B_{\text{global}} = \min_i B_i$

代码位置：`dcbf/models/dcbf_net.py`、`dcbf/safety/compose.py`

### 训练损失（论文 Eq. 8–11）

$$L = \eta_s L_s + \eta_u L_u + \eta_d L_d$$

| 分量 | 含义 |
|------|------|
| $L_s = \operatorname{mean}(\operatorname{ReLU}(-B_t))$ | safe 样本的 barrier 应 ≥ 0 |
| $L_u = \operatorname{mean}(\operatorname{ReLU}(B_t))$ | unsafe 样本的 barrier 应 < 0 |
| $L_d = \operatorname{mean}(\operatorname{ReLU}((1-\gamma)B_t - B_{t+1} + \sigma))$ | 离散 CBF 不变性约束 |

代码位置：`dcbf/training/losses.py`

### 在线 Safety Filter

给定 nominal 动作 $u_\text{nom}$：
- 若 $B_\text{global}(\text{next}(u_\text{nom})) \ge 0$，直接执行
- 否则在 $u_\text{nom}$ 附近采样 $K$ 个候选，从中选取安全且偏移最小的动作

代码位置：`dcbf/safety/filter.py`

### Refinement

1. 从训练数据中选取 near-boundary 样本（$|B| \le \delta$）
2. 从 snapshot 恢复场景，每步选最安全动作 rollout $s$ 步
3. 新生成的数据与原始数据合并后 finetune 模型

代码位置：`dcbf/refinement/refine.py`

### Object-centric 变换

以物体最早历史帧 $o^{t-T}_i$ 为锚点做平移对齐，代码在 `dcbf/utils/geometry.py`。

## 评估方法（论文 6 种）

| 方法 | 说明 |
|------|------|
| Do Nothing | 直线走向目标，不做避障 |
| Backstep | 倾斜超过 14° 时沿接触方向后退 |
| APF | 人工势场 (KP=5.0, η=50.0, influence_dist=1.2m) |
| Initial Model | Refinement 之前的 NCBF |
| Ours σ=0.01 | Refined DCBF，σ=0.01 |
| Ours σ=0.02 | Refined DCBF，σ=0.02 |

评估在 4 / 10 / 20 / 40 个物体的场景中各跑 100 episodes，输出 success rate、violation rate、stalling rate、平均步数四个指标和对应的 bar chart。

## 数据格式

每个 `.npz` 文件包含若干 object-centric 样本：

- 输入：$(r_i^t, O_i^{t-1})$ 和 $(r_i^{t+1}, O_i^t)$
- 标签：`label_safe_obj`（单物体安全）、`label_safe_global`（全局安全）
- Snapshot 字段：`snap_ee` / `snap_goal` / `snap_object_pos` / `snap_object_tilt_rad` 等，用于 refinement 恢复

## 关键超参

| 参数 | 值 | 说明 |
|------|----|------|
| object_radius | 0.05 m | 瓶子半径 |
| object_height | 0.20 m | 瓶子高度 |
| tilt_threshold | 15° | 安全/不安全判据 |
| max_action_step | 0.01 m | 每步最大位移 |
| mass_range | [1.3, 2.0] kg | 随机质量 |
| gamma | 0.98 | CBF 衰减率 |
| sigma | 0.01 / 0.02 | 安全裕度 |
| history_len | 10 | 历史窗口长度 T |

## 当前局限

- mock 后端不涉及真实接触动力学，但数据流、训练、评估逻辑与论文一致
- object-centric 变换仅用平移对齐，未做姿态旋转
- snapshot 恢复是轻量级状态恢复（位置 / 倾角 / EE），不含物理引擎内部状态
- refinement 用采样近似 argmax，后续可替换为 CEM 或梯度优化
