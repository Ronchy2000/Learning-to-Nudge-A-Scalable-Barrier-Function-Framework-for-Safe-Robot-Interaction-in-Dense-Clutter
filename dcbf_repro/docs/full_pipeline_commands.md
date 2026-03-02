# 全流程复现手册

从零开始到拿到论文 Fig.3 / Fig.4 风格的评估图表，一共 9 步。
所有命令在同一个终端窗口里顺序执行，变量会在步骤之间传递。

## 前置条件

- macOS / Linux，已安装 Anaconda3
- conda 环境名 `MARL`（按你的实际环境改）
- 已安装 `requirements.txt` 中的依赖

---

## Step 0. 准备

```bash
conda activate MARL
cd dcbf_repro
```

---

## Step 1. 环境检查（可选）

跑几轮随机动作，确认环境能正常产生碰撞和倾斜。

```bash
python scripts/make_env_check.py --resets 50 --steps 30
cat outputs/env_check/summary.json
```

看到 `violation_rate > 0` 就说明环境没问题。

---

## Step 2. 采集数据

论文用 4 个物体采 1200 条轨迹。mock 环境没有真实接触力学，论文默认的 `tilt_gain=1.8` 产生的 unsafe 样本不够（~1.4%），需要调整两个参数来补偿：

```bash
sh scripts/run_collect.sh \
  --num_objects 4 \                    # 论文设定：4 个瓶子
  --num_traj 1200 \                    # 论文采集量
  --table_half_extent 0.12 \           # 缩小桌面让物体更挤（默认 0.35 太稀疏）
  --tilt_gain 10.0 \                   # 补偿 mock 环境无真实力学（论文 1.8 用于 Isaac Sim）
  --contact_distance 0.10 \            # 接触判定半径，适当放大让碰撞更频繁
  --max_episode_steps 200 \            # 给足步数让 EE 穿越 clutter
  --backstep_margin_deg 1.0            # 采集时安全阈值(15°)附近的裕度
```

> 其余参数（`object_radius=0.05`、`tilt_threshold=15°`、`max_action_step=0.01` 等）走 `env.yaml` 默认值，与论文一致。

采完之后看一下数据分布：

```bash
LATEST_DATA="$(cat outputs/data/LATEST_RUN)"
echo "数据目录: ${LATEST_DATA}"

python -m dcbf.data.collect stats \
  --data_glob "${LATEST_DATA}/train_*.npz" \
  --output_json "${LATEST_DATA}/stats.json"

cat "${LATEST_DATA}/stats.json"
```

`unsafe_ratio_object` > 0.01 就能训练，> 0.05 更理想。太低的话继续调大 `tilt_gain` 或调小 `table_half_extent`（不低于 0.12，否则放不下 4 个瓶子）。

---

## Step 3. 训练 σ=0.01

```bash
RUN_001="ours_sigma_001"

sh scripts/run_train.sh \
  --sigma 0.01 \
  --run_name "${RUN_001}"

CKPT_001_INIT="outputs/train/${RUN_001}/best.pt"
if [ ! -f "${CKPT_001_INIT}" ]; then
  CKPT_001_INIT="outputs/train/${RUN_001}/latest.pt"
fi
echo "σ=0.01 checkpoint: ${CKPT_001_INIT}"
```

---

## Step 4. Refinement σ=0.01

```bash
sh scripts/run_refine.sh \
  --checkpoint "${CKPT_001_INIT}" \
  --dataset_glob "${LATEST_DATA}/train_*.npz" \
  --output_dir "outputs/refine/sigma_001" \
  --run_name "refined_sigma_001"

CKPT_001_REFINED="$(find outputs/refine/sigma_001 -name best.pt | sort | tail -1)"
if [ -z "${CKPT_001_REFINED}" ]; then
  CKPT_001_REFINED="$(find outputs/refine/sigma_001 -name latest.pt | sort | tail -1)"
fi
echo "σ=0.01 refined: ${CKPT_001_REFINED}"
```

---

## Step 5. 训练 σ=0.02

```bash
RUN_002="ours_sigma_002"

sh scripts/run_train.sh \
  --sigma 0.02 \
  --run_name "${RUN_002}"

CKPT_002_INIT="outputs/train/${RUN_002}/best.pt"
if [ ! -f "${CKPT_002_INIT}" ]; then
  CKPT_002_INIT="outputs/train/${RUN_002}/latest.pt"
fi
echo "σ=0.02 checkpoint: ${CKPT_002_INIT}"
```

---

## Step 6. Refinement σ=0.02

```bash
sh scripts/run_refine.sh \
  --checkpoint "${CKPT_002_INIT}" \
  --dataset_glob "${LATEST_DATA}/train_*.npz" \
  --output_dir "outputs/refine/sigma_002" \
  --run_name "refined_sigma_002"

CKPT_002_REFINED="$(find outputs/refine/sigma_002 -name best.pt | sort | tail -1)"
if [ -z "${CKPT_002_REFINED}" ]; then
  CKPT_002_REFINED="$(find outputs/refine/sigma_002 -name latest.pt | sort | tail -1)"
fi
echo "σ=0.02 refined: ${CKPT_002_REFINED}"
```

---

## Step 7. 全量评估

6 种方法 × 4 种密度（4/10/20/40 objects）× 100 episodes：

```bash
sh scripts/run_eval.sh \
  --methods do_nothing backstep apf initial_dcbf ours_sigma_001 ours_sigma_002 \
  --learned_method "initial_dcbf=${CKPT_001_INIT}" \
  --learned_method "ours_sigma_001=${CKPT_001_REFINED}" \
  --learned_method "ours_sigma_002=${CKPT_002_REFINED}" \
  --num_objects_list 4 10 20 40 \
  --episodes 100
```

`--learned_method` 把方法名映射到对应的 checkpoint，会覆盖 `eval.yaml` 中的默认路径。

---

## Step 8. 查看结果

```bash
EVAL_DIR="$(cat outputs/eval/LATEST_RUN)"

# 终端里看表格
column -t -s, "${EVAL_DIR}/metrics.csv"

# 打开图片（macOS）
open "${EVAL_DIR}/metrics_plot.png"
```

最终的 `metrics_plot.png` 包含 4 个子图：Success Rate / Violation Rate / Stalling Rate / Avg Episode Steps，与论文 Fig.3 & Fig.4 对应。

---

## （可选）Single-σ 消融：Initial vs Refined

只想看 refinement 前后效果对比：

```bash
sh scripts/run_eval.sh \
  --methods do_nothing backstep apf initial_dcbf refined_dcbf \
  --initial_checkpoint "${CKPT_001_INIT}" \
  --refined_checkpoint "${CKPT_001_REFINED}" \
  --num_objects_list 4 10 20 40 \
  --episodes 100

open "$(cat outputs/eval/LATEST_RUN)/metrics_plot.png"
```

---

## 方法参数对照表

| 论文中的名字 | `--methods` 里写 | 说明 |
|---|---|---|
| Do Nothing | `do_nothing` | 直线奔向目标，不避障 |
| Back-stepping | `backstep` | 倾斜超 14° 时后退一步 |
| APF | `apf` | 人工势场 (KP=5, η=50, 作用距离 1.2m) |
| Initial Model | `initial_dcbf` | refinement 前的 NCBF |
| Ours σ=0.01 | `ours_sigma_001` | refined DCBF, σ=0.01 |
| Ours σ=0.02 | `ours_sigma_002` | refined DCBF, σ=0.02 |

## 关键超参

| 参数 | 值 | 意义 |
|---|---|---|
| object_radius | 0.05 m | 瓶子半径 |
| object_height | 0.20 m | 瓶子高度 |
| tilt_threshold | 15° | 安全/不安全的判据 |
| max_action_step | 0.01 m | 一步最多走 1 cm |
| mass_range | [1.3, 2.0] kg | 每瓶随机质量 |
| gamma | 0.98 | CBF 衰减率 |
| sigma | 0.01 or 0.02 | 安全裕度 |
| history_len | 10 | 历史窗口 T |
