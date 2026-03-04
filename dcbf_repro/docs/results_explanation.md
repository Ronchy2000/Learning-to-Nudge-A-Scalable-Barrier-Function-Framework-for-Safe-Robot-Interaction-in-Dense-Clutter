# Results Explanation — "Learning to Nudge" DCBF Reproduction

---

## 1. What is "Do Nothing"?

"Do Nothing" refers to a **greedy goal-reaching controller** with no obstacle avoidance mechanism. At each timestep, it computes the direction vector from the current end-effector position to the goal, clips it to the maximum step size (0.01 m), and executes it directly:

$$u = \text{clip}(p_{\text{goal}} - p_{\text{ee}},\; \|u\|_{\max})$$

It is called "Do Nothing" because **no safety behavior is applied** — the robot does nothing to avoid collisions. It serves as the lower-bound baseline in the paper (Sec. VI-A).

---

## 2. What Does the Contour Line Represent?

Yes, the black dashed contour line on the heatmap represents the **zero level-set of the learned barrier function**, i.e., $B(r^t, O^{t-1}) = 0$. According to CBF theory:

- **$B > 0$ (blue region):** the robot state is **safe** — no objects are at risk of toppling.
- **$B < 0$ (red region):** the robot state is **unsafe** — at least one object is predicted to topple.
- **$B = 0$ (white band / black dashed contour):** the **safety boundary** separating the two regions.

The white color in the heatmap corresponds to $B \approx 0$ due to the `TwoSlopeNorm(vcenter=0)` colormap normalization. The contour line is drawn exactly at $B = 0$.

---

## 3. Mock Environment vs. Isaac Lab — Why Data is Critical

### Current Limitation

Our reproduction uses a **mock (kinematic) environment** rather than NVIDIA Isaac Lab. The mock environment approximates object tilt through a simplified contact model: when the end-effector enters the contact distance ($d_{\text{contact}} = 0.10$ m), a proportional tilt is applied with a tunable gain ($k_{\text{tilt}} = 10.0$). This means we lack realistic physics: no friction, no inertia, no multi-body contact propagation.

### What the Original Paper Says About Data (Sec. V)

The authors emphasize several critical data aspects:

1. **Data collection policy (Sec. V-A):** Training data is collected using an APF (Artificial Potential Field) controller, which naturally produces diverse interactions — some trajectories push objects, some avoid them. This diversity is essential for the network to learn a meaningful boundary.

2. **Free-space sample discarding (Sec. V-A):** Samples where the end-effector is far from all objects are discarded, because they carry no informative gradient for learning the barrier boundary. The authors explicitly state: *"we discard data points where the robot is far from all objects."*

3. **Balanced sampling (Sec. V-A):** Safe and unsafe samples are balanced during training to prevent the network from trivially predicting "always safe."

4. **Boundary refinement with real physics (Sec. V-D):** The refinement procedure specifically re-labels near-boundary states by **executing the safest action in the simulator** for $s = 4$ steps and checking whether any object actually topples. This step fundamentally relies on **accurate physics simulation** — tilting, sliding, and multi-object interactions that only a high-fidelity simulator like Isaac Lab can provide.

### Impact on Our Results

Because our mock environment uses a simplified tilt model, the learned barrier boundary may not accurately reflect true physical safety margins. In the paper, after refinement, the zero contour "invades" into the obstacle circles — meaning the robot learns it can safely nudge objects without toppling them. Our reproduction shows this trend qualitatively (refined model has tighter boundaries), but the exact boundary shape depends on the fidelity of the physics simulation.

---

## 4. Other Important Details

### a) Loss Function (Sec. V-C)

The total loss is:

$$L(\theta) = \eta_s L_s + \eta_u L_u + \eta_d L_d$$

The **derivative loss** $L_d$ (Eq. 10) is critical — it enforces the discrete-time CBF condition:

$$B_{t+1} \geq (1 - \gamma) B_t - \sigma$$

Without $L_d$, the network can learn a correct classification boundary but with **no gradient structure** — it would be a flat step function rather than a smooth barrier field. In our implementation, $\gamma = 0.1$ (barrier must retain 90% of its value per step), $\sigma = 0.02$ (robustness margin).

### b) Object-Centric Decomposition (Sec. IV-B)

The key architectural insight is **scalability**: instead of feeding all $N$ objects into one network, the DCBF evaluates each object independently — the network computes $B_i(r^t, O_i^{t-1})$ and the global barrier is:

$$B_{\text{global}} = \min_i B_i$$

This means a model trained on $N = 4$ objects generalizes directly to $N = 40$ without retraining, which is exactly what our evaluation demonstrates.

### c) Why APF Performs Worst

APF achieves near-zero success rate (2%–10%) because in dense clutter, repulsive forces from surrounding objects create **local minima** (oscillation traps). The robot gets stuck bouncing between objects, triggering the stalling detector (92% stall rate at $N = 4$). This is a well-known limitation of potential field methods in cluttered environments.

### d) Refined DCBF Advantage

The refined model achieves the **lowest violation rate** across all densities (2%–6% vs. Do Nothing's 4%–42%), while maintaining high success rate (86%–98%). The tradeoff is longer episode steps (32–87 vs. 15–20 for Do Nothing), because the safety filter steers the robot around obstacles rather than through them.

### Evaluation Results Summary

| Method | N=4 Success | N=40 Success | N=40 Violation |
|--------|------------|-------------|---------------|
| Do Nothing | 96% | 60% | 40% |
| Backstep | 96% | 58% | 42% |
| APF | 2% | 6% | 40% |
| Initial DCBF | 96% | 88% | 8% |
| **Refined DCBF** | **98%** | **86%** | **6%** |
