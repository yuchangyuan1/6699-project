# Expanded A+ Project Plan (Target ~22 Pages, Under 25 Pages)

## 0. Revised Goal

Since the report can be around **22 pages** and only needs to stay under **25 pages**, the project can be upgraded more ambitiously. The goal is no longer just to fit a compact 15-page version. Instead, the report can become a high-quality deep learning mathematics project centered on:

\[
\text{preconditioned stochastic dynamics}
\rightarrow
\text{implicit solution selection}
\rightarrow
\text{parameter-space geometry}
\rightarrow
\text{loss-level controls}
\rightarrow
\text{basin connectivity}
\rightarrow
\text{function-space and representation-space behavior}
\rightarrow
\text{generalization}.
\]

The project should still avoid becoming a broad optimizer benchmark. The stronger 22-page version should deepen the original question:

> Does Adam’s adaptive diagonal preconditioning merely accelerate optimization, or does it change the solution selected by training? If the selected parameters differ, do those differences correspond to differences in local geometry, basin connectivity, learned functions, learned representations, and generalization?

The expanded version can include several high-quality modules that were previously excluded due to page limits.

---

# 1. Recommended Final Title

Recommended title:

**Adaptive Preconditioned Dynamics, Solution Geometry, and Generalization in Neural Networks**

Subtitle:

**A Matched-Seed Study of SGD and Adam Across Parameter, Basin, Function, and Representation Geometry**

This title communicates mathematical depth and avoids sounding like a simple “SGD vs Adam” benchmark.

---

# 2. Final Report Structure and Page Budget

Target length: **21–23 pages**.  
Hard limit: **under 25 pages**.

| Section | Target Pages | Purpose |
|---|---:|---|
| 1. Introduction | 1.0–1.2 | Research question, framing, contributions |
| 2. Mathematical Framing | 3.0–3.5 | Preconditioned dynamics, local quadratic analysis, implicit bias |
| 3. Related Concepts and Theoretical Context | 1.0–1.5 | Flatness, mode connectivity, function/representation geometry |
| 4. Experimental Design | 1.5–2.0 | Dataset, models, matched seeds, metrics |
| 5. Early Convergence and Learning-Rate Caveat | 1.5 | Speed analysis, threshold crossing, LR caveat |
| 6. Parameter-Space Geometry | 2.0–2.5 | Distance, path length, Hessian, dispersion |
| 7. Loss-Matched and Step-Norm Controls | 2.0 | Fairness/control analysis |
| 8. Perturbation Flatness | 1.5–2.0 | Operational validation of Hessian trace |
| 9. Linear Mode Connectivity | 1.5–2.0 | Basin connectivity between SGD and Adam solutions |
| 10. Function-Space Similarity | 1.5–2.0 | Disagreement, symmetric KL, logit cosine |
| 11. Representation-Space Geometry | 1.5–2.0 | Feature kernel alignment, CKA or feature cosine |
| 12. Generalization and Three-Level Synthesis | 1.5–2.0 | Accuracy/gap, parameter vs function vs representation |
| 13. Limitations and Sharpness Caveats | 1.0–1.5 | Normalized sharpness, scale invariance, scope |
| 14. Conclusion | 0.5–0.8 | Final takeaway |
| **Total** | **~22 pages** | High-depth final version |

---

# 3. Core Modules to Include

The expanded report should include the following modules.

## Essential Modules

1. **Preconditioned dynamics theory**
2. **Matched-seed experimental design**
3. **Early convergence and learning-rate caveat**
4. **Parameter-space geometry**
5. **Loss-matched geometry control**
6. **Step-norm / path-length control**
7. **Perturbation flatness**
8. **Linear mode connectivity**
9. **Function-space similarity**
10. **Representation-space geometry**
11. **Generalization and three-level synthesis**
12. **Normalized sharpness and scale-aware caveat**

The most important additions beyond the previous 15-page plan are:

- **step-norm / path-length control**;
- **representation-space geometry**;
- a stronger **theoretical context** section;
- a more complete **three-level synthesis**.

---

# 4. What to Keep from the Current Project

## 4.1 Matched-Seed Design

Keep as a central experimental control.

For each seed:

\[
\theta_0^{\mathrm{SGD}}=\theta_0^{\mathrm{Adam}},
\qquad
\{B_t^{\mathrm{SGD}}\}=\{B_t^{\mathrm{Adam}}\}.
\]

This isolates optimizer-induced effects from initialization and minibatch-order randomness.

## 4.2 MLP and SmallCNN

Keep both architectures.

Interpretation:

- MLP has weaker image-specific inductive bias, allowing optimizer-induced geometry to appear more strongly.
- SmallCNN has convolutional inductive bias, so parameter differences may be compressed at the function or representation level.

## 4.3 Hessian \(\lambda_{\max}\) and Trace

Keep both.

\[
\lambda_{\max}(H)=\max_{\|u\|_2=1}u^\top H u.
\]

\[
\operatorname{tr}(H)=\sum_i \lambda_i(H).
\]

Connect trace to perturbation:

\[
\mathbb{E}_{\delta\sim\mathcal{N}(0,\sigma^2I)}
[\delta^\top H\delta]
=
\sigma^2\operatorname{tr}(H).
\]

## 4.4 Generalization Results

Keep test accuracy and train-test gap, but do not overclaim.

Correct framing:

> Adam changes parameter-space geometry strongly, but this does not automatically translate into a clear generalization advantage.

---

# 5. What to Compress or Remove

Even with 22 pages, the report should remain focused.

## Compress

### Raw Gradient Norm

Keep only as a trajectory diagnostic. It is weaker than path length, update norm, Hessian, perturbation, or function-space metrics.

### Basic Optimizer Tutorial

Do not spend too much space explaining standard Adam/SGD mechanics. Explain only what is needed for the preconditioning interpretation.

### Long Accuracy Curves

Final accuracy and train-test gap can be summarized compactly.

## Move to Appendix if Needed

- extra training curves;
- full learning-rate search details;
- implementation pseudocode;
- additional perturbation curves;
- additional seed-level tables.

---

# 6. Section-by-Section Plan

## 1. Introduction

### Goal

Frame the project as a mathematical study of optimizer-induced solution selection.

Suggested opening:

> In over-parameterized neural networks, many parameter vectors can achieve similar training loss. Optimizers therefore do not only minimize an objective; they also select a solution from a large low-loss set. This project studies whether Adam’s adaptive diagonal preconditioning changes this selection process relative to SGD with momentum.

### Research Question

> Does adaptive diagonal preconditioning alter the selected solution in parameter space, and do these changes appear in basin structure, function space, representation space, or generalization?

### Contributions

List 5 contributions:

1. We formulate SGD and Adam as different preconditioned stochastic dynamics.
2. We use matched seeds to isolate optimizer-induced effects.
3. We analyze parameter-space geometry using distance, path length, Hessian curvature, inter-seed dispersion, and loss-matched controls.
4. We test geometric interpretation using perturbation flatness and linear mode connectivity.
5. We compare final models in function and representation space using prediction disagreement, symmetric KL, logit cosine similarity, and feature-kernel alignment.

---

# 2. Mathematical Framing

This is the theoretical backbone of the report.

## 2.1 Optimizers as Preconditioned Dynamics

Unified update:

\[
\theta_{t+1}
=
\theta_t
-
P_t g_t.
\]

For SGD:

\[
P_t^{\mathrm{SGD}}=\eta I.
\]

For SGD with momentum:

\[
u_t=\mu u_{t-1}+g_t.
\]

Expanding:

\[
u_t
=
\sum_{k=0}^{t}\mu^{t-k}g_k.
\]

Thus momentum provides temporal smoothing but not coordinate-wise spatial preconditioning.

For Adam:

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\]

\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^{\odot 2}.
\]

Bias-corrected estimates:

\[
\hat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}.
\]

Adam update:

\[
\theta_{t+1}
=
\theta_t
-
\alpha
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
\]

Equivalent preconditioned form:

\[
\theta_{t+1}
=
\theta_t
-
\alpha D_t\hat m_t,
\]

where

\[
D_t=
\operatorname{diag}
\left(
\frac{1}{\sqrt{\hat v_{t,i}}+\epsilon}
\right).
\]

Key statement:

> SGD uses scalar spatial preconditioning, while Adam uses time-varying diagonal preconditioning.

## 2.2 Local Quadratic Error Dynamics

Local quadratic approximation:

\[
\hat L(\theta)
\approx
\hat L(\theta^\star)
+
\frac12(\theta-\theta^\star)^\top H(\theta^\star)(\theta-\theta^\star).
\]

Let:

\[
e_t=\theta_t-\theta^\star.
\]

For preconditioned gradient descent:

\[
\theta_{t+1}=\theta_t-P_t\nabla \hat L(\theta_t),
\]

the local error dynamics become:

\[
e_{t+1}
\approx
(I-P_tH)e_t.
\]

For SGD:

\[
e_{t+1}\approx(I-\eta H)e_t.
\]

For Adam-like diagonal preconditioning:

\[
e_{t+1}\approx(I-\alpha D_tH)e_t.
\]

This is the key mathematical equation.

## 2.3 Eigen-Direction Interpretation

If:

\[
H=Q\Lambda Q^\top,
\]

then under scalar SGD:

\[
\tilde e_{t+1,i}
=
(1-\eta\lambda_i)\tilde e_{t,i}.
\]

All directions are controlled by one scalar learning rate.

For Adam:

\[
I-\alpha D_tH
\]

is generally not diagonal in the eigenbasis of \(H\). The effective preconditioned curvature can be related to:

\[
D_t^{1/2}HD_t^{1/2}.
\]

Interpretation:

> Adam changes the effective metric of optimization, so it can alter the path and the selected low-loss solution.

## 2.4 Implicit Bias as Solution Selection

Define low-loss set:

\[
S_\epsilon=
\{\theta:\hat L(\theta)\leq \epsilon\}.
\]

Define optimizer-induced selection map:

\[
\theta^\star_{\mathcal A}
=
A_{\mathcal A}(\theta_0,\{B_t\},\text{hyperparameters}).
\]

Matched-seed comparison:

\[
\theta^\star_{\mathrm{SGD}}(\xi),
\qquad
\theta^\star_{\mathrm{Adam}}(\xi).
\]

The project compares two algorithm-induced selection maps under identical randomness.

## 2.5 Hessian Geometry and Perturbation Interpretation

At a trained solution:

\[
H=\nabla^2\hat L(\theta^\star).
\]

Worst-case curvature:

\[
\lambda_{\max}(H)=\max_{\|u\|_2=1}u^\top H u.
\]

Aggregate curvature:

\[
\operatorname{tr}(H)=\sum_i\lambda_i(H).
\]

For perturbation:

\[
\Delta\hat L(\delta)
=
\hat L(\theta^\star+\delta)-\hat L(\theta^\star)
\approx
\frac12\delta^\top H\delta.
\]

If \(\delta\sim\mathcal{N}(0,\sigma^2I)\), then:

\[
\mathbb{E}[\Delta\hat L(\delta)]
\approx
\frac{\sigma^2}{2}\operatorname{tr}(H).
\]

This directly motivates perturbation flatness.

---

# 3. Related Concepts and Theoretical Context

This section is optional in a shorter version, but useful in the 22-page version.

## 3.1 Flat Minima and Sharpness

Explain briefly:

- flatness is often associated with robustness and generalization;
- Euclidean sharpness is not invariant to reparameterization;
- therefore sharpness should be treated as a within-protocol diagnostic.

## 3.2 Linear Mode Connectivity

Mode connectivity asks whether two solutions are connected by a low-loss path.

For two final solutions \(\theta_A\) and \(\theta_B\), linear interpolation:

\[
\theta(\lambda)=(1-\lambda)\theta_A+\lambda\theta_B.
\]

If \(\hat L(\theta(\lambda))\) remains low, then Euclidean distance does not imply a loss barrier.

## 3.3 Function and Representation Geometry

Parameter-space geometry may not determine the learned function.

Function-level metrics:

\[
D_{\mathrm{pred}},
\quad
D_{\mathrm{SKL}},
\quad
C_{\mathrm{logit}}.
\]

Representation-level metrics compare hidden features:

\[
h_\theta(x).
\]

This sets up the later representation-space section.

---

# 4. Experimental Design

## 4.1 Dataset

Fashion-MNIST.

Reasonable because it is:

- simple enough for repeated controlled experiments;
- nontrivial enough to show optimizer and architecture effects;
- computationally feasible for Hessian and perturbation diagnostics.

## 4.2 Architectures

1. Batch-normalized MLP.
2. SmallCNN.

MLP tests optimizer-induced effects under weaker spatial prior.  
SmallCNN tests whether convolutional inductive bias mediates those effects.

## 4.3 Optimizers

SGD with momentum and Adam.

Keep optimizer hyperparameters fixed for main matched-seed experiments.

## 4.4 Matched-Seed Protocol

For each seed:

- same initialization;
- same minibatch order;
- same architecture;
- same training data;
- same number of epochs.

## 4.5 Metrics

Organize metrics into four groups.

### Parameter-space metrics

\[
\|\theta_T-\theta_0\|_2,
\quad
\mathcal P_T,
\quad
R_T,
\quad
\lambda_{\max}(H),
\quad
\operatorname{tr}(H),
\quad
\bar d_{\mathrm{seed}}.
\]

### Control metrics

\[
\hat L(\theta_{\mathrm{SGD},t^\star})
\approx
\hat L(\theta_{\mathrm{Adam},T}).
\]

\[
\|\Delta\theta_t\|_2.
\]

### Basin metrics

\[
\hat L((1-\lambda)\theta_S+\lambda\theta_A).
\]

### Function and representation metrics

\[
D_{\mathrm{pred}},
\quad
D_{\mathrm{SKL}},
\quad
C_{\mathrm{logit}},
\quad
A(K_S,K_A).
\]

---

# 5. Early Convergence and Learning-Rate Caveat

## 5.1 Training Loss Curves

Show Adam’s early convergence advantage over default SGD.

## 5.2 Threshold Crossing

For thresholds \(\tau\), compute:

\[
T_\tau
=
\min\{t:\hat L(\theta_t)\leq\tau\}.
\]

Compare Adam and SGD.

## 5.3 Learning-Rate Caveat

Keep the existing learning-rate caveat:

> Adam clearly beats default SGD, but tuned SGD can narrow the early convergence gap.

This prevents overclaiming.

Transition:

> Since convergence speed alone does not establish implicit bias, we next examine the geometry of the selected solutions.

---

# 6. Parameter-Space Geometry

## 6.1 Distance from Initialization

\[
d_{\mathrm{init}}=\|\theta_T-\theta_0\|_2.
\]

Interpretation:

> Adam moving farther suggests a different parameter-space trajectory, but this is not yet a function-space statement.

## 6.2 Cumulative Path Length and Directness Ratio

New high-quality metric.

Cumulative path length:

\[
\mathcal P_T
=
\sum_{t=0}^{T-1}
\|\theta_{t+1}-\theta_t\|_2.
\]

Directness ratio:

\[
R_T=
\frac{\|\theta_T-\theta_0\|_2}{\mathcal P_T}.
\]

Interpretation:

- \(d_{\mathrm{init}}\): net displacement.
- \(\mathcal P_T\): total movement.
- \(R_T\): how direct the trajectory is.

This distinguishes “Adam moves farther because it takes larger steps” from “Adam follows a more directed drift.”

## 6.3 Update Norm Profile

Record per-step update norm:

\[
\|\Delta \theta_t\|_2=\|\theta_{t+1}-\theta_t\|_2.
\]

This is stronger than raw gradient norm because it measures the actual parameter update.

Interpretation:

> Comparing update norms helps determine whether Adam’s distance advantage is mostly step-size-driven or trajectory-direction-driven.

## 6.4 Hessian Curvature

Report:

\[
\lambda_{\max}(H),
\qquad
\operatorname{tr}(H).
\]

Main interpretation:

> In the MLP, Adam may be much flatter by Euclidean Hessian proxies. In the CNN, curvature differences may be mixed, suggesting that architecture mediates optimizer-induced geometry.

## 6.5 Inter-Seed Dispersion

\[
\bar d
=
\frac{1}{\binom{S}{2}}
\sum_{i<j}
\|\theta_i^\star-\theta_j^\star\|_2.
\]

Interpretation:

> Higher dispersion means the optimizer produces a broader distribution over selected parameters.

---

# 7. Loss-Matched and Step-Norm Controls

This is a major A+ section because it addresses fairness.

## 7.1 Loss-Matched Geometry Control

### Motivation

Fixed-epoch comparisons may be confounded by different training losses.

### Method

For each seed, take Adam final checkpoint:

\[
\theta_A=\theta_{\mathrm{Adam},T}.
\]

Choose SGD checkpoint:

\[
t^\star=
\arg\min_t
\left|
\hat L(\theta_{\mathrm{SGD},t})
-
\hat L(\theta_A)
\right|.
\]

Compare:

\[
\theta_{\mathrm{SGD},t^\star}
\quad
\text{vs.}
\quad
\theta_{\mathrm{Adam},T}.
\]

Metrics:

- distance from initialization;
- Hessian \(\lambda_{\max}\);
- trace;
- function-space disagreement if feasible.

Interpretation:

> If Adam remains geometrically different under loss matching, the difference is not merely a final-loss artifact.

## 7.2 Update-Norm / Path-Length Control

Question:

> Does Adam move farther simply because its updates are larger?

Compute:

\[
\overline{\|\Delta\theta\|}
=
\frac{1}{T}
\sum_{t=0}^{T-1}
\|\theta_{t+1}-\theta_t\|_2.
\]

And cumulative path length:

\[
\mathcal P_T=
\sum_t \|\Delta\theta_t\|_2.
\]

Compare with straight-line distance:

\[
\|\theta_T-\theta_0\|_2.
\]

Interpretation:

- If Adam’s path length is much larger, movement may partly reflect larger effective steps.
- If Adam’s directness ratio differs, the trajectory geometry itself differs.
- If geometry differences persist after accounting for path length, this supports true optimizer-induced selection effects.

## 7.3 Optional: Loss-Matched Perturbation

If time permits, evaluate perturbation flatness at loss-matched checkpoints as well.

This is strong but optional.

---

# 8. Perturbation Flatness

## 8.1 Theory

For small perturbations:

\[
\Delta\hat L(\delta)
\approx
\frac12\delta^\top H\delta.
\]

For isotropic Gaussian perturbations:

\[
\mathbb{E}[\Delta\hat L]
\approx
\frac{\sigma^2}{2}\operatorname{tr}(H).
\]

## 8.2 Relative Global Perturbation

Use:

\[
\delta=
\sigma
\frac{\|\theta^\star\|_2}{\|\xi\|_2}
\xi,
\qquad
\xi\sim\mathcal{N}(0,I).
\]

Evaluate:

\[
\Delta\hat L(\sigma)
=
\hat L(\theta^\star+\delta)-\hat L(\theta^\star).
\]

Recommended:

\[
\sigma\in
\{10^{-4},3\times10^{-4},10^{-3},3\times10^{-3},10^{-2}\}.
\]

Use \(K=10\) or \(20\) perturbations per \(\sigma\).

## 8.3 Layer-Normalized Perturbation

Since global Euclidean perturbation is scale-sensitive, add a layer-normalized variant if feasible.

For layer \(l\):

\[
\delta_l=
\sigma
\frac{\|W_l\|_F}{\|\xi_l\|_F}
\xi_l,
\qquad
\xi_l\sim\mathcal{N}(0,I).
\]

This is not fully invariant, but it is more scale-aware.

## 8.4 Interpretation

> Perturbation flatness checks whether Hessian trace corresponds to actual loss robustness under parameter noise. Layer-normalized perturbation provides a scale-aware robustness check.

---

# 9. Linear Mode Connectivity

## 9.1 Motivation

Parameter distance does not reveal whether two solutions are separated by a loss barrier.

## 9.2 Method

For matched seed:

\[
\theta(\lambda)
=
(1-\lambda)\theta_{\mathrm{SGD}}
+
\lambda\theta_{\mathrm{Adam}},
\qquad
\lambda\in[0,1].
\]

Evaluate:

\[
\hat L(\theta(\lambda)).
\]

Use:

\[
\lambda\in\{0,0.1,0.2,\ldots,1.0\}.
\]

## 9.3 Loss Barrier Metric

Define:

\[
B=
\max_{\lambda\in[0,1]}
\hat L(\theta(\lambda))
-
\max\{\hat L(\theta_{\mathrm{SGD}}),\hat L(\theta_{\mathrm{Adam}})\}.
\]

Interpretation:

- \(B\approx 0\): connected low-loss path along linear interpolation.
- \(B>0\): interpolation crosses a higher-loss barrier.

## 9.4 BatchNorm Caveat

For BatchNorm models:

- use eval mode consistently; or
- recalibrate BN statistics along interpolation path if feasible.

Mention this caveat.

---

# 10. Function-Space Similarity

## 10.1 Motivation

Different parameter vectors can represent similar functions.

For each matched seed, evaluate SGD and Adam on the same test set.

## 10.2 Prediction Disagreement

\[
D_{\mathrm{pred}}^{(s)}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}
[
\arg\max_k p_S^{(s)}(k|x_i)
\ne
\arg\max_k p_A^{(s)}(k|x_i)
].
\]

## 10.3 Symmetric KL Divergence

\[
D_{\mathrm{SKL}}^{(s)}
=
\frac{1}{2N}
\sum_i
\left[
D_{\mathrm{KL}}(p_S\Vert p_A)
+
D_{\mathrm{KL}}(p_A\Vert p_S)
\right].
\]

## 10.4 Logit Cosine Similarity

\[
C_{\mathrm{logit}}^{(s)}
=
\frac{1}{N}
\sum_i
\frac{z_S(x_i)^\top z_A(x_i)}
{\|z_S(x_i)\|_2\|z_A(x_i)\|_2}.
\]

## 10.5 Interpretation

Possible outcomes:

- Parameter distance large, function discrepancy small:
  > parameter-space difference does not necessarily imply function-space difference.
- MLP discrepancy higher than CNN:
  > architecture mediates functional impact of optimizer.
- High discrepancy but similar accuracy:
  > accuracy is too coarse to capture optimizer-induced functional differences.

---

# 11. Representation-Space Geometry

This is a strong additional module for the 22-page version.

## 11.1 Motivation

Function-space metrics compare outputs, but two models can have similar outputs while using different internal representations. To deepen the analysis, compare hidden features.

Let:

\[
h_\theta(x)
\]

be the feature vector from the penultimate layer.

## 11.2 Feature Kernel Alignment

For a subset of \(n\) test samples, define normalized feature Gram matrix:

\[
K_\theta(i,j)
=
\frac{
h_\theta(x_i)^\top h_\theta(x_j)
}{
\|h_\theta(x_i)\|_2\|h_\theta(x_j)\|_2
}.
\]

For matched SGD and Adam models, compute alignment:

\[
A(K_S,K_A)
=
\frac{
\langle K_S,K_A\rangle_F
}{
\|K_S\|_F\|K_A\|_F
}.
\]

Interpretation:

- \(A\approx 1\): similar representation geometry.
- Lower \(A\): learned representations differ.

This connects to kernel and representation geometry, which is highly relevant to mathematical deep learning.

## 11.3 Optional: Linear CKA

If feasible, compute centered kernel alignment:

\[
\mathrm{CKA}(K_S,K_A)
=
\frac{
\langle HK_SH,HK_AH\rangle_F
}{
\|HK_SH\|_F\|HK_AH\|_F
},
\]

where:

\[
H=I-\frac{1}{n}\mathbf{1}\mathbf{1}^\top.
\]

CKA is more robust to isotropic scaling of representations.

If time is limited, use feature kernel alignment only.

## 11.4 Interpretation

This module answers:

> Do SGD and Adam learn similar internal representations even when their parameters differ?

Possible outcomes:

- CNN feature alignment high:
  > convolutional architecture stabilizes representation geometry.
- MLP alignment lower:
  > optimizer-induced differences appear more strongly in representation space.
- Alignment high but parameter distance high:
  > different parameterizations can implement similar feature geometry.

This is a major depth upgrade.

---

# 12. Generalization and Three-Level Synthesis

## 12.1 Accuracy and Train-Test Gap

Report:

\[
\text{test accuracy},
\qquad
\text{train-test gap}.
\]

Do not overclaim small differences.

Main interpretation:

> Geometry differences do not necessarily translate into clear generalization differences.

## 12.2 Three Levels of Geometry

Organize final synthesis around three levels.

### Level 1: Parameter Geometry

Metrics:

\[
\|\theta_T-\theta_0\|_2,
\quad
\mathcal P_T,
\quad
\lambda_{\max}(H),
\quad
\operatorname{tr}(H),
\quad
\Delta\hat L(\sigma).
\]

Question:

> Where does the optimizer go, and what is local curvature?

### Level 2: Basin Geometry

Metrics:

\[
B,
\quad
\hat L(\theta(\lambda)).
\]

Question:

> Are the solutions connected by a low-loss path?

### Level 3: Function and Representation Geometry

Metrics:

\[
D_{\mathrm{pred}},
\quad
D_{\mathrm{SKL}},
\quad
C_{\mathrm{logit}},
\quad
A(K_S,K_A).
\]

Question:

> Do parameter differences correspond to different predictors or representations?

## 12.3 Core Synthesis

A strong final synthesis:

> Adam can strongly change parameter-space trajectory and local Euclidean geometry. However, generalization depends more directly on the learned function and representation than on raw parameter distance or Euclidean sharpness. The MLP and CNN comparison shows that architecture can mediate how optimizer-induced geometry appears at the function and representation levels.

---

# 13. Limitations and Normalized Sharpness Caveat

## 13.1 ReLU Scale Symmetry

For a two-layer ReLU network:

\[
f(x)=W_2\sigma(W_1x).
\]

Since:

\[
\sigma(cW_1x)=c\sigma(W_1x),
\]

we have:

\[
W_2\sigma(W_1x)
=
(c^{-1}W_2)\sigma(cW_1x).
\]

Thus different parameter vectors can represent the same function.

## 13.2 Consequence for Sharpness

Euclidean Hessian sharpness:

\[
\lambda_{\max}(H_\theta)
\]

is not invariant under reparameterization.

Therefore:

> Hessian sharpness and perturbation flatness should be interpreted as controlled within-architecture diagnostics, not universal measures of function complexity.

## 13.3 Scope Limitations

Mention:

- Fashion-MNIST only;
- two architectures only;
- five seeds;
- finite hyperparameter search;
- Hessian estimates are proxies;
- mode connectivity with BatchNorm has caveats.

The limitations section should be honest but not self-undermining.

---

# 14. Conclusion

Final message:

> Adam is not simply a faster version of SGD. It can be understood as a stochastic dynamic with a time-varying diagonal preconditioner, which changes the effective geometry of optimization and can select different regions of parameter space. Matched-seed, loss-matched, perturbation, mode-connectivity, function-space, and representation-space analyses show that optimizer-induced parameter geometry is real but not sufficient to predict generalization. Generalization is mediated by loss level, basin connectivity, architecture, learned functions, and learned representations.

---

# 15. Final Figure and Table Plan

To stay under 25 pages, use a small number of high-value figures and tables.

## Main Tables

1. **Table 1: Training protocol and metrics**
2. **Table 2: Threshold crossing**
3. **Table 3: Parameter-space geometry**
   - distance, path length, directness ratio, inter-seed dispersion
4. **Table 4: Hessian curvature**
   - \(\lambda_{\max}\), trace
5. **Table 5: Loss-matched geometry**
6. **Table 6: Function-space similarity**
7. **Table 7: Representation-space alignment**
8. **Table 8: Generalization summary**
   - train/test accuracy, gap

## Main Figures

1. **Figure 1: Training loss curves**
2. **Figure 2: Distance from initialization or path length**
3. **Figure 3: Perturbation flatness**
4. **Figure 4: Linear mode connectivity**
5. **Figure 5: Optional representation alignment visualization**

Do not include too many overlapping plots.

---

# 16. Priority Ranking If Time Is Limited

If there is not enough time to implement everything, prioritize in this order:

## Highest Priority

1. Preconditioned dynamics theory
2. Loss-matched geometry
3. Function-space similarity
4. Perturbation flatness
5. Linear mode connectivity

## Medium Priority

6. Step-norm / path-length control
7. Representation-space feature kernel alignment
8. Layer-normalized perturbation

## Lower Priority

9. Extra learning-rate ablation
10. Raw gradient norm plots
11. Additional datasets
12. Additional optimizers

For an A+ report, the best combination is:

> preconditioned dynamics + matched seeds + loss-matched control + perturbation flatness + mode connectivity + function-space similarity + representation-space alignment.

---

# 17. What Makes This A+ Level

The final project will be strong because it does not stop at one level of evidence.

## Level 1: Optimization Dynamics

\[
\theta_{t+1}=\theta_t-P_tg_t.
\]

## Level 2: Local Geometry

\[
e_{t+1}=(I-P_tH)e_t.
\]

## Level 3: Parameter-Space Evidence

Distance, path length, Hessian, dispersion.

## Level 4: Control Evidence

Loss-matched comparison and update-norm analysis.

## Level 5: Operational Geometry

Perturbation flatness and mode connectivity.

## Level 6: Function and Representation Evidence

Prediction disagreement, symmetric KL, logit cosine, feature kernel alignment.

## Level 7: Generalization Interpretation

Accuracy and train-test gap interpreted through the three geometry levels.

This creates a complete and rigorous project story.

---

# 18. Final Recommended Abstract

Possible abstract for the final report:

> Adaptive optimizers are often viewed as tools for faster convergence, but in over-parameterized neural networks they may also change which low-loss solution is selected. We study this question by comparing SGD with momentum and Adam as two preconditioned stochastic dynamics under a matched-seed protocol on Fashion-MNIST. Mathematically, SGD applies scalar spatial preconditioning, while Adam applies a time-varying diagonal preconditioner. A local quadratic analysis shows that this changes the effective error dynamics through \(e_{t+1}\approx(I-P_tH)e_t\). Empirically, Adam moves farther from initialization and selects more dispersed parameter regions. We then test whether these differences persist under loss-matched controls, correspond to Hessian and perturbation flatness, imply basin separation through linear mode connectivity, and appear in function and representation space. The results show that optimizer-induced parameter geometry can be substantial, but it does not by itself determine generalization. Generalization is mediated by loss level, architecture, basin connectivity, learned functions, and learned representations.
