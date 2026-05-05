# Fix Plan — v2 报告问题清单与修复方案

针对当前 `final_project_report_v2.pdf`（35 页）的诚实评审，按"必须修 / 应该修 / 可选修"三档列出每个问题、修复路径、预计工作量和验证方法。所有路径相对于 `D:\yuchangyuan\Documents\6699 final project all parts\`。

---

## 优先级总览

| 档 | 问题 | 修后 | 工作量 |
|---|---|---|---|
| **P0 必须** | 无 same-optimizer seed-pair baseline | 把"是否优化器效应"这个核心命题回答掉 | 3–4 小时（含 0.5h 编码 + 0.5h 跑 + 2h 写作） |
| **P1 应该** | Hessian 噪声未声明 | 反转结论可信度提高 | 0.5h |
| **P1 应该** | 扰动小 σ 是噪声地板 | 不再过度解读 slope-2 预言 | 0.5h |
| **P1 应该** | 规模限制不在 abstract / scope 段 | 防止读者把结论外推 | 0.5h |
| **P2 可选** | §2 Background 有冗余 | 6 页 → 4–5 页 | 1h |
| **P2 可选** | flatness 反转的口吻偏强 | 单架构对、N=5 的克制版 | 0.2h |

总修复时间：**P0+P1 约 5h，全做约 7h。**

---

## P0：补 Same-Optimizer Seed-Pair Baseline ★★★

### 为什么这是必须修的

当前所有 cross-optimizer 指标（`D_pred = 0.07`、`CKA = 0.86`–`0.91`、mode-connectivity `B_train = 0.07`）都是 SGD-vs-Adam 在**配对相同 seed** 下的结果。读者立刻会问：

> "两个不同 seed 的 SGD 之间这些指标值是多少？"

如果同优化器 seed-pair 也有 `D_pred ≈ 0.05`、`CKA ≈ 0.88`、`B_train ≈ 0.06`，那 cross-optimizer 的差异就主要来自 seed 噪声而不是优化器。这是 implicit bias 论证里的最低门槛 falsification test，不补这个，§VII–IX 三节的论证都站不稳。

### 数据现状

5 seeds × 2 optimizers × 2 archs = 20 runs。已存 `_state.pt`。

可用的对比集：

| 对比类型 | 数量 | 当前是否计算 |
|---|---:|---|
| Cross-opt, matched seed (SGD-vs-Adam, same seed) | 5 / arch | ✅ 已算 |
| Same-opt, different seed (SGD-vs-SGD or Adam-vs-Adam) | C(5,2)=10 / (arch, opt) | ❌ 未算（需补） |
| Cross-opt, unmatched seed | 20 / arch | ❌ 未算（可选） |

**最小补做**：each (arch, opt) 的 10 个 same-opt pair。

### 需要补算的指标

对每个 same-opt pair (i, j)，三个评估都重跑：

1. **Function-space**: $D_{\text{pred}}^{S_i,S_j}$, $D_{\text{SKL}}^{S_i,S_j}$, $C_{\text{logit}}^{S_i,S_j}$
2. **Representation CKA**: $A^{S_i,S_j}$, $\text{CKA}^{S_i,S_j}$
3. **Mode connectivity**: $B_{\text{train}}^{S_i,S_j}$, $B_{\text{test}}^{S_i,S_j}$

### 代码改动

#### 新文件：`part2 & part3/baseline_same_optimizer.py`

```python
"""
Same-optimizer seed-pair baseline for §VII-IX.
For each (arch, opt), enumerate C(5,2)=10 seed pairs and compute:
  - D_pred, D_SKL, C_logit (same as function_space.py)
  - feature kernel alignment A, linear CKA (same as representation_cka.py)
  - linear mode-connectivity barrier B_train, B_test (same as mode_connectivity.py)

Output: results_part2/baseline_same_optimizer.json with structure
{
  "function_space": { "MLP_SGD": {"D_pred_mean", "D_pred_std", ...},
                      "MLP_Adam": {...}, ... },
  "cka":            { "MLP_SGD": {"A_mean", "A_std", "CKA_mean", "CKA_std"}, ... },
  "mode_conn":      { "MLP_SGD": {"barrier_train_mean", ...}, ... }
}
"""
```

实现细节：
- 复用 `function_space.py`、`representation_cka.py`、`mode_connectivity.py` 里的核心函数（提取出 `compute_function_space_pair(model_a, model_b, loader)` 等纯函数，让 baseline 脚本和原脚本都调用）。
- 对 mode-connectivity，BN recalib 仍然对内部 λ 做（保持和 cross-opt 同样的 protocol）。
- 同 (arch, opt) 的 10 个 pair 跑完后，聚合为均值 ± std。

#### 跑时估算

| 模块 | per-pair 时间 | pair 数 | 总时间 |
|---|---|---:|---|
| Function-space | ~3s | 40 (2 archs × 2 opts × 10) | 2 min |
| CKA | ~5s | 40 | 3 min |
| Mode-connectivity（含 BN recalib，11 λ）| ~30s | 40 | **20 min** |

总 ~25 min on 4060 GPU。

### 报告改动

#### 在 §VII–IX 各加一行 baseline 列

例如 §VIII Table 4 改成：

| Model | Comparison | $D_{\text{pred}}$ | $D_{\text{SKL}}$ | $C_{\text{logit}}$ |
|---|---|---|---|---|
| MLP | SGD-vs-Adam (matched) | 0.0738 ± 0.0028 | 0.1749 ± 0.0051 | 0.6122 ± 0.0058 |
| MLP | SGD-vs-SGD (cross-seed) | TBD | TBD | TBD |
| MLP | Adam-vs-Adam (cross-seed) | TBD | TBD | TBD |
| SmallCNN | SGD-vs-Adam (matched) | 0.0671 ± 0.0042 | 0.2676 ± 0.0219 | 0.6516 ± 0.0061 |
| SmallCNN | SGD-vs-SGD (cross-seed) | TBD | TBD | TBD |
| SmallCNN | Adam-vs-Adam (cross-seed) | TBD | TBD | TBD |

CKA Table 5 和 mode-connectivity Table 3 同样加 2 行 baseline。

#### 在每节 Results 段后加判定句

模板：
> "Same-optimizer cross-seed baseline is $D_{\text{pred}}^{\text{SS}} = X \pm Y$, well below the cross-optimizer matched-seed value $0.0738$. The factor of $0.0738 / X \approx Z$ indicates that optimizer choice contributes meaningfully on top of seed noise / **does not** contribute meaningfully on top of seed noise."

按 baseline 实测结果填判定。

#### 在 Three-Level Synthesis 和 Conclusion 里补一句

如果 cross-opt 显著高于 same-opt baseline → 论证立稳。
如果差不多 → **必须**改写 conclusion，承认大多数差异是 seed 噪声而非优化器。

### 期望结果（猜测）

我猜 same-opt baseline 大致是：

- `D_pred^{SS} ~ 0.04–0.05`（两个 88% 准确率分类器随机重叠预期）
- `CKA^{SS} ~ 0.92–0.95`（同 opt 不同 seed 通常很高）
- `B_train^{SS} ~ 0.02–0.05`（同 opt 不同 seed 之间也有非零 barrier）

如果是这个量级，cross-opt 信号还是高于 baseline 的（D_pred 0.07 vs 0.05；CKA 0.86 vs 0.93），但 effect size 比目前报告呈现的要小。三级综合 verdict 可能从 "decisions disagree non-trivially" 变成 "decisions disagree slightly more than seed-baseline expected"。

不管哪个结果，都比现在不报 baseline 强。

---

## P1：声明 Hessian 估计噪声

### 问题

`§V Table 2` 里：

- SmallCNN SGD@t* `λ_max = 6.17 ± 12.34`（**std > mean × 2**）
- SmallCNN Adam@T `λ_max = 61.77 ± 26.43`（std/mean ≈ 0.43）
- MLP SGD@t* `λ_max = 37.15 ± 11.65`（std/mean ≈ 0.31）

Power iteration on 神经网络 Hessian + 5 seed 平均，σ 这么大是预期的，但报告里没说。读者第一眼会觉得 "Adam 10× 比 SGD 尖" 的反转结论统计上不靠谱。

### 修法

#### §V 加一段（Table 2 后）

```latex
\paragraph{Estimator Variance.}
The standard deviations on $\lambda_{\max}(H)$ in Table~\ref{tab:loss-matched} are large relative to the means, particularly for the SmallCNN/SGD entry ($6.17 \pm 12.34$). This reflects two sources of noise: power-iteration estimates on a $\sim 10^5$-parameter neural Hessian are sensitive to the random initialization of the iterate, and the value of $\lambda_{\max}$ itself varies across the five matched seeds. We report mean $\pm$ std rather than median to remain comparable with Part~II. The MLP architecture comparison ($\lambda_{\max} = 37.15$ vs $6.86$) is statistically distinguishable at one-sigma despite this noise, but the SmallCNN reversal ($6.17$ vs $61.77$) should be read as a directional finding from a single architecture pair rather than a tight quantitative claim.
```

#### 可选：补充 median + IQR

如果想做得更实诚，对 5 seeds 报 median + IQR（鲁棒统计）：

| Model | Endpoint | $\lambda_{\max}$ median (IQR) |
|---|---|---|
| MLP | SGD @ t* | TBD |
| MLP | Adam @ T | TBD |
| SmallCNN | SGD @ t* | TBD |
| SmallCNN | Adam @ T | TBD |

实现：在 `loss_matched_geometry.py` 末尾的聚合部分加 `np.median` 和 `np.percentile([25,75])`。

### 工作量
0.5h：算 median/IQR + 加段落。

---

## P1：扰动 Flatness 的诚实重写

### 问题

§VI Table 不存在（仅图），但正文声称：

> "The data are consistent with the local quadratic prediction $\Delta\widehat L \propto \sigma^2\operatorname{tr}(H)$ at small $\sigma$."

实际上：

| σ | MLP SGD ΔL̂ | MLP Adam ΔL̂ |
|---|---|---|
| 1e-4 | +0.00000 | -0.00000 |
| 3e-4 | +0.00000 | -0.00000 |
| 1e-3 | +0.00001 | +0.00001 |
| 3e-3 | +0.00001 | +0.00000 |
| 1e-2 | +0.00023 | +0.00009 |

只有 σ=1e-2 一点有信号，其他全是 ±10⁻⁵ 量级噪声。slope-2 预言根本没真正测——只测了一个端点。

### 修法

#### §VI 把"局部二次预言验证"改写成"single-σ 操作性比较"

替换正文这段：

> ~~The data are consistent with the local quadratic prediction $\Delta\widehat L \propto \sigma^2 \operatorname{tr}(H)$ at small $\sigma$. At the largest probed perturbation, $\sigma=10^{-2}$, we obtain on the MLP $\Delta\widehat L = (2.3\pm 1.2)\times 10^{-4}$ for SGD versus $(0.9\pm 0.9)\times 10^{-4}$ for Adam, a $\sim 2.5\times$ gap in the direction predicted by Part~II's trace ratio.~~

改成：

> "Across $\sigma \in \{10^{-4}, 3\!\times\!10^{-4}, 10^{-3}, 3\!\times\!10^{-3}\}$, $\Delta\widehat L$ is at the $\le 10^{-5}$ level for all four (architecture, optimizer) configurations and is statistically indistinguishable from zero given the seed variability. We therefore restrict the comparison to the largest probed magnitude, $\sigma = 10^{-2}$. On the MLP, $\Delta\widehat L = (2.3\pm 1.2)\times 10^{-4}$ for SGD versus $(0.9\pm 0.9)\times 10^{-4}$ for Adam, a $\sim 2.5\times$ gap in the direction predicted by Part~II's trace ratio. On the SmallCNN, the values are $(0.5\pm 0.7)\times 10^{-4}$ for SGD and $(1.6\pm 0.8)\times 10^{-4}$ for Adam, a directional reversal that mirrors the loss-matched Hessian comparison. We do not interpret the small-$\sigma$ data points as a separate test of the quadratic prediction; they primarily establish a noise floor against which the $\sigma=10^{-2}$ contrast is read."

#### Caption 也要改

```latex
\caption{Mean training-loss increase $\Delta\widehat L(\sigma)$ under relative isotropic Gaussian perturbations, averaged over five matched seeds and $K=10$ perturbations per $\sigma$. Values at $\sigma \le 3\times 10^{-3}$ are at the noise floor ($\le 10^{-5}$); the operational comparison is at $\sigma=10^{-2}$.}
```

### 工作量
0.5h：改两段 + 一个 caption。

---

## P1：声明实验规模限制

### 问题

`abstract` 和 §1 没说：

- 数据集只有 Fashion-MNIST（28×28，10 类，60k 训练）
- 模型只有 256-128-10 MLP 和两层 Conv 的 SmallCNN
- 只跑 30 epoch
- 只 5 seed
- 学习率只用每 optimizer 的"标准"取值（虽然 §4.5 做过 lr search 但仅 Part I 用）

读者完全有可能把 "Adam 改变 implicit bias" 这个结论外推到 ResNet/ImageNet，这不是论文实际支持的。

### 修法

#### 在 §1 末尾加一段 "Scope of Claims"

```latex
\subsection{Scope of Claims}

This study uses Fashion-MNIST as the dataset, a two-hidden-layer MLP and a two-block convolutional network as architectures, and 30 training epochs across five matched seeds per (architecture, optimizer) pair. All optimizer comparisons use the architecture's standard learning rate from a small grid search (Part~I); we do not sweep weight decay, batch size, or extended training horizons. The implicit-bias claims in this report are therefore claims about Adam-vs-SGD trajectory and solution geometry \emph{at this scale and protocol}. We do not claim that the same effects, in the same direction or magnitude, transfer to large-scale settings such as ResNets on ImageNet, transformer language models, or training horizons of $10^4$+ steps. The MLP/SmallCNN architecture comparison is a controlled scale-down designed to expose mechanism, not a representative sample of modern deep learning.
```

#### 在 abstract 末尾加一句限定

```latex
% before:
% Our results indicate that ...

% after:
% Our results indicate that, \emph{at this protocol scale}, ...
```

#### 在 §15 Conclusion 重申

最后一段加一句：

> "The architecture-dependent flatness reversal observed at matched loss is the most novel finding but is supported by a single MLP/SmallCNN comparison; it should be regarded as a directional observation worth replicating on larger architectures rather than a robust trend."

### 工作量
0.5h：3 处文本添加。

---

## P2：§2 Background 压缩

### 问题

§2 总 6 页（pp.3–8），含：

- §2.1 SGD and Adam（基础）
- §2.2 Preconditioned Dynamics View
- §2.3 Eigen-Direction Interpretation（**新加**）
- §2.4 Implicit Bias and Generalization
- §2.5 Why Optimizer Choice Can Act as Implicit Regularization
- §2.6 Interpreting Sharpness Carefully
- §2.7 Project Hypotheses

§2.4 和 §2.5 大量重叠：都在论证"为什么优化器不只是收敛速度问题"。其中 §2.5 三段话基本是 §2.4 末段的扩写。

### 修法

#### 合并 §2.4 + §2.5 为 §2.4 "Implicit Bias and Generalization"

把 §2.5 的核心观点（"convergence speed alone is not enough" + "optimizer hyperparameters change effective noise"）压缩成 §2.4 末尾两段。

预计省 0.7–1 页。

#### 不要动的部分

§2.3 eigen-direction 是新加的核心数学，保留。
§2.6 sharpness caveat 和后面 Hessian 用法直接挂钩，保留。
§2.7 hypotheses 短，保留。

### 工作量
1h：合并段落 + 删冗余 + 重新校验衔接。

---

## P2：Flatness 反转的口吻克制

### 问题

当前 §V 末段：

> "The optimizer-induced flatness ordering is therefore architecture-dependent and not a universal property of Adam-versus-SGD; the convolutional architecture changes which preconditioner ends up selecting the flatter dominant direction."

口吻略强。这是单架构对、N=5 seeds、Hessian std/mean ≥ 2 的发现。

### 修法

把 "is therefore architecture-dependent and not a universal property" 改成更弱的措辞：

```latex
The opposite sign of the curvature gap on the SmallCNN, even after loss matching, suggests that the optimizer-induced flatness ordering is not a universal property of Adam-versus-SGD: at least one of these architecture-optimizer combinations produces a different selection. Given the single architecture pair and the high variance of $\lambda_{\max}$ estimates noted above, we read this as a directional finding that motivates replication on a larger architecture sample rather than as an established architectural rule.
```

并在 Three-Level Synthesis Table 13 的 "Verdict" 列也把 "ordering reverses" 改成 "ordering may reverse, single-architecture-pair observation"。

### 工作量
0.2h：两处文本改。

---

## 推荐执行顺序

```
Day 1 (~3 h)
  1. 写 baseline_same_optimizer.py   (0.5h)
  2. 跑（后台 ~25 min，期间做下面）  (0.5h overlap)
  3. P1 三处文本修改：
     - §V Hessian noise paragraph
     - §VI 扰动 honest framing
     - §1 / abstract / §15 scope statement
                                    (1.5h)
  4. baseline 跑完后 → 拿 same-opt 数字    (0.5h)
  5. 把 baseline 结果写进 §VII–IX 的 3 张表 + 判定句  (1h)

Day 2 (~2 h, 可选)
  6. P2 §2 Background 合并            (1h)
  7. P2 flatness 反转 hedging        (0.2h)
  8. 重编译 + 检查 page count + sanity   (0.3h)
  9. 提 commit + push                 (0.2h)
```

最少做 P0 + P1，报告质量从 A 升到 A+。

---

## 验证清单（修完后）

- [ ] `baseline_same_optimizer.json` 存在，三个模块都填了同 opt seed-pair 数字
- [ ] §VIII Table 4 包含 SS/AA baseline 行
- [ ] §IX Table 5 包含 SS/AA baseline 行
- [ ] §VII Table 3 包含 SS/AA baseline 行
- [ ] §VII–IX 每节 Results 段都有"baseline vs cross-opt"判定句
- [ ] §V 加了 "Estimator Variance" 段
- [ ] §VI 改成了 single-σ 操作性比较口吻
- [ ] Abstract、§1、§15 三处都有 scope of claims 限定
- [ ] §2 Background 不超 5 页（如果做了 P2）
- [ ] PDF 重编译，0 warning，0 undefined ref
- [ ] 总页数 ≤ 38（加 baseline 行可能多 0.5–1 页）

---

## 不要做的事

- 不要补 ResNet/CIFAR 实验。规模升级超出课程项目范围，且让现有数据贬值。
- 不要再加新评估模块。当前 6 个新模块已经够用，过度饱和会让读者抓不到主线。
- 不要把 "implicit bias" 从标题里删掉。这是已经做了的研究方向，加 scope 限定就够。
- 不要重训。所有问题都能用现有 20 个 `_state.pt` 解决。

---

## 一句话总结

**P0 是真问题，P1 是声誉问题，P2 是写作美化。** 做完 P0 + P1，这份报告就从"A 但有方法论漏洞"变成"A+ 真正的 controlled implicit-bias study"。
