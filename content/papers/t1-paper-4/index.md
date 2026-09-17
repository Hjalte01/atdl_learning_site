---
title: "Deep double descent: where bigger models and more data hurt"
shortTitle: "Deep double descent"
topic: "topic-1"
order: 4
description: "Trace the fitting threshold through model width, training time, and dataset size—and understand when bigger models or more data can hurt."
status: "ready"
pdf: "/papers/deep-double-descent.pdf"
source: "https://doi.org/10.1088/1742-5468/ac3a74"
---
## 1 · Why can the next larger model be worse?

You train three classifiers on the same noisy dataset. The small model misses some training labels. The medium model finally fits them all—but predicts new examples worse. A much larger model also fits the labels and predicts new examples better. The usual advice to stop increasing complexity after overfitting begins misses that second improvement.

Nakkiran and colleagues study **double descent**: test error can fall, rise around the point where the training procedure barely fits the data, and fall again. Their contribution is an experimental account across architectures and a proposed unifying quantity, **effective model complexity (EMC)**. They do not introduce a new network or prove that every neural network follows this curve.

This guide uses the supplied **32-page Journal of Statistical Mechanics article, published 29 December 2021, article 124003**, an updated version of the ICLR 2020 paper. Its PDF packaging metadata says August 2026; that is not its publication date. The [journal identity](https://doi.org/10.1088/1742-5468/ac3a74) also appears in this [university-hosted copy of the article](https://cseweb.ucsd.edu/classes/sp26/cse291C-b/scaling/Nakkiran.pdf). All original diagrams below are teaching schematics; the result image in §8 is extracted from the supplied paper. Open any figure to enlarge it. No trained model runs here.

[![A schematic test-error curve falls, rises near the fitting threshold, and falls again as effective complexity grows.](/figures/double-descent/problem.svg)](/figures/double-descent/problem.svg)

**Read left to right, keeping the dataset fixed.** The high point is near interpolation: achieving approximately zero training error. This is not necessarily zero cross-entropy, nor the point where parameter count numerically equals sample count. The drawing has no measured vertical scale.

**A reconstruction of the reasoning, not the authors’ private thought process:** if width alone cannot explain why training longer changes generalization, measure what the entire training procedure can fit. Then ask whether changes in width, time, or sample size move a run toward or away from that boundary.

<details><summary>Predict: does a first rise in validation error establish that longer training can never help?</summary>

No. Some large models in the paper improve again after an intermediate peak. That possibility does not guarantee a recovery in your run or eliminate the need for a held-out validation set.

</details>

## 2 · Architecture: a predictor inside an experiment

[![CIFAR images pass through four Conv-BatchNorm-ReLU-Pool stages with widths k, 2k, 4k, and 8k, then a linear classifier.](/figures/double-descent/architecture.svg)](/figures/double-descent/architecture.svg)

**Follow A → B → C → D → E → F**, including the return direction along the bottom. This is the paper’s five-layer CNN from Appendix B.1, one of its three architecture families. For a batch of $B$ CIFAR images, the input tensor is $B\times3\times32\times32$. Each convolution uses a $3\times3$ kernel, stride 1 and padding 1. Channels are $[k,2k,4k,8k]$; pooling factors are $[1,2,2,8]$. The spatial sizes shown are derived from those operations, not additional undocumented layers. The last pooled representation has $8k$ features per image. A fully connected layer produces $C$ logits, where $C=10$ for CIFAR-10 and $C=100$ for CIFAR-100.

The paper also studies width-scaled preactivation ResNet18s with channel groups $[k,2k,4k,8k]$; $k=64$ is its standard-width reference. These residual networks are separate from the simple CNN in the diagram. For translation, it uses a six-layer encoder-decoder Transformer specification with eight attention heads per layer and varies embedding dimension $d_{\mathrm{model}}$, with feed-forward width $d_{\mathrm{ff}}=4d_{\mathrm{model}}$. No undocumented decoding recipe is needed to explain the reported token-prediction metric.

At F, logits become class probabilities. At G, the training objective compares them with the **observed** labels:

$$
p_{ic}=\frac{e^{z_{ic}}}{\sum_{j=1}^{C}e^{z_{ij}}},\qquad
\mathcal L(\theta)=-\frac1B\sum_{i=1}^{B}\log p_{i\tilde y_i}.
$$

Here $z_{ic}$ is class $c$’s logit for image $i$, $\tilde y_i$ is its possibly corrupted training label, and $\theta$ collects trainable weights, biases and normalization parameters. This is teaching notation for the cross-entropy used in §4 and Appendix B.2, not a numbered paper equation. It gives a smooth training signal even when classification accuracy stays unchanged. The SGD update is $\theta_{t+1}=\theta_t-\eta_t\nabla_\theta\mathcal L_t$, with step index $t$, learning rate $\eta_t$, and the current batch loss $\mathcal L_t$. Adam uses a different update with carried optimizer state; it should not be described as this plain SGD rule.

**Why this architecture comparison?** Widening changes fitting ability without replacing the entire task. But optimizer, augmentation and training budget still influence whether a given width can fit it. The study therefore varies those factors too.

## 3 · Training and inference are different paths

[![Training uses observed labels, backpropagation and optimizer updates; inference freezes the predictor and measures held-out predictions.](/figures/double-descent/training.svg)](/figures/double-descent/training.svg)

**Read the upper row as a loop and the lower row as a forward evaluation.** During training, a batch goes through A–F, its labels supply the loss at G, and backpropagation updates the network. Weights and optimizer state carry into the next step; BatchNorm running statistics also evolve. At inference, the trained model and its normalization state are fixed. Class prediction is $\hat y=\arg\max_c p_c$. Held-out labels are used at H to score predictions, never as predictor inputs.

For a model-size sweep, each width gets a separately trained model. For an epoch sweep, one width is held fixed and its checkpoints are compared. For a sample-size sweep, architecture and the specified training procedure are held fixed while the number of sampled examples changes. A fixed number of updates is not a fixed number of epochs when dataset size changes.

The classification setups use batches of 128. A common ResNet setting uses Adam at $10^{-4}$ for 4,000 epochs with random crops and horizontal flips. The standard CNN SGD setup uses an inverse-square-root learning-rate schedule, starting at 0.1, typically for 500,000 updates; individual figures have different budgets. These are reported experimental settings, not recommended universal defaults.

**Label noise is sampled once.** With probability $p$, the correct label is replaced uniformly by one of the other $C-1$ classes; otherwise it stays correct. All augmentations of an image keep that sampled label. Resampling labels every epoch would define a different problem.

Section 4 says Figure 1 evaluates on the noisy distribution, while the other figures use clean test labels. If $e$ is clean classification error and $e_p$ is error under that symmetric corruption, a teaching derivation gives

$$
e_p=(1-e)p+e\left(1-\frac{p}{C-1}\right)
=p+e\left(1-\frac{Cp}{C-1}\right).
$$

For a clean-correct prediction, corruption makes it wrong with probability $p$. For a clean-wrong prediction, corruption makes it correct only when it selects that particular wrong class, with probability $p/(C-1)$. Thus with $C=10$, $p=0.15$ and $e=0.20$, $e_p\approx0.3167$. This affine conversion explains why raw vertical values from Figure 1 should not be compared directly with clean-test plots. It applies to classification error under the stated noise model, not cross-entropy or arbitrary label corruption.

## 4 · Representation close-up: width is not capacity by itself

[![Doubling CNN width k doubles the final pooled feature vector and widens all earlier convolutions, while the output class count stays fixed.](/figures/double-descent/representation.svg)](/figures/double-descent/representation.svg)

**Compare the two rows at box E.** At $k=8$, the pooled vector has 64 entries; at $k=16$, it has 128. The classifier’s output still has $C$ entries. Earlier convolutions also have larger input and output channel counts, so their parameter counts often scale quadratically in $k$. Appendix B.1 reports 1,558,026 parameters for its $k=64$ CNN. Counting final features alone would miss most of the model.

Now hold every tensor shape fixed and train longer. The parameter count remains unchanged, but the optimizer may fit more difficult label patterns. This is why the paper defines the capacity of a **procedure**, rather than treating width or parameter count as a complete explanation.

## 5 · Effective model complexity: how much can this procedure fit?

Before asking whether a procedure generalizes, ask how large a dataset it can typically fit. Definition 2.1 on page 5 states

$$
\operatorname{EMC}_{\mathcal D,\epsilon}(\mathcal T)
=\max\left\{n:\mathbb E_{S\sim\mathcal D^n}
\left[\operatorname{Error}_S(\mathcal T(S))\right]\leq\epsilon\right\}.
$$

$\mathcal D$ is the labeled-data distribution, $S$ a dataset of $n$ examples sampled from it, $\mathcal T$ the full training procedure, and $\mathcal T(S)$ the resulting predictor. $\operatorname{Error}_S$ measures mean training error; $\epsilon>0$ is the tolerated error. The expectation concerns fresh sampled training sets. Randomized training can be evaluated across seeds too, but the displayed definition explicitly writes the expectation over $S$. The authors heuristically use $\epsilon=0.1$ and do not give a principled universal choice. “Near zero” therefore does not mean literal perfect fitting in this formal proxy.

This equation lives **outside the architecture**: repeat A–G on different dataset sizes and score training error at H. It is neither a differentiable loss added to cross-entropy nor a hidden tensor passed between layers.

[![A toy EMC measurement fixes the procedure, averages training error over sampled datasets, and compares the averages with a tolerance.](/figures/double-descent/emc.svg)](/figures/double-descent/emc.svg)

**Read the table as illustrative measurements, not paper results.** For $n=100,200,300,400$, suppose average errors are $0.02,0.06,0.09,0.14$. At $\epsilon=0.1$, the largest passing *tested* size is 300. That coarse grid does not establish the exact maximum: sizes between samples are untested, errors are estimated, and monotonicity has not been proved. The main plots study interpolation behavior; they do not provide an exact EMC oracle for each neural network.

Hypothesis 1 proposes three regimes relative to the actual training-set size $n$: well below $n$, greater EMC helps; near $n$, it may help or hurt; well above $n$, it helps again. **This is an informal hypothesis**, not a proven guarantee. The authors do not precisely define how far “well below” or “well above” must be, or predict the width of the critical interval.

Why might the critical region be fragile? The authors’ interpretation is that a barely fitting procedure has little freedom to accommodate noisy or mismatched labels without damaging useful structure. A larger model has many possible fits and optimization can select a better one. Their phrase about effectively having only one fitting model is intuition, not a neural-network uniqueness theorem. Linear-model theory motivates it; their deep-network evidence does not prove the mechanism.

<details><summary>Predict: do identical parameter counts imply identical EMC?</summary>

No. Changing training duration, optimization, augmentation, or the labeled-data distribution can change fitting ability without changing the number of parameters. EMC is also distinct from a model family’s worst-case ability to fit arbitrary labels.

</details>

## 6 · Trace a noisy example through the same boxes

[![Toy pooled features produce probabilities 0.75 and 0.25; a corrupted class-two label gives loss 1.386 even though clean class-one prediction is correct.](/figures/double-descent/numbers.svg)](/figures/double-descent/numbers.svg)

**This is a two-feature, two-class teaching reduction of E–H, not a trained CIFAR model.** Suppose E produces $h=(1,2)$. At F let the classifier matrix have rows $(\log3,0)$ and $(0,0)$, with zero biases. Then $z=Wh=(\log3,0)$ and softmax gives $p=(0.75,0.25)$.

If the clean label is class 1 but the fixed observed label is class 2, G computes $-\log0.25\approx1.3863$. The derivative at the logits is

$$
\frac{\partial\mathcal L}{\partial z}=p-\operatorname{onehot}(\tilde y)
=(0.75,-0.75),\qquad
\frac{\partial\mathcal L}{\partial W}
=\begin{bmatrix}0.75&1.5\\-0.75&-1.5\end{bmatrix}.
$$

The outer product with $h$ supplies the weight gradient. A plain SGD step of size 0.1 moves the two classifier rows to $(\log3-0.075,-0.15)$ and $(0.075,0.15)$ and the biases to $(-0.075,0.075)$. If we **hold $h$ fixed for this local calculation**, the new logits are $(\log3-0.45,0.45)$ and class 1’s probability falls to about 0.5495. The observed-label loss decreases to about 0.7974, yet the gradient has reduced confidence in the clean-correct class. In the real CNN, gradients also change earlier layers; this is only the classifier’s local contribution.

At H, the original prediction has training classification error 1 against the observed label and clean classification error 0 against the true label. Neither error is the cross-entropy value 1.3863. This example illustrates the conflict introduced by a corrupted label; **one example cannot establish a double-descent curve**. That requires comparing trained procedures and held-out performance across the relevant axis.

## 7 · Three ways to approach the same fitting threshold

[![Small, medium, and large fixed-width models have schematic training-time curves that respectively decrease, form a U, or exhibit a second descent.](/figures/double-descent/epochs.svg)](/figures/double-descent/epochs.svg)

**Each little plot holds width fixed and advances training time.** Section 6 distinguishes small models that remain unable to fit, medium models that only barely fit by the end, and larger models that can pass through the critical regime. The third case can show an initial improvement, an intermediate deterioration, then recovery. No tensor needs to grow during this process. Weights and training state evolve within the same architecture.

The common ResNet comparison in Figure 9 uses CIFAR-10 with 20% label noise, augmentation and Adam at $10^{-4}$. Appendix E.1 also explores Adam, SGD, and SGD with momentum 0.9 under multiple schedules. These results make a single optimizer quirk less plausible, but they do not prove every schedule or task must recover after overfitting.

[![With a fixed toy EMC of 18,000, increasing sample size from 4,000 to 18,000 changes EMC divided by sample size from 4.5 to 1.](/figures/double-descent/samples.svg)](/figures/double-descent/samples.svg)

**Here the procedure is fixed and the denominator changes.** If a toy procedure had EMC 18,000, moving from 4,000 to 18,000 examples would move its capacity-to-data ratio from 4.5 to 1. This arithmetic illustrates a direction of movement only; it does not estimate the translation model’s EMC or predict its perplexity.

More samples generally improve the learning signal, but can also require more fitting capacity. On a model-size plot, the critical peak shifts toward larger models. At a fixed width, that shift can offset the benefit of more data—or temporarily reverse it. This is a claim about a specified learner and budget, not a reason to throw useful data away. An algorithm that can elect to ignore additional data has options that the fixed training procedure in a comparison does not exercise.

## 8 · What the experiments actually establish

[![Source Figure 3 plots translation test loss versus Transformer embedding dimension for 4,000 and 18,000 sentence samples; the larger-data curve is worse in an intermediate width range.](/figures/double-descent/source-figure-3.png)](/figures/double-descent/source-figure-3.png)

**Measured source evidence, not a generated curve.** This is Figure 3 cropped from page 5 of the supplied journal PDF, by Nakkiran et al. The horizontal axis is Transformer embedding dimension. The two series use 4,000 and 18,000 IWSLT’14 German-to-English sentence pairs, a 4.5-fold data increase. Compare curves at the **same** horizontal coordinate: the larger dataset is worse in part of the intermediate-width region, even though it is better elsewhere. The baseline is the smaller-data version of the same architecture family and procedure, not a different state-of-the-art model. No exact point values are inferred from pixels here.

**Metric caveat:** the source axis says “Cross-Entropy Test Loss,” while its caption and Figure 8 describe per-token perplexity. Appendix B.3 defines mean negative log token probability. To keep the evidence faithful, this guide calls the plotted quantity *test loss* and preserves the original labels rather than converting its numbers. For unambiguous natural-log token loss $L$, perplexity is $\exp(L)$; the units cannot be interchanged numerically.

Translation training predicts target tokens conditioned on the source and preceding target tokens. Appendix B.3 factors the model probability as

$$
p_M(y\mid x)=\prod_{i=1}^{|y|}p_M(y_i\mid y_{<i},x),\qquad
L=-\frac1{|S|}\sum_{(x,y,i)\in S}\log p_M(y_i\mid y_{<i},x).
$$

$M$ is the Transformer, $x$ the source sentence, $y$ the target sequence, $y_{<i}$ its prefix, $i$ the predicted position, and $S$ the set of scored token triples. This loss operates at the token-output head, analogous to G; the paper’s training additionally uses 10% label smoothing and no dropout. For two scored tokens with probabilities 0.5 and 0.25, the teaching example gives $L\approx1.0397$ and perplexity $\sqrt8\approx2.8284$. Evaluation of known target prefixes is different from free-running translation quality such as BLEU. The paper also notes that its actual token triples are grouped by sentence rather than independently sampled as in its abstract distribution model.

| Source evidence | Matched setting and comparison | What it supports |
| --- | --- | --- |
| Figures 1–2 | Width-scaled ResNet18, CIFAR-10, 15% fixed label noise, Adam, up to 4,000 epochs | Model-width and training-time slices can both exhibit a peak near fitting. Figure 1 uses noisy-test error. |
| Figures 4 and 19 | ResNet18 on CIFAR-100, including no added label noise; Adam at $10^{-4}$, augmentation, 4,000 epochs | Added random label noise is not necessary. Figure 19 retains model-wise double descent even with optimal early stopping. |
| Figure 9 | ResNet18, CIFAR-10, 20% noise; compare training trajectories across widths | Small, critical and large models can have different time-dependent behavior. |
| Figures 3, 11(b), 23 | IWSLT’14 German-to-English, 4k versus 18k sentences, width-scaled Transformers | More data can worsen final test loss locally; the shown reversal disappears with optimal early stopping. |
| Figures 11(a) and 12 | Width-scaled CNNs, subsampled CIFAR-10 with label noise | Near-critical sample-size comparisons can instead show a plateau; more data need not create a sharp peak. |
| Figures 28–29 | Ensembles of five models versus individual predictors | Reduced error around the critical region is consistent with sensitivity, but does not prove the proposed mechanism. |

**Version-specific inconsistencies:** §4 and Figure 8 specify **80k** Transformer updates; Appendix B.1 prints **80**. This guide follows the main text and figure and records the discrepancy. Appendix B.4 says **500k epochs** for Figure 11(a), whereas the standard CNN recipe uses **500k gradient steps**; that figure’s exact budget is therefore ambiguous in the PDF. The early-stopping appendix heading is categorical, but §8 and Figure 19 explicitly give an exception. The translation-direction shorthand also varies in §2; the dataset specification and Figure 3 identify German-to-English. These ambiguities should be checked against original experiment records before exact replication.

## 9 · What would a convincing follow-up establish?

A useful ablation changes the proposed cause while checking that the fitting boundary moves with the observed peak. Sweep width at several fixed noise rates, track both training error and clean held-out error, and repeat seeds. If the peak tracks the boundary rather than a fixed width, that supports the EMC interpretation. It does not by itself explain how the optimizer selects a good interpolant.

To separate time from capacity effects, report both update count and epochs, optimizer and schedule, augmentation, and checkpoint-selection rule. Compare final checkpoints with a separately validation-selected checkpoint; keep test data for final evaluation. The paper’s “optimal early stopping” curves are a diagnostic comparison, not evidence that an oracle best-test checkpoint can be selected without extra data.

To test the sensitivity story, perturb labels or initialization and measure changes in predictions around and away from the fitting boundary. Figure 28’s five-model ensemble also changes sampled training-label noise across models, so it is not an isolated test of initialization alone. Ensembling is supporting evidence, not a uniqueness or causality proof.

The open questions matter: EMC has no general efficient estimator here, its tolerance and critical interval are heuristic, and larger models or longer runs have compute costs. The hypothesis does not guarantee that an affordable training budget reaches the second descent. The authors did not observe sample-wise harm with optimal early stopping, but explicitly do not prove it impossible.

<details><summary>Predict: does no added random label noise rule out double descent?</summary>

No. The paper reports noiseless examples, including CIFAR-100 and translation. The authors interpret noise as one way of increasing mismatch between the task and the learner, not a necessary ingredient for all double descent.

</details>

## 10 · Idea to carry forward

Ask **“How close is this whole training procedure to barely fitting this dataset?”** before interpreting an unexpected scaling result. Width, optimization time and sample count move that relationship in different ways. Near the boundary, local evidence that “bigger is worse” or “more data is worse” can coexist with improvements farther away. Locate the regime and name the metric before turning one curve into advice.

| Reading goal | Supplied journal source |
| --- | --- |
| Formal EMC and its deliberately informal hypothesis | §2, Definition 2.1 and Hypothesis 1, pp. 5–6 |
| Architecture and losses | §4, Appendix B.1–B.3, pp. 8, 17–19 |
| Model-size effects and sensitivity intuition | §5, Figures 4–8, pp. 8–12 |
| Training-time effects | §6, Figures 9–10; Appendix E.1, Figures 16–18 |
| More-data reversal versus plateau | §7, Figures 3 and 11–12; Appendix D for random features |
| Early stopping and limits | §8; Figures 19 and 23–24; Appendix E |

[Read the exact course PDF](/papers/deep-double-descent.pdf). The eight cards below test the mechanism and its limits.
