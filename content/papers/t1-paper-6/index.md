---
title: "Binarized neural networks converge toward algorithmic simplicity: empirical support for the learning-as-compression hypothesis"
shortTitle: "Binarized networks and compression"
topic: "topic-1"
order: 6
description: "Trace binary-network training, 4×4 weight patterns, and BDM measurements to understand the evidence for learning as compression."
status: "ready"
pdf: "/papers/binarized-networks.pdf"
source: "https://doi.org/10.3389/fncom.2026.1791546"
---
## 1 · Can a network learn a shorter description?

Imagine two networks with the same number of positive and negative weights. One organizes its signs into repeating structures; the other looks irregular. Counting individual signs cannot tell them apart. Would measuring patterns inside their weight matrices reveal more about learning?

Sakabe and colleagues ask whether **Block Decomposition Method (BDM) scores track training loss more closely than block entropy** in small binarized neural networks. Their entropy baseline already counts **4×4 patterns**, not just individual bits. BDM adds an estimated description cost for each distinct pattern. This is a more demanding comparison than structure versus a histogram of signs.

This guide follows the supplied **15-page Frontiers in Computational Neuroscience article, published 29 May 2026**, DOI 10.3389/fncom.2026.1791546. Its identity was checked against the [publisher record](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2026.1791546/full). The local course PDF supplies the methods and numbers below. Its referenced Supplementary Tables S1–S2 are not included, so exact configuration-specific learning rates, batch sizes and patience are not transcribed here.

[![Two matrices with equally frequent block types have identical block entropy but can have different pattern description costs.](/figures/bnn/problem.svg)](/figures/bnn/problem.svg)

**Read across the two dictionaries.** Both have two equally frequent block types, hence one bit of block entropy. Their BDM values need not agree because the actual patterns can have different estimated description costs. This is a conceptual comparison, not a measured pair of networks. All figures open at full size; the final plot redraws reported measurements. No trained model or interactive simulation runs on this page.

**A plausible reasoning path, not the authors’ private thoughts:** choose a network whose signs can be inspected directly; compare two diagnostics using the same partition; measure both during ordinary learning; then destroy the label relationship to test whether the difference persists. The experiment measures a proposed signature of compression rather than optimizing that signature.

<details><summary>Predict: would a falling BDM score alone establish better generalization?</summary>

No. A simpler weight pattern can be unhelpful, and training loss can fall through memorization. The relevant evidence includes the task, held-out performance, preprocessing and random-data controls.

</details>

## 2 · Architecture: one predictor, a separate measuring instrument

[![An MNIST image feeds a binary MLP with hidden widths 32 and 16 and ten outputs; a separate weight-snapshot branch computes BDM and entropy.](/figures/bnn/architecture.svg)](/figures/bnn/architecture.svg)

**A → B → C predicts; D → E → F measures.** In the representative MNIST configuration, resize a 28×28 image to 10×10, normalize using dataset statistics, and flatten to $x\in\mathbb R^{100}$. The MLP has hidden widths 32 and 16 and ten class outputs. With an output-by-input matrix convention, the linear weight shapes are $32\times100$, $16\times32$, and $10\times16$. These are derived shapes, not additional source specifications.

The paper binarizes weights and activations and applies batch normalization after each hidden layer (§4.2). It tests one to four hidden layers. The main text does not fully specify activation/normalization ordering, output-layer binarization exceptions, bias handling or the exact straight-through rule. The diagram therefore stays functional. Table 1 lists 3,872 parameters for (32,16); the three matrices alone total 3,872, so that number should not be treated as a verified count of every possible normalization parameter or bias.

At C, let $z=f_\theta(x)\in\mathbb R^{10}$ be the class logits and $\theta$ the learned parameters. This teaching expansion of the cross-entropy objective named in §4.3 is not a numbered paper equation:

$$
p_c=\frac{e^{z_c}}{\sum_{k=1}^{10}e^{z_k}},\qquad
\mathcal L=-\frac1B\sum_{b=1}^{B}\log p_{b,y_b}.
$$

$B$ is mini-batch size, $b$ indexes examples, $y_b$ is the observed class and $p_{b,y_b}$ its predicted probability. Natural logarithms give loss in nats. Assigning the target probability 0.25 incurs $-\ln(0.25)\approx1.386$; probability 0.5 gives about 0.693. Cross-entropy rewards the observed label, including when that label is corrupted. It does not directly reward low BDM.

At D, extract weight matrices **excluding batch-normalization layers**. At E, map positive weights to 1 and nonpositive weights to 0, partition each matrix into nonoverlapping 4×4 blocks, and count their identities. At F, compute entropy and BDM from that shared decomposition using `pybdm`. No diagnostic gradient returns to B.

**Reproducibility boundary:** §3.1 defines per-matrix calculations but does not clearly specify aggregation across matrices or incomplete edge-block handling. The 10-output matrix is not divisible by four in both dimensions; UCI HAR also has 561 input features. Do not silently assume padding, discarding or overlapping boundaries when reproducing the experiment.

## 3 · Training and inference follow different paths

[![Training uses cross-entropy, straight-through gradients and Adam; inference freezes the selected checkpoint and needs no BDM computation.](/figures/bnn/training.svg)](/figures/bnn/training.svg)

**The upper row repeats; the lower row predicts once.** Adam trains the network using a **straight-through estimator (STE)** through discrete operations. A literal sign function has derivative zero away from its jump, preventing ordinary gradient learning through it. An STE substitutes a surrogate derivative on the backward pass while retaining discrete forward values.

For intuition only, if $q=\operatorname{sign}(w)$ and the upstream gradient is $g$, an identity surrogate would pass $\widetilde{\partial\mathcal L/\partial w}=g$. Other STEs clip or gate it. The supplied main text does not identify the exact surrogate, clipping rule or real-valued weight-update implementation, so this identity is **not a claim about the experimental code**.

Parameter state and Adam moments carry between updates. Validation selects the checkpoint with minimum validation loss. Inference freezes learned state; evaluation uses normalization in evaluation mode rather than updating training statistics. Held-out prediction requires A–C, with no target, optimizer update or complexity measurement. Choosing the largest class logit is the standard classifier interpretation; BDM is not a decoding rule.

For MNIST, 10,000 of the 60,000 original training images form a stratified validation set. From the remaining pool of 50,000, each of 200 runs receives a separately sampled stratified subset of 25,000. Those subsets can overlap each other; they are disjoint from validation. The standard 10,000-image test set is reserved for final accuracy.

<details><summary>Predict: should adding BDM to the training loss reproduce this study?</summary>

No. That would be a new intervention. Cross-entropy trains the network; BDM and entropy observe its weights. Directly differentiating the discrete BDM pipeline is also not ordinary backpropagation.

</details>

## 4 · Representation close-up: frequency and description cost

Distinguish an ideal from its computable approximation. **Kolmogorov complexity** $K(x)$ is the length of the shortest prefix-free program that makes a fixed universal machine output object $x$. It is not generally computable. Paper Eq. 1 relates it to algorithmic probability:

$$
K(x)=-\log_2 m(x)+O(1).
$$

$m(x)$ is a universal algorithmic semimeasure; the additive constant depends on the reference machine, not $x$. Short generative programs make some patterns algorithmically probable. This is a theoretical relationship: the experiment neither searches all programs nor computes exact $K$.

The **Coding Theorem Method (CTM)** estimates a short pattern’s cost from its frequency $D_n(r)$ among outputs of considered halting small Turing machines:

$$
\operatorname{CTM}(r)=-\log_2 D_n(r).
$$

$r$ is a binary pattern and $n$ indexes machine size. This finite lookup approximation differs from universal $m$. A teaching frequency of $1/32$ yields five bits, but we do not claim that frequency for any real block. The experiment uses precomputed costs for 4×4 binary patterns; enumerating larger machine spaces is expensive.

[![Checkerboard and striped four-by-four tiles occur three times and once; entropy is 0.811, while invented CTM costs five and nine yield BDM 15.585.](/figures/bnn/blocks.svg)](/figures/bnn/blocks.svg)

**Read the tiles, then their multiplicities.** Let $r_j$ be a distinct tile, $n_j$ its count, $N=\sum_j n_j$ the number of tiles, and $p_j=n_j/N$ its empirical frequency. At F, paper Eqs. 2–3 become, for a fixed partition:

$$
\operatorname{BDM}(W)=\sum_{j\text{ distinct}}\left[K_m(r_j)+\log_2 n_j\right],
\qquad H(W)=-\sum_j p_j\log_2 p_j.
$$

$W$ is a binarized matrix; $K_m(r_j)$ is its tile’s CTM-based approximate cost from method $m$. Pay each distinct tile’s cost **once**; repetition adds a logarithmic multiplicity term. Entropy instead summarizes the distribution of tile identities. Entropy is in bits per sampled block; BDM is an estimated description score, so their raw magnitudes are not interchangeable. Section 3.1.4 mistakenly points to Eq. 3 for BDM; its definition is Eq. 2.

For four tiles with counts $(3,1)$:

$$
H=-\tfrac34\log_2\tfrac34-\tfrac14\log_2\tfrac14\approx0.811.
$$

With **invented CTM costs** $K_m(r_1)=5$ and $K_m(r_2)=9$:

$$
\operatorname{BDM}=5+\log_2 3+9+\log_2 1\approx15.585.
$$

These are computed teaching values, not actual lookup entries or measured network scores. Four copies of $r_1$ alone give $H=0$ and toy BDM $5+\log_2 4=7$. Zero entropy means a single observed tile identity, not zero information or no weights.

**What the partition loses:** rearranging whole tiles preserves both summaries. A coordinated hidden-neuron permutation can preserve the predictor while changing which weights share a tile. Thus BDM depends on representation and partition; it is not a complete invariant of network function. The paper’s “entropy-like behavior” at large scale should not be read as numerical equality of these formulas, whose units and scaling differ.

## 5 · Trace one example and its weight snapshot

[![The prediction branch assigns probability 0.25 and loss 1.386; the diagnostic branch maps weight signs to bits and collects full tiles.](/figures/bnn/numbers.svg)](/figures/bnn/numbers.svg)

**Follow A–C first, then D–F.** A teaching image with class $y=3$ receives $p_y=0.25$, hence loss 1.386. Its logit gradient is $\partial\mathcal L/\partial z_c=p_c-\mathbf1[c=y]$: at the target logit it is $-0.75$, so gradient descent tends to raise that logit. This standard cross-entropy derivative explains the training arrow, not BDM’s dynamics.

A simultaneous weight snapshot might contain $(-0.2,0,0.7)$; §3.1.1 encodes these as $(0,0,1)$. Three bits illustrate only the sign rule. The full tiles in §4 supply the metric example: $H\approx0.811$ and toy BDM $15.585$.

If $-0.2$ changes to $-0.1$, that bit remains zero. Crossing to $+0.01$ changes it and can replace a tile identity. Loss and BDM need not move smoothly together at each update. Epoch averaging and smoothing materially affect the reported comparison.

<details><summary>Predict: does halving cross-entropy halve BDM?</summary>

No. They operate on different objects through different formulas. Positive correlation along a processed trajectory is not proportionality or a causal update law.

</details>

## 6 · Training over time: what gets correlated?

[![Step measurements are averaged into epochs, trimmed, transformed, correlated and bootstrapped over model runs.](/figures/bnn/timeline.svg)](/figures/bnn/timeline.svg)

**This is an analysis pipeline, not another optimizer.** The source records loss, BDM and entropy every optimization step, evaluates validation loss every 20 steps, and averages measurements per epoch. It removes final patience epochs before early stopping. Runs shorter than five epochs are excluded from correlation analysis, leaving fewer than 200 in some configurations.

Each metric’s trajectory is independently transformed per run: `log1p`, Gaussian smoothing with $\sigma=1$, then min–max scaling to $[0,1]$. A teaching expression is

$$
u_t=\operatorname{Smooth}_{\sigma=1}(\log(1+v_t)),\qquad
\widetilde v_t=\frac{u_t-\min_su_s}{\max_su_s-\min_su_s}.
$$

$v_t$ is the original epoch metric; $u_t$ is its smoothed log value; $s$ ranges over retained epochs. This operates after F on recorded trajectories. A constant series makes the denominator zero and correlation undefined; the main text does not specify handling of that case. Positive affine scaling alone preserves Pearson correlation, but log transformation and smoothing generally do not. Smoothing can also alter ranks.

Pearson $r$ describes linear association; Spearman $\rho$ describes rank association. If loss and complexity both decline, their correlation is **positive**. The toy sequences $(3,2,1)$ and $(9,6,3)$ have $r=\rho=1$ although both decrease. Common trends with time can correlate without one causing the other.

The paper obtains 95% bootstrap intervals by resampling model runs. It describes correlations between mean per-epoch trajectories, but does not fully specify whether each bootstrap aggregates trajectories before correlation or aggregates runwise coefficients. Exact reproduction requires that detail, unequal-length handling and the supplementary/code specification. Figure 2 shows active-run counts and confidence bands only where at least five runs remain: later epochs can represent a different subset of models.

Controls use a fixed **14 epochs**, the mean best-validation epoch of MNIST (32,16) runs, without early stopping. This avoids using uninformative random-label validation loss to define control durations, but differs from the main early-stopped analysis.

## 7 · Evidence: where does the advantage hold?

[![Table 3 Pearson intervals favor BDM on MNIST, Fashion-MNIST and sincos; both UCI HAR variants have overlapping intervals and lower BDM midpoints.](/figures/bnn/results.svg)](/figures/bnn/results.svg)

**Read interval positions, not their lengths as accuracy.** These are Table 3’s **95% bootstrap intervals for Pearson correlation with processed training loss**. Green is block entropy, purple BDM. Both observe the same networks. The plot redraws supplied measurements; it is not a new experiment.

| Dataset and input | Hidden widths | Entropy $r$ interval | BDM $r$ interval | Test accuracy, mean ± SD |
| --- | --- | --- | --- | --- |
| MNIST, resized 10×10 | 32, 16 | [0.74, 0.79] | [0.92, 0.94] | 79.5 ± 1.0% |
| Fashion-MNIST, resized 10×10 | 32, 16 | [0.03, 0.17] | [0.30, 0.46] | 73.1 ± 0.9% |
| sincos, 48 samples, four classes | 64, 32 | [0.36, 0.45] | [0.71, 0.77] | 96.5 ± 0.7% |
| UCI HAR, 561 engineered features | 8, 4 | [0.68, 0.77] | [0.63, 0.74] | 73.7 ± 6.5% |
| UCI HAR raw, 1,152 values | 4, 16 | [0.57, 0.70] | [0.49, 0.64] | 50.9 ± 2.6% |

Accuracy comes from the checkpoint selected by minimum validation loss. It is not a comparison between a “BDM network” and an “entropy network.” Fashion-MNIST shows a relative advantage but weak absolute correlations. Both HAR settings lack a consistent advantage; the raw-signal result weakens the suggestion that engineered features alone explain the exception.

Table 1 varies MNIST widths from (8,4) to (128,64). The reported Pearson midpoint gap shrinks from 0.28 to 0.06. These are rounded differences of interval midpoints, not paired confidence intervals for a metric difference. Table 2 checks one to four hidden layers at similar parameter counts and retains the MNIST advantage. Neither establishes a universal scale threshold.

Controls preserve MNIST inputs but randomly permute example labels, or use random 10×10 inputs with independent random labels. **Permuting labels across examples differs from consistently renaming classes**, which preserves a learnable task. In the fixed-window controls BDM’s relative advantage largely disappears (§5.3, Table 4). This supports dependence on task structure, but is not a calibrated generalization detector.

## 8 · What would stronger evidence require?

The result is an association in **small binarized MLPs**, consistent with learning as compression. It is not an exact Kolmogorov-complexity measurement, complete MDL code length, generalization bound or proof of a data-generating causal mechanism. The authors distinguish empirical support from a formal compression proof in §7.

They offer two possible explanations for the shrinking advantage: fixed CTM blocks may miss structure in larger matrices, or higher capacity may encourage memorization (§6.4.3). The experiment does not resolve them. Algorithmic probability does not by itself turn a loss correlation into a causal test.

Useful **proposed follow-ups**, not reported results:

- Permute hidden neurons and compensate in the next layer. Predictions stay fixed; BDM changes would reveal matrix-layout sensitivity.
- Compare raw, detrended and processed trajectories with documented bootstrapping and matched windows, testing sensitivity to shared trends and preprocessing.
- Vary capacity and quantization separately at fixed task and training budget to distinguish estimator resolution from constraints on memorization.
- Evaluate a BDM-based stopping rule prospectively against validation stopping on untouched test data, testing selection value beyond retrospective correlation.

A BDM regularizer would need a discrete optimization method or validated surrogate. This diagnostic study does not show that adding such a term improves accuracy.

## 9 · Reading map and the idea to carry forward

| Mechanism | Supplied source |
| --- | --- |
| Program length, algorithmic probability, CTM | §§2.1–2.2; Eq. 1 |
| Shared tiles, BDM, block entropy | §2.3 Eqs. 2–3; §3.1 |
| Dimensions, architecture, STE and Adam | §§4.1–4.3; Figure 1 |
| Transformations and bootstrap | §4.4; Figure 2 caption |
| Capacity, depth and dataset evidence | §§5.1–5.2; Tables 1–3 |
| Fixed-window controls | §5.3; Table 4 |
| Limitations and alternative explanation | §§6.4–7 |

**The idea to carry forward:** a network can change in ways a frequency summary misses. Local pattern-description costs can reveal a stronger training signal in some settings. Follow the whole chain—predictor, encoding, partition, estimator and statistical analysis—before interpreting that signal as learning.

Compare [information bottleneck](/papers/t1-paper-1), which measures information in activations, and [grokking](/papers/t1-paper-5), which traces input representations. Here the observed object is the **weight matrix**, and the compression interpretation remains bounded by the estimator and experiment.
