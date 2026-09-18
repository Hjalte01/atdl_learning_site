---
title: "Learning through the lens of Complexity · Lecture 2"
shortTitle: "Algorithmic complexity and learning"
topic: "topic-1"
order: 2
materialKind: "slides"
description: "From shortest programs and two-part codes to bit-plane diagnostics and reusable weight motifs: what complexity measures, and what it changes."
status: "ready"
pdf: "/papers/complexity-lecture.pdf"
---
## The puzzle: same size, different structure

Two networks have the same number of stored weights and the same training accuracy. One weight array repeats a few patterns; the other looks irregular. **Should they cost the same number of bits to describe? And would a shorter description imply better predictions?** These are different questions.

This guide follows Raghavendra Selvan’s **Characterizing Learning in Deep Neural Networks using the lens of Complexity**, the supplied **43-slide September 3, 2026** lecture (`slides/2_T1L2_raghav.pdf`). [Read the course PDF](/papers/complexity-lecture.pdf). Slide numbers below are PDF page numbers. Allow 30–40 minutes; prerequisites are basic probability, binary numbers, gradients and train/validation splits. Start with [Lecture 1](/papers/t1-theories-lecture) for the broader six-paper map.

The lecture progresses from a description-length motivation to computable diagnostics, then to a constrained training algorithm. Colored diagrams and worked numbers below are **teaching constructions**, except the explicitly identified source result slide. No trained model or CTM lookup service runs on this page. We preserve the source’s distinction between a promising research lens and a proved explanation of generalization, and qualify several abbreviated mathematical claims.

## 1 · Short programs and practical codes

[![A repeated twelve-bit string can be described by a motif and repeat count, while a literal description stores every bit; decoder overhead must also be counted.](/figures/complexity-lecture/codes.svg)](/figures/complexity-lecture/codes.svg)

Read the rows as alternative descriptions, not compression measurements. Slide 12 contrasts `100111010110` with `100100100100`. The second has an obvious repetition rule. At twelve bits, program syntax and the decoder may cost more than the saving; neither visual irregularity nor failure of one compressor proves algorithmic randomness.

The ideal is the shortest halting program that prints the object. The source’s Eq. (1) is

$$
C_U(w)=\min_{p:U(p)=w}|p|.
$$

Here $w$ is a finite binary object, $p$ a program, $|p|$ its length in bits, and $U$ a fixed universal prefix-free machine. The subscript emphasizes the reference language. Changing universal machines changes complexity by an additive constant; short-object comparisons can be sensitive to it. Prefix-free complexity is also commonly written $K$; we retain the deck’s $C$.

This equation belongs to the **ideal description box**, not backpropagation. There is no general exact algorithm for it. Running programs in parallel can find progressively shorter descriptions, but cannot certify that no shorter program will eventually halt. Thus complexity is **upper semicomputable** (approachable from above); slide 21’s footnote incorrectly calls this “lower semi-computable.”

Minimum description length (MDL) restricts the code family. Slides 13–15 use

$$
L_{\mathcal M}(w)=\min_{h\in\mathcal M}\{L(h)-\log_2 P(w\mid h)\},\qquad
C_U(w)\leq L_{\mathcal M}(w)+c_{\mathcal M}.
$$

$\mathcal M$ is a chosen model/code family, $L(h)$ describes the selected member, $P(w\mid h)$ supplies its residual code, and $c_{\mathcal M}$ describes the fixed decoding machinery. The inequality requires a valid effective code, with integer-length/decoding overhead accounted for. Even within a computable family, finding the best member need not be efficient. Adding candidate codes can improve the minimum when existing code lengths remain fixed; changing families can also change overhead.

For supervised learning with inputs treated as shared side information, use $L(h)+L(y_{1:n}\mid x_{1:n},h)$. In an illustrative comparison, model A costs 20 model bits plus 80 label bits; model B costs 60 plus 20. Their totals are 100 and 80 bits: the larger model description wins by explaining labels better. Cross-entropy equals an ideal label code only when it is the **summed negative log likelihood in bits**. Mean loss needs multiplication by sample count; natural-log loss needs division by $\ln2$.

<details><summary>Predict: does a small weight file alone establish a short two-part training-data code?</summary><p>No. You must include the decoder, model selection and remaining label uncertainty. A tiny constant predictor can still require a long label code.</p></details>

## 2 · A distribution is not a point with free precision

[![A Gaussian prior is compared with two posterior spreads; concentrating the posterior increases its relative entropy, while approaching the prior makes KL zero.](/figures/complexity-lecture/kl.svg)](/figures/complexity-lecture/kl.svg)

Slides 16–19 replace a precisely specified weight vector with a distribution $Q$ relative to a fixed prior $P$. Read the figure as a **distribution cost**, not the bytes needed to serialize a floating-point array. For continuous densities $q,p$, in bits,

$$
\mathrm{KL}_2(Q\Vert P)=\int q(w)\log_2\frac{q(w)}{p(w)}\,dw.
$$

$P$ is the reference chosen independently of the training sample for the usual PAC-Bayes setting; $Q$ can depend on training. A likelihood-based variational objective is

$$
F(Q)=\mathbb E_Q[-\log_2 p(S\mid w)]+\mathrm{KL}_2(Q\Vert P)
=-\log_2 p(S)+\mathrm{KL}_2(Q\Vert P(\cdot\mid S)).
$$

$S$ is the observed data, $p(S)$ its evidence and $P(\cdot\mid S)$ the exact Bayesian posterior. The identity explains the fit-plus-complexity interpretation: minimizing over all admissible distributions reaches negative log evidence; a restricted variational family generally leaves a gap. This objective is **not itself a high-probability test-error bound**. Slide 19’s shorthand needs a PAC-Bayes theorem, bounded-loss assumptions, confidence terms and the appropriate nonlinear conversion; see [paper 2](/papers/t1-paper-2).

A useful correction to slides 16–17: a continuous point has probability zero, and a density value is not its probability mass. For $Q=\mathcal N(\mu,\sigma^2)$ and $P=\mathcal N(0,1)$,

$$
\mathrm{KL}_2(Q\Vert P)=\frac{\mu^2+\sigma^2-1-\ln\sigma^2}{2\ln2}.
$$

With $\mu=0$, $\sigma=1$ gives zero, $\sigma=0.5$ gives about **0.459 bits**, and $\sigma=0.1$ gives about **2.608 bits**. As $\sigma\to0$, KL diverges; it does not approach a finite negative log prior density. For a *discrete* prior with positive mass at $\hat w$, a point distribution instead has KL $-\log_2P(\hat w)$. Wider is not always cheaper either: spreading far beyond the prior increases cost.

Compression can support generalization bounds under a prespecified valid code, appropriate sampling and loss assumptions. It does not ensure transfer under arbitrary distribution shift. Likewise, curvature alone is not evidence: prior density, coordinates, dimension and multiple modes matter. [Paper 3](/papers/t1-paper-3) develops the Bayesian analogy and its limits. Slide 43’s Gaussian-prior/L2 correspondence concerns a negative log prior in a MAP objective; it is not an identity between all stochastic KL objectives and weight decay.

## 3 · From small programs to repeated blocks

[![CTM estimates complexity for short blocks; BDM pays once for each distinct block plus a logarithmic repeat count, losing their large-scale ordering.](/figures/complexity-lecture/bdm.svg)](/figures/complexity-lecture/bdm.svg)

Slides 21–23 introduce a tractable approximation. Algorithmic probability sums $2^{-|p|}$ over halting programs that output an object; the coding theorem relates its negative logarithm to prefix complexity up to a constant. **CTM**, the Coding Theorem Method, substitutes an empirical output frequency $D_n(r)$ from a finite small-machine enumeration:

$$
C_{\mathrm{CTM}}(r)=-\log_2D_n(r).
$$

$r$ is a short block; $n$ indexes the enumeration convention. This is the **lookup box**, not a neural-network probability. Finite tables and halting cutoffs do not recover exact universal complexity. The deck describes tables for short binary strings and $4\times4$ binary blocks; this guide uses its supplied scope, not a claim about the latest available tables.

BDM, the Block Decomposition Method, combines lookups for larger objects. Source Eq. (3) becomes, using a separate block symbol to avoid ambiguity,

$$
C_{\mathrm{BDM}}(w)=\sum_{r\in\Pi_u(w)}\left[C_{\mathrm{CTM}}(r)+\log_2 n_r\right].
$$

$\Pi$ partitions $w$ into blocks; $\Pi_u$ contains distinct blocks, and $n_r$ counts occurrences. The sum operates at the **distinct-block aggregation box**. Repetition costs a logarithmic multiplicity term instead of another full lookup.

Toy lookup: let $C_{\mathrm{CTM}}(A)=5$ and $C_{\mathrm{CTM}}(B)=7$ in illustrative units. The block sequence $A,A,A,B$ scores $(5+\log_2 3)+(7+\log_2 1)=13.585$. These values are invented for arithmetic, not drawn from a CTM table. Rearranging it to $A,B,A,A$ preserves this score. This exposes a limitation: block counts alone lose ordering and long-range structure. A BDM score is not automatically a complete decodable compression format or a certified upper bound for the original object.

[Paper 6](/papers/t1-paper-6) studies binarized networks. Slide 25’s “SGD” shorthand should not replace that paper’s documented Adam/straight-through training. BDM is a diagnostic there, not its optimized loss; correlations with loss do not establish a cause of generalization.

## 4 · QuBD keeps several bit planes

[![Weights are quantized to finite integers, split into binary planes, partitioned into blocks and scored; training and inference use a separate predictor path.](/figures/complexity-lecture/pipeline.svg)](/figures/complexity-lecture/pipeline.svg)

The lecture’s next move is to retain more than each weight’s sign. **Quantized Block Decomposition (QuBD)** applies the block estimator separately to each bit position, then adds the scores (slides 27–30, Eqs. 4–6):

$$
q=Q_b(w)\in\{0,\ldots,2^b-1\}^d,\qquad
q_i=\sum_{\ell=0}^{b-1}2^\ell\beta_{i,\ell},\qquad
C_{\mathrm{QuBD}}(w)=\sum_{\ell=0}^{b-1}C_{\mathrm{BDM}}(\beta_\ell).
$$

$w\in\mathbb R^d$ is a flattened weight object, $Q_b$ an affine uniform $b$-bit quantizer, $q_i$ its integer symbol, and $\beta_\ell=(\beta_{1,\ell},\ldots,\beta_{d,\ell})$ the binary plane. Plane 0 is least significant; plane $b-1$ is most significant. These equations respectively label the **quantize**, **split**, and **aggregate** arrows. The guide leaves layer sizes symbolic because the lecture covers multiple architectures.

The top row of the figure is ordinary supervised training: $x\in\mathbb R^a\to f_w(x)\in\mathbb R^K\to\ell(f_w(x),y)\to$ optimizer update. Here $a$ is input dimension and $K$ output dimension. At checkpoints, QuBD reads $w$; it does not send a gradient back in this diagnostic pipeline. At inference, freeze $w$ and evaluate a fresh input. Post-training quantization would replace the predictor’s weights and requires a separate accuracy evaluation.

[![Four toy weights become integers zero through three, whose two bit planes reconstruct them exactly; dropping a plane merges pairs of values.](/figures/complexity-lecture/planes.svg)](/figures/complexity-lecture/planes.svg)

Follow one numerical object through those same boxes. **Toy quantizer only:** for values in $[-1,1]$, take $b=2$ and

$$
q_i=\operatorname{round}\!\left(3\frac{w_i+1}{2}\right),\qquad
\hat w_i=-1+\frac{2q_i}{3}.
$$

Out-of-range values would be clipped; this example has no clipping or rounding ties. For $w=(-1,-1/3,1/3,1)$, the symbols are $(0,1,2,3)$, the high plane is $(0,0,1,1)$ and the low plane is $(0,1,0,1)$. Reading the columns reconstructs the integers as $2\beta_1+\beta_0$. If their illustrative BDM scores are 6 and 8, QuBD is 14. With a same-protocol random reference scoring 20, normalized QuBD is $14/20=0.7$, or 70%. Neither these scores nor this quantizer range is a reported experiment.

Slide 30 normalizes by $C_{\mathrm{QuBD}}(\tilde w)$ for a random reference $\tilde w$ of the same length. To compare meaningfully, also match precision, layout, quantizer and partition rules. The ratio is not a probability or a universal compression fraction, need not lie below one, and depends on the random reference. The source itself notes finite-size offsets: all-zero $8\times8$, $16\times16$ and $32\times32$ objects score approximately 0.19, 0.05 and 0.01 relative to its reference protocol.

<details><summary>Predict: if two bit planes are identical, does summing their separate complexities exploit that dependence?</summary><p>No. A joint description could say “copy the first plane”; the separate sum does not explicitly encode this relationship. More retained precision does not make QuBD exact.</p></details>

## 5 · Precision and structure are different losses

Slide 29 discusses an ideal **finite-precision target**, not an arbitrary exact real vector. If $w^\star$ is an integer-coded target at $b^\star$ bits and $q$ retains its high $b$ bits in the same convention,

$$
\delta_b=w^\star-2^{b^\star-b}q,\qquad
R_b=C(\delta_b\mid q).
$$

$\delta_b$ contains omitted lower bits; $R_b$ is their conditional algorithmic complexity. The lecture’s residual decomposition relates $C(w^\star)$ to $C(q)+R_b$ up to logarithmic overhead and bounds the residual by roughly $(b^\star-b)d$ bits. This is a statement about ideal description complexity; it does not certify the error of a particular CTM/BDM estimate. Applying it to real weights also requires the shared scale/offset and finite encoding convention.

In our toy, retaining the high bit gives $q'=(0,0,1,1)$ and $\delta=(0,1,0,1)$ because $(0,1,2,3)=2q'+\delta$. Dropping the low bit loses information even though reconstruction from *both* planes was lossless. Separately, BDM can miss structure *inside* a plane; summing planes can miss structure *between* them. These are three distinct approximation issues.

A high-complexity low bit plane can behave like noise, but complexity alone does not establish that it is safe to discard. The prediction effect depends on sensitivity and downstream computation. Slide 36’s pruning/quantization message is an empirical diagnostic suggestion, not a universal bit-removal rule.

## 6 · What the lecture actually measures

[![Original lecture slide 32 plots FashionMNIST validation accuracy against normalized QuBD for MLPs at several training sample counts and parameter counts.](/figures/complexity-lecture/source-results.png)](/figures/complexity-lecture/source-results.png)

**Source evidence, not a teaching plot:** the supplied slide 32 compares MLPs on **FashionMNIST**, with **10,000 validation examples**. Its vertical axis is accuracy (%) and horizontal axis normalized QuBD (%); the legend lists training sample counts from 100 to 40,000 and parameter counts 0.6M, 1.5M, 4.2M and 13.7M. Read the marks as a relationship across experimental settings, not a universal function mapping complexity to accuracy. No individual points have been reverse-engineered into an invented numerical table.

| Source | Observation offered | What the slide does not establish |
| --- | --- | --- |
| 32 | Accuracy and QuBD across MLP/sample-size settings | Causality, a calibrated error bound, or complete training details |
| 33 | QuBD compared visually with BDM and GZIP | A universal ranking of estimators or exact KCS errors |
| 34 | Complexity alongside generalization/grokking traces | That declining complexity causes the later accuracy rise |
| 35 | Eight planes of 100 pretrained models, 1M–100M parameters | Comparability independent of quantizer and layout |
| 36 | Plane scores and post-training quantization accuracy for named architectures | A dataset-independent accuracy guarantee from a plane score |
| 39 | Motif assignment shares over epochs for MLP, ResNet20 and Tiny-ViT | A fully specified accuracy/storage/latency benchmark |

Slide 34’s embedded Figure 5 shows MLP and Tiny-ViT learning curves, plus a grokking trace labeled as a modulo task with $P=97$. In the MLP panel, complexity first falls and later rises alongside worsening validation behavior; this is not a claim that complexity decreases throughout every training run. The slide does not spell out a complete task/training protocol for that grokking trace.

The named post-training models on slide 36 are ResNet-18, ResNet-50, ViT-B/16, EfficientNet-B0 and MobileNetV3; the plotted baseline is FP32. The slide does not specify enough evaluation detail to quote a reproducible cross-model leaderboard here. Slide 39 describes performance as comparable to dense models but provides no full accuracy table on that slide. We therefore do not interpret its 10%, 13%, 10% annotations as accuracy improvements or compression ratios.

A useful **proposed ablation** would fix architecture, sample split, optimizer, training budget and seeds, compare true versus shuffled labels, and log validation accuracy plus prespecified complexity settings. Then vary quantization range, precision, block partition and layout. A finding that disappears under these changes is evidence of estimator sensitivity. For compression, actually remove planes or constrain motifs and measure retained accuracy, stored bytes and runtime; a diagnostic correlation cannot substitute for that intervention.

## 7 · MoMos turns a diagnostic idea into a constraint

[![MoMos alternates an optimizer step with repartitioning, motif sampling, nearest-motif assignment and reconstruction; inference uses the final reconstructed weights.](/figures/complexity-lecture/momos.svg)](/figures/complexity-lecture/momos.svg)

Slides 37–38 introduce **Mosaic-of-Motifs (MoMos)**. Instead of merely measuring repeatable weight blocks after training, restrict blocks to a shared dictionary. This is one way to induce local reuse; low ideal algorithmic complexity does not in general require repeated local blocks (a short program can produce globally structured, locally diverse output).

Here is the actual sequence visible in slide 38’s Algorithm 1, with renamed symbols to avoid colliding with bit planes. There are $N$ weights partitioned by $\psi$ into $m=\lceil N/s\rceil$ blocks of size $s$; capacity $c\in[0,1]$ sets $k=\max\{1,\lfloor cm\rfloor\}$ motifs. The slide does not detail incomplete-block padding, so our example uses $s$ dividing $N$.

1. Apply one optimizer step to the **previous constrained weights**, producing temporary weights $w^{(t)}$.
2. Partition them into blocks $b_1^{(t)},\ldots,b_m^{(t)}$.
3. Set motif $z_1^{(t)}=0_s$. If $k>1$, uniformly sample $k-1$ distinct block indices and use those blocks as the other motifs. Distinct indices can still contain identical values.
4. Assign each block its nearest motif and reconstruct the constrained array:

$$
M^{(t)}(i)=\arg\min_{r\in\{1,\ldots,k\}}\|b_i^{(t)}-z_r^{(t)}\|_2^2,\qquad
\hat w^{(t)}=\psi^{-1}\!\left(z_{M^{(t)}(1)}^{(t)},\ldots,z_{M^{(t)}(m)}^{(t)}\right).
$$

$M$ is the mosaic of motif indices, $z_r$ a length-$s$ motif, and $\psi^{-1}$ restores the original arrangement. The squared distance lives at the **assignment box**; it favors small local weight perturbations, not necessarily the smallest increase in prediction loss. Set $w_t=\hat w^{(t)}$ and repeat. Motifs and assignments are rebuilt each iteration in this slide’s algorithm; it is not a single fixed codebook fitted once, nor a claim of exact differentiable optimization through discrete choices. Tie handling and optimizer-state details are not specified here.

At inference, freeze the final motifs and mosaic, reconstruct weights and evaluate $f_{\hat w}(x)$. A specialized execution engine could avoid dense reconstruction, but the slide does not establish such runtime savings. The optimizer is an input to the algorithm; we do not invent its schedule or layer dimensions.

[![Three two-weight blocks choose between a zero motif and a sampled motif; two blocks share the sampled motif, and a storage calculation includes both dictionary and index costs.](/figures/complexity-lecture/motif-example.svg)](/figures/complexity-lecture/motif-example.svg)

Follow a **single illustrative projection**. After an optimizer step, let the three blocks be $(0.1,0.2)$, $(0.9,1.1)$ and $(1.0,1.0)$. Choose $s=2$, $c=0.8$, hence $m=3$ and $k=2$. Condition on sampling the third block: motifs are $z_1=(0,0)$ and $z_2=(1,1)$. Squared distances to these motifs are respectively $(0.05,1.45)$, $(2.02,0.02)$ and $(2,0)$. Thus $M=(1,2,2)$ and the constrained weights are $(0,0,1,1,1,1)$. This is a projection trace, not a measured training run.

A teaching storage estimate makes the benefit conditional. For $m$ blocks, $s$ scalars per block, $k$ motifs and $r$ stored bits per scalar, dense storage is $msr$ bits; a simple fixed-width dictionary representation uses about

$$
ksr+m\lceil\log_2 k\rceil
$$

bits plus metadata. For $m=16,s=4,k=2,r=32$, that is **2048 dense bits versus 272 dictionary-plus-index bits**, before headers/alignment. This illustrative format even stores the zero motif explicitly; it is not the paper’s compression measurement. Accuracy can fall when reuse is too restrictive, and a small code need not make dense inference faster.

<details><summary>Predict: is c = 1 guaranteed to reproduce an arbitrary dense optimizer step exactly?</summary><p>No. The displayed algorithm reserves one of k = m motifs for zero and samples only m − 1 block indices. A missing nonzero block need not have an identical motif available. At k = 1, every block is projected onto zero.</p></details>

## The idea to carry forward

Ask three questions in order: **what object is described, what information the estimator discards, and whether complexity is merely measured or actually constrains learning**. KCS is an ideal shortest-program quantity. MDL commits to a practical code. QuBD measures quantized bit-plane structure. MoMos changes the allowed weight arrays. None of these names alone supplies a generalization theorem.

The reasoning path reconstructed here is pedagogical: perfect fit leaves many solutions; description length suggests a preference; computability requires approximation; an observed diagnostic motivates an intervention. It is not a claim about the lecturer’s private reasoning. Use the eight cards below to test whether you can keep these roles separate.

## Source and slide reading map

The [local September 3, 2026 course deck](/papers/complexity-lecture.pdf) is the source for this guide; its original research references remain in the PDF. No newer paper version or external benchmark is merged into the lecture evidence.

- **Slides 1–11:** identity, recap, six readings and Occam motivation. The fit-plus-complexity framing is a research lens, not a proof that both terms must be monotone in parameter count.
- **Slides 12–15, 41–42:** KCS, MDL, prefix codes and information; §1 separates ideal descriptions, overhead and supervised label coding. Entropy is an average under a source; describing that source can enter the additive complexity constant.
- **Slides 16–19, 43:** distribution codes, variational evidence and prior choice; §2 distinguishes discrete mass from continuous density and objectives from certificates.
- **Slides 21–25:** CTM, BDM and the binarized-network reading; §3 corrects the semicomputability direction and gives a labeled toy lookup.
- **Slides 27–30:** QuBD, residual precision and normalization; §§4–5 trace a four-weight example. The quantizer graphic is schematic, not a numerical calibration table to copy.
- **Slides 31–36:** empirical motivation, comparisons, grokking and bit-plane diagnostics; §6 preserves the evidence scope and provides the source result slide.
- **Slides 37–40:** MoMos, source algorithm, motif reuse and open questions; §7 follows the displayed resampling/projection loop. No supplementary training protocol is inferred.

Prediction disclosures and cards supply practice; there is no custom numerical simulator, so slider endpoints do not apply.
