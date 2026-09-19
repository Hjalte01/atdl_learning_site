---
title: "Algorithmic Simplification of Neural Networks: Mosaic-of-Motifs"
shortTitle: "Mosaic-of-Motifs · Lecture 3"
topic: "topic-1"
order: 3
materialKind: "slides"
description: "Replace independent weight blocks with reusable motifs: trace the representation, projected training loop, description budget and evidence limits."
status: "ready"
pdf: "/papers/momos-lecture.pdf"
---
## The puzzle: can a network learn with a small vocabulary of weights?

Imagine sixteen four-weight blocks. A dense model stores all 64 numbers independently. Another model uses only three distinct four-number patterns, placing them in sixteen positions. It still has 64 weight slots and can run the same forward computation. **What has become smaller: the architecture, the description, or the computation?**

This guide follows Pedram Bakhtiarifard’s supplied *Algorithmic Simplification of Neural Networks: Mosaic-of-Motifs (MoMos)* course deck, `slides/3_atdl_momos_pedram.pdf`. [Open the course PDF](/papers/momos-lecture.pdf). The title page says **September 2, 2026**, while a quotation on PDF page 3 refers to September 3. We retain the title-page date without resolving that inconsistency. There are **42 PDF pages**, including incremental builds of slides numbered through 24. References below use **PDF page numbers**, not the footer numbers. “Lecture 3” follows its course filename/order.

Allow 25–35 minutes. You need gradients, vectors and logarithms; [the complexity lecture](/papers/t1-complexity-lecture) supplies the description-length background. The learning route here follows the deck: analogy → representation → constrained optimization → evidence. All colored diagrams and worked values are teaching constructions except the explicitly reproduced source result slide. No trained network runs on this page.

## 1 · Reuse needs a pattern and its locations

[![Independent weight blocks store every value; a motif bank stores shared blocks and a mosaic records where to reuse them.](/figures/momos-lecture/reuse.svg)](/figures/momos-lecture/reuse.svg)

Read the two rows as alternative representations of one weight array, not different network widths. The deck’s music example (PDF pp. 10–12) replaces sixteen notes by a repeated four-note motif. Its chess example (pp. 13–15) illustrates recognizing a reusable configuration. These motivate structure as a useful description; they do not prove that all learning is lossless compression or that a short description always predicts well.

For weights, locations matter. Knowing that block A occurs seven times, B five times and C four times does not say which block goes in each layer position. Page 22 counts only the twelve motif scalars; page 23 explicitly raises the missing-location problem. The **mosaic** supplies those locations.

A plausible design argument is: repeated structure can shorten a description; waiting for exact repetition may be unreliable; constrain optimization to produce reuse. This is our teaching reconstruction, not a claim about the authors’ private reasoning. Local repetition is one compressible structure, not a complete characterization of algorithmic simplicity.

<details><summary>Predict: if two arrays have the same motif counts but different block order, must their predictions match?</summary><p>No. Positions determine which connections receive each weight. Counts can agree while reconstructed arrays and predictions differ.</p></details>

## 2 · The representation: a bank plus a mosaic

[![A bank with motifs A, B and C combines with a four-by-four grid of indices; looking up each index reconstructs its four-weight block.](/figures/momos-lecture/representation.svg)](/figures/momos-lecture/representation.svg)

The **bank** holds values; the **mosaic** holds indices. Pages 25–30 describe partitioning a weight vector $W\in\mathbb R^n$ into $m=\lceil n/s\rceil$ blocks of $s$ scalars. A $2\times2$ block has $s=4$, not $s=2$. Write

$$
\psi(W)=(b_1,\ldots,b_m),\quad b_i\in\mathbb R^s,\qquad
Z=(z_1,\ldots,z_k)\in\mathbb R^{k\times s},\qquad
M\in\{1,\ldots,k\}^m.
$$

Here $\psi$ is a fixed partition/layout rule, $Z$ is the bank of $k$ motifs, and $M_i$ identifies the motif used at block position $i$. The **reconstruction arrow** performs

$$
\widehat W=\psi^{-1}(z_{M_1},\ldots,z_{M_m}).
$$

The inverse restores the original tensor arrangement; it does not learn a new layer. For example, $Z=((0,0),(1,1))$ and $M=(1,2,2)$ reconstruct $(0,0,1,1,1,1)$. An array with $n$ slots can thus have at most $k$ distinct block values. Motifs themselves can coincide, so the actual number can be smaller.

The deck does not specify all tensor-boundary, incomplete-block or padding conventions. Our examples choose $s$ dividing $n$ and fix one partition. A real implementation must specify these conventions and any unmodified parameters. The abstract and diagrams do not supply an exact layer-by-layer network configuration; our predictor remains $f_{\widehat W}(x)$ rather than an invented backbone.

For fixed $M$, the bank contains $ks$ continuous scalars. Allowing $M$ to change creates a union of parameter-sharing arrangements. This is more informative than saying simply “there are only $ks$ parameters”: the discrete mosaic also carries information.

## 3 · Count the entire description

[![A 64-weight example compares 2048 dense bits with 384 motif bits plus 32 index bits, totaling 416 before metadata.](/figures/momos-lecture/storage.svg)](/figures/momos-lecture/storage.svg)

Page 30 bounds a description of the reconstructed weights by storing the bank and an index for each block. At $q$ bits per scalar, its expression is

$$
K(\widehat W)\leq ksq+m\lceil\log_2 k\rceil+O(1).
$$

$K$ denotes ideal Kolmogorov description complexity of the **finite encoded object**, not an exactly computed statistic for arbitrary real numbers. The first term is the **bank box**, the second the **mosaic box**. The constant presumes fixed decoding conventions; if architecture, $n,s,k,q$, layout or quantization metadata vary, their descriptions must also be counted. This gives an available code and therefore an upper bound, not the shortest possible code or a measured file size.

For the page-22 example, $n=64$, $s=4$, $m=16$, $k=3$ and occurrence counts $(7,5,4)$. The removed duplicate scalar counts are $4(7-1)=24$, $4(5-1)=16$, and $4(4-1)=12$. That leaves $64-24-16-12=12$ stored scalars. With **teaching assumption** $q=32$:

$$
L_{\rm dense}=64\cdot32=2048,\qquad
L_{\rm motif}=3\cdot4\cdot32+16\lceil\log_2 3\rceil=416\text{ bits}.
$$

The bank-only value is 384 bits; omitting the 32 index bits overstates the saving. The simple complete payload is about **4.92 times smaller** than dense storage before metadata, alignment and decoder costs. This is our arithmetic, not the slide’s experimental simplification metric. In the small example, a two-bit index has an unused fourth codeword.

Let capacity be $c=k/m$, as on PDF p. 38. When $n=ms$, dividing our fixed-width payload by dense storage gives the teaching identity

$$
\frac{L_{\rm motif}}{L_{\rm dense}}
=c+\frac{\lceil\log_2 k\rceil}{sq}.
$$

Thus 5% capacity is not automatically a 20-fold storage reduction: indices still cost bits. This model also ignores inference workspace and optimizer state. For $k=1$, indices need zero bits but all blocks share one motif. The displayed training construction reserves that sole motif for zero, so this extreme erases the represented weights. For $k=m$, bank-plus-index storage can exceed dense storage; capacity alone does not guarantee useful compression.

<details><summary>Predict: does the bound certify better test accuracy?</summary><p>No. It describes an encoding. Generalization guarantees need an appropriate coding or statistical theorem, sampling and loss assumptions, and the model’s fit. A compact model can still underfit or fail under distribution shift.</p></details>

## 4 · Train by proposing, assigning and reconstructing

[![Training alternates loss-based optimizer updates with block partition, zero-plus-sampled motif construction, nearest assignment and reconstruction; inference freezes the final representation.](/figures/momos-lecture/training.svg)](/figures/momos-lecture/training.svg)

Pages 31–34 show an ordinary optimizer proposal followed by a motif constraint. At each iteration start with the **previous reconstructed weights**. For a teaching SGD specialization,

$$
U_t=\widehat W_{t-1}-\eta\nabla_W\ell(f_W(x_t),y_t)\big|_{W=\widehat W_{t-1}}.
$$

$U_t$ is the temporary dense proposal, $\eta>0$ the learning rate, $(x_t,y_t)$ a training example or batch, and $\ell$ its task loss. This equation labels the **optimizer arrow**, not an additional source-numbered equation. The lecture permits an ordinary optimizer; it does not specify SGD as the experimental choice or give a schedule.

Partition $U_t$ into blocks $b_i^{(t)}$. Construct $k$ motifs from a zero block and sampled blocks of the proposal. With this bank fixed for the current assignment, choose

$$
M_i^{(t)}\in\arg\min_{j\in\{1,\ldots,k\}}
\|b_i^{(t)}-z_j^{(t)}\|_2^2,\qquad
\widehat W_t=\psi^{-1}(z_{M_1^{(t)}}^{(t)},\ldots,z_{M_m^{(t)}}^{(t)}).
$$

The **assignment box** minimizes squared Euclidean block error; the **reconstruction box** replaces blocks by selected motifs. Carry $\widehat W_t$ into the next iteration. The bank is selected within this repeated loop, not frozen at initialization. This deck says “sampled blocks” but does not define sampling distribution, replacement, tie-breaking, capacity rounding or optimizer-state handling. We do not silently import those details from another lecture or a newer implementation.

For a fixed bank, independent nearest assignments minimize the sum of squared block errors. They need not minimize task loss, and choosing a bank by sampling is not a global nearest projection over every possible bank. The diagram’s “project” should therefore be read as this concrete constrain-and-reconstruct operation. No derivative through the discrete argmin is specified.

At inference, freeze the final bank and mosaic, reconstruct $\widehat W_T$ and compute $f_{\widehat W_T}(x_*)$ for a new input $x_*$. Targets, optimizer steps and motif resampling are absent. The architecture and dense operation count need not shrink. Executing compressed blocks directly could help in a suitable implementation, but the deck supplies no latency benchmark establishing that benefit.

## 5 · Trace one update, then the next

[![Three two-weight blocks have squared distances to zero and a sampled motif; nearest assignments produce the mosaic 1,2,2 and reconstruct six constrained weights.](/figures/momos-lecture/distances.svg)](/figures/momos-lecture/distances.svg)

Follow a complete **toy** step through the same boxes. Start with $\widehat W_{t-1}=(0,0,1,1,1,1)$, choose $\eta=0.1$, and suppose the task gradient is $g=(-1,-2,1,-1,0,0)$. The optimizer gives

$$
U_t=\widehat W_{t-1}-0.1g=(0.1,0.2,0.9,1.1,1,1).
$$

Partition into $m=3$ blocks with $s=2$ and choose $k=2$ (capacity $2/3$). Condition on selecting the third block as the nonzero motif: $Z_t=((0,0),(1,1))$. This sampling outcome and gradient are stipulated, not experimental measurements.

| Proposal block | Squared distance to $(0,0)$ | Squared distance to $(1,1)$ | Assigned index |
| --- | ---: | ---: | ---: |
| $(0.1,0.2)$ | 0.05 | 1.45 | 1 |
| $(0.9,1.1)$ | 2.02 | 0.02 | 2 |
| $(1,1)$ | 2 | 0 | 2 |

The first distance is $0.1^2+0.2^2=0.05$; every other entry follows the same rule. Reconstruction returns $(0,0,1,1,1,1)$. Total squared projection error is $0.05+0.02+0=0.07$. An optimizer proposal can be entirely undone by the constraint even though its task gradient was nonzero.

[![The first illustrative proposal is erased by projection; a second proposal forms a different sampled motif and the carried-forward weights change.](/figures/momos-lecture/iterations.svg)](/figures/momos-lecture/iterations.svg)

To see what carries forward, suppose the next gradient at those constrained weights is $(0,0,-2,-2,-2,-2)$. At the same learning rate the proposal is $(0,0,1.2,1.2,1.2,1.2)$. Sampling its third block yields $Z_{t+1}=((0,0),(1.2,1.2))$, still with mosaic $(1,2,2)$, and reconstruction now preserves the proposal exactly. **A repeated index need not imply an unchanging motif value across training.** These two stipulated gradients illustrate the algorithm, not convergence or loss improvement.

Small weight error also need not mean small output error. In a separate linear teaching example, changing a two-weight block by $(-0.1,-0.2)$ changes its dot product with $x=(10,10)$ by $-3$. Nearest-weight assignment is a convenient local distortion rule, not an input-aware guarantee. Testing actual prediction quality remains necessary.

## 6 · Capacity and block size are different controls

[![For a fixed sixteen-scalar array and two motifs, two-scalar blocks allow 256 index strings while four-scalar blocks allow 16, with different bank costs.](/figures/momos-lecture/domain.svg)](/figures/momos-lecture/domain.svg)

Pages 37–38 argue that smaller blocks allow more arrangements and that greater capacity enriches the optimization domain. A simple teaching count explains part of this intuition. With a **fixed bank of distinct motifs** and $m$ independent block positions, there are $k^m$ index strings. For $n=16$ and $k=2$, $s=2$ gives $m=8$ and $2^8=256$ strings; $s=4$ gives $m=4$ and $2^4=16$. Smaller blocks permit finer spatial recombination.

This comparison fixes $k$, not capacity or total storage: the bank sizes are four versus eight scalars, and capacities are $1/4$ versus $1/2$. If motifs coincide, distinct strings can reconstruct the same array. If the bank varies, the domain also includes continuous choices. The count is not a theorem that one training setting generalizes better.

Increasing $k$ at fixed layout permits more distinct block values and generally costs more bank and index bits. Increasing $s$ makes each reused pattern larger, which can improve description efficiency but couples more weights in every assignment. A fair “fixed budget” comparison must adjust $k$ to account for both terms in the code length and measure accuracy; it cannot hold every knob constant simultaneously.

A useful **proposed ablation** would hold architecture, task, split, optimizer, training steps and seeds fixed; sweep $s$ and $k$; and report task accuracy, actual encoded bytes, projection distortion, peak training memory and inference time separately. Compare with the dense baseline, one-time post-training motif assignment, and a fixed-bank training variant. These comparisons would test whether repeated rebuilding matters and whether savings survive implementation overhead. They are experiments to run, not results in the deck.

## 7 · What the supplied results support

[![Original PDF page 36 compares accuracy against capacity for Tiny-ViT, ResNet20 and MLP, then reports simplification factors at block size two and capacity five percent.](/figures/momos-lecture/source-results.png)](/figures/momos-lecture/source-results.png)

This is **source evidence**, reproduced from PDF p. 36 (numbered slide 18), not a generated result. Read the top curves as accuracy versus capacity, with a dashed baseline and different block sizes. The visible rendered slide does not show readable numeric capacity tick labels; no values have been reconstructed from hidden overlapping PDF text. The explicit callout reports **near-baseline performance at $s=2$, $c=5\%$**, with these *algorithmic simplification* factors:

| Architecture | Factor printed in the deck | Baseline comparison |
| --- | ---: | --- |
| Tiny-ViT | $3.02\times$ | Dashed baseline in its accuracy panel |
| ResNet20 | $3.95\times$ | Dashed baseline in its accuracy panel |
| MLP | $3.52\times$ | Dashed baseline in its accuracy panel |

The slide does **not identify the dataset, evaluation split, exact baseline accuracies, seed count, uncertainty-band definition or full training protocol**. It also does not fully define how these factors were evaluated. We therefore preserve its label rather than translating the factors into serialized file ratios, speedups or guaranteed accuracy retention. “Near” is not a numerical tolerance supplied by the deck. Larger blocks visibly incur a stronger low-capacity accuracy cost in these panels; this is evidence for the shown settings, not a universal ranking.

PDF p. 39 (slide 21) plots **normalized cross-entropy loss** on an $\alpha,\beta$ plane for MLP, ResNet20 and Tiny-ViT, with baseline, $s=2$ and $s=4$ markers. The caption argues for similar decision geometry despite different weights. A two-dimensional loss visualization is not a proof of identical decisions, global landscape equivalence or identical calibration; the deck does not fully specify the plane construction or normalization.

PDF p. 40 (slide 22) shows assignment shares for ranked motif groups at epochs 1 and 200. Concentration suggests a nonuniform index code might improve on the worst-case fixed-width mosaic. The 10%, 13% and 10% labels sit in the bottom-50%-of-motifs segments at epoch 200 for MLP, ResNet20 and Tiny-ViT respectively: they indicate assignment shares, not accuracy or byte savings. A teaching entropy code would ideally approach $mH(p)$ index bits for frequencies $p_j$, where $H(p)=-\sum_jp_j\log_2p_j$, but frequency tables and coding overhead must also be stored. For two equally likely motifs, $H=1$ bit per index; for proportions $(0.9,0.1)$, $H\approx0.469$. That calculation illustrates why concentration can help; it is not the source’s measured compressed size.

<details><summary>Predict: if motif use becomes concentrated, can unused motifs always be deleted during training without consequence?</summary><p>No. Current assignment frequency does not establish future usefulness. At a frozen checkpoint, unused motifs can be removed with corresponding index remapping; during optimization, changing the available bank can change later assignments and learning.</p></details>

## The idea to carry forward

MoMos separates **weight slots**, **reusable values**, and **the map that places those values**. Its training loop repeatedly turns an ordinary dense proposal into a structured array. The code-length argument explains why that representation can be shorter; the task evaluation must establish whether it still works. Storage, optimization freedom, accuracy and execution speed are related questions with different evidence requirements.

Use the eight study cards to check that you can reconstruct a weight array, count its indices, follow the carried-forward state and qualify the empirical claims. This guide uses static worked traces and prediction disclosures; no custom numerical lab or slider is needed to inspect these exact examples.

## Source reading map

The [supplied course deck](/papers/momos-lecture.pdf) is the sole source for lecture claims. Its embedded abstract on PDF p. 24 is dated April 2026 and names Bakhtiarifard, Chen, Wenshøj, Dam and Selvan. It does not supply a complete paper or experimental appendix. No results from a different paper version are merged here.

| PDF pages | Numbered slide / role | Guide connection |
| --- | --- | --- |
| 1–3 | Title, overview, compression motivation | Version/date caveat and puzzle |
| 4–16 | Everyday, music and chess analogies | §1: reusable structure as motivation |
| 17–23 | Weight descriptions; missing locations | §§1–3: bank alone is insufficient |
| 24 | Embedded research abstract, slide 13 | Scope of the constrained parameterization |
| 25–30 | Partition, bank, mosaic and bound, slides 14–15 | §§2–3: representation and complete payload |
| 31–34 | Optimizer, sampling, assignment, reconstruction, slide 16 | §§4–5: training and worked iterations |
| 35–38 | Capacity, block size and accuracy, slides 17–20 | §§6–7: domain tradeoffs and qualified evidence |
| 39–40 | Loss geometry and motif shares, slides 21–22 | §7: what the diagnostic plots establish |
| 41–42 | Takeaways and closing | The idea to carry forward |
