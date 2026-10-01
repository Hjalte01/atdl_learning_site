---
title: "Active Learning for Vision-Language Models"
shortTitle: "CEC"
topic: "topic-4"
order: 2
description: "Which images deserve a scarce annotation budget? Trace calibrated entropy, neighborhood uncertainty and weighted clustering into frozen-encoder prompt tuning."
status: "ready"
source: "https://openaccess.thecvf.com/content/WACV2025/html/Safaei_Active_Learning_for_Vision_Language_Models_WACV_2025_paper.html"
pdf: "/papers/cec.pdf"
---
## 1 · The puzzle: the most confused image may be the wrong purchase

You have a pretrained CLIP model, a pool of unlabeled aircraft images, and money to label only a small fraction. Class names are known; individual image labels are hidden. Should you buy the labels of the images with the most uncertain predictions?

That sounds sensible, but uncertainty can come from several places. CLIP may systematically favor some class descriptions. An unusual image may confuse the model without representing many other images. And the ten most uncertain images may all look alike. **Calibrated Entropy-weighted Clustering (CEC) changes the acquisition rule: adjust class probabilities, borrow uncertainty from nearby images, and query representatives from uncertainty-weighted clusters.** Oracle labels then train a small prompt rather than the entire model.

This is Topic 4 paper 2, by Bardia Safaei and Vishal M. Patel. The guide follows the supplied **11-page WACV 2025 open-access proceedings paper, pp. 4902–4912**. Its title is *Active Learning for Vision-Language Models*; “CEC” is the method, not a different paper. All colored diagrams are original teaching constructions; only the result diagram contains reported measurements. Allow about 20 minutes. Open any figure to enlarge it.

[![Three routes from an unlabeled pool to annotations compare random sampling, raw entropy and CEC.](/figures/cec/problem.svg)](/figures/cec/problem.svg)

**Read each row as a decision rule for the same budget.** Random sampling ignores the current model. Entropy sampling uses its probability distribution. CEC uses both predictions and feature geometry. The diagram is schematic; it does not claim a particular set of images will always be best.

The design logic is our interpretation: repair the signal used to rank images, ask whether nearby images share the confusion, then avoid spending the whole batch in one region. These choices target different failure modes; none supplies an oracle label before selection.

## 2 · Architecture: two frozen encoders and a changing prompt

[![Full CEC architecture labels image and text embeddings, class probabilities, calibration, neighbor scores, clustering, oracle queries and context-only training.](/figures/cec/architecture.svg)](/figures/cec/architecture.svg)

**A makes predictions, B assigns acquisition weights, C buys labels, and D learns from them.** The next acquisition cycle returns to A with updated prompts. The two encoder branches converge on the probability comparison; they are not a serial image-to-text encoder chain. The diagram specifies functional tensors and arrows, not undisclosed transformer layers.

CLIP has an image encoder $f$ and a text encoder $h$. For image $x$, the normalized image representation is $z=f(x)/\|f(x)\|_2\in\mathbb R^d$. For class $i$, CoOp forms a text representation

$$
t_i=h([V_1,\ldots,V_M,C_i])\in\mathbb R^d.
$$

$V_m$ is learnable context vector $m$; $M$ is their count; $C_i$ denotes the tokens of class $i$'s name; brackets denote a token sequence. The default setup shares context across classes and places the class name at the end (§3.2, §5). The supplied main paper does not specify a full token-width/layer diagram or the exact learned context count, so we leave those symbolic. For the main experiments it identifies **CLIP ViT-B/16**, rather than a newly trained vision-language backbone.

At A’s image–text comparison arrow, paper Eq. 1 computes

$$
P_i(x)=P(y=i\mid x)=
\frac{\exp(\operatorname{sim}(f(x),t_i)/T)}
{\sum_{j=1}^{K}\exp(\operatorname{sim}(f(x),t_j)/T)}.
$$

$K$ is the number of known classes, $y$ a possible class label, $\operatorname{sim}$ cosine similarity, and $T=0.01$ the paper’s temperature. Thus $P(x)$ is a $K$-entry probability vector, while $z$ is a $d$-entry visual feature. They answer different questions: *which class?* versus *where in image-feature space?*

For a two-class illustration, similarities $0.2000$ and approximately $0.186137$ give a logit gap of $\log 4$ at $T=0.01$, hence $P=(0.8,0.2)$. A small similarity gap can yield a sharp distribution at this temperature. A sharp prediction is not proof of correctness.

<details><summary>Predict: if CLIP is frozen, can its class probabilities still change?</summary><p>Yes. Changing the context vectors changes the text encoder’s input, hence the class text embeddings and image–text similarities. Frozen weights do not make the encoder’s output independent of its input.</p></details>

## 3 · Calibration: compensate for unequal class tendencies

Before choosing an image, CEC estimates a class-specific correction from the **unlabeled pool**. At B’s first box, let $S_i$ be the $N$ pool images with the highest $P_i(x)$ (paper Eq. 3). Then paper Eqs. 4–5 are

$$
Q(i)\approx\frac1N\sum_{x\in S_i}P_i(x),\qquad
\widehat P_i(x)=\frac{P_i(x)/Q(i)}{\sum_{j=1}^{K}P_j(x)/Q(j)}.
$$

$N$ is the number of top-scoring examples per class; the experiments use $N=10$. $Q(i)$ is the paper’s “contextualized prior,” estimated separately for each class. **It is not a measured class frequency or an oracle label distribution, and its entries need not sum to one.** The sets $S_i$ may overlap. Division reduces the relative influence of a class whose top predictions already tend to be large; the denominator renormalizes the corrected scores.

Uncertainty is entropy, paper Eq. 2 applied to the calibrated probabilities:

$$
\widehat{\mathcal H}(x)=-\sum_{i=1}^{K}\widehat P_i(x)\log\widehat P_i(x).
$$

Here $\log$ is the natural logarithm in our examples, so entropy is in nats. It ranges from zero for a concentrated distribution to $\log K$ for a uniform one. Entropy summarizes disagreement among possible classes, not whether the model’s most likely class is actually correct.

[![Toy calibration divides probabilities by class-specific top-score means and normalizes; one example gains entropy and another loses it.](/figures/cec/calibration.svg)](/figures/cec/calibration.svg)

**Follow the middle row through B.** With $P=(0.8,0.2)$ and teaching values $Q=(0.8,0.4)$, division gives $(1,0.5)$ and normalization gives $(2/3,1/3)$. Raw entropy is $0.5004$ nats; corrected entropy is $0.6365$. The bottom row holds $Q$ fixed but starts at $(0.5,0.5)$: calibration now *reduces* entropy from $0.6931$ to $0.6365$. These are toy values, not measured CLIP outputs.

Why choose this operation? It addresses systematic differences between class scores before using them for acquisition. Its limitation is equally important: a top-$N$ heuristic cannot distinguish every model bias from a genuine downstream class difference. The paper’s term “calibration” does not establish a general guarantee of statistically calibrated probabilities. Our displayed division assumes positive $Q(i)$; numerical clipping or tiny-pool handling is not specified here by the supplied paper.

## 4 · Neighbors: is confusion shared locally?

Self-entropy alone may favor isolated difficult images. CEC also examines nearby normalized visual features. At B’s second box, paper Eq. 6 defines

$$
\mathcal H_{\mathrm{NN}}(x)=\frac1k
\sum_{x_j\in k\mathrm{NN}(x)}
\exp\!\left(-\alpha\|z-z_j\|_2^2\right)
\widehat{\mathcal H}(x_j),
\qquad
U(x)=\widehat{\mathcal H}(x)+\mathcal H_{\mathrm{NN}}(x).
$$

The second expression is paper Eq. 7. $k\mathrm{NN}(x)$ is the set of $k$ nearest neighbors in visual-feature space; $k$ is its size; $z_j=f(x_j)/\|f(x_j)\|_2$; $\alpha$ controls how quickly distant neighbors lose influence. $U$ is an acquisition weight, not a probability or a training loss.

[![Two neighbors contribute attenuated entropy values to a query image; normalized visual features, class probabilities and true labels have distinct roles.](/figures/cec/neighbors.svg)](/figures/cec/neighbors.svg)

**Read the first row as two terms of the sum, then the second as the addition.** Continue the previous toy image with $k=2$, $\alpha=1$, squared feature distances $0.1$ and $0.4$, and neighbor entropies $0.6$ and $0.4$:

$$
\mathcal H_{\mathrm{NN}}(x)=
\frac{e^{-0.1}(0.6)+e^{-0.4}(0.4)}2
\approx0.4055,\qquad U(x)\approx0.6365+0.4055=1.0420.
$$

The geometrical distances and entropies are illustrative. **The divisor is $k$, not the sum of similarity weights.** With $\alpha=0$, this becomes a plain neighbor-entropy average. With positive $\alpha$, far neighbors contribute less. Identical features have weight one; increasing $\alpha$ cannot remove their influence. For unit features, squared distance equals $2-2\cos\theta$, connecting this geometry to angular similarity.

Although the prose motivates samples uncertain both individually and locally, Eq. 7 is an **addition**, not a product or hard AND rule. A high self-entropy sample is not rejected just because its neighbors are confident. This is a bias toward locally shared confusion, not an outlier-removal guarantee. If the representation groups images poorly, neighbors may share the wrong signal.

<details><summary>Predict: does U = 1.042 mean a 104.2% chance that the prediction is wrong?</summary><p>No. U adds entropy terms. Its scale depends on the number of classes, the neighbor entropies and distance weights; it is used to weight clustering, not to estimate error probability.</p></details>

## 5 · Weighted clustering: move the centers, then query real images

A batch should cover more than one confusing region. At C, CEC partitions visual features into $b$ clusters using uncertainty-weighted K-means, then selects the image nearest each center (§§4.3–4.4, Algorithm 1). $b$ is the number of labels purchased in the current cycle; it is distinct from $K$, the class count. Clusters need not correspond to classes.

A conventional weighted K-means objective makes the mechanism explicit:

$$
\min_{\mu_1,\ldots,\mu_b,\,a}
\sum_{x\in D_U} U(x)\|z_x-\mu_{a(x)}\|_2^2,
\qquad
\mu_r=\frac{\sum_{x:a(x)=r}U(x)z_x}{\sum_{x:a(x)=r}U(x)}.
$$

This is a **teaching expansion** of the named algorithm, not an additional numbered CEC equation. $D_U$ is the remaining unlabeled pool; $a(x)$ assigns an image to a cluster; $\mu_r$ is center $r$. The center identity holds for fixed assignments and positive total cluster weight. Large $U$ makes a point more costly to ignore, moving centers toward uncertain regions. The algorithm’s initialization and local optimum still matter.

[![Toy weighted mean moves to 1.5 between positions 0 and 2; selection chooses the closest real point, contrasting with greedy within-cluster uncertainty.](/figures/cec/clustering.svg)](/figures/cec/clustering.svg)

**Follow one cluster through C.** In a one-dimensional teaching projection, positions $0$ and $2$ have weights $1$ and $3$. Their center is $(1\cdot0+3\cdot2)/4=1.5$. The real point at $2$ is nearest, so its label is queried. These coordinates are an arithmetic illustration, not normalized CLIP embeddings or the same two-class image used above.

CEC does **not** first form ordinary clusters and then choose the maximum-entropy image within each. Weighting changes cluster geometry before representative selection; the representative itself need not have the largest $U$. Nor does it query a synthetic center: the oracle needs an actual image.

The supplied main paper names weighted K-means but leaves details such as initialization, empty-cluster handling and ties unspecified. Algorithm 1 updates the sets; the surrounding description supplies the oracle annotation step. Selected images move out of $D_U$ and into the labeled set $D_L$ with their **true acquired targets**, not their current predicted classes.

## 6 · Learn prompts, then classify without acquisition

[![Selection uses unlabeled scores and oracle queries; training updates only context vectors; inference predicts classes without calibration, clustering or oracle queries.](/figures/cec/paths.svg)](/figures/cec/paths.svg)

**The three rows have different inputs and outputs.** Selection consumes unlabeled images and returns query IDs. Training consumes collected labels and changes context vectors. Inference consumes a new image and returns a class. This is a schematic of the main CoOp setting; the paper also experiments with VPT and MaPLe, whose prompt mechanisms differ.

At D, the paper specifies supervised cross-entropy on $D_L$. Written as a teaching objective,

$$
\mathcal L(V)=-\frac1{|D_L|}\sum_{(x,y)\in D_L}\log P_y(x;V).
$$

$V$ denotes the collection of learned context vectors; $|D_L|$ is the number of labeled images; $P_y$ is Eq. 1’s probability of the acquired true class. The encoders’ **weights stay fixed**, but gradients pass through the text encoder to its context inputs. The calibrated $\widehat P$ is introduced for acquisition; the source does not introduce an extra calibrated-entropy training objective or require its use at test time.

For the toy raw prediction $(0.8,0.2)$, suppose the oracle reveals class 2. That example’s training loss is $-\log0.2=1.6094$. The familiar softmax-logit derivatives are $(0.8,-0.8)$; backpropagation then maps them through similarities and $h$ to the context vectors. This derivative is a teaching identity, not a full numerical prompt update: we have not invented a text-encoder Jacobian or context dimensions. The oracle can overturn the model’s confident guess.

Section 5 says each round reloads zero-shot CLIP and fine-tunes on the accumulated labeled set. This guide therefore shows **reset-and-retrain**, rather than assuming prompts are warm-started indefinitely. It reports initialization with “a photo of a {}”, 200 epochs, SGD learning rate 0.002, momentum 0.9, weight decay 0.005, cosine annealing and batch size 32. Those are source settings, not recommendations for every dataset; the later discussion mentions training for 50–200 epochs.

## 7 · The annotation loop carries data forward

[![Six-round toy loop starts from zero labels, acquires ten per round and retrains on accumulated oracle labels.](/figures/cec/loop.svg)](/figures/cec/loop.svg)

**Read downward through the changing datasets.** For an illustrative initial pool of 1,000 images and a fixed query batch of ten, cycle 1 leaves 10 labeled and 990 unlabeled; cycle 2 leaves 20 and 980; cycle 6 leaves 60 and 940. New prompts change probabilities, top-$N$ calibration estimates and acquisition weights. Frozen image features need not change, but uncertainty scores should be recomputed. These are bookkeeping values, not a recreation of the benchmark splits.

The paper uses six AL rounds and describes querying 1% each round. It presents 1%, 2% and 5% rows, while Table 1 calls $B$ a “per-cycle” budget. The figures and discussion treat these as successive annotation levels. We retain the published labels below without inventing the exact denominator/rounding behavior; the fixed ten-image toy explicitly uses the initial pool size. Six rounds are not six independently class-balanced few-shot draws.

Known class names allow zero-shot acquisition before $D_L$ exists. Some comparison methods instead start with random sampling because they require labeled initialization (§6). CEC does not guarantee class coverage or a minimum number of labels per class. A real system must also account for oracle errors and unequal annotation costs; the supplied experiments evaluate ordinary labels from dataset ground truth.

<details><summary>Predict: after a queried aircraft receives its label, should it remain a candidate next round?</summary><p>No. It moves into the accumulated labeled set and leaves the unlabeled pool. The next batch queries new images, while prompt training reuses all collected labeled examples.</p></details>

## 8 · Evidence: label efficiency is conditional on the setting

[![Measured Textures test accuracy bars compare Random and CEC at published 1%, 2% and 5% budgets with three-seed standard deviation whiskers.](/figures/cec/results.svg)](/figures/cec/results.svg)

**The bars are measured, unlike the earlier diagrams.** They redraw selected Table 1 means and standard deviations for **Textures / CoOp / CLIP ViT-B/16**, evaluated on the held-out test set over three seeds. Bars start at zero; whiskers show seed SD, not confidence intervals. The zero-shot reference is 44.3%.

| Published budget label | Random accuracy (%) | Raw entropy accuracy (%) | CEC accuracy (%) |
| --- | ---: | ---: | ---: |
| 1% | 38.4 ± 0.2 | 35.2 ± 0.8 | **47.9 ± 1.2** |
| 2% | 44.2 ± 2.9 | 40.9 ± 2.0 | **52.8 ± 1.0** |
| 5% | 54.1 ± 2.9 | 49.3 ± 1.4 | **58.2 ± 2.0** |

At 2%, CEC exceeds random by **8.6 percentage points**, and ALFA-Mix’s 49.6% by **3.2 points**. These are differences between reported accuracies, not relative percentage improvements. The raw entropy baseline underperforms random in these rows; that supports examining the uncertainty signal rather than assuming entropy ranking is reliable.

Table 1 covers six datasets: Textures, Caltech101, EuroSAT, FGVC-Aircraft, Flowers102 and UCF101. Mean accuracies across those datasets are 60.2%, 65.5% and 71.8% for CEC at the three displayed budgets, versus 57.1%, 62.1% and 69.2% for random and a 55.7% zero-shot average. **A better average is not a win on every dataset.** On Caltech101 at 2%, CEC is 89.0% versus CoreSet’s 91.1% and zero-shot’s 91.3%; on Aircraft at 1%, CEC’s 20.3% is below zero-shot’s 24.9%. Prompt tuning with very few labels can overfit; the authors explicitly discuss performance below zero-shot.

Table 2 extends the comparison to VPT and MaPLe on three datasets. Table 3 studies ViT-L/14 and ResNet-50 with CoOp. These support portability across the tested setups, rather than an architecture-independent guarantee. Figure 3 ablates entropy calibration and neighbor uncertainty on Textures and contrasts weighted clustering with ordinary K-means and greedy within-cluster selection; Figure 4 varies $N$, $k$ and $\alpha$. We do not assign exact numerical ablation gains by guessing pixels from the plot.

### What would you check before generalizing?

- **Hyperparameter selection:** $N=10$ is fixed; the paper grid-searches $k\in\{10,20,50\}$ and $\alpha\in\{0.05,0.1,1\}$ per dataset. The main paper names validation splits but does not fully document this selection protocol. Count validation-label costs when comparing total annotation effort.
- **Calibration versus imbalance:** test whether correcting $Q$ helps on deliberately imbalanced pools, where prediction asymmetry might reflect real class proportions. Evaluate probability calibration separately from acquisition accuracy.
- **Neighbor reliability:** perturb image features or include out-of-distribution images, then compare self-only and neighbor-augmented scores at equal label budgets. An additive neighbor term does not prove an outlier cannot be selected.
- **Compute and oracle realism:** nearest-neighbor search and repeated clustering cost time; prompt tuning is still training. The main paper reports an A5000 GPU but no complete acquisition-time benchmark. Study noisy labels and variable human labeling cost before treating a fixed image count as fixed effort.

These are proposed tests and reading limitations, not additional reported results. The supplied 11-page PDF ends with references; it points to supplementary material for details that are not included in that file. This guide uses the supplied main paper and does not silently import supplementary or newer-version settings.

## 9 · The idea to carry forward

**A useful annotation is uncertain enough to matter, representative enough to help other examples, and complementary enough to justify its place in a batch.** CEC implements this intuition through three distinct mathematical operations: class-score correction, local entropy aggregation and weighted geometry. It then uses acquired truth to learn a prompt while retaining pretrained encoders.

Compare with [LabelBench](/papers/labelbench): both separate acquisition from prediction, but LabelBench’s efficient path uses cached features and a selection proxy before final full fine-tuning. CEC’s main path uses prompt tuning with frozen encoders. Do not transfer one paper’s final-training architecture into the other.

Try drawing A–D from memory. Mark where $P$ becomes $\widehat P$, where $U$ changes cluster centers, where true labels first enter, and where gradients change $V$. If you can distinguish those four changes, you understand the mechanism.

## Sources and reading map

- [Supplied course PDF: WACV 2025, pp. 4902–4912](/papers/cec.pdf). §3 / Eq. 1 and §3.2: CLIP probabilities and CoOp; Figure 2: architecture; §4.1 / Eqs. 2–5: calibrated entropy; §4.2 / Eqs. 6–7: neighbor uncertainty; §§4.3–4.4 / Algorithm 1: weighted clustering and query loop; §§5–6 / Tables 1–3: training settings and evidence; Figures 3–4 / §6.2: ablations, hyperparameters and failure cases.
- [Official CVF proceedings record](https://openaccess.thecvf.com/content/WACV2025/html/Safaei_Active_Learning_for_Vision_Language_Models_WACV_2025_paper.html): verifies authors, venue, year and page range for the supplied paper.

The worked arithmetic, weighted K-means objective and cross-entropy derivatives are teaching explanations. No trained CLIP model runs on this page; the reported results are historical paper evidence, not a local reproduction.
