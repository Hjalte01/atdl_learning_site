---
title: "LabelBench: A Comprehensive Framework for Benchmarking Adaptive Label-Efficient Learning"
shortTitle: "LabelBench"
topic: "topic-4"
order: 1
description: "Follow the annotation loop: pretrained features, semi-supervised proxy training, active selection, and final full fine-tuning. Learn what label savings and pool accuracy actually measure."
status: "ready"
source: "https://openreview.net/forum?id=Y2QcZfwHE7"
pdf: "/papers/labelbench.pdf"
---
## The puzzle: can you afford to learn which labels to buy?

You have a million images, a strong pretrained image encoder, and money to annotate only a fraction of the images. Random sampling is cheap, but may spend labels on examples the model already understands. Active learning asks the model which examples would be informative. The catch: updating a large model after every annotation batch can consume the compute budget before you exhaust the labeling budget.

**LabelBench combines transfer learning, semi-supervised learning, and active learning in one benchmark. Its efficient path uses a small model to choose annotations, then trains the full model for the final prediction task.** It is a framework and evaluation protocol, not a single new acquisition score.

This is Topic 4, paper 1. We follow the supplied **43-page course PDF, published March 2024, revised February 2024**, including its appendices. Its first page says “Journal of Data-centric Machine Learning Research”; that is the header in this supplied copy. The [authors’ repository](https://github.com/EfficientTraining/LabelBench) confirms the matching project and separate selection/evaluation workflows. All seven local diagrams are original teaching constructions; the result chart transcribes selected Table 2 values. Allow about 20 minutes.

## 1 · Three ingredients answer three different questions

[![Transfer supplies features, semi-supervision uses unqueried inputs, and active learning chooses which examples receive oracle labels.](/figures/labelbench/problem.svg)](/figures/labelbench/problem.svg)

Read each row as a different decision. **Transfer learning** supplies an encoder already trained on a large external corpus. **Semi-supervised learning (Semi-SL)** uses both labeled and unlabeled downstream inputs during adaptation. **Active learning (AL)** chooses which currently unlabeled inputs should receive a human or oracle label next. The latter changes the dataset; Semi-SL changes how the model learns from that dataset.

Let $L$ be the set of labeled input–label pairs and $U$ the set of still-unlabeled inputs. The annotation budget counts oracle labels, not pseudo-labels. A pseudo-label is a model prediction used as a training target; it can be wrong. Benchmark “annotation” reveals stored ground truth to simulate an oracle (§4.1); it does not measure the time a real person spends on a difficult image.

A plausible design interpretation is: use the pretrained representation to make learning from few labels possible, use unqueried inputs to inform training, and use the resulting model to allocate future annotation. Then reduce the cost of the repeated model updates. This reconstructs the design logic in §§1–3, rather than claiming knowledge of the authors’ private reasoning.

<details><summary>Predict: does active learning require semi-supervised training?</summary><p>No. A model trained only on L can still select examples from U. LabelBench studies their combination because training on unqueried inputs can change both the final predictor and the selection decisions.</p></details>

## 2 · Architecture: cache features, learn a proxy, query the oracle

[![Frozen encoder computes five augmentation feature sets; a trainable proxy maps d-dimensional features to K-class probabilities, which feed acquisition before labels are revealed.](/figures/labelbench/architecture.svg)](/figures/labelbench/architecture.svg)

The top row is **precomputation**. An image augmentation $a_s(x)$ passes through a fixed encoder $E_\phi$ to produce a vector $z_s(x)\in\mathbb R^d$. Here $x$ is an input image, $s$ indexes an augmentation set, $\phi$ denotes encoder weights, and $d$ is the feature dimension. Section 3.1 caches **five sets of randomly augmented embeddings using different seeds**, cycling through them over epochs. This saves repeated encoder forward passes as well as avoiding encoder backpropagation during proxy training.

The middle row is **proxy training**. A head $h_\theta$ maps the cached vector to $K$ logits, where $K$ is the number of downstream classes and $\theta$ is the head’s trainable weights. For a linear proxy, the following is our standard functional expansion, not a numbered LabelBench equation:

$$
z_s(x)=E_\phi(a_s(x)),\qquad
\ell(x)=Wz_s(x)+b,\qquad
p_k(x)=\frac{\exp(\ell_k(x))}{\sum_{j=1}^K\exp(\ell_j(x))}.
$$

$W\in\mathbb R^{K\times d}$ and $b\in\mathbb R^K$ belong to $\theta$; $\ell\in\mathbb R^K$ is the logit vector; $p_k$ is the predicted probability of class $k$. The encoder is fixed on this path. The nonlinear proxy instead has **one hidden layer with dimension equal to the embeddings** (§4.1). We do not invent further layer widths or activation details.

The bottom row is **acquisition**. The selection strategy receives model predictions and/or features for $U$, chooses a batch $Q$, and sends its IDs to the oracle. Only then are the corresponding target labels revealed. A high uncertainty score is a request for information, not proof that the selected image is representative or misclassified.

### Where the training loss acts

A useful schematic of the middle row is:

$$
\mathcal L(\theta)=
\frac{1}{|L|}\sum_{(x,y)\in L}-\log p_y(x)
+\lambda\frac{1}{|U|}\sum_{x\in U}
\mathbf 1[\max_k p_k^w(x)\ge\tau_{\hat y}(t)]
\bigl[-\log p_{\hat y}^s(x)\bigr],
\qquad \hat y=\arg\max_k p_k^w(x).
$$

This is a **teaching abstraction of confidence-gated consistency/pseudo-label training**, grounded in §4.1 and Appendix C, not the paper’s complete FlexMatch specification. Superscripts $w,s$ mean weak and strong augmentation views; $\hat y$ is a guessed class from the weak view; $t$ is training progress; $\tau_{\hat y}(t)$ is a class- and time-dependent threshold; $\lambda$ balances the two losses. The indicator accepts or rejects that unlabeled example. Actual mini-batch sampling, threshold updates and loss normalization need the chosen trainer implementation. The expression assumes both sets are nonempty.

**Intuition first:** known labels teach the head directly; sufficiently confident predictions ask it to remain consistent under augmentation. For a labeled example whose correct class gets probability 0.4, its supervised contribution is $-\ln 0.4\approx0.916$. For an unlabeled example with weak-view maximum 0.92 and toy threshold 0.90, the gate opens. If the strong-view probability for the guessed class is 0.80, its accepted loss is $-\ln0.80\approx0.223$ before weighting. A maximum of 0.85 would close this toy gate. These thresholds and probabilities are illustrative, not tuned LabelBench settings.

Why gate? Early pseudo-labels can be unreliable. Why make thresholds adaptive? Classes need not become learnable at the same speed. The limitation is feedback: a confidently wrong guess can reinforce itself. This mechanism does not turn pseudo-labels into human ground truth.

## 3 · The representation close-up: a feature is not a label

[![Three toy examples have identical probability vectors but different two-dimensional feature locations, separating diversity from uncertainty and oracle truth.](/figures/labelbench/representations.svg)](/figures/labelbench/representations.svg)

This diagram is a two-dimensional toy, not a projection measured from CLIP. A and B have nearly identical feature locations; C lies far away. All have the same three-class prediction. An uncertainty-only score ties them, while a diversity strategy can prefer a batch that covers separated feature locations. The actual feature space has dimension $d$; “near” is defined by the representation and distance rule, not necessarily by task-relevant similarity.

Appendix B describes several interchangeable strategies:

- **Confidence:** query the smallest maximum class probability.
- **Entropy:** query the largest uncertainty across the probability vector.
- **Margin:** query the smallest gap between the top two class probabilities.
- **CORESET:** approximate a k-center coverage objective in feature space.
- **BADGE:** combine uncertainty and diversity through k-means++ in hallucinated gradient space.
- **BAIT:** use a Fisher-information selection objective; **GALAXY:** use a graph-based strategy combining uncertainty and diversity.

These strategies need not choose the same images. Probability calibration, feature quality, class imbalance and duplicate examples all affect whether their scores identify useful annotations. BAIT is evaluated only on CIFAR-10, and GALAXY is omitted on ImageNet because of computational costs (Appendix F).

## 4 · Follow one annotation through the numbers

[![A toy proxy predicts three candidate probability vectors; margin selects B, the oracle supplies class 2, and retraining precedes the next acquisition.](/figures/labelbench/trace.svg)](/figures/labelbench/trace.svg)

At the acquisition box, define three scores (Appendix B; our notation):

$$
c(x)=\max_k p_k(x),\qquad
H(x)=-\sum_{k=1}^K p_k(x)\ln p_k(x),\qquad
m(x)=p_{(1)}(x)-p_{(2)}(x).
$$

$c$ is confidence; $H$ is entropy in natural-log units; $p_{(1)}\ge p_{(2)}$ are the largest two probabilities; $m$ is their margin. Query low $c$, high $H$, or low $m$ respectively. Use the limit $0\ln0=0$ if a probability is zero. None of these scores accesses the hidden label.

| Toy candidate | Predicted probabilities | Confidence ↓ | Entropy ↑ | Margin ↓ |
| --- | --- | ---: | ---: | ---: |
| A | (0.80, 0.15, 0.05) | 0.80 | 0.613 | 0.65 |
| B | (0.45, 0.40, 0.15) | 0.45 | 1.010 | 0.05 |
| C | (0.60, 0.20, 0.20) | 0.60 | 0.950 | 0.40 |

For a one-label batch, all three strategies select B in this particular example. For B, entropy is $-(0.45\ln0.45+0.40\ln0.40+0.15\ln0.15)\approx1.010$, and margin is $0.45-0.40=0.05$. The model predicts class 1, but suppose the oracle says class 2. Store the oracle label, not the guess. That changes the next supervised training loss, which changes future predictions and potentially future selected examples.

<details><summary>Predict: would copying B many times make a large uncertainty-selected batch more diverse?</summary><p>No. The copies can all have the same uncertainty score. Selecting them spends several labels on similar information. Diversity-aware acquisition addresses a different property; its success still depends on the representation and objective.</p></details>

### A scalable mathematical trick inside BADGE

Appendix E speeds up distances between gradient embeddings. If $v_i\in\mathbb R^d$ is an example’s feature and $q_i\in\mathbb R^K$ its class residual, set $g_i=\operatorname{vec}(q_iv_i^\top)$. The paper uses $q_{i,k}=\mathbf1[k=\hat y_i]-p_{i,k}$; $\hat y_i$ is the model’s guessed label, not an oracle label. This sign convention is the negative of the usual cross-entropy logit gradient, but a global sign flip preserves pairwise distances.

Instead of constructing every $Kd$-dimensional vector, calculate:

$$
\|g_i-g_j\|^2=
\|q_i\|^2\|v_i\|^2+\|q_j\|^2\|v_j\|^2
-2(q_i^\top q_j)(v_i^\top v_j).
$$

This identity lives in the **BADGE selection box**, not the training loss. The Frobenius inner product of two outer products factorizes, so each distance uses $O(K+d)$ rather than $O(Kd)$ work, with norms reusable. A tiny check: $q_i=q_j=(1,-1)$, $v_i=(1,0)$ and $v_j=(0,1)$ give squared distance $2+2-0=4$. The paper reports a 512-fold computation reduction for this distance computation in its ImageNet setting; that is not a 512-fold end-to-end training speedup.

## 5 · The annotation loop carries data, not a frozen decision list

[![Toy pool of ten examples progresses from two random labels to four and six adaptive labels, then stops at its budget and trains the full model.](/figures/labelbench/loop.svg)](/figures/labelbench/loop.svg)

Read left to right, then continue on the next row. This is a toy pool of ten inputs, total budget $B=6$, annotation batch size $b=2$. The initial random batch counts toward $B$. The paper uses a uniformly random first batch, then adaptive batches (§4.1).

For round $r$, the dataset-state update is:

$$
Q_r\subseteq U_r,\qquad
L_{r+1}=L_r\cup\{(x,y_{\mathrm{oracle}}(x)):x\in Q_r\},\qquad
U_{r+1}=U_r\setminus Q_r.
$$

$Q_r$ is the selected batch; $y_{\mathrm{oracle}}$ supplies its true labels. This is our set notation for the protocol, not a paper-numbered equation. The proxy is trained **from scratch** for the updated dataset; its weights are not simply carried forward as a warm start (§4.1). Dataset partitions, cached features and collected annotations survive the loop.

Smaller annotation batches mean more chances to adapt decisions, but also more retraining and oracle interactions. Section 4.4 reports that proxy selection with batches of 200 reached 97.75% CIFAR-10 accuracy with 2,750 labels, compared with random sampling’s 10,000-label reference. That corresponds to a calculated 72.5% label saving for that reported target, not a guarantee for a new task. It is distinct from the introductory “more than four-fold” claim for the full-selection setting.

## 6 · Training and inference are different paths

[![Repeated proxy learning updates only the head; final FlexMatch fine-tuning updates the full encoder and head; prediction on a test image queries no oracle.](/figures/labelbench/paths.svg)](/figures/labelbench/paths.svg)

The first row is repeated many times. The second is the final model training after collecting labels. The third predicts a new image. **A proxy selects the data; it is not the final evaluated predictor.** Final end-to-end FlexMatch uses the collected labeled examples **and the remaining unlabeled inputs**, regardless of whether acquisition used a proxy or full fine-tuning (§4.1). The full-selection baseline retrains the large model during acquisition as well.

There is a subtle cost distinction. Producing curves at several label budgets may require separate full-model evaluations at checkpoints. The efficient deployed annotation workflow can reserve full training for the last batch; do not describe an entire multi-budget benchmark as literally needing only one full-model training run.

The full fine-tuning model uses a zero-shot prediction linear head at initialization; proxy heads start with random weights (§4.1). A “linear probe” in this paper refers to the head’s architecture and frozen encoder, not necessarily exclusively supervised training.

<details><summary>Predict: if the proxy is cheap, can we reuse its cached features while updating the final encoder?</summary><p>Not as current outputs of that changing encoder. Cached embeddings came from fixed weights. Final end-to-end fine-tuning must permit the encoder to change; using only its old cached vectors would remove that part of the update.</p></details>

## 7 · Evidence: label efficiency, compute, and two accuracies

[![Selected Table 2 bars show test and whole-pool accuracy for Random and Confidence selection with full, shallow, and linear selection models.](/figures/labelbench/results.svg)](/figures/labelbench/results.svg)

This chart contains **measured historical values transcribed from Table 2**, not simulated outcomes. Bars start at zero; use the table for the smaller differences. All rows use ImageNet, CLIP ViT-B32, a 400,000-label budget, and final end-to-end FlexMatch. Selection models differ. The reported uncertainty is standard error over two trials, not a 95% confidence interval.

| Acquisition / model used to select | Test accuracy (%) | Whole-pool accuracy (%) |
| --- | ---: | ---: |
| Random | 76.12 ± 0.14 | 83.35 ± 0.01 |
| Confidence / full fine-tuning | 77.38 ± 0.13 | 90.11 ± 0.01 |
| Confidence / shallow proxy | 76.96 ± 0.12 | 88.93 ± 0.01 |
| Confidence / linear proxy | 77.23 ± 0.10 | 89.01 ± 0.02 |

Confidence selection with a linear proxy improves test accuracy by **1.11 percentage points** over random in this setting, and trails full-model Confidence selection by 0.15 points. Its pool accuracy trails the latter by 1.10 points. The source’s “Best” row chooses winners separately across columns; it is not a single acquisition method, so this guide uses the actual Confidence row instead.

### Why pool accuracy is different

Test accuracy evaluates predictions on unseen test inputs. Pool accuracy evaluates the completed labels for the training pool: oracle labels for queried inputs and model labels for the remainder (§4.2). Under a perfect-oracle assumption, with $N$ total pool inputs, $n_L$ queried inputs and residual unqueried prediction accuracy $a_U$, our teaching identity is:

$$
a_{\mathrm{pool}}=\frac{n_L+(N-n_L)a_U}{N}.
$$

For $N=100$, $n_L=20$ and $a_U=0.75$, pool accuracy is $(20+80\times0.75)/100=0.80$. It can improve because more labels are known exactly, even if residual prediction accuracy stays constant. This identity applies to ordinary accuracy; macro F1 does not decompose this way. Human annotation errors would also invalidate the perfect-oracle numerator.

The benchmark includes CIFAR-10, CIFAR-100, ImageNet, iWildCam and fMoW. Although the latter two come from WILDS, the paper evaluates **in-domain test performance**, leaving distribution-shift evaluation for future work (§§4.2–4.3, 5). For imbalance it includes balanced accuracy and macro F1, defined in Appendix A. Balanced accuracy averages per-class recall; macro F1 averages per-class F1. Neither lets a common class dominate simply through its count. Figure 4 has inconsistent metric labels for CIFAR-100; use the explicitly labeled tables rather than infer its metric from the caption alone.

### What is the compute saving?

Table 1 estimates training cost for ImageNet, 600,000 collected labels, 20 annotation iterations, **12 trials**, CLIP ViT-B32 and FlexMatch. It lists full-selection stage GPU hours of 0 precompute, 1,900 retraining and 100 final training, versus shallow-proxy stage hours of 5, 57 and 100: totals of 2,000 versus 162, about **12.35×** by our division.

The table’s dollar column is internally inconsistent: full-selection stages $5,700 + $300 sum to $6,000, but the printed total is $6,300. The proxy stages $15 + $180 + $300 sum to its printed $495. Treat these as historical source estimates, not current cloud prices or a precise pricing model. The broad GPU-hour reduction is supported by the listed hours; we do not silently repair the printed dollar total.

## 8 · Limits and experiments worth doing

The supplied benchmark supports comparing combinations under specified settings. It does not establish that the most uncertain examples always have the best learning value, or that proxy selection always matches full selection.

- **Representation dependence:** a fixed encoder may miss distinctions the downstream task requires. Compare proxy and full acquisition with the same budget and final trainer; inspect whether their selected sets cover different rare classes.
- **Trainer dependence:** Figure 6 and Appendix F.7 show that Semi-SL choice changes label efficiency. Hold acquisition, backbone, seeds, batches and final evaluation fixed when comparing trainers.
- **Small samples and imbalance:** ImageNet results use two trials; other reported experiments often use four, with exceptions explicitly labeled. Appendix F notes high iWildCam variability and rare classes with no annotations. Report per-class coverage and repeated-seed uncertainty.
- **Hidden annotation overhead:** Appendix D uses a separate validation set, typically around 10% of the pool, and tunes on a randomly queried subset around 10%. Annotation-budget curves alone are not an accounting of every label used for development or of the external pretraining labels.
- **Augmentation and cache limits:** five cached feature sets approximate augmentation diversity. Compare against more cached views or fresh encoder passes under a fixed compute budget before attributing all gains to acquisition.
- **Real annotation and shift:** the oracle is perfect and evaluation is in-domain. Test noisy annotators, unequal annotation times, and shifted inputs before applying the headline savings elsewhere.

These are limitations and proposed tests, not additional experiments we ran. This learning guide does not train LabelBench or reproduce its reported models.

## 9 · The idea to carry forward

**Spend cheap computation repeatedly to decide where expensive information should come from; spend expensive model adaptation when the collected information justifies it.** The proxy and final model have different jobs, while their shared annotation history connects them.

Redraw the two training paths. Mark which weights change, where hidden labels become visible, and which unlabeled inputs still contribute to the final loss. Then ask whether your metric measures prediction on new inputs or completion of the existing pool. Those distinctions explain the mechanism and prevent misleading claims about label savings.

## Sources and reading map

- [Supplied course PDF, March 2024 publication, 43 pages](/papers/labelbench.pdf): §§1–2 for motivation; Figure 2 and §§3.1, 4.1 for architecture and loop; §3.2 for modularity; §4.2 and Appendix A for metrics; Table 1 for estimated compute; Table 2 for the chart; §4.4 and Figures 5–6 for tradeoffs; §5 for proposed extensions.
- Appendices B–C define acquisition and Semi-SL strategies; Appendix D describes validation/tuning; Appendix E gives the BADGE distance factorization and BAIT optimization; Appendix F gives additional settings and results.
- [Paper’s review record](https://openreview.net/forum?id=Y2QcZfwHE7). The supplied PDF is the source for version-specific values; the review site required browser verification during authoring.
- [Authors’ LabelBench repository](https://github.com/EfficientTraining/LabelBench) documents separate selection and final evaluation entry points. This guide follows the supplied paper, not an asserted reproduction of the repository’s current revision.

Equations outside the numbered appendix metrics and explicitly identified BADGE identity are pedagogical expansions. The teaching schematics use a repeated teal → purple → green palette to make the successive boxes easy to follow; those colors are not class labels. Open the linked diagrams to enlarge them.
