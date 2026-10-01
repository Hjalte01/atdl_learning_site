---
title: "SUGFW+: An Uncertainty-guided Feature Weighting Framework for Cold Start Active Adaptation of SAM in Medical Image Segmentation"
shortTitle: "SUGFW+"
topic: "topic-4"
order: 5
description: "Select one unlabeled image batch with uncertainty-weighted SAM features, then turn uncertainty into an automatic decoder prompt for low-budget adaptation."
status: "ready"
source: "https://arxiv.org/abs/2608.16110v1"
pdf: "/papers/sugfw-plus.pdf"
---
## The puzzle: choose useful masks before you have a task model

You have a pool of prostate MRI slices, no target-domain annotations, and enough time to delineate only a few slices. Training an uncertainty model first would already require labels. Training a self-supervised encoder costs compute and does not directly tell you which regions are ambiguous for segmentation. Can a pretrained segmentation foundation model supply both a useful geometry and a useful initial signal?

**SUGFW+ uses pretrained SAM to select one image batch, obtains human masks for that batch, and then adapts SAM with uncertainty as a learned spatial prompt.** The plus matters: the earlier SUGFW selected data for a separate U-Net; this extension also changes downstream model training.

This guide follows the supplied **14-page arXiv:2608.16110v1 PDF, dated 17 August 2026**, rather than the 2025 conference predecessor. The [authors’ official repository](https://github.com/HiLab-git/SUGFW-plus) confirms the method identity and its selection/adaptation stages; its citation block still points to the earlier conference paper. The supplied PDF controls all results here. All diagrams except the explicitly measured result chart are teaching schematics. No SAM model runs in this guide. Allow about 25 minutes; prerequisites are entropy, softmax, clustering and supervised segmentation.

## 1 · Cold start is one annotation round

[![Conventional active learning needs initial target labels and repeated query rounds; SUGFW+ uses pretrained SAM to query once and adapt afterward.](/figures/sugfw-plus/problem.svg)](/figures/sugfw-plus/problem.svg)

Read the rows as contrasting workflows. The target pool is $\mathcal D=\{X_i\}_{i=1}^N$, where $N$ is the number of available training images or slices. A selected subset $\widehat{\mathcal D}$ contains $M\ll N$ images. The human supplies dense task masks only after all $M$ images have been selected (§3.1).

Cold start means no initial **target-domain** labeled set. SAM still brings substantial pretrained knowledge. And one query round can contain several **computational selection steps**: choosing the next image greedily does not imply annotating and retraining after each choice.

A plausible design interpretation is: use SAM’s spatial features to preserve coverage, use its response to perturbations to emphasize ambiguity, avoid trusting that ambiguity as exact task uncertainty, then reuse it to guide adaptation. This explains the design logic; it is not a claim about the authors’ private reasoning.

<details><summary>Predict: can the selection stage inspect a candidate’s prostate mask?</summary><p>No. The initial target pool is unlabeled. SAM supplies class-agnostic region predictions, while the true target masks arrive only after selection. Dense test masks are used to evaluate performance, not to choose the query batch.</p></details>

## 2 · Follow the full architecture

[![PFUC produces a spatial feature tensor and uncertainty grid, PGDR produces a feature vector and scalar, GSCU selects one batch, and UPFT adapts SAM.](/figures/sugfw-plus/architecture.svg)](/figures/sugfw-plus/architecture.svg)

**A → B:** PFUC, or Patch-level Feature and Uncertainty Calculation, produces spatial features and surrogate uncertainty from fixed pretrained SAM. **B → C:** PGDR, or Patch-based Global Distinct Representation, summarizes them into two different image-level objects. **C → D:** GSCU, or Greedy Selection with Cluster and Uncertainty, picks images for human annotation. UPFT, or Uncertainty-Prompted Fine-Tuning, uses those labels and stored uncertainty maps to adapt SAM.

At A, paper Eq. 1 defines

$$
F_i=E_{\mathrm{img}}(X_i),\qquad F_i\in\mathbb R^{C_0\times H'\times W'}.
$$

$X_i$ is an image with spatial size $H\times W$; $E_{\mathrm{img}}$ is SAM’s image encoder; $C_0$ is its feature channel count; $H'\times W'$ is the encoder’s spatial grid. A patch feature $F_i(r)$ is a vector in $\mathbb R^{C_0}$ at grid position $r$. This is a spatial representation, not one global classification token.

For an automatic point grid $\mathcal P$, paper Eq. 2 applies SAM’s original prompt encoder $E_p$ and mask decoder $D_m$:

$$
M'_i=D_m\bigl(E_{\mathrm{img}}(X_i),E_p(\mathcal P)\bigr).
$$

$M'_i$ represents a collection of predicted regions, rather than target-class probabilities. The main experiment uses SAM **ViT-Base**, a $32\times32$ grid of 1,024 prompt points, and $K=10$ augmented views (§4.2). Preprocessing resizes images/slices to $512\times512$ and clips intensities above the 99.5th percentile. These documented values do not specify every internal tensor dimension or decoder implementation; the diagram stays functional.

## 3 · Turn class-agnostic regions into surrogate uncertainty

[![Four perturbed binary union masks yield mean foreground 0.5 and entropy 0.6931, while a stable foreground patch has zero entropy without guaranteed correctness.](/figures/sugfw-plus/representation.svg)](/figures/sugfw-plus/representation.svg)

The representation close-up follows A. “Everything mode” returns many regions. To reject noise and presumed background, paper Eq. 3 unions only regions whose area is **strictly smaller than half the image**:

$$
M_i=\bigcup_{n:\,|M'_i(n)|<HW/2} M'_i(n).
$$

$n$ indexes a predicted region and $|M'_i(n)|$ is its area. $M_i$ is a binary foreground union. It is not a prostate or liver label, and an area filter cannot guarantee semantic correctness. An actual target occupying more than half the image can make this heuristic especially questionable.

Apply intensity and spatial transformations to $X_i$ and obtain union masks $M_i^k$ for views $k=1,\ldots,K$. Paper Eq. 4 averages them:

$$
\overline M_i=\frac1K\sum_{k=1}^{K}M_i^k.
$$

Averaging binary values produces fractional foreground agreement. The paper downsamples this mean mask to match the feature grid; call the downsampled value $q_i(r)\in[0,1]$. Paper Eq. 5 computes binary entropy at each grid position:

$$
U_i(r)=-q_i(r)\ln q_i(r)-\bigl(1-q_i(r)\bigr)\ln\bigl(1-q_i(r)\bigr).
$$

$U_i\in\mathbb R^{H'\times W'}$ is the surrogate uncertainty map. Natural logarithms give nats; $0\ln0$ is defined by its limiting value zero. $q=.5$ gives $U=\ln2\approx.6931$, while $q=0$ or $1$ gives zero. Thus changing agreement changes the weights at B even though no human target is available.

**Why surrogate?** The signal describes class-agnostic foreground agreement under perturbations, not calibrated uncertainty about the requested anatomy. SAM can agree confidently on the wrong region. It also is not BALD: no expected per-model entropy is subtracted. Spatially transformed masks need a common coordinate frame for this averaging to be meaningful; §3.2.1 does not spell out inverse-transform handling, interpolation or overlapping-mask resolution. We do not invent those details.

<details><summary>Predict: ten views all miss the same organ. Is its surrogate uncertainty high?</summary><p>No. If the averaged foreground value is zero, entropy is zero. Low perturbation uncertainty can coexist with a serious systematic segmentation error.</p></details>

## 4 · Weight the feature, average the uncertainty

[![Two scalar patch features 2 and 10 have uncertainties zero and log 2; lambda 6 gives weights 1/65 and 64/65, feature 9.8769 and scalar uncertainty 0.3466.](/figures/sugfw-plus/numeric.svg)](/figures/sugfw-plus/numeric.svg)

This is a worked teaching example through the same A → B boxes, using two scalar patch features instead of a real SAM grid. PGDR gives ambiguous patches more influence on the image representation. With $R=H'W'$, paper Eq. 6 is

$$
a_i(r)=\frac{\exp\bigl(\lambda U_i(r)\bigr)}{\sum_{s=1}^R\exp\bigl(\lambda U_i(s)\bigr)},\qquad
f_i=\sum_{r=1}^R a_i(r)F_i(r).
$$

$a_i(r)$ is the normalized patch weight; $\lambda\ge1$ amplifies uncertainty differences; $f_i\in\mathbb R^{C_0}$ is the resulting global feature. $s$ is merely a second grid index in the denominator. The source uses $R$ in that denominator without a separate definition; here it explicitly equals the number of spatial patches.

Take $F_i(1)=2$, $F_i(2)=10$, $U_i(1)=0$, $U_i(2)=\ln2$, and the main setting $\lambda=6$. Then

$$
[e^{6U_i(1)},e^{6U_i(2)}]=[1,64],\quad
[a_i(1),a_i(2)]=[1/65,64/65],\quad f_i=\frac{2+64\cdot10}{65}\approx9.8769.
$$

Uniform pooling would give 6. The uncertainty weighting moves the feature toward the ambiguous patch. The weights are nonnegative and sum to one, so this operation forms a mixture of existing features. It does not manufacture a new ground-truth label or necessarily normalize the feature vector’s length.

The second output at B is deliberately different. Paper Eq. 7 uses an **unweighted** mean:

$$
u_i=\frac1R\sum_{r=1}^R U_i(r)=\frac{\ln2}{2}\approx0.3466.
$$

$u_i$ is one scalar uncertainty per image; it is not the feature vector $f_i$ and not an uncertainty-weighted entropy average. GSCU clusters using $f_i$ but chooses within clusters using $u_i$.

**Why normalize the weights?** Otherwise feature magnitudes would also scale with the total exponential mass. If all patch uncertainties are equal, softmax gives uniform pooling. Raising $\lambda$ concentrates weights, potentially magnifying an irrelevant SAM ambiguity. Figure 3 analyzes this sensitivity; §4.2 chooses $\lambda=6$ from the ablation results.

## 5 · Greedy selection covers two different spaces

[![A toy greedy trace chooses uncertainty 0.3 from cluster A, 0.6 from B and 0.05 from C by maximizing nearest uncertainty distance.](/figures/sugfw-plus/selection.svg)](/figures/sugfw-plus/selection.svg)

Read the figure chronologically. At C, K-means partitions the global feature vectors into $C=M$ clusters, where $M$ is the annotation budget. This $C$ is a cluster count, not a semantic class count; $K$ earlier counted augmented views. Paper Eq. 8 assigns an image to its closest centroid:

$$
c_i=\arg\min_{c\in\{1,\ldots,C\}}\|f_i-\mu_c\|^2.
$$

$\mu_c$ is cluster $c$’s centroid and $c_i$ is the image’s assignment. The method selects one **actual image** per cluster. It does not annotate a mathematical centroid.

First visit a random cluster and choose its median-uncertainty image. Then visit a new random cluster $\mathcal C$. Paper Eq. 9 selects

$$
X_{m+1}=\arg\max_{X_i\in\mathcal C}\ \min_{X_j\in\mathcal D^m}|u_i-u_j|.
$$

$\mathcal D^m$ is the set of images already selected after $m$ picks. The inner minimum measures how close a candidate’s scalar uncertainty is to its nearest selected uncertainty. The outer maximum chooses the largest such distance. The absolute value matters: this covers the uncertainty spectrum, including low and high values, rather than repeatedly taking the largest uncertainty.

In the toy, cluster A has uncertainties $[.1,.3,.5]$, so its median is $.3$. Cluster B offers $.2$ and $.6$, at distances $.1$ and $.3$ from the selected value; pick $.6$. Cluster C offers $.05$, $.4$, $.65$. Their nearest distances to $\{.3,.6\}$ are $.25$, $.1$, $.05$, so select $.05$. The third pick is the **least uncertain** candidate in C.

After all $M$ picks, annotate the batch and begin D. Cluster visitation randomness and unspecified tie handling can change the batch. One-per-cluster encourages feature coverage, while max-min encourages scalar-uncertainty coverage; neither guarantees true-class balance or complete anatomical diversity. Randomly choosing within uncertainty-weighted clusters is the paper’s “w/o GSCU” ablation, not its full algorithm.

## 6 · Train with uncertainty as a decoder prompt

[![UPFT sends image features and encoded uncertainty to the mask decoder, while a separate predictor learns to imitate the prompt embedding with MSE.](/figures/sugfw-plus/training.svg)](/figures/sugfw-plus/training.svg)

The training figure expands D. SAM’s original prompt encoder is replaced by an uncertainty encoder $E_{\mathrm{un}}$. It maps stored surrogate map $U_i$ to embedding $F_i^u=E_{\mathrm{un}}(U_i)$, which is injected as queries in the mask decoder’s cross-attention (§3.3). Here superscript $u$ marks an uncertainty **embedding**, unlike scalar $u_i$.

Paper Eq. 12 predicts the task segmentation:

$$
P_i=D_m\bigl(E_{\mathrm{img}}(X_i),F_i^u\bigr).
$$

$P_i$ is a task segmentation prediction supervised by human mask $y_i$. The old automatic foreground union was an acquisition signal; it is not substituted for $y_i$ as a pseudo-label. The paper names Dice and cross-entropy losses without documenting every smoothing, reduction and class-decoding choice. A binary teaching CE at one labeled pixel with correct-class probability $.8$ is $-\ln(.8)\approx.2231$; this illustrates the human-supervised loss arrow, not the full implementation.

To make the same prompt available efficiently at inference, a lightweight transformer $g$ predicts its embedding from image features and a pool-wide mean map. Paper Eqs. 10–11 are

$$
\widehat F_i^u=g(F_i,\overline U),\qquad
\mathcal L_u=\|\widehat F_i^u-F_i^u\|_2^2,\qquad
\overline U=\frac1N\sum_{i=1}^N U_i.
$$

$\overline U$ is the average surrogate map across the **training pool**, including its unlabeled images; it is not a test mask or a new test-image uncertainty map. $\widehat F_i^u$ is the predicted embedding. The target $F_i^u$ is itself produced by a trainable uncertainty encoder, so this is learned embedding imitation, not a calibrated entropy regression target. The paper does not specify stop-gradient handling or all widths/depths of these modules.

If a toy target embedding is $(.2,.6)$ and the predictor returns $(.1,.4)$, the squared-error sum is $.01+.04=.05$. This loss trains the predictor path; segmentation losses train the decoder path receiving $F_i^u$.

Paper Eq. 13 combines them:

$$
\mathcal L=\omega\mathcal L_{\mathrm{Dice}}+(1-\omega)\mathcal L_{\mathrm{CE}}+\alpha_t\mathcal L_u,\qquad
\alpha_t=\min\left(\alpha_{\max},\alpha_{\max}\frac tT\right).
$$

$\omega$ balances the supervised losses, $t$ is training iteration, $T$ is the warm-up length, and $\alpha_{\max}$ caps the imitation-loss weight. The main setting uses $\omega=.8$, $\alpha_{\max}=.02$, $T=300$. Thus $\alpha_t$ is 0 at $t=0$, $.01$ at 150 and $.02$ at 300 and afterward. With illustrative losses $.2$, $.3$, $.05$ at iteration 150, the total is $.8(.2)+.2(.3)+.01(.05)=.2205$. Ramp-up delays the full auxiliary contribution while the representations begin adapting; it does not prove stability. **Source inconsistency:** §4.2 specifies alpha-max 0.02, but §4.4.3 says the chosen optimum is 0.20. The example follows the explicit implementation setting of 0.02; reproducing the study requires resolving that tenfold discrepancy.

Training updates **rank-32 LoRA in the frozen image encoder, the mask decoder, the uncertainty encoder and the uncertainty predictor**. Section 4.2 specifies 300 epochs, batch size 16, AdamW, initial learning rate $.0008$, weight decay $.1$ and optimizer betas $(.9,.999)$. Figure 1 also depicts an adapter; the written update/implementation description names LoRA and the three other trainable modules. We follow that description without inventing an extra adapter configuration or parameter schedule.

## 7 · Inference replaces a branch, not a target mask

[![Training encodes a stored uncertainty map; inference discards that encoder and uses the learned embedding predictor with image features and the training-pool mean map.](/figures/sugfw-plus/inference.svg)](/figures/sugfw-plus/inference.svg)

Read the top two rows as training versus inference. At test time, keep the adapted image encoder, predictor $g$, mask decoder and training-pool mean $\overline U$. Discard $E_{\mathrm{un}}$. For a new image $X$, compute features $F$, predict $\widehat F^u=g(F,\overline U)$, and supply that predicted embedding to $D_m$ in place of $F^u$.

This removes the need to produce a fresh $K$-view automatic-mask uncertainty map for each test image. There is no manual point/box, no new annotation query and no gradient update. The trained predictor must approximate the prompt that training taught the decoder to use. A small training embedding error does not guarantee low segmentation error on an unusual anatomy or modality.

<details><summary>Predict: does inference run the uncertainty encoder on the average map?</summary><p>No. The average map is an input to g together with the image features. The uncertainty encoder is discarded; the predictor’s output replaces its embedding in the mask decoder.</p></details>

## 8 · Evidence: budgets count slices and images

[![Measured DSC means compare Random and SUGFW+ for Promise12 at 3 percent, UTAH at 0.10 percent, Liver at 0.05 percent and ISIC at 0.50 percent, all using UPFT training.](/figures/sugfw-plus/results.svg)](/figures/sugfw-plus/results.svg)

The chart transcribes mean DSC from v1 Tables 1–4 at the lowest reported budget for each dataset. Bars start at zero and omit spread; the table below retains the printed ± values. These are reported experimental measurements, not generated segmentation outputs or a reproduction run.

| Dataset / target | Training pool | Query budget | Random DSC (%) | SUGFW+ DSC (%) |
| --- | ---: | ---: | ---: | ---: |
| Promise12 / prostate MRI | 960 slices | 3.00% | 84.03 ± 5.87 | 86.38 ± 5.60 |
| UTAH / left atrium MRI | 5,041 slices | 0.10% | 70.96 ± 11.96 | 78.70 ± 5.66 |
| MSD Liver / liver CT | 13,792 slices | 0.05% | 68.99 ± 9.51 | 82.27 ± 4.67 |
| ISIC 2018 / skin lesion | 2,595 images | 0.50% | 83.67 ± 15.75 | 86.36 ± 13.50 |

Promise12 uses train/validation/test proportions 5:2:3, UTAH uses 80/20/54 volumes, and Liver uses 7:1:2. ISIC’s listed splits are 2,595/100/1,000 images. The paper says ISIC contains 3,694 images, although those split counts sum to 3,695; this source inconsistency does not alter the stated 2,595-image training denominator. Fractional budgets require integer rounding; §4.2 does not specify its exact rule, so we do not assert exact selected counts from these percentages.

The segmentation backbone is **the same UPFT-based SAM** across the main selection comparisons. Competing selection methods use the same SAM ViT-Base feature encoder with global average pooling (§4.3). The CEC implementation here is a SAM-feature comparator; it is not a replication of the previous guide’s original CLIP prompt-learning protocol. This comparison tests sample selection under a common downstream trainer, while Table 6 separately changes training methods with fixed SUGFW-selected samples.

DSC measures overlap, $2|P\cap G|/(|P|+|G|)$, for predicted and ground-truth foreground sets $P,G$. HD95 measures the 95th-percentile boundary distance; lower is better. The 3D datasets are queried and trained as **2D slices**, then predictions are stacked and scored in 3D. ISIC’s HD95 is reported in pixels; the three volumetric datasets use millimeters. No voxel budgets or measured expert-hour savings are established by the slice/image fractions.

**Read the actual table exceptions.** At Liver budget 0.20%, TypiClust has DSC **91.58** versus SUGFW+ **91.53**. At Liver 0.05%, ProbCover has HD95 **22.48 mm** versus SUGFW+ **29.00 mm**; at 0.06%, it has **11.19 mm** versus **16.14 mm**. These qualify the narrative claim of consistent superiority. A method can improve overlap while giving worse boundary distances.

The table captions mark paired-test $p<.05$ against the second-best results where a star is printed. They do not make all differences significant. The method/implementation sections do not clearly state the number of repeated selection/training runs or the aggregation behind every ± value; do not interpret them as confidence intervals or a documented five-seed protocol. Hyperparameters were chosen from ablation performance, and the text does not fully explain a separate validation-selection protocol for those choices.

### What do the ablations isolate?

Table 5 keeps UPFT training and changes the query strategy. At Promise12 3%, full SUGFW gives DSC **86.38**, uncertainty-only **85.73**, no PGDR **85.13**, and no GSCU **81.22**, compared with Random **84.03**. In particular, the no-GSCU variant is *below* Random, despite the accompanying narrative calling Random lowest. Uncertainty-weighted representation alone is not sufficient here.

Table 6 fixes SUGFW-selected examples and changes the trainer. On Promise12 3%, SAM adaptation without uncertainty gives **85.01** versus full UPFT **86.38**; on ISIC 0.5%, it gives **86.02** versus **86.36**. Trainable parameter counts are **28.67M** versus **29.86M**. This supports the complete uncertainty module in these settings, but does not individually isolate the encoder, predicted-prompt approximation and ramp-up loss. The prose mistakenly refers to Table 5 while discussing this trainer comparison; the relevant table is 6.

Section 4.5 compares the extension with U-Net-based SUGFW and full-supervision baselines. Beating a fully supervised U-Net at a small label fraction combines a changed pretrained backbone and a changed training strategy; it is not proof that fewer labels inherently beat full supervision with identical SAM adaptation.

## 9 · Limits and the idea to carry forward

The authors identify selection overhead from $K$ augmented everything-mode passes and the restriction to 2D SAM (§5). Additional reading questions follow from the mechanism: does the foreground-area filter preserve the anatomy of interest, does surrogate uncertainty track useful annotation value, and how well does the predicted prompt match the training prompt out of distribution?

Useful proposed tests are to hold the trainer fixed while varying spatial-alignment handling and the foreground filter, repeat random cluster orders and training seeds, compare measured annotation time at matched quality, and replace the test-time predictor with the actual encoded uncertainty map to estimate approximation cost. These are proposed experiments, not findings of this paper.

**The idea to carry forward:** an uncertain region can influence both **which example you label** and **how the model processes that example**. SUGFW+ uses a surrogate twice: first as a feature-pooling and selection signal, then as a decoder prompt learned well enough to be predicted at inference. Explain where each use happens, which human targets supervise training, and which branch is replaced at test time; that is the core mechanism.

## Sources and reading map

- [Supplied course PDF, arXiv v1, 17 August 2026](/papers/sugfw-plus.pdf): §3.1 for the one-query setting; §3.2.1 / Eqs. 1–5 for features, region filtering and entropy; §3.2.2 / Eqs. 6–7 for weighted pooling; §3.2.3 / Eqs. 8–9 for GSCU; §3.3 / Eqs. 10–13 and Figure 1 for training/inference; §4.1–4.2 for data and hyperparameters; Tables 1–4 for results; Tables 5–6 / Figure 3 for ablations; §4.5 / Figure 4 and §5 for comparisons and limitations.
- [Version-specific arXiv record](https://arxiv.org/abs/2608.16110v1): identifier/date transcribed from the supplied PDF. The record could not be retrieved during authoring, so no online text or later version was substituted.
- [Official SUGFW+ implementation](https://github.com/HiLab-git/SUGFW-plus): confirms identity and the two-stage workflow; code-level reproduction is outside this guide.

All toy arithmetic and colored diagrams are pedagogical constructions; all stated benchmark values come from the supplied version.
