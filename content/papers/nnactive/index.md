---
title: "nnActive: A Framework for Evaluation of Active Learning in 3D Biomedical Segmentation"
shortTitle: "nnActive"
topic: "topic-4"
order: 4
description: "How partial-label 3D training, stronger random baselines and foreground-aware cost accounting change the verdict on active learning."
status: "ready"
source: "https://openreview.net/forum?id=AJAnmRLJjJ"
pdf: "/papers/nnactive.pdf"
---
## The puzzle: did the query method improve learning—or just find an organ?

Imagine a CT volume containing a small kidney inside a large background. You can afford to label only a few 3D regions. Uniform random sampling repeatedly lands in empty space; a model-driven method finds uncertain organ boundaries. It wins at an equal patch count. But did it save expert time, or simply request more structures that take longer to delineate?

**nnActive turns this into an evaluation question.** It extends nnU-Net to train on partial 3D annotations, builds foreground-aware random baselines, and measures several aspects of quality and cost. Its contribution is a framework and a systematic comparison, rather than a single new acquisition score.

This guide follows the supplied **62-page TMLR 08/2025 course PDF**, including its appendices. Allow about 25 minutes; prerequisites are categorical probabilities, entropy, supervised segmentation and train/test separation. The [authors’ repository](https://github.com/MIC-DKFZ/nnActive) identifies the August 2025 paper and a distinct January 2026 follow-up; it points reproduction of this paper to the `nnactive/benchmark` branch. We use the supplied 2025 results throughout. All diagrams except the explicitly measured Table 3 chart are original teaching constructions. No model or clinical experiment runs on this page.

## 1 · Make the baseline and the annotation bill credible

[![Plain random queries often hit background, foreground-aware random simulates structure screening, and uncertainty sampling adds model computation and potentially repetitive boundaries.](/figures/nnactive/problem.svg)](/figures/nnactive/problem.svg)

Read each row as a different annotation policy, not as three stages of one algorithm. Sections 2–3 identify four evaluation pitfalls: too few datasets and budgets; 2D training that discards 3D context; a random baseline poorly suited to sparse foreground; and annotation cost counted only in voxels. The responses are multiple settings, partial-label 3D training, stronger random policies and foreground-efficiency accounting.

A plausible design interpretation is: establish a strong segmentation pipeline first, then ask whether sophisticated acquisition adds value beyond cheap task knowledge, then account for what the annotator actually encounters. This is our reconstruction of the logic, not the authors’ private reasoning.

The main study compares **eight** policies: Predictive Entropy, BALD, PowerBALD, SoftrankBALD, PowerPE, plain Random, Random 33% FG and Random 66% FG. For Random 66% FG, roughly one third of patches are fully random. The foreground-aware portion is split equally between patches centered on a randomly chosen foreground class and patches centered on its border. Random 33% FG uses roughly two thirds fully random queries and one third foreground-aware queries (§4).

**66% FG is a sampling-policy mixture, not a guarantee that 66% of annotated voxels are foreground.** The policy simulates a person roughly locating structures before detailed annotation; in this retrospective benchmark, dense labels provide that information. It should not be interpreted as a hidden-label-free model predicting the true locations without help. Screening has a practical cost that foreground voxel counts alone do not measure.

<details><summary>Predict: is a random baseline automatically fair because it is unbiased?</summary>

No. Uniform spatial sampling ignores known task structure. A practically cheap screening policy can be a much stronger comparator. Conversely, a retrospective oracle-assisted baseline must disclose the information it uses and the practical effort needed to obtain it.

</details>

## 2 · Follow the full 3D learning architecture

[![A trains five 3D nnU-Net models using partial labels; B converts probability volumes to patch scores; C acquires segmentation masks; D preserves labels while retraining weights.](/figures/nnactive/architecture.svg)](/figures/nnactive/architecture.svg)

The diagram is a functional architecture, not a fabricated layer-by-layer nnU-Net. **A** receives image intensities and the available target mask. **B** predicts voxel probabilities over training/pool images and scores spatial patches. **C** obtains labels for selected regions. **D** incorporates the new masks and retrains; its output returns to B at the next acquisition step. Labels and queried locations persist. Fitted model weights are replaced each round.

Let a scan be $X\in\mathbb R^{M\times H\times W\times D}$: $M$ modalities and three spatial dimensions $H,W,D$. Write $K=C+1$ for $C$ foreground classes plus background. A segmentation model $f_\theta$ returns probabilities $p_m(c\mid v)$ for class $c$ at voxel $v$, with ensemble member index $m$. A full predicted scan has $K\times H\times W\times D$ probabilities per member. These are semantic segmentation targets, rather than one class label per scan.

Section 4’s main setup uses the **3D full-resolution nnU-Net configuration**, 200 epochs per training and an ensemble of five models trained via five-fold cross-validation. Experiments use four seeds for dataset-related choices; the authors explicitly say model training itself is not seeded. Thus four experimental seeds do not imply completely deterministic neural training.

nnU-Net configures each dataset’s architecture and forward-pass patch size. The paper does not prescribe one universal layer count or feature width; we do not invent those. A query patch defines the region an annotator labels. A training patch defines the image context the network sees. They need not have the same size (Appendix C).

## 3 · Missing labels are not background labels

[![A four-voxel teaching crop contains known foreground, known background and two missing targets; only the first two contribute target loss, while all image intensities provide context.](/figures/nnactive/representation.svg)](/figures/nnactive/representation.svg)

This schematic is a tiny projection of a 3D crop. At A’s loss arrow, Appendix B defines the partial supervised objective as a sum over labeled positions:

$$
\mathcal L(\theta)=\sum_{i=1}^{N}\sum_{v\in S^{(i)}}
\ell\bigl(f_\theta(X^{(i)})_v,\widetilde Y^{(i)}_v\bigr).
$$

$N$ is the number of scans; $S^{(i)}$ is the set of labeled voxel positions in scan $i$; $\widetilde Y$ holds available class targets; $\ell$ is a supervised loss. $\theta$ contains trainable model weights. This restates the source’s unnumbered partial-loss formulation with voxel index $v$. It explains the valid-label restriction, not every implementation detail of nnU-Net’s loss.

For a **teaching cross-entropy component**, take four voxels with targets $[\mathrm{FG},\mathrm{BG},?,?]$. If the correct-class probabilities at the first two are $0.8$ and $0.9$, their mean CE is

$$
\mathcal L_{\mathrm{toy}}=\frac{-\ln(0.8)-\ln(0.9)}{2}\approx0.1643.
$$

The denominator is the two labeled voxels in this toy, not the four visible voxels. No target loss is applied at either missing position. Setting their targets to background would inject false supervision and favor erasing unannotated structures. Known background is a valid target and does receive loss.

**Context still matters.** Appendix C’s region sampler ensures a forward crop contains at least one labeled voxel, but places that voxel uniformly within the valid crop region rather than always at its center. Surrounding unlabeled image intensities can inform predictions and gradients at labeled voxels through the network. That is not a claim that these locations have ground-truth labels, nor a separate pseudo-label or consistency-loss method.

<details><summary>Predict: does ignoring a voxel’s target mean ignoring its image intensity?</summary>

No. The image is an input; the target mask controls direct supervision. Unlabeled context can influence a labeled prediction while its own target remains unknown.

</details>

## 4 · Turn ensemble probabilities into a patch score

[![Two ensemble members that confidently disagree yield predictive entropy 0.6931 and BALD 0.3681; two members that agree on a uniform distribution have the same entropy but BALD zero.](/figures/nnactive/uncertainty.svg)](/figures/nnactive/uncertainty.svg)

At B, the network has already fitted weights. Acquisition computes scores without a gradient update. The following equations are standard definitions used to explain the named methods in §4, **not numbered equations reproduced from the paper**. For $E$ ensemble members, define

$$
\bar p(c\mid v)=\frac1E\sum_{m=1}^E p_m(c\mid v),
\qquad H(p)=-\sum_{c=0}^{C}p(c)\ln p(c).
$$

$H$ is categorical entropy, in nats with natural logarithms; the $0\ln0$ contribution is zero by continuity. Predictive Entropy scores the average predictive distribution. BALD subtracts the members’ average entropy:

$$
U_{\mathrm{PE}}(v)=H(\bar p(\cdot\mid v)),
\qquad U_{\mathrm{BALD}}(v)=H(\bar p(\cdot\mid v))-
\frac1E\sum_{m=1}^{E}H(p_m(\cdot\mid v)).
$$

PE asks whether the ensemble prediction is uncertain. BALD highlights disagreement among members: confidently different predictions contribute more than unanimous ambiguity. The finite ensemble is an approximation, not an exact Bayesian posterior or a certificate of future errors.

**Worked trace through B:** two toy members predict $(0.9,0.1)$ and $(0.1,0.9)$. The average is $(0.5,0.5)$, so PE is $0.6931$. Each member has entropy $0.3251$, giving BALD $0.3681$. If both instead output $(0.5,0.5)$, PE stays $0.6931$ and BALD becomes zero. The main study has five members; using two here only makes the arithmetic visible.

The source uses mean aggregation with aggregation size equal to query patch size. For patch region $R$,

$$
s(R)=\frac1{|R|}\sum_{v\in R}U(v).
$$

$|R|$ counts voxels in the patch; $s$ is the score passed to patch selection. For a toy two-voxel patch with PE values $0.6931$ and $0.3251$, $s=0.5091$. An average favors broadly uncertain regions and can dilute a very small ambiguous structure inside a large patch. It measures neither foreground cost nor class coverage directly.

The greedy PE and BALD methods select high scores. PowerPE, PowerBALD and SoftrankBALD perturb selection to promote diversity; the main study sets their noise parameter $\beta=1$. This guide does not invent a perturbation formula absent from the supplied method exposition. Section 5.3 studies how noise strength changes results.

## 5 · Enforce spatial eligibility before global selection

[![In image A, a high scoring overlapping patch is rejected; surviving patches join image B’s candidates, and global ranking chooses the top two eligible regions.](/figures/nnactive/selection.svg)](/figures/nnactive/selection.svg)

Read this as a trace of the source’s two-stage uncertainty query procedure (§4; Appendix C, Algorithm 1). For each image, compute voxel uncertainty, aggregate it to patch scores, process candidates in descending score order, and retain patches whose overlap with previous labels and already retained patches stays within $o$. Combine retained patches across images, rank globally and obtain the requested batch. The main study sets **$o=0$**.

In the toy, A1 scores $0.5091$, A2 scores $0.49$ but overlaps A1, and disjoint A3 scores $0.30$. A2 is rejected. Image B supplies an eligible B1 scoring $0.45$. For query size $n=2$, the global choices are A1 and B1. C acquires their segmentation masks; D preserves their coordinates and targets for future rounds.

**Source pseudocode gap:** Algorithm 1 takes query size $n$ as input, but its printed final lines only sort and return $Q$; they do not explicitly truncate to $n$. The surrounding text describes top-$k$ selection and fixed query sizes. Our explanation includes that budget step as the intended study behavior and flags the omission instead of presenting the pseudocode as a complete budget-enforcing implementation.

Avoiding physical overlap does not remove semantic redundancy. Two disjoint patches may contain almost identical organ boundaries, including across different scans. The noisy query methods address that concern; a spatial filter alone is not a true-class balancing mechanism.

## 6 · Separate training, acquisition and held-out inference

[![Training updates all segmentation weights with partial targets; acquisition uses a fixed ensemble to choose annotations; test inference segments a held-out scan without querying labels or retraining.](/figures/nnactive/paths.svg)](/figures/nnactive/paths.svg)

These are three information paths. Unlike [CEC](/papers/cec) or [Active Prompt](/papers/active-prompt), nnActive does not adapt context tokens behind a frozen vision-language encoder. It retrains segmentation models from scratch at each annotation level (§4). All accumulated partial labels are available to the next training; the previous round’s fitted weights are not the initialization carried forward.

At acquisition time, ensemble probabilities lead to annotation requests. In the retrospective benchmark, a requested region’s mask is revealed from existing dense annotation. At test time, fixed fitted models produce voxel segmentations on held-out scans; dense test masks evaluate the output. Test labels do not enter acquisition or partial-loss fitting. We leave exact prediction fusion and dataset-specific backbone details to the supplied nnU-Net configuration rather than inventing a decoder recipe.

[![ACDC Low evaluates models after 30, 60, 90, 120 and 150 queried patches; old labels persist, but models are retrained each time.](/figures/nnactive/loop.svg)](/figures/nnactive/loop.svg)

This is a study-protocol trace, not a measured learning curve. Section 5.1 specifies five evaluated budgets per label regime: the shared starting budget and each added query batch are 20% of the total. The start puts every foreground class in at least two patches; its remaining patches follow Random 33% FG. For ACDC Low, total budget 150 means 30 starting patches, then four additions of 30. These choices matter: the study does not start with arbitrary empty data and hope that uncertainty discovers every class.

| Dataset | Query patch dimensions | Total patches: Low / Medium / High |
| --- | --- | --- |
| ACDC | $4\times40\times40$ | 150 / 300 / 450 |
| AMOS2022, task 2 | $32\times74\times74$ | 200 / 1000 / 2500 |
| KiTS2021 | $64\times64\times64$ | 200 / 1000 / 2500 |
| MSD Hippocampus | $20\times20\times20$ | 100 / 200 / 300 |

All settings use a 75% training/pool split and 25% test split, held fixed across seeds and experiments; scans are resampled to median dataset spacing. nnU-Net fingerprints come from training/pool data. Appendix E supplies anatomy and dataset details. In particular, AMOS has 15 foreground organs, whereas the Hippocampus task separates two neighboring structures.

Changing **query size** changes how many patches are requested per acquisition round. Changing **query patch size** changes their geometry. Halving all three axes leaves one eighth as many voxels per patch; keeping patch counts constant then changes total annotation volume. The patch-size ablation cannot be read as an equal-voxel-budget comparison.

## 7 · Put quality and effort on different axes

Final Dice emphasizes the final budget; AUBC summarizes the entire trajectory. For one foreground class with predicted voxel set $P$ and reference set $G$, the teaching Dice definition is $2|P\cap G|/(|P|+|G|)$. The source uses Mean Dice per 3D image; this simple identity is not a full statement of its class aggregation or empty-set conventions. Dice quantifies segmentation overlap, not time spent annotating.

Appendix D.2 uses a normalized trapezoidal AUBC. A teaching expression matching that normalization for evaluated budgets $b_1<\cdots<b_T$ and Mean Dice $d_t$ is

$$
\mathrm{AUBC}=\frac{1}{b_T-b_1}\sum_{t=1}^{T-1}
(b_{t+1}-b_t)\frac{d_t+d_{t+1}}2.
$$

The denominator normalizes the evaluated budget interval. If all $d_t=0.8$, AUBC is $0.8$. For toy budgets $(30,60,90)$ and Dice $(0.5,0.6,0.7)$, the result is $(30\times0.55+30\times0.65)/60=0.6$. These values are arithmetic examples, not ACDC results. AUBC treats this budget axis as cost; it does not fix differences in labor per patch.

[![Synthetic foreground-efficiency curves start at Dice 0.5 and approach 0.8; gamma 10 shrinks the gap faster than gamma 5 for the same foreground fraction.](/figures/nnactive/efficiency.svg)](/figures/nnactive/efficiency.svg)

At the evaluation box, **source Eq. (1)** fits performance against foreground annotation:

$$
y(t)=\bigl(\widehat y(\widehat t_0)-\widehat y_{\mathrm{full}}\bigr)
\exp[-\gamma(t-\widehat t_0)]+\widehat y_{\mathrm{full}}.
$$

$t\in[0,1]$ is the fraction of annotated foreground voxels; the source calls it a percentage but uses this fraction scale in its formula. $\widehat t_0$ is the mean starting fraction; $\widehat y(\widehat t_0)$ is initial Mean Dice; $\widehat y_{\mathrm{full}}$ comes from full-data training of identical length. These quantities are estimated first, then $\gamma$ is fitted. Higher $\gamma$ means faster gap reduction per unit foreground annotation: the **FG-Eff** metric (Appendix D.4).

**Toy substitution:** with start $t_0=0.1$, $y_0=0.5$, $y_{\mathrm{full}}=0.8$, $\gamma=5$ and $t=0.2$, the fitted Dice is $0.8-0.3e^{-0.5}=0.6180$. Using $\gamma=10$ gives $0.6896$. The figure draws this equation, not experimental measurements. Changing from fractions to percentage points changes the fitted parameter’s scale.

The fit assumes a roughly exponential approach below the full-data score; a finite exponential does not exactly equal its asymptote at $t=1$. FG-Eff is meaningful only with the same model/training, starting budget and annotation budget. A method that selects almost no foreground may look efficient while attaining poor Dice. Foreground amount also misses boundary complexity, screening time and interaction costs. It is a proxy that supplements the other metrics, not a measured time-saving guarantee.

The Pairwise Penalty Matrix (PPM; Appendix D.3) records the fraction of comparisons where a row method significantly beats a column method in Mean Dice, using two-sided $t$-tests at $\alpha=0.05$ plus the direction of the mean difference. It aggregates settings without reporting effect sizes. A PPM value of 27% is neither 27 percentage points of Dice improvement nor a 27% chance of winning on a new dataset. Pairwise thresholded comparisons also deserve caution when many tests are inspected.

<details><summary>Predict: can one method win on Final Dice and lose on FG-Eff?</summary>

Yes. It may achieve excellent segmentation by querying substantially more foreground per patch. The benchmark’s Predictive Entropy results illustrate why quality, patch budget and a labor proxy should all be reported.

</details>

## 8 · What the experiment actually establishes

Across four datasets, three regimes and four data-related seeds, §5.2 reports that all five AL policies beat plain Random overall at equal patch budgets. Yet no AL method reliably dominates foreground-aware Random across tasks and budgets. Random 66% FG has the best mean AUBC rank overall; Predictive Entropy has the best AUBC and Final Dice ranks **among AL methods**, but poor foreground efficiency and substantial variability. These are different comparisons, not contradictory universal winners.

The dataset changes the verdict. AL benefits over Random 66% FG are more favorable on KiTS and Hippocampus, more neutral on ACDC, and unfavorable on AMOS in the main settings. The authors link AMOS difficulties to small-organ coverage and repetitive uncertainty queries (§5.2 Q4); that explanation is an interpretation of observed behavior, not a controlled causal proof. Noisy methods tend to help earlier under small budgets. Greedy methods become more competitive later.

[![Reported KiTS Medium Table 3 gains in Final Dice times 100 are 1.73 for BALD, 2.30 for PowerBALD, 1.41 for SoftrankBALD, 0.58 for Predictive Entropy and 1.89 for PowerPE.](/figures/nnactive/results.svg)](/figures/nnactive/results.svg)

This is a **measured-result transcription**, restricted to all five KiTS Medium rows of Table 3, TMLR 08/2025. The dataset’s Medium budget is 1000 patches of $64^3$ voxels. The metric is $(\mathrm{Final\ Dice}_{500\ \mathrm{Epochs}}-\mathrm{Final\ Dice}_{\mathrm{Precomputed}})\times100$, in Dice percentage points. Both final training procedures use 500 epochs. “Precomputed” reuses the query trajectories acquired by 200-epoch models; “500 Epochs” acquires using longer-trained models too. Thus this comparison tries to isolate **better queries** from merely training longer on the same labels. It is not a comparison to full-data training, plain Random, or clinical annotation time.

All displayed differences are positive, from $+0.58$ for PE to $+2.30$ for PowerBALD. Table 3 marks significance with a two-sided $t$-test at $\alpha=0.1$; the chart intentionally omits that shading and does not label every positive bar statistically significant. The source reports four experimental seeds; these bars are reported differences, not error bars or confidence intervals.

The other ablations (§5.3; Appendix G) vary batch query size, noise strength and query geometry. Smaller query batches and longer training can improve AL, while increasing computation. Noise settings depend on dataset and budget; the paper does not establish one optimal value for unseen tasks. The halved-axis patch ablation changes labeled voxel volume as well as spatial precision.

**Limits of the conclusion:** the study emphasizes uncertainty methods and excludes architecture-changing Learning Loss and diversity-based Core-Set from its main comparison (§6). Costs are retrospective proxies, not prospectively timed annotations. Foreground-aware policies use simulated screening, and the starting protocol already ensures class presence. Fixed held-out scans are not the separate held-out development-to-deployment **datasets** proposed in §2. Neither a weak-baseline win nor this four-dataset study proves universal superiority of any policy.

A useful proposed follow-up would compare PE, a noisy method and foreground-aware Random with identical initial masks, nnU-Net plans, training length and patch geometry; log measured screening plus delineation time, model computation, class coverage, AUBC and Final Dice. Repeat on genuinely unseen datasets. A separate partial-loss/region-sampling ablation would be needed to isolate the value of context from acquisition choice. These are proposed experiments, not results reproduced here.

## 9 · Reading map and the idea to carry forward

| Mechanism or claim | Supplied paper location |
| --- | --- |
| Evaluation requirements and four pitfalls | §§2–3, pp. 3–6; Figure 1 |
| Ensemble training, query policies and baseline mixtures | §4, pp. 6–7 |
| Dataset splits, geometry and label budgets | §5.1, p. 7; Appendix E, p. 28 |
| Main policy comparisons and task dependence | §5.2, pp. 8–11; Figures 2–4; Appendix F |
| Compute, noise and patch-size ablations | §5.3, pp. 11–13; Tables 2–3; Appendix G |
| Partial-loss math and spatial sampler | Appendix B, p. 24; Appendix C, p. 25, Algorithm 1 |
| Metric definitions and FG-Eff Eq. (1) | Appendix D, pp. 26–27; Figure 7 |
| Limits and future directions | §6, pp. 14–15; Appendix H ranking analysis |

References use printed page numbers, which match this PDF’s page sequence. The paper is [available locally](/papers/nnactive.pdf); its primary review record is [OpenReview](https://openreview.net/forum?id=AJAnmRLJjJ).

**Idea to carry forward:** a query method’s benefit depends on the training pipeline, the baseline’s task knowledge and the cost axis. First make partial annotations usable by a strong 3D model; then compare acquisition against credible cheap policies; finally ask whether the improvement survives accounting for foreground work and computation. This connects [LabelBench’s separation of acquisition and training](/papers/labelbench) to a setting where each label is a spatial region rather than a single image class.
