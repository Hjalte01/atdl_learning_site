---
title: "Dataset-Aware Cold-Start Active Learning for Annotation-Efficient 3D Medical Image Segmentation"
shortTitle: "CSCS"
topic: "topic-4"
order: 6
description: "Select the first annotated volumes by combining embedding density, reconstruction disagreement and a pool-aware pacing rule."
status: "ready"
source: "https://arxiv.org/abs/2606.20765v1"
pdf: "/papers/cscs.pdf"
---
## The puzzle: which scans should receive the first labels?

You have a collection of unlabeled 3D scans and can afford to annotate only a few. Selecting the hardest scans sounds useful—but reconstruction failures may indicate artifacts rather than useful anatomical variation. Selecting only typical scans sounds safe—but could miss informative difficult cases. How do you make a first choice before a segmentation model exists?

**CSCS separates two decisions: cover different regions of the embedding space, then choose within each region using a dataset-aware balance of typicality and difficulty.** Its name expands to **Curriculum-Stratified Cold-Start**.

This guide follows the supplied **20-page arXiv v1, 18 June 2026**, not a later implementation revision. Allow about 25 minutes. All colored diagrams are original teaching schematics; only the result chart contains measured paper values. Click figures to enlarge them. No trained model runs on this page.

## 1 · Cold start is a selection stage before supervised training

[![Three workflows distinguish random initialization, iterative active learning and CSCS single-batch initialization.](/figures/cscs/problem.svg)](/figures/cscs/problem.svg)

Read each row as a workflow. Ordinary warm-start active learning already has labels and a task model with which to score the next query. CSCS constructs that initial set from a fully unlabeled pool. Self-supervised training is allowed: **no task labels does not mean no learned model**.

Our interpretation of the design logic is: obtain label-free signals, avoid concentrating the entire budget in one region, and temper difficult-case preference using the pool's structure. This reconstructs the mechanism's rationale, not the authors' private reasoning. The evaluated operation is one-shot initialization; later active-learning rounds are outside the reported evaluation (§§1–2.1, 3.7).

<details><summary>Predict: does “curriculum” mean annotating easy scans first and harder scans in later rounds?</summary><p>No. The pacing rule sets one global mixture weight for a single annotation batch. It varies with the supplied budget and pool statistics, not with an evaluated sequence of training epochs or query rounds.</p></details>

## 2 · Architecture: two models with different jobs

[![Functional CSCS architecture from 3D volumes to SSL features and two scores, pool calibration, clustered query and downstream nnU-Net.](/figures/cscs/architecture.svg)](/figures/cscs/architecture.svg)

Read A through D. **A** learns representations using the entire unlabeled training pool. A Swin-UNETR masked-inpainting encoder–decoder produces reconstructions and volume embeddings. **B** freezes the SSL model and extracts uncertainty and typicality. **C** uses pool statistics to set the score weight and clusters embeddings to distribute queries. **D** obtains expert masks for the selected volumes and trains a separate 3D full-resolution nnU-Net.

A volume $X_i$ is a spatial image tensor, possibly with multiple imaging channels; its embedding $z_i\in\mathbb R^{768}$ comes from global average pooling followed by L2 normalization (§3.2). The acquisition output is a list of volume identifiers, not a voxelwise segmentation. Uncertainty $U_i$, typicality $T_i$, and score $S_i$ are scalars per volume.

The supplied implementation protocol uses $128^3$ SSL patches, masking ratio 0.3, AdamW with learning rate $10^{-4}$ for 5,000 iterations. Downstream nnU-Net uses Ranger21, learning rate $10^{-3}$, and 250 epochs. The paper does not document a complete layer-by-layer reconstruction architecture, view count $R$, or all view-alignment details. These diagrams deliberately show the documented functional boxes. They do not claim that nnU-Net inherits the SSL encoder's weights.

## 3 · What the two signals actually measure

[![Representation close-up shows reconstruction disagreement and inverse neighbor distance as distinct scalar summaries of each scan.](/figures/cscs/representation.svg)](/figures/cscs/representation.svg)

**Reconstruction uncertainty lives at B's reconstruction branch.** Repeat reconstruction with masked inputs under different geometric views, then average voxelwise variance (paper Eq. 1):

$$
U_i=\frac{1}{|\Omega_i|}\sum_{v\in\Omega_i}\operatorname{Var}_{r=1,\ldots,R}\left[\hat X_i^{(r)}(v)\right].
$$

$\hat X_i^{(r)}(v)$ is reconstruction $r$ at voxel $v$; $\Omega_i$ is the set of reconstructed voxel positions; $R$ is the number of reconstructions. Greater disagreement is a **difficulty proxy**, not calibrated segmentation error, class entropy, or reconstruction error against the original scan. Systematically wrong but identical reconstructions have zero variance. Geometric views must correspond spatially for a meaningful voxelwise comparison; v1 does not specify the complete alignment implementation.

**Typicality lives at B's embedding branch.** A point close to many neighbors is locally representative (paper Eq. 2):

$$
T_i=\left(\frac1k\sum_{j\in\operatorname{KNN}_k(i)}\|z_i-z_j\|_2\right)^{-1},\qquad k=20.
$$

$\operatorname{KNN}_k(i)$ indexes neighboring volumes in embedding space, and $\|\cdot\|_2$ is Euclidean distance. Mean distance 0.25 gives raw typicality 4; mean distance 0.5 gives 2. This is an inverse-distance density proxy, not a measured anatomical quality label. A rare useful anatomy may have low typicality.

Before scoring, each signal is separately min-max normalized over the full pool. For a raw signal $a_i$, the teaching expression is $\tilde a_i=(a_i-\min_j a_j)/(\max_j a_j-\min_j a_j)$. Below, $T_i,U_i$ mean those normalized values unless marked raw. Min-max normalization affects the geometric score; it is not a probability calibration.

**Reproduction gaps:** constant signals make the normalization denominator zero, duplicate embeddings can make raw typicality singular, and a pool of 18 DIANE volumes cannot provide 20 distinct other-volume neighbors. V1 does not give the small-pool $k$ fallback, self-neighbor convention, variance divisor, or numerical safeguards. We expose these gaps rather than silently picking an implementation.

## 4 · Pool geometry chooses the mixture weight

[![Pacing diagram shows positive and negative rank alignment and budget-dependent conservative changes around gamma one half.](/figures/cscs/pacing.svg)](/figures/cscs/pacing.svg)

A useful question is whether difficult scans tend to be typical or peripheral. CSCS summarizes their **relative orderings** using the Difficulty-Coverage Ratio, despite its name a correlation rather than a quotient (paper Eq. 3):

$$
\mathrm{DCR}=\rho_s\big((U_i)_{i=1}^N,(T_i)_{i=1}^N\big).
$$

$\rho_s$ is Spearman rank correlation and $N$ the number of training-pool volumes. DCR near $+1$ means high-uncertainty scans also tend to be typical; DCR near $-1$ means they tend to be isolated. The statistic is computed across the whole pool, not separately within small clusters. It does not certify the uncertainty proxy's correctness or directly pick any scan. Strictly increasing rescalings preserve ranks; ties and constant rankings need care.

The budget affects how far to move from a balanced score. With annotation count $B$, paper Eqs. 5–7 define:

$$
\begin{aligned}
\alpha_\beta&=\frac{B}{N^\beta},\quad \beta\in[0,1],\\
\alpha_{\mathrm{eff}}&=\frac{B}{\sqrt N},\\
m&=\frac{\alpha_{\mathrm{eff}}}{1+\alpha_{\mathrm{eff}}},\\
\gamma&=0.5+\frac{\mathrm{DCR}}4m.
\end{aligned}
$$

$\beta$ connects count-only ($\beta=0$) and fraction-only ($\beta=1$) budget descriptors; the chosen midpoint is $1/2$. $m$ is a saturating confidence factor, and $\gamma$ controls uncertainty's exponent. This computation lives in C's pool-calibration box. **$B/\sqrt N$ is not the annotation fraction $B/N$** and can exceed one.

For a finite positive budget, $0<m<1$ and $0.25<\gamma<0.75$. DCR zero gives $\gamma=0.5$ at every budget. Positive DCR moves upward as $B$ grows; negative DCR moves downward. Thus increasing budget does **not** universally mean harder-first selection. As effective budget tends to zero the rule approaches 0.5, not pure typicality. The square-root scaling and divisor four are design choices, not a theorem of optimal acquisition (§3.7).

<details><summary>Predict: with DCR = −0.8, does a larger budget raise the uncertainty weight?</summary><p>No. A larger budget increases m, so the negative correction becomes larger in magnitude and gamma falls further below 0.5. The direction comes from DCR; budget changes its strength.</p></details>

## 5 · Select locally, then repair the budget after merging

[![Selection process shows B clusters, small-cluster merging, score winners and global filling of leftover slots.](/figures/cscs/selection.svg)](/figures/cscs/selection.svg)

C first runs k-means++ on normalized embeddings with $B$ clusters. Clusters smaller than $s_{\min}=3$ merge into the nearest neighboring cluster, using centroid Euclidean distance. The surviving cluster count $M$ may be less than $B$.

The mixture score prefers scans strong on both signals (paper Eq. 4):

$$
S_i=T_i^{1-\gamma}U_i^\gamma,\qquad
 i_m^*=\arg\max_{i\in C_m}S_i.
$$

$C_m$ is a surviving cluster, and $i_m^*$ its chosen scan. At $\gamma=0.5$, $S_i=\sqrt{T_iU_i}$. A weak factor suppresses a strong other factor, unlike a simple additive score. For strictly positive signals, the teaching identity $\log S_i=(1-\gamma)\log T_i+\gamma\log U_i$ shows precisely where the weight acts. Increasing $\gamma$ raises $S_i$ when $U_i>T_i$ and lowers it when $U_i<T_i$; it need not increase every score.

For the adaptive interior range, a zero in either normalized signal makes the score zero. That is a preference, **not a guarantee that zero-score samples are never selected**: a cluster could contain only zero-score candidates, or budget filling could require them. Conceptual ablation endpoints $\gamma=0$ and $1$ use pure $T$ and pure $U$; they should not evaluate ambiguous $0^0$ literally.

Algorithm 1 then:

1. Select the maximum-score candidate from each surviving cluster.
2. If $M<B$, rank all unselected scans globally by the same score.
3. Add the top remaining scans until exactly $B$ have been selected.
4. Ask experts to annotate those volumes; the returned masks form the supervised initial set $L_0$.

Global filling can add multiple scans from the same surviving region. The paper's $L_0$ notation calls the output labeled before the expert step; operationally selection supplies identifiers first. V1 does not specify merge-order details or score tie-breaking. No segmentation training happens between the individual picks.

## 6 · Follow one toy query through the boxes

[![Worked numerical trace computes variance, typicality normalization, rank correlation, effective budget, scores and a cluster winner.](/figures/cscs/numeric.svg)](/figures/cscs/numeric.svg)

This teaching example uses population variance and a three-volume pool with $k=2$, **not the paper's $k=20$ experiment**. Three embeddings are unit vectors in a plane at angles $0$, $60$, and $180$ degrees. Their pairwise distances are $1$, $2$, and $\sqrt3$. Name the volumes A, B, C in that order.

Raw typicalities are $[1/1.5,\ 2/(1+\sqrt3),\ 2/(2+\sqrt3)]\approx[0.6667,0.7321,0.5359]$. Pool min-max normalization gives $T=[2/3,1,0]$.

At a single illustrative reconstructed voxel, use two views $[0.3,0.5]$ for A, $[0.2,0.6]$ for B, and $[0.4,0.4]$ for C. Their population variances are $[0.01,0.04,0]$, giving normalized $U=[0.25,1,0]$. The rankings agree perfectly, so DCR is 1. This one-voxel example is not a real volume uncertainty estimator.

For $N=3,B=1$, $\alpha_{\mathrm{eff}}=1/\sqrt3\approx0.5774$, $m\approx0.3660$, and $\gamma\approx0.5915$. One cluster contains all three scans and meets the minimum size of three:

$$
S_A=(2/3)^{0.4085}(0.25)^{0.5915}\approx0.3732,\qquad S_B=1,\qquad S_C=0.
$$

Select B, obtain its expert mask, then train the task model. The score is not a predicted Dice and does not itself generate that mask. Reproducible figure generation checks this arithmetic; the displayed rounding does not change the winner.

## 7 · Training changes weights; inference no longer selects

[![Separate training and test paths show SSL reconstruction training, frozen acquisition, expert masks, nnU-Net optimization and direct test segmentation.](/figures/cscs/paths.svg)](/figures/cscs/paths.svg)

Read the paths separately. SSL updates the reconstruction model using unlabeled images; acquisition uses its frozen outputs. Supervised training updates nnU-Net using selected images and expert masks. At inference a held-out image enters the trained nnU-Net and yields a segmentation. **DCR, k-means and the query score do not run on each test patient to produce the prediction.**

V1 reports the training configurations but does not spell out the complete SSL loss or downstream optimizer/loss implementation. We do not substitute a guessed objective for a paper equation. The experimental comparison shares the encoder, embeddings, normalization and KNN definition across acquisition methods, and shares the k-means partition for cluster-based methods (§§3.2–3.3). This helps isolate selection while leaving sensitivity to representation quality open.

## 8 · Evidence: useful robustness, with exceptions

[![Reported Table 2 Dice means compare Random and CSCS at the middle operating point with exact annotation counts for four datasets.](/figures/cscs/results.svg)](/figures/cscs/results.svg)

The chart is measured evidence: **Table 2 means over three independent selection-and-training runs on a fixed split**, with no invented error bars. BraTS averages WT/TC/ET composite-region Dice; FeTA and DIANE average foreground classes; Spleen uses one foreground class. Train/validation splits are 387/97, 39/10, 34/8 and 18/5 volumes respectively (§3.1).

| Dataset, selected count / pool | Random Dice (%) | CSCS Dice (%) | Full-pool nnU-Net (%) |
| --- | ---: | ---: | ---: |
| BraTS, 116/387 (29.97%) | 85.2 | 85.4 | 86.4 |
| FeTA, 12/39 (30.77%) | 64.1 | 73.7 | 75.3 |
| Spleen, 10/34 (29.41%) | 93.3 | 95.7 | 97.5 |
| DIANE, 7/18 (38.89%) | 75.3 | 83.4 | 90.2 |

**Budget-label exception:** Table 1’s caption, Table 2’s columns and the figure use nominal 20/30/40% labels, but DIANE selects 5/7/9 of 18 volumes: actual fractions **27.78/38.89/50%**. Table 1’s numeric fraction column correctly rounds these to 0.28/0.39/0.50. These differ beyond nearest-integer rounding of 20/30/40%. We report absolute counts and actual fractions rather than silently repairing the source. The chart uses the middle nominal operating point for every dataset.

At the lowest nominal budget, CSCS is highest or tied highest in Table 2. At the highest, Spleen's Random is 97.2 versus CSCS 96.9, and DIANE's FPS is 85.6 versus CSCS 85.1. This supports the authors' emphasis on cross-setting robustness rather than universal wins. They report an indicative Friedman/Nemenyi rank analysis across 12 configurations; budgets share pools, and three runs are limited evidence. Detailed standard deviations and rank matrices are in a separate ancillary supplement, absent from the supplied course PDF; this guide does not reproduce those statistics from rounded cells.

### What the ablations isolate—and what they do not

At the middle nominal budget, Table 3's fixed $\gamma=0.5$ gives Spleen 96.8 Dice and HD95 1.6 mm, while predicted $\gamma$ gives 95.7 and 12.1 mm. On DIANE predicted pacing improves over $0.5$: 83.4 versus 82.2 Dice, and 14.7 versus 19.7 mm HD95. Adaptation helps in that tested setting; it is not always better than a balanced mixture.

Table 4 compares random-per-cluster against CSCS: FeTA is 73.6 versus 73.7 Dice, whereas DIANE is 74.9 versus 83.4. Diversity can explain much of one dataset's gain and less of another's.

**Source inconsistencies:** Table 3 says its $\gamma=0$ row is copied from the main TypiClust experiments, but FeTA Dice is 71.8 versus Table 2's 72.3, and Spleen is 88.9 versus 90.7. Predicted FeTA HD95 is 4.6 in Table 3 and 4.8 in Table 4. The text says fixed $\gamma=0.7$ leads on DIANE, although predicted pacing is higher in Table 3. These discrepancies remain explicit; Table 2 supplies our main comparison and chart.

### Reading the metrics through a numerical example

Overlap tells one part of the story (paper Eqs. 8, 11–12):

$$
\mathrm{Dice}(P,G)=\frac{2|P\cap G|}{|P|+|G|},\qquad
\mathrm{IoU}(P,G)=\frac{|P\cap G|}{|P\cup G|}.
$$

$P$ and $G$ are predicted and reference voxel sets. If both have 100 voxels and 80 overlap, Dice is 0.8 and IoU is $80/120=2/3$. For one region, $\mathrm{IoU}=\mathrm{Dice}/(2-\mathrm{Dice})$, using fractional Dice. Transform each class separately before averaging; transforming mean Dice need not give mean IoU.

HD95 (paper Eqs. 9–10) takes each surface point's nearest distance to the other surface, computes a 95th percentile in each direction, and takes the larger percentile. Lower millimeters is better. This reduces sensitivity to isolated distant errors compared with maximum Hausdorff distance, but may overlook rare severe errors. Precision $|P\cap G|/|P|$ and recall $|P\cap G|/|G|$ (Eq. 13) distinguish excess foreground from missed foreground.

### What would you test next?

Change SSL backbones while holding selection budgets and task training fixed; compare DCR stability across pool resamples; test artifact-heavy pools where reconstruction difficulty is misleading; vary the square-root exponent, divisor four and small-cluster threshold. These are proposed experiments. The source has no expert-time study: selected-volume fractions do not establish proportional reductions in annotation hours. DIANE's images are not public. Broader datasets, repeated splits and later AL rounds would be needed to extend the conclusions.

## 9 · The idea to carry forward

**Adapt the balance globally, select locally, and evaluate the trained task model separately.** DCR describes pool geometry; pacing turns it and the budget into a conservative weight; clustering distributes the first query; human masks supply the truth that SSL cannot provide.

Compare [SUGFW+](/papers/sugfw-plus), which uses pretrained SAM signals and a prompt-adaptation path, with CSCS's pool-specific SSL and separate nnU-Net. Compare [nnActive](/papers/nnactive) for iterative acquisition and careful annotation-cost accounting. Redraw the architecture and explain which weights change, which scalar sets the trade-off, and where the first human label enters.

## Sources and reading map

- [Supplied course PDF: CSCS, 18 June 2026 arXiv v1](/papers/cscs.pdf): §§2.1–2.2 / Eqs. 1–3 for signals; §2.3 for diversity and merging; §2.4 / Eqs. 4–7 and Algorithm 1 for pacing and acquisition; §§3.1–3.4 / Table 1 for protocols and metrics; §§3.5–3.7 / Tables 2–4 and Figures 2–3 for evidence and limitations.
- [Official arXiv v1 record](https://arxiv.org/abs/2606.20765v1) confirms source identity and lists the separate ancillary supplement. Our explanation uses the supplied main PDF.
- [Authors' implementation](https://github.com/rhattat/CSCS-AL) is a further reading link, not a claim that this guide reproduces a particular code revision.

All toy computations and colored mechanism diagrams are teaching constructions. Reported results, source gaps and proposed experiments are identified separately.
