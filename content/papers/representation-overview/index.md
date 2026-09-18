---
title: "Principles and Practice of Deep Representation Learning: a reading roadmap"
shortTitle: "Deep representation learning"
topic: "general"
order: 1
materialKind: "overview"
description: "Follow the textbook from low-dimensional geometry to denoising, coding-rate objectives, unrolled networks, consistent representations and conditional inference."
status: "ready"
pdf: "/papers/representation-overview.pdf"
---
## The puzzle: which details should a representation keep?

Imagine photographs of the same object under changing lighting and viewpoint. A pixel array has many coordinates, but its meaningful variations may occupy a much smaller set. Keeping every pixel is expensive; throwing everything away is cheap but useless. A useful representation must preserve structure while making it easier to learn, reconstruct or predict.

This is a **reading roadmap**, not a chapter-by-chapter replacement for the textbook by **Sam Buchanan, Druv Pai, Peng Wang and Yi Ma**. It follows the supplied **Version 2.0, dated March 1, 2026**, including its revised nine-chapter organization. The catalog correctly treats the book as general material, not as one of the numbered topic papers. Allow about 20 minutes for this guide; the reading routes below lead back to the relevant sections.

The book develops a compression-centered account of representation learning. Its mathematical constructions, experiments, and proposals about intelligence have different evidential roles. We keep those roles separate. All diagrams here are original schematics except the explicitly labeled redraw of selected Table 8.5 values. Numerical examples are teaching calculations, not trained models.

## 1 · A map of the book

[![A chapter map links geometry and linear models to denoising, informative codes, unrolled architectures, consistency and inference.](/figures/representation-overview/map.svg)](/figures/representation-overview/map.svg)

Read the three rows as a suggested progression. Chapter 3 learns a distribution through denoising; Chapter 4 transforms data into structured features. Those goals are related but not interchangeable. Chapter 8 supplies implementations and evaluations; Chapter 9 sets out the authors’ open research directions.

| Before you start | What it unlocks |
| --- | --- |
| Vectors, orthogonal projections and eigenvalues | PCA and subspace geometry in Chapter 2. |
| Conditional expectation and Gaussian noise | Optimal denoising in Chapter 3. |
| Determinants, logarithms and covariance | The coding-rate surrogate in Chapter 4. |
| Gradients, constraints and iterative optimization | Layers derived from updates in Chapter 5; Appendix A is a refresher. |
| Bayes’ rule and likelihood | Using a learned prior under observations in Chapter 7. |

A plausible design path is: start with a solvable linear model, generalize the learning objective, turn its iterative computation into a network, then check whether the code preserves what matters. This is our pedagogical interpretation of the chapter order, not a reconstruction of the authors’ private thought process.

## 2 · Start with a representation you can calculate

A low-dimensional representation need not be mysterious. In PCA, the encoder records coordinates along a learned subspace, and the decoder places them back in the ambient space. For centered data, the book’s Eq. (2.1.6) gives the functional architecture

$$
x\in\mathbb R^D \quad\xrightarrow{\ U^\top\ }\quad z=U^\top x\in\mathbb R^d
\quad\xrightarrow{\ U\ }\quad \hat x=Uz\in\mathbb R^D.
$$

Here $D$ is the input dimension, $d$ the subspace dimension, and $U\in\mathbb R^{D\times d}$ has orthonormal columns, so $U^\top U=I_d$. The **encoder** computes $z$; the **decoder** computes $\hat x$. Learning estimates $U$ from training samples, using the leading eigenvectors of their second-moment matrix after centering. For nonzero-mean raw data, subtract the training mean before encoding and restore it after decoding.

[![A two-coordinate input passes through an orthogonal projection to one latent coordinate and reconstructs on a diagonal line; the residual lies across that line.](/figures/representation-overview/pca.svg)](/figures/representation-overview/pca.svg)

Follow the upper row through the same encoder and decoder boxes. The lower row distinguishes reconstruction from the discarded residual. This schematic uses an assumed basis to expose the arithmetic; it does not estimate that basis from a dataset.

Take $D=2$, $d=1$, $U=(1,1)^\top/\sqrt2$, and $x=(3,1)^\top$. Then

$$
z=\frac{3+1}{\sqrt2}=2\sqrt2,\qquad
\hat x=\frac{(1,1)^\top}{\sqrt2}\,2\sqrt2=(2,2)^\top.
$$

The residual is $x-\hat x=(1,-1)^\top$, with squared length $2$. We have retained the component along the line and removed the component perpendicular to it. This removes noise **only to the extent that the assumed subspace is an appropriate signal model**; meaningful variation outside that subspace is also lost. A single linear subspace cannot represent every curved or multimodal distribution. Sections 2.2–2.3 move to mixtures and sparse dictionaries.

<details><summary>Predict: is a one-number code always a better representation than a two-number code?</summary><p>No. Compression is useful only relative to the structure or task to preserve. If the discarded direction carries the class label, this projection can harm classification despite reducing dimension.</p></details>

## 3 · Denoising learns more than a clean-looking image

For complex distributions, it may be easier to learn how noise perturbs data than to prescribe one global subspace. In Chapter 3 the corruption convention is $x_t=x+t\epsilon$, where $x\in\mathbb R^D$ is clean data, $\epsilon\sim\mathcal N(0,I_D)$ is independent noise, and $t>0$ is its standard deviation. This is the book’s additive-noise convention; it is not the interpolation time convention used in every flow paper.

[![Training draws data and noise, predicts clean data and updates weights; inference fixes weights and updates a noisy state repeatedly using a chosen sampler.](/figures/representation-overview/paths.svg)](/figures/representation-overview/paths.svg)

Read the two rows independently. During **training**, many noisy examples help fit one denoiser. During **sampling**, fixed weights guide a sequence of changing states. A trained denoiser is a component of a sampler, not a complete sampling algorithm.

The squared-error teaching objective is

$$
\mathcal L(\theta)=\mathbb E_{x,t,\epsilon}\big[\|D_\theta(x+t\epsilon,t)-x\|^2\big].
$$

$D_\theta$ is the denoiser with parameters $\theta$; the expectation averages over training data, selected noise levels and noise draws. This operates at the diagram’s **loss → weights** box. For a fixed noise level, the population-optimal squared-error predictor is the conditional mean. The book connects that mean to a density gradient via **Theorem 3.2, Eq. (3.2.23)**:

$$
\mathbb E[x\mid x_t]=x_t+t^2\nabla_{x_t}\log p_t(x_t).
$$

Here $p_t$ is the density of the noisy data, and its log-density gradient is the **score**. This identity explains the **denoiser output**: its displacement from the noisy input reveals a score scaled by the noise variance. A learned finite-data denoiser only approximates the optimal mean. The identity by itself supplies neither a noise schedule nor an ODE/SDE solver.

For a scalar teaching example, let $x\sim\mathcal N(0,1)$ and $t=1$. Then $p_t=\mathcal N(0,2)$ and its score at $x_t=2$ is $-2/2=-1$. The posterior mean is $2+1^2(-1)=1$. This is a mean estimate, not a fresh draw from the posterior. At exactly $t=0$, recovering a score by dividing the residual by $t^2$ is undefined; the displayed score relationship is used for positive noise levels.

Chapter 3.3 explicitly studies memorization and generalization. A low training denoising error alone does not demonstrate recovery of the population distribution. Continue to the [generative-model roadmap](/papers/t2-generative-overview) for model families and to [REPA](/papers/t2-paper-1) for the distinction between a denoising objective and useful intermediate representations.

## 4 · Compression must preserve differences

If every input receives the same code, compression is extreme and discrimination disappears. Chapter 4 therefore balances two aims: compact structure within each group and distinguishable structure across groups. The book measures this through **coding-rate reduction**, rather than minimizing one global code size in isolation.

[![Two normalized groups share one axis or occupy different axes. Each group is compact in both cases, but distinct axes give a larger whole-set coding rate.](/figures/representation-overview/rate.svg)](/figures/representation-overview/rate.svg)

Read left-to-right within each row. The whole-set rate and the weighted within-group rate feed a subtraction. The values use the synthetic, unit-norm four-sample example below; they are not measured information in real images.

For $N$ feature columns in $Z\in\mathbb R^{d\times N}$, the book’s **Eq. (4.2.10)** uses

$$
R_\varepsilon(Z)=\frac12\log\det\left(I_d+\frac{d}{N\varepsilon^2}ZZ^\top\right).
$$

Here $d$ is feature dimension, $N$ sample count, $\varepsilon>0$ coding precision, $I_d$ the identity, and $ZZ^\top$ the unnormalized second-moment matrix. This lives in the **whole-set rate** box. It is a Gaussian-inspired lossy coding surrogate, not an exact file length for arbitrary data. We use natural logarithms in the example; changing to base 2 rescales values into bits.

For groups $Z_k$ with $N_k$ columns, define the weighted within-group rate and the objective (Eqs. 4.2.11–4.2.13) as

$$
R^c_\varepsilon(Z)=\sum_{k=1}^{K}\frac{N_k}{N}
\left[\frac12\log\det\left(I_d+\frac{d}{N_k\varepsilon^2}Z_kZ_k^\top\right)\right],
\qquad \Delta R_\varepsilon=R_\varepsilon-R^c_\varepsilon.
$$

$K$ counts groups and $N=\sum_kN_k$. Our brackets make the weighting explicit; the book folds $N_k/N$ into its definition of each group’s contribution. The **within-group rate** box uses these terms; **subtract → learn encoder** maximizes their difference. The encoder maps data to features, $Z=f_\theta(X)$. Feature normalization is essential for comparing geometries: the book constrains each group’s squared Frobenius norm to $N_k$ or normalizes individual features.

**Work the same boxes.** Let $d=2$, $N=4$, $\varepsilon=1$, and two equally sized groups. Group A contains $(1,0)^\top$ and $(-1,0)^\top$; group B contains $(0,1)^\top$ and $(0,-1)^\top$. Thus $ZZ^\top=2I_2$, giving $R=\tfrac12\log\det(2I_2)=\log2\approx0.6931$. Each group has unweighted rate $\tfrac12\log3$; after weighting and summing, $R^c=0.5493$. The reduction is **0.1438**.

If both groups instead occupy the same horizontal axis, $ZZ^\top=\operatorname{diag}(4,0)$, so $R=\tfrac12\log3=0.5493$. The within-group rate stays $0.5493$ and the reduction becomes **0**. Every sample still has unit norm. The contrast measures group separation under this model; it does not certify that labels are meaningful, a representation generalizes, or its dimension is correctly chosen.

<details><summary>Predict: could you fairly compare coding rates after multiplying only one model’s features by 100?</summary><p>No. The second moments and rates change with scale. Fix the normalization and coding precision before interpreting a difference as better representation geometry.</p></details>

## 5 · Turn an update rule into a layer

Chapter 5 asks whether layers can implement steps toward an explicit objective. For supervised rate reduction, its first construction updates a whole feature matrix; then it develops a feature map that can operate on a new input whose class is unknown. That second step matters: training labels cannot simply be supplied at inference.

[![An optimization-inspired network repeatedly transforms token features with subspace attention and sparse coding, while training adjusts its parameters through a separate outer loop.](/figures/representation-overview/unroll.svg)](/figures/representation-overview/unroll.svg)

Read the upper row as the functional CRATE path: embeddings become $Z^0\in\mathbb R^{d\times n}$, then repeated blocks transform $Z^\ell$ into $Z^{\ell+1}$, and a task head consumes the result. Here $n$ counts tokens within an example, unlike $N$ training examples in the preceding calculation. The second row opens one block; the last row shows a separate loop over training batches. This diagram omits normalization and implementation-specific dimensions and is not a layer specification for every transformer.

A generic **teaching update**, illustrating the principle rather than specifying CRATE, is

$$
z^{\ell+1}=z^\ell-\eta\nabla F(z^\ell).
$$

$z^\ell$ is the current feature state, $F$ a differentiable objective to minimize, $\eta>0$ a step size, and $\ell$ the iteration/layer index. The gradient lives on the arrow between feature states. For rate **maximization**, the sign reverses; constraints also require appropriate projection or normalization. For the toy scalar $F(z)=\tfrac12(z-2)^2$, starting at $z^0=0$ with $\eta=0.5$ gives $z^1=1$, $z^2=1.5$, and $z^3=1.75$. This arithmetic is not a coding-rate experiment.

In §§5.2.1–5.2.2, CRATE combines **multi-head subspace self-attention (MSSA)** with an **ISTA-inspired sparse-coding operation**. The compression term acts through candidate subspaces; the sparsifying stage encourages structured coordinates. These replace the conventional attention/MLP block roles in the book’s construction. The token objective uses projected subspaces instead of the known class memberships of §4.2, so the four-sample supervised example is not its full loss.

The distinction to remember is **inner feature transformation versus outer parameter learning**. A network inspired by optimization can still have its parameters learned by backpropagation. Finite learned blocks and approximate updates do not automatically inherit a convergence guarantee from an ideal iterative algorithm.

## 6 · Check the round trip

A representation can separate classes yet lose information needed for reconstruction. Chapter 6 adds a decoder, then checks the reconstructed data through the encoder again.

[![Data are encoded to Z, decoded to reconstructed data, and re-encoded to Z hat; feedback compares the two feature representations.](/figures/representation-overview/loop.svg)](/figures/representation-overview/loop.svg)

Follow the top row from $X$ to $\hat X$, then the lower row back into the **same encoder** to obtain $\hat Z$. The comparison occurs in feature space. This redraws the functional loop of Eq. (6.2.14); it does not claim that ordinary squared reconstruction error is the full closed-loop objective.

The relevant relationships are $Z=f_\theta(X)$, $\hat X=g_\eta(Z)$, and $\hat Z=f_\theta(\hat X)$, where $f_\theta$ is the encoder and $g_\eta$ the decoder. Here $\eta$ denotes decoder parameters, as in the source, rather than the step size in the previous teaching update. Section 6.2 distinguishes **sample-wise** agreement of codes from **distributional** agreement: matching feature distributions need not pair every input with its own reconstruction.

Section 6.2.1 frames training as a game: the encoder can expose discrepancies in feature space while the decoder learns to reduce them, alongside representation-structuring objectives. A frozen, constant encoder would report perfect feature agreement for every pair, which is why agreement alone is insufficient. The book also notes that choosing feature dimension remains a model-selection problem (§6.2.1, footnote 11). Closing the loop does not magically identify the right dimension or prove that every semantic detail survived.

<details><summary>Predict: if two sets of reconstructions have the same feature distribution, must every image be reconstructed correctly?</summary><p>No. Samples could be permuted or some information could be invisible to the encoder. Distributional matching is weaker than a faithful sample-by-sample round trip.</p></details>

## 7 · Use the representation under an observation

Chapter 7 turns a learned distribution into a prior for inference. Suppose the measurement is $y=h(x)+w$ (Eq. 7.1.1), where $h$ is an observation function and $w$ measurement noise. For an image-completion task, $h$ could retain only visible pixels.

Bayes’ rule combines the prior and measurement model:

$$
p(x\mid y)\propto p(y\mid x)\,p(x).
$$

Here $p(x)$ is the prior distribution, $p(y\mid x)$ the likelihood of the observed measurement, and $p(x\mid y)$ the posterior. The omitted normalizing factor depends on fixed $y$, not on candidate $x$. This operates where the **learned prior** and **measurement agreement** meet during inference. It requires a likelihood consistent with the observation process, not just an attractive generator.

The posterior supports different outputs: its mean, a mode (MAP), or a sample. For a discrete toy posterior that puts equal mass at $-2$ and $2$, the mean is $0$, although $0$ has no posterior mass; a sample is one of the two supported values. Thus “denoised estimate” and “plausible generated sample” are different requests. More generally, a prior helps disambiguate missing information but can also impose an incorrect guess when observations are weak.

Read §§7.1–7.3 before the [DiffAtlas guide](/papers/t2-paper-4) to ask how observed information constrains a changing sample. Read §7.4 before [FLUX.1 Kontext](/papers/flux-kontext) for paired conditioning. These are course connections, not claims that those papers implement the textbook’s exact algorithms.

## 8 · Use experiments to test the story

A principled design is a reason to investigate a model, not proof it wins every metric. One useful checkpoint is **Table 8.5, printed page 345**, in the supplied version.

[![A measured bar chart shows ImageNet-1K linear-probe accuracy 69.2 percent for CRATE-S and 72.4 percent for ViT-S; parameter counts are 13.12 and 22.05 million.](/figures/representation-overview/evidence.svg)](/figures/representation-overview/evidence.svg)

This is a redraw of **two entries** from Table 8.5, with the accuracy axis starting at zero. It is not a new experiment. The caption calls these linear-probing classification accuracies after supervised ImageNet-1K pretraining. Section 8.4.5 reports 150 pretraining epochs, LION, batch size 2048, 224×224 crops and patch size 16; probes use AdamW. Keep this setting separate from Table 8.6’s patch-size-8 MaskCut evaluation.

| Model | Parameters | ImageNet-1K linear-probe accuracy |
| --- | ---: | ---: |
| CRATE-S | 13.12 million | 69.2% |
| ViT-S | 22.05 million | 72.4% |

In this comparison CRATE-S has **about 40.5% fewer parameters** and **3.2 percentage points lower accuracy**. The table supplies no uncertainty interval for these values. Its other datasets and model sizes offer different comparisons; this pair does not justify a universal efficiency or accuracy ranking. Parameter count is also not latency or training energy.

An ablation would need to isolate the subspace-attention and sparse-coding changes under controlled data, schedule and evaluation, ideally checking both equal architecture settings and comparable resource budgets. Chapter 8’s broader applications are useful follow-up reading, but this overview does not reproduce or audit all of their experiments.

## 9 · Choose your reading route

| Your question | Textbook route | Then connect to the course |
| --- | --- | --- |
| What structure survives compression? | §§1.4, 2.1, 4.2, 6.1 | [Information bottleneck](/papers/t1-paper-1): do not equate coding-rate surrogates with exact mutual information. |
| Why does a denoiser help generation? | §§3.2–3.3, 7.1, 8.6 | [Generative overview](/papers/t2-generative-overview), then [Mean Flows](/papers/t2-paper-2). |
| Can an objective suggest a network? | §§2.3.3, 5.1–5.2, 8.4 | Compare feature updates with the parameter dynamics in [Bayesian SGD](/papers/t1-paper-3). |
| What does reconstruction verify? | §§6.1–6.2, 7.3, 8.5 | [DiffAtlas](/papers/t2-paper-4): identify observed and generated quantities. |
| How should I assess generalization? | §3.3, §8.4.5 and Chapter 9 | [Generalization bounds](/papers/t1-paper-2) and [double descent](/papers/t1-paper-4): assumptions and evaluation matter. |

The recall cards below are grouped with Topic 1’s foundational study material for practice and printing; this does not reclassify the textbook into Topic 1. No new interactive simulator is needed here: each toy example can be checked directly from its equations.

## 10 · The idea to carry forward

For a new model, draw **data → code → output**, label what its objective rewards, and identify what its evaluation actually measures. Then ask whether you are learning the data distribution, transforming its representation, or doing inference with an already learned model. All three can appear in one system, but evidence for one is not automatically evidence for the others.

The authors’ Chapter 9 proposals about autonomous, natural and scientific intelligence are a research agenda. Their broader claims should be read as the authors’ positions, distinct from the assumptions of a PCA result or the measured entries in a classification table. Compactness, consistency and a useful prior provide concrete questions to test, rather than a universal guarantee of intelligence.

## Sources and reading map

[Open the supplied textbook](/papers/representation-overview.pdf). The cover dates this copy March 1, 2026; the preface names Version 2.0. Page references below are **printed page numbers**, which differ from PDF viewer indices. This guide targets the overview and selected mechanisms, not an exhaustive review of the entire book.

| Source location | Role in this guide |
| --- | --- |
| Prefaces, contents; §§1.2–1.4 | Low-dimensional structure, chapter map, prerequisites and design perspective. |
| §2.1.1, pp. 53–57, Eq. 2.1.6 | PCA encoder/decoder and its assumptions. |
| §3.2.1, pp. 96–102, Theorem 3.2; §3.3 | Denoising, score identity and generalization questions. |
| §4.2.3, pp. 157–160, Eqs. 4.2.10–4.2.15 | Coding-rate surrogate, group weighting, normalization and objective. |
| §§5.1–5.2, pp. 177–204 | Feature updates, unknown membership at inference and CRATE’s functional construction. |
| §§6.1–6.2, pp. 225–244, Eq. 6.2.14 | Reconstruction, self-consistency and the encoder–decoder game. |
| §7.1, pp. 260–262; §§7.3–7.4 | Measurements, posterior means/modes/samples and conditioning routes. |
| §8.4.5, pp. 344–345, Table 8.5 | The selected quantitative comparison and its protocol. |
| Chapter 9, pp. 437–449 | Open problems and the authors’ broader proposals. |

The PCA coordinates, Gaussian calculation, normalized group example, scalar optimization trace and discrete posterior are original teaching examples. The displayed network paths are functional schematics; consult Chapter 8 for implementation-specific choices. No weights are trained or inference run by this page.
