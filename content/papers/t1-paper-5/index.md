---
title: "Towards Understanding Grokking: An Effective Theory of Representation Learning"
shortTitle: "Understanding grokking"
topic: "topic-1"
order: 5
description: "Why can a network memorize first and generalize much later? Trace embedding geometry, constraint rank, and the competition between representation and decoder learning."
status: "ready"
pdf: "/papers/understanding-grokking.pdf"
source: "https://proceedings.neurips.cc/paper_files/paper/2022/hash/dfc310e81992d2e4cedc09ac47eff13e-Abstract-Conference.html"
---
## 1 · Perfect training accuracy, unfinished learning

A network learns addition from a small table of examples. It answers every training question correctly, yet fails on held-out pairs. Thousands of updates later, without receiving new examples, it starts answering those too. What changed after the training set was already fitted?

Liu and colleagues call this delayed generalization **grokking**. Their explanation centers on the input representation: a flexible decoder can first memorize arbitrary embedding combinations, while useful geometry develops more slowly. Once equivalent calculations share representations, learned answers become reusable.

This guide follows the supplied **13-page NeurIPS 2022 conference paper**, whose identity is confirmed by the [official proceedings record](https://proceedings.neurips.cc/paper_files/paper/2022/hash/dfc310e81992d2e4cedc09ac47eff13e-Abstract-Conference.html). The course PDF ends after the references and checklist; its cited Appendices A–K are not included. We do not infer an arXiv revision, missing decoder widths, or an exact phase-labeling rule from those absent appendices. Original diagrams are teaching schematics; Figure 4 in §8 is an actual source crop. Every image opens at full size. No trained model or interactive simulation runs here.

[![Three schematic stages: random representations, training memorization, and later reusable structure.](/figures/grokking/problem.svg)](/figures/grokking/problem.svg)

**Read the three stages as different checkpoints of one training run.** Perfect training accuracy does not mean that every parameter has stopped moving, nor that the training loss is exactly zero. Classification loss can continue changing after the argmax answers are correct. The drawing is conceptual, with no measured times.

**A plausible reasoning path, not a claim about the authors’ private thoughts:** first look for what changes internally during the delay; then define a measurable structure; then ask how many training constraints identify it and how fast those constraints can be satisfied. Finally, vary how readily the decoder can memorize.

<details><summary>Predict: does a long plateau at perfect training accuracy prove that further learning is impossible?</summary>

No. Held-out accuracy can improve much later. But a finite run that has not generalized does not establish whether it will eventually grok; the reported memorization phase depends on the observation budget.

</details>

## 2 · Architecture: lookup, add, decode

The main theory studies **ordinary addition**, not modular addition. Inputs are symbols $i,j\in\{0,\ldots,p-1\}$. Swapping them produces the same example, so the complete dataset $D_0$ has $p(p+1)/2$ unordered pairs, including repeats. For $p=10$, there are 55 examples and 19 possible sums, from 0 through 18. Training set $D$ and held-out set $D'$ are disjoint and cover $D_0$; the training fraction is $r=|D|/|D_0|$.

[![Symbols select two vectors from a shared p by d-in embedding table, sum them, and feed an MLP decoder; loss gradients update both.](/figures/grokking/architecture.svg)](/figures/grokking/architecture.svg)

**Follow A → B → C → D, then the update path E.** The trainable table $E\in\mathbb R^{p\times d_{\rm in}}$ returns two rows. Box B adds them, giving $z=E_i+E_j\in\mathbb R^{d_{\rm in}}$. Box C is a trainable MLP decoder with parameters $\theta$:

$$
\widehat Y_{i+j}=\operatorname{Dec}_{\theta}(E_i+E_j)\in\mathbb R^{d_{\rm out}}.
$$

This is the architecture of §2, not a new numbered paper equation. For regression, each sum $c$ has a **fixed random target vector** $Y_c$. For classification, the target is one-hot; the decoder produces class logits and training uses cross-entropy. The decoder’s hidden widths and activation recipe are not specified in the supplied main text, so the diagram leaves them functional.

At D, a teaching convention for the regression objective is

$$
\mathcal L_{\rm task}=\frac1{|D|}\sum_{(i,j)\in D}
\left\|\operatorname{Dec}_{\theta}(E_i+E_j)-Y_{i+j}\right\|^2.
$$

The norm sums squared output-coordinate errors; dividing by $d_{\rm out}$ would change the scale, not the best fit. This displayed normalization is for teaching, not a claim about the code’s loss reduction. Both $E$ and $\theta$ receive gradients. The **effective loss** introduced later instead operates only on embeddings and is an analytical surrogate, not the network’s training objective.

**Why sum?** It builds commutativity into the architecture. It does not encode the full addition table: randomly initialized symbol vectors still have to acquire useful relations. The same symmetric input construction alone cannot represent a genuinely order-sensitive operation; the paper discusses a different construction for non-abelian operations.

## 3 · Training and inference follow different paths

[![Training updates embedding and decoder parameters with separate optimizer states; inference freezes both and uses no target.](/figures/grokking/training.svg)](/figures/grokking/training.svg)

**The upper path repeats; the lower path is a prediction.** A training pair and its target pass through A–D. Backpropagation supplies gradients to both the selected embeddings and the shared decoder. Embedding weights, decoder weights and optimizer moments carry forward; targets and the train/validation split stay fixed. At inference, both learned modules are frozen. Classification uses the largest logit; regression produces a target-space vector. The main text does not specify the regression-to-accuracy decoding rule, so we do not invent one.

Section 4.1 updates 1D embeddings with Adam, no weight decay, and learning rates in $[10^{-5},10^{-2}]$. It updates the decoder with AdamW over the same learning-rate range, varying weight decay over $[0,10]$ for addition regression and $[0,20]$ for classification. The addition phase diagrams use 45 training and 10 validation examples. The 90% training/validation accuracy milestones concern those experiments; the earlier representation experiments use an RQI threshold instead.

These two optimizers can run on different timescales. A fast decoder may fit arbitrary representations before the embedding table develops useful structure. An extremely slow decoder can also delay success because it cannot exploit a good representation quickly. The paper’s lesson is a balance, not “make the decoder as slow as possible.”

## 4 · Representation close-up: equal sums become equal inputs

[![Training pair 6 plus 8 and held-out pair 5 plus 9 have the same embedding midpoint and sum under an affine representation.](/figures/grokking/representation.svg)](/figures/grokking/representation.svg)

**Compare the two routes into the same decoder.** Suppose $(6,8)$ is in training and $(5,9)$ is held out. If

$$
E_5+E_9=E_6+E_8,
$$

the decoder gets exactly the same input for both. If it correctly maps the training input to $Y_{14}$, it must return that same target for the held-out pair. This is the transfer mechanism in §3.1, not a claim that it can predict targets never represented in training.

The paper calls $(i,j,m,n)$ a $\delta$-parallelogram when $\|(E_i+E_j)-(E_m+E_n)\|\leq\delta$, with tolerance $\delta\geq0$. The equal-sum condition corresponds to coincident midpoints; a 1D example is a degenerate parallelogram. Figure 2 in the paper uses **averaged** embeddings for visualization, $(E_i+E_j)/2$, while the main architecture uses sums. Consistent rescaling preserves the equality; mixing the two inputs without changing the decoder would not.

To measure this structure, paper Eqs. (1)–(3) define

$$
P_0(D)=\{(i,j,m,n):(i,j),(m,n)\in D,\ i+j=m+n\},
$$
$$
P(R,\delta)=\{q\in P_0(D_0):\|E_i+E_j-E_m-E_n\|\leq\delta\},
\qquad \operatorname{RQI}(R)=\frac{|P(R,\delta)|}{|P_0(D_0)|}.
$$

Here $R$ is the whole representation, $q=(i,j,m,n)$, and $|P|$ counts constraints. RQI is a **geometry diagnostic**, not accuracy or the loss optimized at D. It measures the fraction of permissible equal-sum relations that are realized. It uses the known full task structure, including held-out relations, for analysis.

An affine representation $E_k=a+kb$, for fixed vectors $a,b$, satisfies every relation because $E_i+E_j=2a+(i+j)b$. It therefore has RQI 1. But $b=0$ collapses every symbol to the same point and also satisfies those equalities; a useful fitted decoder must additionally distinguish different target sums.

**A counting caveat:** the literal printed set includes self-comparisons such as $(i,j,i,j)$. These always pass, so the text’s claim that random representations have RQI 0 requires omitting trivial relations or treating their contribution as a baseline. We retain the printed definition and do not fabricate an exact random baseline from an unspecified counting convention.

**Two directions, different assumptions.** For two fitted training examples, identical decoder inputs cannot produce different targets. Conversely, equal targets force identical inputs only if the decoder is injective on the inputs under consideration. Proposition 2 assumes both zero training loss and an injective decoder; arbitrary trained MLPs do not automatically meet those assumptions. The proof of Proposition 1 likewise applies to fitted examples, not arbitrary held-out pairs.

<details><summary>Predict: if two embedding sums are only close, must their predictions be identical?</summary>

No. Equality gives the exact transfer argument. A small distance needs additional control of decoder sensitivity to imply close outputs; a classification boundary could lie between the points.

</details>

## 5 · A worked pair through the same boxes

Take a **teaching example**, not learned weights: $p=10$, $d_{\rm in}=2$, and $E_k=(k,1)$. This is the affine representation above with $a=(0,1)$ and $b=(1,0)$.

[![Numerical trace: E6 plus E8 and E5 plus E9 both equal 14 comma 2, so the same decoder emits the same target.](/figures/grokking/numbers.svg)](/figures/grokking/numbers.svg)

**Use the architecture labels again.** At A, $(6,8)$ selects $(6,1)$ and $(8,1)$. At B, their sum is $(14,2)$. Suppose C has learned $\operatorname{Dec}_{\theta}(14,2)=Y_{14}=(0.2,-0.4)$. Then D’s squared regression error on that example is zero. At inference, A selects $(5,1)$ and $(9,1)$, B again produces $(14,2)$, and C outputs the same target without seeing the held-out label.

Now perturb only $E_9$ to $(9.2,1)$. The held-out input becomes $(14.2,2)$ and its equal-sum residual is $(0.2,0)$. That relation contributes $0.04$ to an **unnormalized** squared-residual sum. If $\delta=0.1$, it fails the RQI test. Neither fact tells us its classification accuracy without the decoder. Also, a held-out relation is not included in the training-derived effective objective unless both constituent pairs belong to $D$.

To see how a constraint moves embeddings, suppose both pairs are training examples and temporarily keep just their one relation. Write $r_q=E_5+E_9-E_6-E_8$ and $L_q=\|r_q\|^2$. A teaching gradient step with $\eta=0.1$ moves $E_5,E_9$ by $-2\eta r_q=(-0.04,0)$ and $E_6,E_8$ by $+2\eta r_q=(0.04,0)$. The new residual is $(0.04,0)$ and $L_q=0.0016$. This calculation isolates the attraction imposed by one relation. It is neither the full normalized effective flow nor an Adam update on the task loss.

## 6 · Effective dynamics: which structure is determined?

[![Training pairs generate equal-sum constraints; their nullspace determines remaining freedom and their slow modes determine convergence time.](/figures/grokking/dynamics.svg)](/figures/grokking/dynamics.svg)

**Follow a second, analytical path.** Training pairs supply equal-target relations. Those relations define constraints on the table, bypassing the decoder. The squared mismatch penalizes broken parallelograms; normalization prevents reducing the objective merely by shrinking every embedding. Paper Eqs. (4)–(5) propose

$$
\ell_0=\frac1{|P_0(D)|}\sum_{(i,j,m,n)\in P_0(D)}
\|\widetilde E_i+\widetilde E_j-\widetilde E_m-\widetilde E_n\|^2,
\qquad Z_0=\sum_k\|\widetilde E_k\|^2,
\qquad \ell_{\rm eff}=\frac{\ell_0}{Z_0},
$$
$$
\frac{d\widetilde E_i}{dt}=-\frac{\partial\ell_{\rm eff}}{\partial\widetilde E_i}.
$$

Here $\widetilde E$ denotes normalized embeddings, $Z_0>0$ is their total squared norm, and $t$ is effective continuous time. This is a hypothesized description of representation evolution, not an exact reduction of Adam plus MLP training. If there are no usable constraints or the representation is identically zero, the expression needs separate treatment.

At zero effective loss, each coordinate solves a linear system $AR=0$: one row per equal-sum relation, with coefficients $+1,+1,-1,-1$ combined when indices repeat. Here $R\in\mathbb R^p$ holds one coordinate of all embeddings, and $A$ is the constraint matrix. The nullity $n_0=p-\operatorname{rank}(A)$ counts free directions. Constant vectors and the vector $(0,1,\ldots,p-1)$ always solve the system, so $n_0\geq2$.

If $n_0=2$, all solutions are affine, $E_k=a+kb$, coordinate by coordinate. If $n_0>2$, extra unstructured solutions remain. More training pairs can add independent constraints and remove those freedoms, but sample count alone does not determine rank: which pairs were sampled matters.

**Small rank example.** For $p=3$, the nontrivial relation $(0,2)$ versus $(1,1)$ gives the row $[1,-2,1]$. Its rank is 1, nullity is 2, and it enforces $E_1=(E_0+E_2)/2$. Without that relation, all three scalar embeddings are free. This is about determining geometry; it does not train a decoder on missing output classes.

The paper next introduces a linear relaxation (Eqs. (8)–(9)):

$$
\frac{dR}{dt}=-HR,\qquad
R(t)=\sum_s a_s v_s e^{-\lambda_s t},\qquad
n_h=\frac1{\eta\lambda_3}.
$$

$H$ is the scaled curvature matrix of the quadratic constraint loss, $v_s$ its eigenvectors, $\lambda_s$ its ordered eigenvalues, and $a_s$ the initial coefficients. With exactly two zero modes, $\lambda_3>0$ is the slowest decaying mode; $n_h$ estimates one e-fold of decay at embedding step size $\eta$. If additional null modes exist, $\lambda_3=0$ and this relaxation cannot remove all unwanted structure.

For example, $\eta=10^{-3}$ and $\lambda_3=0.2$ give $n_h=5000$ steps. After three such times, that mode retains $e^{-3}\approx0.050$ of its initial amplitude. Halving $\lambda_3$ doubles this timescale. These are illustrative numbers, not measured accuracy milestones. In discrete Euler descent the mode instead multiplies by $1-\eta\lambda_s$ per step; exponential decay approximates small stable steps.

**Read the normalization carefully.** The supplied page 5 defines a variance-like $\sigma_t$ without a square root, while page 6 describes unit-variance normalization. More substantively, differentiating the displayed quotient does not give only $-HR$. A teaching derivation, writing $\ell_0=R^TQR$ and $Z_0=R^TR$ for a symmetric quadratic matrix $Q$, gives

$$
-\nabla_R\left(\frac{R^TQR}{R^TR}\right)
=-\frac{2QR}{Z_0}+\frac{2\ell_0 R}{Z_0^2}.
$$

The second term is radial and preserves the norm under this flow: $R^T\dot R=0$. The fixed quadratic linear flow omits it and generally changes the norm. On centered embeddings, their directions can be related through normalization and time rescaling, but they are not literally the same evolution in the same coordinates and time. We therefore present the paper’s $1/(\eta\lambda_3)$ result as a qualitative spectral timescale, consistent with its stated empirical limitations, not an exact theorem for the trained network. Missing Appendix F prevents checking its fuller conservation argument here.

## 7 · Four outcomes from the same learning problem

[![Four learning phases compare whether training succeeds, whether validation succeeds, and whether validation is substantially delayed.](/figures/grokking/phases.svg)](/figures/grokking/phases.svg)

**Read each box as an outcome within an observation window.** Comprehension and grokking both generalize; their difference is the delay relative to fitting training examples. Memorization fits training examples without attaining the validation criterion during the run. Confusion fails even to fit the training set. This diagram summarizes the labels, not measured phase boundaries or a replacement for the missing Appendix A threshold table.

The phase sweeps in Figure 6 support a competition picture. A fast, flexible decoder can memorize before representation structure develops. Constraining it can encourage reusable representations, but too much restriction can prevent fitting. Comprehension occupies a useful balance; grokking appears near memorization. Neither “more weight decay always helps” nor “waiting longer always works” follows.

The transformer experiment in §4.2 uses **addition modulo 53**, 256-dimensional trainable symbol embeddings, a decoder-only transformer, concatenation of its two final token outputs, and a linear classifier. Operation symbols are omitted. This is a separate architecture from the sum-plus-MLP toy model; the main text does not specify its layer count or head count. Figure 7 scans decoder learning rate and weight decay with embedding learning rate fixed at $10^{-3}$ and embedding weight decay zero.

PCA projections of successful transformer embeddings form a circle. That is evidence of a structured representation for modular arithmetic, not a proof that the ordinary-addition affine constraint system governs the transformer. Absence of visible PCA structure also cannot establish absence of all structure.

The PCA diagnostic in §4.2 is

$$
S=-\sum_s q_s\log q_s,\qquad d_{\rm eff}=e^S,
$$

where $q_s$ is the fraction of embedding variance explained by principal component $s$, with $0\log0=0$. This lives in an **analysis branch on the embedding table**, not in the training loss. Equal variance in two components gives $d_{\rm eff}=2$; equal variance in four gives 4. A two-dimensional plot does not imply that the full representation has effective dimension exactly 2. Figure 7’s reported values are much larger.

<details><summary>Predict: would increasing decoder weight decay necessarily turn memorization into comprehension?</summary>

No. The phase diagrams include confusion at strong restriction. Useful regularization depends on decoder learning rate, representation learning, task and training budget; a sweep is evidence about that setup, not a universal setting.

</details>

## 8 · What the evidence establishes

[![Original paper Figure 4 shows a critical data fraction around 0.4, empirical steps to RQI above 0.95, and theoretical versus neural representation trajectories.](/figures/grokking/source-figure-4.png)](/figures/grokking/source-figure-4.png)

**This is measured and theoretical source material, not generated data.** The crop preserves all four panels and the original caption from supplied page 5. In (a), the effective constraint theory predicts the probability of an affine structure. In (b), blue points summarize successful neural runs and red triangles mark runs that did not reach RQI $>0.95$ within $10^4$ steps. Those triangles are censored observations, not exact convergence times or proof of never generalizing. Panels (c) and (d) compare representation trajectories under theory and actual training; qualitative agreement is the claim.

| Source and setting | Comparison and metric | Supported conclusion and limit |
| --- | --- | --- |
| Figure 3, toy ordinary addition | Full-dataset accuracy measured from the network versus predicted from representation; diagonal memorization reference | Geometry predicts useful transfer in this setup. The full-dataset metric includes training examples; it is not held-out accuracy alone. The detailed predictor is in absent Appendix D. |
| Figure 4, $p=10$, 55 unordered pairs | Constraint-based probability of affine structure versus steps to RQI $>0.95$ | Transition near training fraction $r_c\approx0.4$, roughly 22 examples. It is a distribution over sampled datasets, not a guarantee for every set of 22 pairs. |
| Figure 5, addition, embedding rate $10^{-3}$ | Neural steps to RQI $>0.95$ versus the effective spectral rate $\lambda_3$ | More data generally increases the rate and shortens the delay above the threshold. The text emphasizes qualitative, not exact quantitative, agreement. |
| Figure 6, addition, 45/10 split | Encoder/decoder rate and decoder weight-decay sweeps, 90% accuracy milestones | Four phases and a balance between fitting and representation learning. Exact phase-labeling details are not in the supplied PDF. |
| Figure 7, modulo-53 transformer | PCA effective dimension over 100 seeds; dropout and decoder hyperparameter comparisons | Structure and generalization coincide; decoder restriction can reduce delay. The text reports generalization below $10^3$ steps with substantial dropout, while the center axis says epochs; we do not silently equate them. |
| Figure 8, MNIST example | Training/validation curves; displayed 1,000 training points and initialization scale 9.0 | Delayed generalization also occurs in a mainstream image task under this setup. The missing Appendix J contains the fuller recipe; this is not an MNIST benchmark improvement claim. |

**What would strengthen the mechanism claim?** A proposed ablation would freeze random embeddings while training the same decoder, compare jointly learned embeddings at matched optimization budgets, and repeat across data splits. Another would compare training sets of equal size but different constraint ranks and spectral gaps. Report training accuracy, held-out accuracy, RQI and uncertainty over seeds separately. These are suggested tests, not additional results we ran.

Important limitations are the injectivity idealization, Euclidean geometry’s limited suitability for modular tasks, the omission of decoder dynamics from the effective theory, finite observation windows, and sensitivity to hyperparameters. The authors themselves discuss replacing Euclidean distance with a decoder-sensitive metric: distinct inputs can fall into the same output region even without an exact Euclidean parallelogram.

## 9 · Idea to carry forward

**Fitting examples and learning reusable structure are different milestones.** In the toy architecture, the reusable object is an embedding sum. Equal-sum training constraints can make a held-out calculation land on an already learned decoder input. Constraint rank describes what structure is identifiable; slow modes describe why discovering it can take time; decoder flexibility helps explain why fitting can happen first.

Carry that sequence into another model as a question, not a universal law: what representation would let two examples share a solution, what training signal could create it, and what competing path could fit the data without it?

| Reading question | Supplied paper location |
| --- | --- |
| What computation does the toy network perform? | §2, p. 2; Figure 2 uses midpoint visualization |
| Why can a new pair reuse an old target? | §3.1, pp. 3–4; Propositions 1–2; Eqs. (1)–(3) |
| Where do the effective constraints and nullity come from? | §3.2, pp. 5–6; Eqs. (4)–(7) |
| Why is a spectral gap associated with delay? | §3.2, p. 6; Eqs. (8)–(9), Figures 4–5 |
| What hyperparameters change the outcome? | §4.1, pp. 7–8; Figure 6 |
| What transfers beyond the toy example? | §4.2–4.3, pp. 8–10; Figures 1, 7–8 |
| What is unavailable in the local course copy? | Appendices A–K cited by the main text; not present in its 13 pages |

Return to [Topic 1](/topics/topic-1), compare delayed generalization with [epoch-wise double descent](/papers/t1-paper-4), or open the [supplied course PDF](/papers/understanding-grokking.pdf).
