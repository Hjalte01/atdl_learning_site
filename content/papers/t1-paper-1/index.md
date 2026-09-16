---
title: "Opening the black box of Deep Neural Networks via Information"
shortTitle: "Information bottleneck"
topic: "topic-1"
order: 1
description: "Follow a hidden layer into the information plane: what it remembers, what it predicts, and what measured compression can—and cannot—explain."
status: "ready"
source: "https://arxiv.org/abs/1703.00810v3"
pdf: "/papers/information-bottleneck.pdf"
---
## The puzzle: why keep training after fitting the labels?

Imagine a classifier that recognizes a shape but also remembers its irrelevant orientation. Both kinds of information occupy its hidden representation. A low training loss alone does not tell us whether the network has learned a reusable rule or retained accidental details.

Shwartz-Ziv and Tishby propose watching **what each whole layer remembers about the input and the label**. Their experiments show an early fitting phase followed by a longer phase of measured input compression. They interpret the latter through stochastic gradient noise and the information bottleneck (IB).

This guide follows the supplied **19-page arXiv 1703.00810v3, 29 April 2017** course PDF. It explains a historical experimental argument, not a universal law of neural-network training. Allow about 25 minutes. Colored figures are original teaching schematics; the evidence figure uses explicitly reported milestones. No trained model runs here. Click any diagram to enlarge it.

## 1 · Keep the rule, forget the nuisance

[![Four inputs differing in relevant bit A and irrelevant bit B are grouped into two representations according to A.](/figures/information-bottleneck/problem.svg)](/figures/information-bottleneck/problem.svg)

Read left to right: inputs differing only in nuisance bit $B$ share a representation. In this **teaching example**, $X=(A,B)$ consists of two independent fair bits and the label is $Y=A$. Keeping $X$ retains two bits; keeping $A$ retains one bit and still predicts $Y$ perfectly. Keeping only $B$ uses one bit but loses all predictive information. Compression needs a relevance constraint.

A plausible reasoning path is: inspect representations rather than individual weights; measure retained information independently of coordinate choices; then ask whether training approaches a good compression–prediction tradeoff. This is our reconstruction of the design logic, not the authors’ private reasoning.

The paper's actual task is more elaborate: 12 binary inputs describe points on a sphere, giving $2^{12}=4096$ uniformly likely patterns. A rotation-invariant function $f(x)$ determines a softened binary label (Eq. 10):

$$
p(y=1\mid x)=\sigma(f(x)-\theta),\qquad \sigma(u)=\frac{1}{1+e^{-u}}.
$$

Here $x$ is one pattern, $\theta$ is the decision threshold, and $\sigma$ makes the label stochastic. The authors choose the threshold for approximately balanced labels and a sufficiently sharp sigmoid for $I(X;Y)\approx0.99$ bits (§3.1). The rule is known, allowing evaluation over the full distribution. Our two-bit example is not their dataset.

## 2 · Architecture: train a predictor, measure its representations

[![Training sends input through learned fully connected layers to sigmoid prediction and cross-entropy; an analysis branch bins frozen layer activations to estimate information. Inference uses only the frozen predictor.](/figures/information-bottleneck/architecture.svg)](/figures/information-bottleneck/architecture.svg)

**A → B → C** is an ordinary fully connected classifier. For a batch of $m$ inputs, a hidden activation tensor has shape $m\times d_i$, with $d_i$ the width of layer $i$. A functional layer is $T_i=\tanh(W_iT_{i-1}+b_i)$ for column-vector inputs; $W_i$ has shape $d_i\times d_{i-1}$. These are learned weights and biases. The final sigmoid produces a binary-label probability $q_\theta(y=1\mid x)$; here the subscript $\theta$ denotes all model parameters, distinct from the rule threshold above.

Section 3.1 lists widths **12–10–7–5–4–3–2** for its general setup, with tanh neurons and a sigmoid final output. Different experiments use different architectures: Figure 5 explicitly describes widths beginning at 12 and decreasing by two, with a final two-neuron layer in every panel. Do not splice these descriptions into one supposedly universal backbone. Section 3.2 also calls the binned activations “arctan,” whereas §3.1 says tanh; this source inconsistency is retained as a caveat rather than silently resolved.

**C → D:** training minimizes binary cross-entropy, using SGD and no additional explicit regularization (§3.1). In teaching notation:

$$
\mathcal L_B=-\frac1m\sum_{j=1}^{m}\left[y_j\log q_j+(1-y_j)\log(1-q_j)\right],\qquad
\theta_{k+1}=\theta_k-\eta g_B,\quad g_B=\nabla_\theta\mathcal L_B.
$$

$B$ denotes the sampled training batch, $y_j\in\{0,1\}$ its labels, $q_j$ the predicted probabilities, $k$ the update index, and $\eta$ the learning rate. For $y=1,q=0.8$, the single-example loss is $-\ln(0.8)\approx0.223$ nats. These are standard training equations, not additional numbered equations from the paper.

**The analysis branch E → F is separate.** At a checkpoint, hold weights fixed, enumerate inputs, collect each layer's activation vector, discretize it, and calculate information coordinates. These measurements do not supply the training loss. **Inference** freezes weights and runs A → B → C for a new input; labels, binning, the information plane and an IB solver are not required to predict.

## 3 · What do the two coordinates mean?

[![Information plane has retained input bits on the horizontal axis and label bits on the vertical axis. A toy trajectory first rises then moves left; ideal two-bit examples are labeled.](/figures/information-bottleneck/plane.svg)](/figures/information-bottleneck/plane.svg)

The horizontal coordinate asks how much the representation tells us about the input; the vertical coordinate asks how much it tells us about the label:

$$
I_X=I(X;T),\qquad I_Y=I(T;Y).
$$

A whole layer is one multivariate random variable $T$, not a sum of separate neurons’ information. Correlated neurons can encode redundant information; simply adding their individual scores double-counts it.

The analysis box F computes mutual information (paper Eqs. 1–2):

$$
I(X;T)=\sum_{x,t}p(x,t)\log_2\frac{p(x,t)}{p(x)p(t)}=H(X)-H(X\mid T).
$$

$p(x,t)$ is the joint probability, $p(x)$ and $p(t)$ are marginals, $H(X)$ is input entropy, and $H(X\mid T)$ is remaining input uncertainty after seeing the layer. Zero-probability terms contribute zero. Base-two logarithms give bits; this is separate from the natural-log loss above. Independence makes the ratio one and the information zero.

In our example, $H(X)=2$. If $T=A$, knowing $T$ leaves one unknown fair bit, so $I_X=2-1=1$ bit; $I_Y=1$ bit. If $T=X$, the point is $(2,1)$. If $T=B$, it is $(1,0)$. Moving left is useful only if enough label information survives. A constant representation reaches $(0,0)$: maximal forgetting, useless prediction.

### Depth is not training time

At **fixed weights**, the Markov chain $Y\to X\to T_1\to T_2$ gives the data-processing inequalities (paper Eqs. 5–6):

$$
I(Y;T_2)\le I(Y;T_1)\le I(Y;X),\qquad
I(X;T_2)\le I(X;T_1)\le H(X).
$$

A later layer cannot manufacture information about the true label absent from its input. Yet a simpler representation can make that information easier for a restricted decoder to use. During **training**, the mapping itself changes, so one layer can move upward across checkpoints without violating this inequality. Also, independently binning each layer need not preserve the Markov chain between the *binned* variables; do not treat arbitrary estimated coordinates as exact DPI guarantees.

<details><summary>Predict: does doubling every activation necessarily double its information?</summary><p>No. An invertible transformation preserves exact mutual information (paper Eq. 3). A fixed-width binning procedure can nevertheless change its measured information when values cross bin boundaries. Coordinate invariance of the mathematical quantity is not invariance of every estimator.</p></details>

## 4 · Representation close-up: the bins matter

[![Four distinct scalar activations occupy four bins initially and two bins later, illustrating a fall from two bits to one bit in quantized entropy even when all exact values remain distinct.](/figures/information-bottleneck/binning.svg)](/figures/information-bottleneck/binning.svg)

Section 3.2 uses **30 equal intervals between −1 and 1**, then treats the tuple of discretized neuron values as the layer state. Denote this measured variable $\widetilde T=Q(T)$, with $Q$ the quantizer. Over all 4096 uniformly likely inputs, the method constructs $p(\widetilde t,x)$ and, using the known stochastic rule:

$$
p(\widetilde t,y)=\sum_x p(x,y)p(\widetilde t\mid x).
$$

The sum aggregates inputs that fall into the same bin tuple. It belongs at **E → F** in the architecture. Labels sharing a tuple determine its decoder distribution, $p(y\mid\widetilde t)$.

Follow the miniature version in the diagram. Four equally likely inputs initially produce activations $[-0.75,-0.25,0.25,0.75]$. Our **four-bin toy quantizer**, with edges at $-1,-0.5,0,0.5,1$, distinguishes all four, yielding $H(\widetilde T)=2$ bits. Later activations $[-0.90,-0.80,0.80,0.90]$ occupy two bins with equal probability, so $H(\widetilde T)=1$ bit. If the first two inputs have label 0 and the last two label 1, label information remains one bit.

For a fixed deterministic network on this finite input set, $H(T\mid X)=0$ and therefore $I(X;T)=H(T)$. All four *exact* activations remain distinct in the toy, so their exact information remains two bits. The measured one-bit reduction comes from quantization. This is a mathematical diagnostic, not an extra experiment from the paper. It explains why noise, precision and the definition of the random variable must accompany any compression claim.

The paper repeats its calculations across 50 randomized initializations and training samples. Averaging runs summarizes variability; it does not make the underlying binning choice disappear. Full-distribution $I_Y$ measures retained label information, not classification accuracy itself: bits and percent correct are different metrics.

## 5 · The bottleneck objective and its encoder

[![A candidate input is compared with two representation groups by label-distribution divergence. The resulting stochastic assignment updates group masses and decoders in an iterative IB consistency loop.](/figures/information-bottleneck/bottleneck.svg)](/figures/information-bottleneck/bottleneck.svg)

The theoretical IB problem asks for a compact representation with predictive value (paper Eq. 8):

$$
\min_{p(t\mid x)}\ \mathcal L_{\mathrm{IB}}=I(X;T)-\beta I(T;Y),\qquad \beta\ge0.
$$

$p(t\mid x)$ is a possibly stochastic encoder and $\beta$ sets the importance of label retention. The marginal $p(t)$ and decoder $p(y\mid t)$ must be consistent with that encoder and the fixed rule $p(x,y)$. Larger $\beta$ rewards preserving predictive information more strongly. This objective describes the theoretical comparison in box F; **the classifier was trained with cross-entropy, not this explicit loss**.

For the toy encoders $T=X$, $T=A$, and constant $T$, at $\beta=2$ the objectives are respectively $2-2=0$, $1-2=-1$, and $0$. Retaining the relevant bit wins. At $\beta=0$, the constant wins. An objective without a relevance reward has no reason to retain the label.

The stationary encoder compares **label distributions**, not Euclidean input distances (paper Eq. 9):

$$
p(t\mid x)=\frac{p(t)}{Z(x;\beta)}\exp\left[-\beta D_{\mathrm{KL}}\big(p(y\mid x)\Vert p(y\mid t)\big)\right],
$$

$$
p(t)=\sum_x p(t\mid x)p(x),\qquad
p(y\mid t)=\sum_x p(y\mid x)p(x\mid t).
$$

$D_{\mathrm{KL}}(p\Vert q)=\sum_y p(y)\ln[p(y)/q(y)]$ measures the label-distribution mismatch; $Z$ normalizes assignments over $t$. $p(x\mid t)$ follows Bayes’ rule. Natural-log KL is used in this exponential form; consistently changing information units rescales $\beta$.

**A numerical assignment through the boxes:** suppose $p(y=1\mid x)=0.9$, and two equally probable groups predict 0.9 and 0.1. Their KL mismatches are 0 and $0.8\ln9\approx1.758$ nats. With $\beta=1$, their unnormalized assignment weights are $0.5$ and $0.5e^{-1.758}\approx0.086$. Normalization gives approximately **0.853 versus 0.147**. This is one illustrative encoder update with provisional groups, not a converged IB solution or a neural activation from the experiment.

Read the loop in the figure: update assignments, recompute group mass, recompute label mixtures, repeat. The equations are self-consistent because changing one distribution changes the others; stationary conditions alone do not certify a globally optimal numerical solution.

In §3.8, the authors use a converged layer's decoder to construct an IB encoder and choose $\beta$ to minimize its mean KL discrepancy from the empirical encoder (Eq. 12). Figure 6 reports proximity to the IB curve, with standard errors across 50 networks. This is their numerical optimality evidence, not proof that arbitrary SGD networks solve IB exactly. On a smooth part of the optimal information curve, the slope is $1/\beta$; tangent slope describes a tradeoff, not an SGD learning rate.

## 6 · Training over time: drift, then diffusion?

[![Early SGD updates align in a common direction; later updates fluctuate while mean drift is small. Each update carries the weights forward and the analysis records a new information point.](/figures/information-bottleneck/dynamics.svg)](/figures/information-bottleneck/dynamics.svg)

The trajectory diagram in §3 represents **one layer across training time**, schematically rising during fitting and moving left during measured compression. Figures 2–3 show this behavior in the paper. To explain it, Figure 4 compares the norms of gradient means and standard deviations, normalized by layer weight norms.

A teaching decomposition writes a batch gradient as $g_B=\mu+\xi_B$, where $\mu=\mathbb E_B[g_B]$ and $\mathbb E_B[\xi_B]=0$. Then:

$$
\theta_{k+1}=\theta_k-\eta\mu-\eta\xi_B.
$$

This expands the update in box D. The carried state is the model's weights; minibatch sampling supplies fluctuating gradients. Early on, a large mean relative to fluctuations produces directed **drift**. Later, small mean relative to fluctuations motivates a **diffusion** approximation. This does not imply that every update is independent Gaussian noise or that training has stopped.

For a scalar illustration, $\mu=0.1$, noise standard deviation $0.01$, and $\eta=0.1$ give mean step magnitude 0.01 and step standard deviation 0.001. Later, $\mu=0.001$ and standard deviation 0.02 give 0.0001 and 0.002. The second regime is noise-dominated. These numbers are invented to explain the comparison, not digitized from Figure 4.

The authors argue that weight diffusion under a training-error constraint encourages compressed representations (§3.5). The identity $I(X;T)=H(X)-H(X\mid T)$ shows what compression means when $H(X)$ is fixed. It does **not**, by itself, prove that increasing entropy of the weight distribution increases $H(X\mid T)$. That bridge requires assumptions about induced representations, noise and dynamics. The paper explicitly defers a rigorous stochastic-relaxation analysis.

Their depth argument (§3.7, Eq. 11) proposes that splitting a large compression task into smaller stages reduces a superlinear relaxation cost. For a teaching cost $C(\Delta)=e^\Delta$, one four-unit task costs $e^4\approx54.6$, whereas two two-unit tasks cost $2e^2\approx14.8$. This is a toy cost model, not measured seconds; it assumes an appropriate cost regime and stages that can perform the required compression. It does not prove that adding arbitrary layers always speeds training.

<details><summary>Predict: does a late move to the left guarantee better generalization?</summary><p>No. Figure 3's small-sample setting loses label information during compression. Forgetting can remove signal as well as nuisance. Check the vertical coordinate, the estimator, and out-of-sample predictive performance.</p></details>

## 7 · What the experiments establish

[![Reported training milestones compare failure of a one-hidden-layer network to reach good label information after ten thousand epochs with a six-hidden-layer network reaching full relevant output information within four hundred epochs; thresholds are not claimed equal.](/figures/information-bottleneck/evidence.svg)](/figures/information-bottleneck/evidence.svg)

Read these as **reported milestones**, not bars estimating a precise speedup ratio. Section 3.6 compares one through six hidden layers on random samples covering 80% of the synthetic patterns, with 50 randomized runs. It says the one-hidden-layer network failed to reach good $I_Y$ even after $10^4$ epochs, while the six-hidden-layer network reached the full relevant output information within 400 epochs. The endpoints use the paper's qualitative descriptions; there is no matched numerical stopping threshold or wall-clock measurement here.

| Source | Setting and measurement | Reported observation | Interpretation limit |
| --- | --- | --- | --- |
| Figures 2–3; §§3.3–3.4 | Synthetic 4096-pattern rule; binned information in bits; 50 randomized networks; 5%, 45%, 85% training coverage | Early fitting followed by compression; at 5%, label information can fall during compression | Compression is not sufficient for generalization |
| Figure 4; §3.5 | Gradient mean/standard-deviation norms, normalized by weight norms | Transition around 350 epochs in that experiment | An observed transition, not a universal training schedule |
| Figure 5; §3.6 | Same synthetic rule; 80% coverage; depth comparison above | Large reduction in epochs with depth | Different architectures; epochs are not equal compute |
| Figure 6; §3.8 | Fitted IB encoder–decoder consistency; 50-network standard errors | Converged layers near the theoretical curve | Depends on numerical distributions and quantization |
| Figure 8; §4 | Non-symmetric committee-machine rule | Similar two-phase trajectories and gradient behavior | Still a controlled setting |

Section 4 additionally reports a gradient-statistics transition for MNIST. It explicitly leaves direct information-plane estimation for large problems to more sophisticated estimators. Thus MNIST is not a demonstrated replication of every information-plane claim in this PDF.

### What would make the explanation stronger?

These are proposed checks, not reported results: vary bin count and precision while holding trained activations fixed; compare saturation-prone activations with other choices; vary batch size or controlled noise while matching optimization progress; repeat the depth comparison at matched compute; measure held-out prediction alongside information estimates. Each separates a potential cause from an accompanying observation.

The known full rule distribution is an experimental advantage, but it also limits immediate transfer to real datasets where $p(x,y)$ is unknown. Neither one neuron’s apparent meaning nor a two-coordinate summary fully describes a representation's geometry, robustness or accessibility to a chosen decoder.

## 8 · The idea to carry forward

**Ask what a representation preserves, what it discards, and how you measured the difference.** The information bottleneck supplies a precise tradeoff, and the information plane offers a useful view of changing representations. The paper's stronger claim—that SGD noise drives general networks toward IB-optimal representations—must be assessed separately from those definitions.

Redraw the architecture and label three different operations: cross-entropy updates weights; binning defines a measured representation; IB compares candidate encoders. Explain why none is interchangeable with the others. Then locate $(2,1)$, $(1,1)$, and $(0,0)$ for the two-bit toy without looking back.

## Sources and reading map

- [Supplied course PDF, arXiv v3, 29 April 2017](/papers/information-bottleneck.pdf): §§2.1–2.3 / Eqs. 1–9 for information and IB; §§3.1–3.2 / Eq. 10 for architecture, rule and estimator; Figures 2–4 / §§3.3–3.5 for trajectories and gradient statistics; Figure 5 / §§3.6–3.7 / Eq. 11 for depth; Figure 6 / §3.8 / Eq. 12 for encoder consistency; Figures 7–8 and §4 for sample size and scope.
- [Version-specific arXiv record](https://arxiv.org/abs/1703.00810v3): confirms paper identity and revision. All experimental claims above use the supplied course version.

All toy values, colored schematics and mathematical diagnostics are teaching constructions. The source's activation-name and architecture-description differences are stated where they matter; no training hyperparameters or missing benchmark scores are invented.
