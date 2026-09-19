---
title: "Deep Generative Modeling · a mechanism-first textbook roadmap"
shortTitle: "Deep Generative Modeling"
topic: "topic-2"
order: 2
materialKind: "extra"
description: "Trace probability through autoregressive models, flows, VAEs, energies, GANs and scores, with worked training and sampling examples."
status: "ready"
pdf: "/papers/deep-generative-modeling.pdf"
---
## The puzzle: why is making a sample different from scoring one?

A network can transform random numbers into an image without telling you the probability of that image. Another model can score a complete sequence efficiently but must generate it one symbol at a time. A third can reconstruct training examples convincingly while generating poorly from its prior. **A generative model is a probability construction plus a computational strategy.** Its architecture alone does not explain what it learns.

This guide follows Jakub M. Tomczak’s **Deep Generative Modeling, second edition, Springer, 2024**, ISBN 978-3-031-64087-2. The supplied copy’s second-edition preface is dated May 2, 2024. [Open the exact course PDF](/papers/deep-generative-modeling.pdf). This is a selective roadmap through Chapters 1–9, centered on the mechanisms in Chapters 3–5 and 9; it does not replace the book’s code exercises, advanced variants, compression chapter or application survey.

Allow 30 minutes. Bring conditional probability, logarithms, Gaussian distributions and basic derivatives. Every diagram below is an original teaching schematic; all numerical examples are constructed, not benchmark results. Open any diagram to enlarge it. Equation references refer to the supplied edition; unnumbered calculations are teaching derivations.

## 1 · Decide which computation you can afford

[![Six model families grouped by whether they compute densities, bound integrals, or learn energies, adversarial feedback and scores.](/figures/deep-generative-modeling/map.svg)](/figures/deep-generative-modeling/map.svg)

Read the boxes as alternatives, not a historical sequence. Chapter 1’s taxonomy asks how to represent a joint distribution. Chapter 2 first makes the underlying tasks explicit: define a model, estimate its parameters from data, test generalization, then perform inference. Here, **training** changes parameters; **generation** draws an observation with trained parameters fixed. Inference can also mean conditioning or estimating a latent posterior, so it is broader than generation.

The shared likelihood idea is to make observed data less surprising:

$$
\mathcal J(\theta)=-\frac1N\sum_{n=1}^N\log p_\theta(x_n).
$$

Here $x_n$ is example $n$, $N$ is the dataset size, $\theta$ denotes learned parameters, and $p_\theta$ is a probability mass function for discrete data or a density for continuous data. This negative log-likelihood operates after a model’s scoring computation. Minimizing it is maximum likelihood; it is not the actual training objective of every family below. Natural logs give nats. A continuous density is not a point probability and can exceed one.

| Family | What makes learning possible? | What happens when generating? | Main computational price |
| --- | --- | --- | --- |
| Autoregressive | Normalize each conditional | Draw and extend a prefix | Sequential dependencies |
| Normalizing flow | Invertible transform with tractable Jacobian | Transform a base sample | Invertibility and determinant constraints |
| VAE | Optimize a lower bound using an encoder | Prior sample → decoder distribution | Approximate posterior and likelihood bound |
| Energy model | Learn relative preferences | Usually run a sampling procedure | Normalization and mixing |
| GAN | Learn from a discriminator | Prior sample → generator | Coupled training and coverage failures |
| Score model | Regress a log-density gradient | Iteratively move a noisy state | Score error and sampling discretization |

This is our teaching reconstruction of the design tradeoffs, not a claim about the author’s private reasoning. Chapter 2’s mixtures and probabilistic circuits are useful prerequisites: mixtures sum over hidden alternatives; circuit restrictions can make inference tractable. They are not obsolete versions of neural models.

## 2 · Autoregression: turn one joint into many conditionals

[![Observed prefixes train a causally masked model; generated prefixes are sampled and carried forward, with a three-bit probability calculation.](/figures/deep-generative-modeling/autoregressive.svg)](/figures/deep-generative-modeling/autoregressive.svg)

The top row scores observed data; the bottom row generates unknown data. Chapter 3, Eq. 3.4 applies the probability chain rule:

$$
p_\theta(x)=\prod_{d=1}^D p_\theta(x_d\mid x_{<d}),\qquad
-\log p_\theta(x)=-\sum_{d=1}^D\log p_\theta(x_d\mid x_{<d}).
$$

$x=(x_1,\ldots,x_D)$ has $D$ entries; $x_{<d}$ is the prefix before entry $d$. The first conditional has an empty prefix. A shared network supplies distribution parameters; a causal architecture prevents access to later entries. This is where the product becomes a sum of training losses. The factorization itself is exact; limited context and finite network capacity are modeling restrictions.

**Follow the three boxes.** For observed bits $(1,0,1)$, suppose the probabilities assigned to the actual next bits are $0.6$, $0.75$ and $0.8$. Their joint mass is $0.36$ and their total negative log-likelihood is $-\log(0.36)\approx1.022$ nats. Each conditional also assigns complementary probability to the other bit. These are hypothetical outputs, not a trained model.

At training time all observed prefixes are available. A masked Transformer can compute many positions in parallel without leaking future tokens. At generation time the second prefix does not exist until the first symbol has been sampled. Freeze the network, sample the next conditional, append that symbol, and repeat. RNN computation has its own sequential constraints; parallel scoring is not universal to every autoregressive architecture (§§3.2–3.3).

<details><summary>Predict: does a causal mask make the variables independent?</summary>

No. It blocks future information while allowing dependence on the past. Independence would require each conditional to ignore its prefix.

</details>

## 3 · Flows: account for the space a transformation stretches

[![An affine flow doubles interval widths and halves density, while likelihood evaluation reverses the map and subtracts the log determinant.](/figures/deep-generative-modeling/flow.svg)](/figures/deep-generative-modeling/flow.svg)

A flow generates by transporting a simple random variable. To score the output, reverse that transport and correct for its local volume change (Chapter 4, Eqs. 4.7–4.14):

$$
x=f_\theta(z),\qquad
\log p_\theta(x)=\log p_Z(f_\theta^{-1}(x))-\log\left|\det J_{f_\theta}(z)\right|.
$$

$z,x\in\mathbb R^D$ have the same dimension in this ordinary bijective construction; $p_Z$ is the base density, $f_\theta$ is an invertible differentiable map, and $J_{f_\theta}$ is its Jacobian matrix. The density formula assumes the usual nonsingular change-of-variables conditions. The determinant arrow measures expansion, not image quality.

For the scalar map $x=2z+1$, $z\sim\mathcal N(0,1)$, the observation $x=1$ maps back to $z=0$. The derivative is $2$, so $p_X(1)=p_Z(0)/2\approx0.1995$ and the negative log-density is about $1.612$ nats. Doubling interval width halves density while preserving probability mass. This affine calculation illustrates the rule; it is not a learned image flow.

An arbitrary neural network does not guarantee an inverse or a cheap determinant. RealNVP’s coupling layer (§4.1.3) deliberately creates a triangular Jacobian:

$$
y_a=x_a,\qquad y_b=e^{s(x_a)}\odot x_b+t(x_a),\qquad
\log|\det J|=\sum_j s_j(x_a).
$$

The subscripts $a,b$ partition the input, $s$ and $t$ are scale and shift networks, and $\odot$ is elementwise multiplication. Since $x_a=y_a$ is preserved, recover $x_b=(y_b-t(y_a))\odot e^{-s(y_a)}$. The subnetworks need not themselves be invertible. Permutations let later layers transform other coordinates. Exact density evaluation still depends on a suitable data representation; the book treats dequantization and discrete flows separately.

Do not equate this recipe with [flow matching](/papers/t2-flow-matching): that guide trains a time-dependent velocity field using regression. Both transport distributions, but their training computations differ.

## 4 · VAE architecture: use an encoder to learn a decoder

[![VAE training encodes an observation into a latent distribution, reparameterizes a sample and decodes it; generation starts from the prior and omits the encoder.](/figures/deep-generative-modeling/vae.svg)](/figures/deep-generative-modeling/vae.svg)

The decoder makes it easy to draw $x$ once a latent $z$ is given. Scoring an observed $x$ is harder because all plausible $z$ values contribute:

$$
p_\theta(x)=\int p_\theta(x\mid z)p(z)\,dz.
$$

Here $p(z)$ is a prior over $M$ latent coordinates and $p_\theta(x\mid z)$ is a decoder likelihood over $D$ observation coordinates. The integral, from §5.3.1, is generally intractable for a nonlinear decoder. A VAE adds an encoder $q_\phi(z\mid x)$ to approximate the posterior, with its own parameters $\phi$.

The practical objective balances fit to the observation against the latent distribution’s departure from the prior (Eq. 5.17):

$$
\mathcal L(x)=\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-\mathrm{KL}(q_\phi(z\mid x)\Vert p(z))\leq\log p_\theta(x).
$$

$\mathcal L$ is the evidence lower bound, or ELBO; KL denotes Kullback–Leibler divergence. The reconstruction term is evaluated at decoder **C** against the original observation. The KL term compares encoder **A** with the prior. Maximize this bound, or minimize its negative, updating both networks. The reconstruction term must match the observation model: squared error is not a universal replacement for a likelihood.

Why a bound? Insert $q_\phi/q_\phi$ into the latent integral and use the concavity of $\log$ to move the logarithm inside the expectation. The more informative identity (Eqs. 5.24–5.25) is:

$$
\log p_\theta(x)-\mathcal L(x)=
\mathrm{KL}(q_\phi(z\mid x)\Vert p_\theta(z\mid x)).
$$

The nonnegative gap compares the encoder with the **true posterior**, not the prior. It vanishes when they agree, under the relevant support assumptions. Thus an improved bound need not represent an equally large improvement in exact likelihood; the approximation gap can change too. Figure 5.2 in the book illustrates how a loose bound can shift an optimum.

During unconditional generation, use the bottom row: draw $z\sim p(z)$, run the frozen decoder, then draw $x$ from its output distribution. No observed image or encoder is required. Returning a decoder mean instead of sampling is a different output choice. Good reconstructions alone do not prove that prior samples land in useful latent regions (§5.3.6).

## 5 · Latent close-up: trace one sample and its loss

[![A one-dimensional encoder with mean one and standard deviation one-half transforms noise minus two to latent zero, then computes reconstruction and Gaussian KL terms.](/figures/deep-generative-modeling/latent.svg)](/figures/deep-generative-modeling/latent.svg)

The encoder supplies a distribution, not just a compressed vector. For the diagonal Gaussian in §§5.3.3.1–5.3.3.2:

$$
q_\phi(z\mid x)=\mathcal N(\mu_\phi(x),\operatorname{diag}(\sigma_\phi(x)^2)),\qquad
z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon,
\quad\epsilon\sim\mathcal N(0,I_M).
$$

$\mu_\phi,\sigma_\phi\in\mathbb R^M$ are encoder outputs, each standard deviation is positive, and $I_M$ is an identity covariance. **Box B** moves randomness into independent $\epsilon$, leaving a differentiable path through $\mu$ and $\sigma$ to the encoder parameters. The variance is $\sigma^2$, not $\sigma$.

In the diagram’s scalar toy, $\mu=1$, $\sigma=0.5$, and $\epsilon=-2$, so $z=0$. Suppose decoder **C** assigns Bernoulli probability $0.8$ to an observed bit $x=1$. Its one-sample reconstruction loss is $-\log0.8\approx0.2231$ nats. For a standard Gaussian prior, the scalar analytic KL is:

$$
\mathrm{KL}(\mathcal N(\mu,\sigma^2)\Vert\mathcal N(0,1))
=\tfrac12(\mu^2+\sigma^2-1-\log\sigma^2)
\approx0.8181.
$$

Adding them gives a **one-sample estimate** of the negative ELBO, about $1.0413$ nats. This is not the exact marginal negative log-likelihood, nor is a single Monte Carlo estimate guaranteed to obey the bound pointwise. The expectation defines the bound. Summing coordinatewise KLs works for this diagonal Gaussian setup; more flexible priors and posteriors require different calculations.

<details><summary>Predict: if the encoder variance approaches zero, does this KL vanish?</summary>

No. With the standard Gaussian prior, the term $-\log\sigma^2$ diverges. Making the encoder nearly deterministic is not a free route to a better ELBO.

</details>

## 6 · Energies and GANs: two different ways to avoid direct likelihood computation

An energy model gives an observation a scalar preference. The basic form underlying Chapter 7 is:

$$
p_\theta(x)=\frac{e^{-E_\theta(x)}}{Z_\theta},\qquad
Z_\theta=\int e^{-E_\theta(u)}du.
$$

$E_\theta$ is energy, $u$ is an integration variable, and $Z_\theta$ is the partition function; use a sum for discrete states. Lower energy gives greater relative density, but the distribution exists only when the normalizer is finite. The book’s §7.2 develops a joint observation–label version and derives a classifier by conditioning. A softmax over a few labels can be easy even when normalization over all possible observations is hard.

For two discrete states with energies $0$ and $\log3$, their unnormalized weights are $1$ and $1/3$, giving probabilities $3/4$ and $1/4$. A score can remove a normalizer independent of $x$ because $\nabla_x\log p_\theta(x)=-\nabla_xE_\theta(x)$. That does not make exact likelihood or rapid sampling automatic. Maximum-likelihood energy training must also account for the parameter dependence of $Z_\theta$; simply lowering energy on training examples is insufficient (§7.3).

[![GAN discriminator and generator take alternating updates, while generation uses only a frozen generator and fresh prior noise.](/figures/deep-generative-modeling/gan.svg)](/figures/deep-generative-modeling/gan.svg)

A GAN instead uses a generator $G_\beta:\mathbb R^M\to\mathbb R^D$ and discriminator $D_\alpha:\mathbb R^D\to(0,1)$. Chapter 8, Eq. 8.9 gives the original minimax objective:

$$
\min_\beta\max_\alpha\left[
\mathbb E_{x\sim p_{\mathrm{data}}}\log D_\alpha(x)
+\mathbb E_{z\sim p(z)}\log(1-D_\alpha(G_\beta(z)))\right].
$$

$\alpha$ and $\beta$ are separate learned weights. The first row updates the discriminator using real and generated examples; the second row holds discriminator parameters fixed while differentiating **through** it to update the generator. Freezing parameters does not mean blocking its input gradients. Generation uses only the third row: draw noise, run the frozen generator. The induced distribution need not have a tractable density.

For hypothetical outputs $D(x)=0.8$ and $D(G(z))=0.2$, the one-pair objective is $2\log0.8\approx-0.4463$. Holding the discriminator fixed, a generator move that raises its fake score to $0.4$ changes the fake term from $\log0.8$ to $\log0.6$, decreasing the minimax objective. Practical generator objectives can differ; this example explains the displayed original game, not every GAN variant. Realistic individual samples do not establish coverage of all data modes.

## 7 · Scores: learn a local direction, then run a sampler

[![Score training constructs noisy inputs and conditional targets; inference repeatedly updates a state using a score and fresh noise, with a labeled drift-only trace.](/figures/deep-generative-modeling/score.svg)](/figures/deep-generative-modeling/score.svg)

The score $s(x)=\nabla_x\log p(x)$ is a vector shaped like $x$, not a probability or a scalar quality rating. Chapter 9 motivates learning it directly. The true data score is unavailable, but Gaussian corruption supplies a target (Eqs. 9.4–9.12):

$$
\widetilde x=x+\sigma\epsilon,\qquad
\nabla_{\widetilde x}\log\mathcal N(\widetilde x;x,\sigma^2I)
=-\frac{\widetilde x-x}{\sigma^2}=-\frac\epsilon\sigma,
\quad\epsilon\sim\mathcal N(0,I).
$$

Here $x$ is clean data, $\widetilde x$ the network input, and $\sigma>0$ a fixed noise standard deviation. The first row of the figure trains $s_\theta(\widetilde x)$ against this **conditional** score with squared error. At the population optimum, regression averages possible targets given $\widetilde x$, yielding the score of the Gaussian-smoothed data mixture. It does not identify the unique clean example that produced each noisy input.

For $x=2$, $\sigma=0.5$ and $\epsilon=-1$, the noisy input is $1.5$ and target score is $2$. If the network predicts $1.6$, the squared error is $0.16$ (or $0.08$ with a half prefactor). If it predicts noise $\widehat\epsilon_\theta$ instead, convert using $s_\theta=-\widehat\epsilon_\theta/\sigma$.

**Notation check on the supplied edition.** Substitution gives the teaching identity

$$
\tfrac12\|s_\theta+\epsilon/\sigma\|^2
=\frac{1}{2\sigma^2}\|\widehat\epsilon_\theta-\epsilon\|^2.
$$

The book prints $1/(2\sigma)$ in Eq. 9.13 rather than the $1/(2\sigma^2)$ obtained by this substitution. Its Eq. 9.14 and surrounding prose also alternate score and rescaled noise notation. We use the explicitly defined score here. At a single fixed $\sigma$ a positive rescaling does not change an ideal regression minimizer, but across noise levels it changes relative loss weighting.

For an illustrative fixed-density sampler, choose the consistent Langevin discretization

$$
X_{k+1}=X_k+h\,s_\theta(X_k)+\sqrt{2h}\,\xi_k,
\qquad\xi_k\sim\mathcal N(0,I).
$$

$X_k$ is the carried sampling state, $h>0$ a step size, and $\xi_k$ fresh independent noise each step. This is a teaching specialization of the Langevin idea in §9.2, not a complete multi-noise diffusion sampler. Freeze $\theta$, repeatedly evaluate the score, update $X$, and return the final state. Finite step size, imperfect scores and incomplete mixing prevent a promise of exact samples.

For an oracle standard-normal score $s(X)=-X$ and $h=0.1$, a **zero-noise drift illustration** takes $2\to1.8\to1.62\to1.458$. Setting noise to zero forever would collapse toward the mode, not reproduce the Gaussian. With a stipulated first noise draw $\xi_0=0.5$, the first update would instead be $1.8+\sqrt{0.2}(0.5)\approx2.0236$. Both traces are arithmetic demonstrations, not model results.

<details><summary>Predict: can a predicted noise vector be inserted directly as the score?</summary>

No. For this corruption convention the score is the negative noise prediction divided by the standard deviation. Omitting the sign or scale changes the update direction or magnitude.

</details>

Chapter 5 (§5.5.3) approaches diffusion through hierarchical latent variables and fixed forward corruption. Chapter 9 approaches it through scores, noise schedules and SDEs/ODEs. They are connected views, not an instruction to replace an entire diffusion sampler with the fixed-density update above. Continue with the [Lecture 2 guide](/papers/t2-generative-lecture-2) and [MIT flow-matching notes](/papers/t2-flow-matching) for time-dependent paths.

## 8 · Evidence: compare the task, not just the model label

The book contains illustrative experiments and generated examples throughout its implementation sections. This roadmap does not reproduce them or invent a cross-family leaderboard. In particular, **Table 5.1, printed p. 157, is a qualitative comparison**, not a benchmark: it lists training, likelihood, reconstruction, invertibility and bottleneck properties for the discussed diffusion models, VAEs and flows. Its own introduction calls the criteria rather arbitrary, and the following discussion acknowledges numerical issues. “Stable” is not a guarantee for a new dataset or optimizer.

Its no-bottleneck entry for diffusion describes that chapter’s formulation; it must not be generalized to latent diffusion systems that first compress images. Likewise, an ordinary flow’s equal-dimensional bijection differs from a VAE bottleneck. The course’s [FLUX.1 Kontext](/papers/flux-kontext) and [Qwen-Image](/papers/t2-paper-6) guides show why inspecting the full system matters.

A proposed comparison should fix the dataset and split, observation representation, training budget and evaluation protocol. Use held-out likelihood only when its definition and estimator are comparable; label a VAE bound as a bound. Measure sample quality and coverage separately from likelihood, and report sampling compute. To study a VAE posterior change, hold decoder capacity and optimization budget fixed and examine bound tightness as well as generated outputs. To study a score sampler, vary step size and score quality separately. These are proposed experiments, not findings from this textbook.

## 9 · Reading map and the idea to carry forward

| Question to take to the source | Supplied-book location |
| --- | --- |
| What is being modeled, learned and inferred? | §§1.3 and 2.1; mixtures and circuits in §§2.3–2.4 |
| Why does a prefix define a joint distribution? | §3.1, Eq. 3.4; causal networks and Transformers in §§3.2–3.3 |
| Where does a flow’s determinant enter? | §§4.1.1–4.1.3, Eqs. 4.7–4.20, Fig. 4.2 |
| Why is the VAE loss only a bound? | §§5.3.1–5.3.2, Eqs. 5.17 and 5.25, Fig. 5.2 |
| How do gradients cross a latent sample? | §5.3.3, Eq. 5.30, Fig. 5.3 |
| What fails in reconstruction or prior sampling? | §5.3.6; prior/posterior improvements in §5.4 |
| How does diffusion fit latent-variable modeling? | §5.5.3; Table 5.1 on printed p. 157 |
| How do a classifier and a generative model connect? | Chapter 6; §§7.2–7.3 for joint energies and training |
| Which GAN network is needed after training? | §§8.2–8.3, Eq. 8.9, Fig. 8.2 |
| How does denoising produce a score target? | §§9.2.1–9.2.4, Eqs. 9.4–9.14; §9.3 for SDEs/ODEs |

Use Chapters 10 and 11 as subsequent routes into neural compression and applications after these mechanisms are secure; those chapters are outside this focused walkthrough.

**The idea to carry forward:** draw the training path and generation path separately, then write the probability operation beside each arrow. Ask what is normalized, integrated out, approximated or differentiated. That habit explains why an encoder can disappear at generation, why a flow needs a determinant, and why a denoiser needs a carefully defined sampler.
