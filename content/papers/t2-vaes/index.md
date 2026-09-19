---
title: "Auto-Encoding Variational Bayes · learn an inference network"
shortTitle: "VAEs"
topic: "topic-2"
order: 6
materialKind: "extra"
description: "Follow the original VAE through probabilistic encoding, reparameterized gradients, the ELBO, joint training and prior-based generation."
status: "ready"
source: "https://arxiv.org/abs/1312.6114v11"
pdf: "/papers/auto-encoding-variational-bayes.pdf"
---
## The puzzle: how can a generator learn from an unseen cause?

Imagine a decoder that turns a few hidden numbers into a distribution over handwritten digits. Training supplies the digits, but not the hidden numbers that produced them. Summing over every possible latent explanation is expensive. A conventional autoencoder supplies one code for each image, but reconstruction alone does not specify a probabilistic model or a distribution from which useful new codes can be sampled.

**A VAE learns an approximate inference network alongside a probabilistic generator. Reparameterization lets reconstruction gradients pass through a latent sample, while a variational objective connects the training signal to data likelihood.**

This guide follows Kingma and Welling’s **14-page Auto-Encoding Variational Bayes, arXiv v11, December 10, 2022**, supplied as `topic 2/extra reading material/VAEs.pdf`. The work was first submitted in 2013; the supplied revision is not a new 2022 experimental study. [Read the exact course PDF](/papers/auto-encoding-variational-bayes.pdf) or the [versioned arXiv record](https://arxiv.org/abs/1312.6114v11). The record describes v11 as an abstract typo correction.

Allow 25 minutes. You need expectations, natural logarithms and the chain rule. Seven local diagrams are teaching illustrations; the experiment image reproduces source PDF page 7. Open figures to enlarge them. All small numerical examples below are constructed; no trained VAE runs here.

## 1 · Architecture: one model generates, another infers

[![The encoder maps a D-dimensional observation to J-dimensional mean and log-variance vectors; noise and these vectors produce z; the decoder outputs observation-distribution parameters.](/figures/vaes/architecture.svg)](/figures/vaes/architecture.svg)

Follow **A → B → C → D → E**. A is the observed vector $x\in\mathbb R^D$ (or a binary vector). B is the encoder $q_\phi(z\mid x)$, with learned parameters $\phi$. It outputs two vectors in $\mathbb R^J$: mean $\mu_\phi(x)$ and log variance $\ell_\phi(x)=\log\sigma_\phi^2(x)$. Here $J$ is latent dimension and $D$ is observation dimension. C combines these outputs with independent noise to form a latent vector $z\in\mathbb R^J$. D is the decoder $p_\theta(x\mid z)$, with a different learned parameter set $\theta$. E evaluates the observed x under the decoder distribution and combines that score with a KL penalty.

The generative model is

$$
p_\theta(x,z)=p(z)p_\theta(x\mid z),\qquad p(z)=\mathcal N(0,I_J).
$$

$I_J$ is the $J\times J$ identity matrix. The standard Gaussian prior has **no learned parameters** in the paper’s VAE example (§3), although the general framework allows parameterized priors. The encoder is a training and inference tool; it is not a factor in this generative joint distribution.

Appendix C specifies single-hidden-layer fully connected networks with tanh hidden activations. A Gaussian network has separate linear heads for mean and log variance. The binary decoder has sigmoid probabilities, one per observed coordinate. The Frey Face Gaussian decoder constrains its means to $(0,1)$ (§5). Experiment widths vary by evaluation: 500 hidden units for the MNIST lower-bound comparison, 200 for Frey Face, and 100 for the separate three-latent-variable likelihood comparison. The diagram describes the common functional structure, not a single universal network width.

**Design reasoning, as interpretation:** replace an expensive inference optimization for each new image with a shared function that proposes its plausible causes. Then learn that function and the generator through one objective. Sharing encoder weights across examples is often called *amortized inference*. It saves repeated inference work, but a restricted shared encoder can approximate the true posterior poorly.

## 2 · Representation: a distribution is more than a code

[![A standard Gaussian prior and an example posterior with mean 1 and standard deviation 0.5 appear as density curves; noise values minus one, zero and one map to latent values 0.5, 1 and 1.5.](/figures/vaes/representation.svg)](/figures/vaes/representation.svg)

The curves are exact densities for a teaching example, not learned latents. The encoder family in paper Eq. 9 is

$$
q_\phi(z\mid x)=\mathcal N\!\left(\mu_\phi(x),\operatorname{diag}(\sigma_\phi^2(x))\right).
$$

The operator $\operatorname{diag}$ places the J coordinate variances on a diagonal matrix. The source writes a vector variance followed by $I$; we use explicit diagonal notation. This distribution makes latent coordinates independent **conditional on x within the approximation**. It does not assert that the true posterior is diagonal, or that the aggregate mixture over all images has independent coordinates.

In the scalar picture, $\mu=1$ and $\sigma=0.5$. The same image can produce many z values around 1. Its mean is one summary; its spread describes uncertainty in the approximate latent explanation. Noise $\epsilon=1$ gives $z=1.5$, while $\epsilon=-1$ gives $z=0.5$. Neither value is an additional stored training example.

The decoder also outputs a **distribution**, not inherently a completed image. Bernoulli probabilities can be displayed as gray intensities or sampled to produce binary pixels. A Gaussian mean can be displayed or used as the center of a Gaussian observation draw. Displaying a mean and sampling an observation are different operations.

## 3 · The ELBO: fit observations without solving the integral

The marginal likelihood sums over hidden explanations:

$$
p_\theta(x)=\int p(z)p_\theta(x\mid z)\,dz.
$$

This integral is the obstacle before training: evaluating the decoder for one z is easy, but integrating a nonlinear decoder over every z generally is not. The encoder supplies an approximate posterior with which to build a tractable objective.

[![The ELBO equals expected reconstruction log likelihood minus prior KL; log evidence lies above it by posterior KL, with prior KL and posterior KL explicitly separated.](/figures/vaes/objective.svg)](/figures/vaes/objective.svg)

Read the top row as a score assembled at E, and the middle row as its relationship to the unavailable exact evidence. Paper Eqs. 1–3 give

$$
\mathcal L(\theta,\phi;x)
=\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-\operatorname{KL}(q_\phi(z\mid x)\Vert p(z)),
$$

$$
\log p_\theta(x)-\mathcal L(\theta,\phi;x)
=\operatorname{KL}(q_\phi(z\mid x)\Vert p_\theta(z\mid x))\geq0.
$$

$\mathcal L$ is the evidence lower bound, or ELBO, for one x. $\mathbb E$ means an average under the encoder distribution. For distributions q and p, $\operatorname{KL}(q\Vert p)=\mathbb E_q[\log q-\log p]$, assuming the relevant support and integrability conditions. The exact posterior $p_\theta(z\mid x)$ differs from the prior $p(z)$. **The prior KL is a computable term in the objective; the posterior KL is the usually uncomputable gap to evidence.**

The reconstruction term rewards explanations from which D assigns high likelihood to x. The prior term penalizes moving each encoder distribution away from the sampling prior. Maximizing the ELBO balances both. Equivalently, minimize negative ELBO: reconstruction negative log likelihood **plus** prior KL. In this original objective their relative coefficient is fixed by the derivation; changing it arbitrarily changes the objective.

For a teaching example, suppose the expected reconstruction log likelihood is −2 and the prior KL is 0.4. Then the ELBO is −2.4. If the posterior KL is 0.3, the exact log evidence is −2.1. These numbers illustrate the identity; we generally cannot measure the final gap directly for the neural model.

To see why the identity holds, insert Bayes’ rule $\log p_\theta(z\mid x)=\log p_\theta(x,z)-\log p_\theta(x)$ into the posterior KL and rearrange. Nonnegativity then gives the bound. Equality requires the approximate and exact posteriors to match almost everywhere, not merely a small prior KL.

<details><summary>Predict: if the encoder equals the prior, must the bound equal the evidence?</summary>

No. That makes the prior KL zero. The bound is tight only if the encoder equals the posterior conditioned on this observation. A zero prior KL can instead mean the latent carries no information about x.

</details>

## 4 · Reparameterization: move randomness before the learned transform

Sampling from a distribution whose parameters change can make naive differentiation awkward. The paper’s move is to draw noise from a fixed distribution and transform that noise differentiably (Eqs. 4–5):

$$
\epsilon\sim\mathcal N(0,I_J),\qquad
z=\mu_\phi(x)+\exp\!\left(\tfrac12\ell_\phi(x)\right)\odot\epsilon.
$$

$\odot$ is elementwise multiplication; the exponential and division by two act coordinatewise. Since $\ell=\log\sigma^2$, $\exp(\ell/2)=\sigma$, the **standard deviation**. Using $\exp(\ell)$ here would incorrectly multiply noise by the variance.

[![A fixed epsilon draw enters the differentiable mean-plus-standard-deviation transform; reconstruction gradients return through z to both encoder heads, with dz/dmu equal to one and dz/dlog-variance equal to sigma epsilon over two.](/figures/vaes/reparameterization.svg)](/figures/vaes/reparameterization.svg)

The upper row is a forward pass through C. The lower row shows local derivatives during a backward pass. Holding the sampled epsilon fixed during that derivative is legitimate because its distribution does not depend on $\phi$. New noise is drawn for later stochastic estimates.

For a differentiable scalar function f, the pathwise identity is

$$
\nabla_\phi\mathbb E_{q_\phi(z\mid x)}f(z)
=\mathbb E_{\epsilon}\!\left[\nabla_z f(z)\,\nabla_\phi g_\phi(\epsilon,x)\right],
$$

where $g_\phi$ is the transform in C. This expression assumes f has no additional direct dependence on $\phi$; otherwise add that direct derivative. It also requires conditions allowing differentiation under the expectation. The reconstruction term uses $f(z)=\log p_\theta(x\mid z)$ while treating $\theta$ as fixed for the encoder derivative.

For one coordinate, $\partial z/\partial\mu=1$ and $\partial z/\partial\ell=\sigma\epsilon/2$. Thus the decoder’s input gradient can reach both encoder heads. The trick preserves the sampling distribution; it does not make z deterministic across draws, set the noise to zero, or differentiate the random-number generator.

Sections 2.3–2.4 discuss broader transformations, while §3 uses a Gaussian example. This elementary pathwise method does not directly solve hard discrete latent sampling. Diagonal Gaussian q is a modeling choice, not a universal restriction of variational inference. Reparameterization typically reduces variance relative to the naive estimator discussed in §2.2; it does not guarantee zero-variance gradients.

## 5 · Training, reconstruction and generation

[![Training uses observation, encoder, reparameterized z, decoder and both losses; reconstruction uses the encoder with fixed weights; unconditional generation starts from the prior and needs only the decoder.](/figures/vaes/paths.svg)](/figures/vaes/paths.svg)

Read the rows as three different tasks. During **training**, both encoder and decoder weights change. During **reconstruction**, an observed image supplies the encoder input but weights stay fixed. During **unconditional generation**, z is drawn from the prior; there is no input image and no need to run the encoder.

For the diagonal Gaussian and standard normal prior, the KL at E is analytic (Appendix B):

$$
K(x)=\tfrac12\sum_{j=1}^J
\left(\mu_j^2+\sigma_j^2-1-\log\sigma_j^2\right).
$$

$K(x)$ is the prior KL for x; $\mu_j,\sigma_j$ are the encoder’s jth mean and standard deviation. The formula penalizes both shifting the mean and changing the spread. A standard normal encoder has zero KL; concentrating $\sigma_j$ toward zero drives the term upward. The source prints **negative** KL in Eq. 10 because it maximizes the ELBO; K above is positive KL, subtracted in that objective.

Only the reconstruction expectation needs sampling in paper Eq. 7:

$$
\widehat{\mathcal L}(x)
=-K(x)+\frac1L\sum_{l=1}^L\log p_\theta(x\mid z^{(l)}).
$$

$L$ is the number of independent noise draws per observation, not latent dimension. $z^{(l)}$ is produced at C for draw l. The hat marks a Monte Carlo estimate. The expected ELBO is a lower bound; **an individual noisy estimate need not lie below the exact log evidence**. Analytically integrating the KL removes sampling variance from that term, not from the entire objective.

For N observations and a uniformly sampled minibatch of size M, paper Eq. 8 estimates the dataset-sum ELBO by $(N/M)\sum_{i=1}^M\widehat{\mathcal L}(x^{(i)})$. The factor corrects the batch sum to dataset scale. An implementation optimizing a batch average uses a different overall scaling; a weight-prior penalty must be scaled consistently if included.

[![The AEVB loop repeatedly replaces the minibatch and epsilon, computes the sampled objective, and jointly updates encoder and decoder weights while retaining optimizer state.](/figures/vaes/loop.svg)](/figures/vaes/loop.svg)

This schematic expands Algorithm 1. Initialize $\theta,\phi$; draw a minibatch and noise; compute encoder statistics, z and decoder log likelihood; add the exact negative KL; ascend the objective in **both** parameter sets; repeat. Carry weights and optimizer state forward, replacing examples and noise. The paper uses $M=100,L=1$, Adagrad and a small weight decay corresponding to a Gaussian prior on generator parameters, making the reported training criterion an approximate MAP objective. This is distinct from Appendix F’s full variational posterior over weights, which is derived but not used in these experiments.

One sample per example can be useful because a minibatch averages many noisy contributions. It does not mean the latent distribution has collapsed to one point. Unlike a diffusion sampler, generation here has no iterative denoising loop: draw z, evaluate the decoder, then draw x if actual observation samples are wanted.

<details><summary>Predict: should a new random image be generated by feeding random pixels into the encoder?</summary>

No. In the model’s ancestral sampling procedure, draw z from the standard Gaussian prior and use the decoder. The encoder approximates latent inference for an observed x.

</details>

## 6 · Worked example: trace a sample and its gradient

[![For binary x equal to one, the encoder gives mean 0.5 and standard deviation 0.5; epsilon equal to one produces z equal to one, a sigmoid decoder gives probability 0.7311, and reconstruction loss plus KL gives 0.7564.](/figures/vaes/worked.svg)](/figures/vaes/worked.svg)

This scalar toy uses the same boxes as the architecture but does not reproduce the paper’s MLP. Choose **A:** $x=1$; **B:** $\mu=0.5,\sigma=0.5$, so $\ell=\log0.25$; **C:** $\epsilon=1$, giving $z=1$. Let **D** be a Bernoulli decoder with probability $y=s(wz+b)$, where $s(a)=1/(1+e^{-a})$, $w=1$ and $b=0$. Then $y=s(1)\approx0.7311$.

Appendix C.1’s Bernoulli likelihood becomes $\log p_\theta(x\mid z)=x\log y+(1-x)\log(1-y)$. At $x=1$ the reconstruction negative log likelihood is $R=-\log y\approx0.3133$. In multiple dimensions, the factorized decoder sums this expression over the D coordinates; probabilities and likelihood are not interchangeable.

At **E**, the KL is

$$
K=\tfrac12(0.5^2+0.5^2-1-\log0.25)\approx0.4431.
$$

The sampled negative ELBO is $C=R+K\approx0.7564$ nats; the sampled ELBO is approximately −0.7564. Natural logs give nats. This is one noise realization, not the exact expected ELBO or exact evidence.

Now trace the backward arrows. The sigmoid cross-entropy derivative gives $\partial R/\partial z=(y-x)w\approx-0.2689$. The KL derivatives are $\partial K/\partial\mu=\mu$ and $\partial K/\partial\ell=(e^\ell-1)/2$. Combining them with C’s local derivatives gives

$$
\frac{\partial C}{\partial\mu}=(y-x)w+\mu\approx0.2311,
$$

$$
\frac{\partial C}{\partial\ell}
=(y-x)w\frac{\sigma\epsilon}{2}+\frac{e^\ell-1}{2}
\approx-0.4422.
$$

Decoder gradients are $\partial C/\partial w=(y-x)z\approx-0.2689$ and $\partial C/\partial b=y-x\approx-0.2689$. Thus the same objective trains both networks. At these settings, reconstruction alone would push the encoder mean up, but KL’s pull toward zero is stronger, so the combined mean gradient is positive.

For an illustrative gradient descent step of size $\eta=0.1$, treating $\mu,\ell$ as direct scalar parameters gives $\mu_{\rm new}\approx0.4769$, $\ell_{\rm new}\approx-1.3421$, and $w_{\rm new}\approx1.0269$, $b_{\rm new}\approx0.0269$. In an actual encoder, these output derivatives propagate through its shared neural weights; the optimizer does not store a free mean and variance for every image. Do not interpret this one stochastic step as a guaranteed improvement in exact data likelihood.

<details><summary>Predict: why is multiplying epsilon by the variance wrong here?</summary>

It would give z = 0.5 + 0.25 × 1 = 0.75, instead of 1. The desired posterior variance is 0.25, so its sampling amplitude must be the standard deviation 0.5.

</details>

## 7 · Evidence: what was actually compared?

[![Original PDF page 7 shows Figure 2 comparing AEVB and wake-sleep estimated train and test ELBOs across MNIST and Frey Face latent dimensions, with experimental settings below.](/figures/vaes/source-page-7.png)](/figures/vaes/source-page-7.png)

This is the **supplied source page**, not a redrawn or generated result. In Figure 2, red is AEVB and green is wake-sleep; solid curves are training and dashed curves are test. The horizontal axis counts evaluated training examples on a logarithmic scale. The vertical axis is the estimated average ELBO per observation: higher is better. MNIST values are negative; Frey Face uses continuous densities and can have positive log densities, so their numerical scales are not directly comparable.

| Evidence in the supplied v11 | Setting | What it supports |
| --- | --- | --- |
| Figure 2, MNIST | 500 hidden units in each network; latent dimensions 3, 5, 10, 20, 200 | AEVB reaches better displayed lower bounds than wake-sleep and improves sooner in evaluated-example count. |
| Figure 2, Frey Face | 200 hidden units; Gaussian decoder; latent dimensions 2, 5, 10, 20 | The same training-method advantage appears in these plotted continuous-image experiments. |
| Figure 3, MNIST | 100 hidden units, 3 latent variables; training sets of 1,000 and 50,000 | Separate estimated marginal-likelihood comparison with wake-sleep and Monte Carlo EM. |
| Figures 4–5, Appendix A | Two-dimensional manifold displays and MNIST samples at several latent dimensions | Qualitative evidence of learned generation and latent organization. |

The Figure 2 caption reports estimator variance below 1 and omits it from the plots. This is not a confidence interval across independent training seeds. It reports roughly 20–40 minutes per million evaluated examples on its Intel Xeon setup; the plotted x-axis itself is **not elapsed time**. We do not digitize exact endpoint scores or claim a portable speedup from those curves.

The authors report that extra latent dimensions did not produce additional overfitting in these experiments and attribute this to the regularizing objective. That observation is not a theorem that VAEs cannot overfit. A better lower bound also does not by itself isolate better generative likelihood: both the model and the tightness of its variational approximation can change.

Figure 3 instead uses a separate marginal-likelihood estimator described in Appendices D–E. It involves posterior HMC samples and a fitted density estimator. The paper explicitly restricts this evaluation to very low-dimensional latent spaces because estimates become unreliable at higher dimension. Its MCMC evaluation does not mean AEVB needs MCMC for each training example or for unconditional generation. These estimated likelihoods are also not exact evidence values or modern perceptual-quality metrics.

For Figure 4 the authors map a uniform grid through the Gaussian inverse CDF before decoding, respecting the Gaussian prior’s mass allocation. A smooth two-dimensional grid is useful for inspecting organization but does not establish disentanglement, coverage or generalization.

**Source wording note:** the first paragraph of §5 swaps the parenthetical encoder/decoder labels. This guide follows §§2–3 and Appendix C consistently: q is the recognition encoder and p(x|z) is the generative decoder. The source’s terminology slip does not change the computational paths.

## 8 · Limitations and useful tests

A diagonal encoder cannot express every posterior dependency. A shared encoder can also fail to find the best approximation for each observation. These are reasons the posterior KL gap may remain positive even after useful training. Better reconstruction alone does not diagnose either problem.

The prior KL creates a real tradeoff: informative codes can help reconstruction but cost divergence from the prior. If q becomes the prior for every x, z carries no information about the input; a decoder may then fail to use the latent meaningfully. This is a mechanism-level possibility, not a collapse result measured in this paper. Likewise, the basic factorized observation model limits conditional detail even though marginalizing z can induce dependencies between pixels.

Useful follow-ups would hold decoder architecture, data split and compute fixed while varying the posterior family, then separately estimate inference quality and data likelihood. To study gradient variance, compare repeated independent noise draws at identical network weights rather than confounding the estimator with training progress. To test the role of the KL term, report reconstruction and prior-sampled outputs alongside the objective, while recognizing that deleting or reweighting KL changes the original bound. These are proposed tests, not reported ablations.

## 9 · The idea to carry forward

**Learn how to infer hidden causes, then use a differentiable random sample to teach both inference and generation.** Keep three objects distinct: prior for generating, approximate posterior for encoding, and exact posterior for interpreting the variational gap.

The [generative modeling roadmap](/papers/t2-deep-generative-modeling) places this method among other density models. The [GAN guide](/papers/t2-gans) learns a discriminator instead of an inference network. The [DDPM guide](/papers/t2-ddpms) uses a multistep latent chain with a fixed forward process. The [neural compression guide](/papers/t2-neural-compression) explains why a useful latent representation still needs a coding protocol before it becomes a real bitstream.

## Source reading map

| Guide question | Supplied paper location |
| --- | --- |
| Which distribution generates, and which infers? | Figure 1; §§2.1 and 3; Appendix C |
| Why is the objective a lower bound? | §2.2, Eqs. 1–3 |
| How does noise become differentiable? | §§2.3–2.4, Eqs. 4–7 |
| What changes during training? | Algorithm 1 and Eq. 8, PDF p. 4 |
| What is the Gaussian KL and its sign? | Eq. 10; Appendix B, PDF pp. 10–11 |
| What do networks actually output? | Appendix C, Eqs. 11–12 |
| What do the experiments establish? | §5, Figures 2–3, PDF pp. 6–8 |
| How are samples and marginal likelihoods evaluated? | Appendices A, D–E, PDF pp. 9–12 |
| Are weights themselves variational random variables? | Appendix F derives that extension; main experiments do not use it |

Continue with the guide’s eight study cards below, or use the [Topic 2 printable cards](/study/print?topic=topic-2).
