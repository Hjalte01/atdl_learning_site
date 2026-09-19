---
title: "An Introduction to Flow Matching and Diffusion Models"
shortTitle: "Flow Matching · MIT notes"
topic: "topic-2"
order: 1
materialKind: "extra"
description: "Why regressing against simple conditional velocities learns a generative flow: probability paths, training, sampling, scores and guidance."
status: "ready"
pdf: "/papers/flow-matching.pdf"
---
## The puzzle: how do you teach a direction without knowing the destination?

A generator starts with random noise and ends with a plausible image. You could train by simulating its entire trajectory, judging the final image, and differentiating through every step. That makes each training update expensive. **Flow matching instead manufactures a local velocity target at a randomly chosen time.** The surprising part is why learning these local targets produces the right distribution of endpoints.

The catalog calls this extra reading *Flow Matching*. Its actual source is Peter Holderrieth and Ezra Erives’s **84-page MIT 6.S184 notes, *An Introduction to Flow Matching and Diffusion Models*, 2026**. The supplied PDF has no numbered revision; its export metadata is March 18, 2026. [Read the exact course copy](/papers/flow-matching.pdf). This is a focused guide to §§2–5, with a bridge into §§6–7, not an exhaustive treatment of all appendices or the original flow-matching research paper.

Allow 30 minutes. You need vectors, derivatives, expectation and conditional probability; Appendix A of the notes refreshes probability. All seven diagrams and numerical values here are teaching constructions, not trained-model results. Open a diagram to enlarge it. Our time convention follows the notes: **0 is noise; 1 is data**. Some diffusion papers reverse it.

## 1 · Choose a path of distributions

[![Conditional Gaussian clouds collapse toward one endpoint while their mixture evolves from noise into a distribution of data.](/figures/flow-matching/paths.svg)](/figures/flow-matching/paths.svg)

Read each row left to right. The upper row fixes a clean example $z\in\mathbb R^d$. The lower row averages over all possible $z$ drawn from the data distribution $p_{\mathrm{data}}$. The diagram is schematic: its boxes represent distributions, not three images from a trained model.

For independent noise $\epsilon\sim\mathcal N(0,I_d)$ and differentiable scalar schedules $\alpha_t,\beta_t$, construct (notes Eqs. 15–16):

$$
x_t=\alpha_t z+\beta_t\epsilon,\qquad
p_t(x\mid z)=\mathcal N(x;\alpha_t z,\beta_t^2 I_d),\qquad
p_t(x)=\int p_t(x\mid z)p_{\mathrm{data}}(z)\,dz.
$$

Here $x_t$ is a noisy data-shaped vector; $I_d$ is the identity covariance. Choose $\alpha_0=0,\beta_0=1$ and $\alpha_1=1,\beta_1=0$. Thus $p_0$ is standard Gaussian and the endpoint $p_1$ is the data distribution. At $t=1$, the conditional Gaussian is understood as a point mass at $z$, not an ordinary positive-variance density.

The simplest choice is $\alpha_t=t$, $\beta_t=1-t$, the notes’ **Gaussian CondOT path**. At $t=0.25$, its conditional mean is $0.25z$ and variance per coordinate is $0.75^2=0.5625$. The coefficients multiply values, not variances. The marginal distribution is a mixture and generally is not Gaussian.

A design interpretation: specifying easy distributions first gives us a tractable training recipe. We can sample a state at any time directly, without simulating how it got there. This is a reconstruction of the method’s logic, not a claim about the authors’ private thought process.

## 2 · Architecture: where the training target comes from

[![Training takes data, noise and time, mixes a state, predicts its velocity, compares with data minus noise and updates network weights.](/figures/flow-matching/training.svg)](/figures/flow-matching/training.svg)

**A** samples a clean vector, noise and time. **B** makes the noisy input. **C** is a trainable network $u_\theta(x_t,t)\in\mathbb R^d$ with parameters $\theta$. Its output has the same shape as the modeled state. The target branch uses $z$ and $\epsilon$, but the unconditional network sees only $x_t,t$. The figure is a functional architecture; flow matching does not prescribe a layer count or hidden width.

The target should describe how a fixed noise–data pair moves. Differentiate its interpolation:

$$
v_t=\dot\alpha_t z+\dot\beta_t\epsilon.
$$

A dot means a derivative with respect to time. To write this velocity as a function of a current position $x$ and endpoint $z$, substitute $\epsilon=(x-\alpha_tz)/\beta_t$ (notes Eq. 20):

$$
u_t(x\mid z)=\left(\dot\alpha_t-\frac{\dot\beta_t}{\beta_t}\alpha_t\right)z
+\frac{\dot\beta_t}{\beta_t}x.
$$

This equation lives in the **target branch**, not the optimizer. It applies where $\beta_t\ne0$. For the linear schedule it becomes $(z-x)/(1-t)$, and at the sampled state $x_t=tz+(1-t)\epsilon$ it simplifies to **$z-\epsilon$**. Computing that direct target avoids dividing by $1-t$. It is a velocity, not a clean-image prediction and not just the added noise.

Algorithm 3 and Eq. 31 minimize:

$$
\mathcal L_{\mathrm{CFM}}(\theta)=
\mathbb E_{t,z,\epsilon}\left[
\left\|u_\theta(\alpha_tz+\beta_t\epsilon,t)
-(\dot\alpha_tz+\dot\beta_t\epsilon)\right\|^2\right],
\qquad t\sim\mathrm{Unif}[0,1].
$$

The expectation averages random times, data examples and independent Gaussian noise. The norm sums squared errors over coordinates; a mean convention rescales it. Backpropagation changes $\theta$ at the final box. Neither $z$ nor the fixed schedules are learned by this objective. **Simulation-free describes training; generation still solves a differential equation.**

## 3 · Why many conflicting targets teach one useful field

At a particular noisy state $x$, multiple clean endpoints could have produced it. Their target velocities can disagree. Squared-error regression learns their conditional mean (notes Eq. 18):

$$
u_t(x)=\mathbb E[u_t(x\mid z)\mid x_t=x]
=\int u_t(x\mid z)\frac{p_t(x\mid z)p_{\mathrm{data}}(z)}{p_t(x)}\,dz.
$$

The weighting factor is the posterior probability density of an endpoint given this noisy state. It is **not** a uniform average of all endpoint directions. This marginal field is what the network should learn; computing its integral explicitly would be intractable for realistic data.

[![Two possible endpoints have conditional velocities minus three and plus one; posterior weights give a marginal velocity of about 0.5232.](/figures/flow-matching/posterior.svg)](/figures/flow-matching/posterior.svg)

For this one-dimensional toy, let $z=-1$ or $+1$ with equal prior probability, $t=0.5$, and $x=0.5$. Their conditional Gaussian means are $-0.5,+0.5$, both with standard deviation $0.5$. The likelihood ratio for the negative versus positive endpoint is $e^{-2}$, so their posterior probabilities are approximately $0.1192,0.8808$. Their velocities are $(-1-0.5)/0.5=-3$ and $(1-0.5)/0.5=1$. The marginal velocity is $0.1192(-3)+0.8808(1)\approx0.5232$.

Read the figure as two alternative explanations combined into one prediction; the arrows are not two sequential moves. These probabilities are calculated for the stipulated toy, not measured on image data.

### Why the loss is enough

Let $V$ denote the sampled conditional target, and $m(x,t)=\mathbb E[V\mid x,t]$. The squared-error decomposition gives this teaching form of Theorem 12:

$$
\mathbb E\|u_\theta-V\|^2
=\mathbb E\|u_\theta-m\|^2+
\mathbb E\|V-m\|^2.
$$

The cross term vanishes by conditional expectation. The second term is independent of $\theta$, so conditional and marginal losses have the same parameter gradient. Their **values need not match**. Irreducible disagreement between conditional targets can leave a positive training loss even when the marginal prediction is perfect. The equality assumes finite moments and the fixed sampling recipe; it is not a guarantee that a finite network or optimizer reaches the optimum.

Why does that mean field transport the right distribution? The continuity equation (Eq. 23) says:

$$
\partial_t p_t(x)=-\nabla\!\cdot\!\big(p_t(x)u_t(x)\big).
$$

The left side is density change; the right side is negative net outward probability flux. Averaging the conditional fluxes gives exactly the marginal flux, which is the mechanism behind Theorem 9. This is a distribution-level statement under the required regularity assumptions, not a claim that a marginal trajectory follows its original noise–data straight line. Collapsing paths at singular endpoints require limiting care.

<details><summary>Predict: can the same input have two different training targets?</summary><p>Yes. The network is not told the clean endpoint. Squared-error regression averages plausible conditional velocities using their posterior weights. It can learn the correct marginal direction despite variation among targets.</p></details>

## 4 · Follow one example through every training box

[![A two-dimensional example mixes endpoint two minus one and noise minus two one at time one quarter, then computes a squared prediction error of two.](/figures/flow-matching/numbers.svg)](/figures/flow-matching/numbers.svg)

Use $z=(2,-1)$, $\epsilon=(-2,1)$, $t=0.25$. In box B:

$$
x_t=0.25(2,-1)+0.75(-2,1)=(-1,0.5),\qquad v=z-\epsilon=(4,-2).
$$

Stipulate that box C predicts $(3,-1)$. Its error is $(-1,1)$, so the squared-error sum is $2$ (or coordinate mean $1$). The derivative of the summed loss with respect to the prediction is $2(u_\theta-v)=(-2,2)$; backpropagation carries this through the network to its parameters.

Read the lower row as the loss calculation attached to the same example, not another sample. None of these values is a reported training result. A small local regression error does not directly give an image-quality metric.

## 5 · Sampling: now the state moves and the weights stay fixed

[![Inference starts from Gaussian noise and repeatedly carries an Euler-updated state forward; a constant-velocity toy reaches its endpoint in four steps.](/figures/flow-matching/sampling.svg)](/figures/flow-matching/sampling.svg)

The inference row has no supplied clean $z$. Freeze $\theta$, sample $X_0\sim\mathcal N(0,I_d)$, and integrate $dX_t/dt=u_\theta(X_t,t)$. Euler’s method (Algorithm 1) approximates the update with step size $h>0$:

$$
X_{t+h}=X_t+h\,u_\theta(X_t,t).
$$

Here $X_t$ is the carried sample, $u_\theta$ its predicted velocity, and $h$ elapsed time. This equation operates at the **state-update arrow**. Sampling a new unrelated state each iteration would destroy the trajectory.

The lower row reuses the previous pair only as an **oracle illustration**: hold velocity $(4,-2)$ constant and take $h=0.25$. The states are $(-2,1)\to(-1,0.5)\to(0,0)\to(1,-0.5)\to(2,-1)$. Euler is exact for this constant field. A learned marginal field generally curves, changes with time, and does not know the endpoint in advance. Four steps here are not a recommendation for image generation.

There are two distinct errors: a learned field can differ from the target field, and a numerical solver can approximate the learned ODE poorly. Increasing the number of steps addresses the latter; it does not repair arbitrary training error. The name CondOT refers to conditional transport; independent noise–data pairing does not establish globally optimal marginal transport or a one-step generator.

<details><summary>Predict: is an ODE generator deterministic?</summary><p>Given its starting noise and fixed deterministic solver, yes. Its output distribution remains random because the starting noise is sampled. An SDE can introduce additional randomness during the trajectory.</p></details>

## 6 · Scores connect flow matching to diffusion

A score $s_t(x)=\nabla_x\log p_t(x)$ points toward increasing log density; it is not a scalar quality score. For a conditional Gaussian (Eq. 40):

$$
s_t(x\mid z)=-\frac{x-\alpha_tz}{\beta_t^2}
=-\frac{\epsilon}{\beta_t}\quad\text{at }x=x_t.
$$

Thus another tractable target exists for a score-predicting network. Conditional score regression learns the marginal score by the same posterior averaging principle (§4.3). The denominator becomes singular as $\beta_t\to0$; schedules, weighting and endpoint handling matter.

Proposition 1 connects marginal velocity and score:

$$
u_t(x)=a_t s_t(x)+b_t x,\qquad
a_t=\beta_t^2\frac{\dot\alpha_t}{\alpha_t}-\dot\beta_t\beta_t,
\qquad b_t=\frac{\dot\alpha_t}{\alpha_t}.
$$

$a_t,b_t$ are scalar schedule-derived coefficients, not network parameters. This conversion operates on the network output; it requires valid denominators and $a_t\ne0$ to recover the score. For the linear path and $0<t<1$, $a_t=(1-t)/t$, $b_t=1/t$. At $t=0.5$, $u=s+2x$. In the posterior toy, $s\approx-0.4768$, $x=0.5$, hence $u\approx0.5232$. Do not substitute $t=0$ into the divided formula.

### Add noise without changing the intended marginals

[![The ODE and score-corrected SDE share ideal time marginals but differ in trajectories; a worked Euler-Maruyama increment is 0.04.](/figures/flow-matching/sde.svg)](/figures/flow-matching/sde.svg)

Theorem 17 adds a paired score drift and Brownian noise to the flow:

$$
dX_t=\left[u_t(X_t)+\frac{\sigma_t^2}{2}s_t(X_t)\right]dt+\sigma_t\,dW_t.
$$

$W_t$ is standard Brownian motion; $\sigma_t\ge0$ is a chosen time-dependent noise amplitude. This uses the notes’ **noise-to-data** time direction. The score correction counteracts the spreading caused by the noise in the Fokker–Planck equation. Adding noise alone would generally change the marginal distribution.

For a numerical step, the noise increment is $\sigma_t\sqrt h\,\xi$ with an independent standard normal vector $\xi$. In the figure’s scalar example, $u=0.5,s=-2,\sigma=1,h=0.04,\xi=0.3$: drift contributes $-0.02$, noise contributes $0.06$, and the net increment is $0.04$. These are stipulated local values. The diagram’s equal-marginal claim is an exact-field, continuous-time idealization; trained models and finite steps need not preserve it exactly. Setting $\sigma_t=0$ recovers the ODE.

## 7 · Conditioning is not the same as the conditional training target

The clean endpoint $z$ used to construct a target and a prompt $y$ serve different roles. For text-conditioned training, sample paired data $(z,y)$ and pass $y$ into $u_\theta(x_t,t,y)$. The target remains $z-\epsilon$ for the linear path. At inference, $y$ stays available while $z$ is unknown (§5.1).

Classifier-free guidance trains a null-prompt branch by sometimes replacing $y$ with $\varnothing$. At sampling time, combine two predictions from the same trained network (§5.2):

$$
\widetilde u_\theta=(1-w)u_\theta(x,t,\varnothing)+w u_\theta(x,t,y).
$$

$w$ is the guidance scale: $w=0$ gives the unconditional prediction, $w=1$ the ordinary conditional prediction, and $w>1$ extrapolates their difference. If the two scalar velocities are $1$ and $3$, $w=2$ yields $5$. This combination enters the sampler’s velocity box; it does not update weights. No separate classifier is required, but ordinary CFG requires both branch predictions. Larger guidance is a heuristic tradeoff, not a guarantee of exact conditional sampling or steadily improving quality.

## 8 · From vectors to image generators—and beyond

[![A staged latent generator first trains an autoencoder, then trains a velocity network on latent endpoints, and finally samples latent noise before decoding.](/figures/flow-matching/latent.svg)](/figures/flow-matching/latent.svg)

Read each row as a different stage. Section 6 discusses U-Nets and diffusion transformers as choices for the velocity/score network; the regression objective does not select one automatically. In a staged latent pipeline, an encoder turns an image $I$ into a smaller representation $z$, the flow learns this latent distribution, and a decoder maps a generated latent back to image space. The figure is a schematic staged recipe, not an assertion that every model freezes or trains every module identically.

A transformer can patchify a state of shape $C\times H\times W$. With patch side $P$, there are $N=(H/P)(W/P)$ tokens, each with $CP^2$ entries before projection to hidden width $D$. Time and prompt embeddings condition the processing; the output is projected and unpatchified to a velocity of shape $C\times H\times W$ (§6.1). For a **toy** $C=4,H=W=8,P=2$, this is 16 tokens with 16 entries each before projection. These are teaching dimensions, not an SD3 or Movie Gen configuration. We use $N$ for token count and a separate layer count to avoid the notes’ overloaded notation around p. 44.

Compression reduces the state size but can lose details. Learning the encoded-data distribution remains necessary even when autoencoder regularization encourages Gaussian latents; the actual aggregated latent distribution need not equal a standard Gaussian. Appendix D discusses reconstruction versus generation and the division of labor between models.

Section 7 extends the path idea to discrete sequences. Token identities cannot be updated by adding a continuous velocity. A continuous-time Markov chain instead uses nonnegative off-diagonal jump rates and a diagonal rate that cancels total outgoing rate (Eqs. 85–86). Its marginalization trick averages conditional rate matrices over endpoint posteriors (Theorem 36). This is a route for further reading, not a complete discrete-model implementation.

## 9 · What evidence supports this story?

These notes are an instructional synthesis, not one controlled benchmark report. **Figure 7 on PDF p. 23** compares ground-truth time-marginal histograms for a two-dimensional chessboard distribution with histograms from a model trained using Algorithm 3. It illustrates approximate agreement after training. **Figures 6 and 9** contrast conditional and marginal ODE/SDE trajectories. They are useful mechanism demonstrations, not evidence of a universal image-quality or speed advantage. [Inspect the supplied figures in the PDF](/papers/flow-matching.pdf#page=23).

We do not invent FID values, solver budgets or confidence intervals for these plots. The equal-gradient theorem establishes a population-objective relationship; the figures illustrate it with a trained example. Neither establishes perfect finite-network optimization or generalization beyond the training distribution.

A useful proposed experiment would fix architecture, training data, training budget and random seeds, then compare schedules under the same number of function evaluations. Measure distribution fidelity and runtime separately. Next hold the trained field fixed and vary solver accuracy to separate integration error from field error. For stochastic sampling, vary $\sigma_t$ with the score correction present; removing the correction is a separate intervention. These are proposed checks, not extra reported findings.

## 10 · The idea to carry forward

**Make the training problem tractable by conditioning on information available during training; use conditional expectation to recover the field needed at generation time.** The endpoint teaches the target without becoming an input the unconditional generator needs later.

Read [Mean Flows](/papers/t2-paper-2) next to compare instantaneous velocity with interval-average velocity, or [FLUX.1 Kontext](/papers/flux-kontext) to see how a fixed reference can condition an evolving state. [Lecture 2](/papers/t2-generative-lecture-2) supplies the wider generative-model map. Flow matching explains a training mechanism; it does not alone specify prompt encoding, model architecture, a one-step sampler or evaluation quality.

<details><summary>Final check: what changes during training, and what changes during inference?</summary><p>Training samples noisy states directly and changes network parameters using known conditional targets. Inference freezes those parameters and repeatedly changes a sampled state using predicted marginal velocities. For a prompt-conditioned model, the prompt remains fixed while the generated state evolves.</p></details>

## Sources and reading map

The source of this guide is the [supplied MIT notes, 2026, PDF export March 18](/papers/flow-matching.pdf), retained byte-for-byte. Page references use PDF/printed pages, which coincide here.

| Question | Source location |
| --- | --- |
| What is a flow and how do we simulate it? | §2.1, pp. 7–10, Eqs. 1–4 and Algorithm 1 |
| How do we define a probability path? | §3.1, pp. 14–16, Eqs. 11–16 |
| Why does posterior averaging work? | §3.2, pp. 16–19, Theorems 9 and 11 |
| Why can we regress against conditional targets? | §3.3, pp. 19–24, Theorem 12, Algorithm 3, Figure 7 |
| How do scores enable SDE sampling? | §§4.1–4.3, pp. 25–33, Proposition 1, Theorems 17 and 22 |
| How is a prompt used and reinforced? | §5, pp. 34–40, guided training and CFG |
| What network and latent representation can implement it? | §6, pp. 41–53; Appendix D, pp. 77–82 |
| What changes for discrete tokens? | §7, pp. 54–65, rates and Theorem 36 |
| Why do other papers reverse time or use a noising process? | Appendix E, pp. 81–84 |

The notes’ figures and external model examples should be read in their original context. This guide adds its own derivations and deliberately labeled toy calculations; it runs no trained generator.
