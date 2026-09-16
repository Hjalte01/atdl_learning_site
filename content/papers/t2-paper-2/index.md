---
title: "Mean Flows for One-step Generative Modeling"
shortTitle: "Mean Flows"
topic: "topic-2"
order: 2
description: "Learn an interval's average velocity: trace the MeanFlow identity, JVP training target, and one-call latent generation through seven illustrated explanations."
status: "ready"
pdf: "/papers/meanflows.pdf"
---
## The puzzle: can one direction replace an entire journey?

A flow generator starts with noise and follows a learned velocity field toward an image. A local velocity tells it where to move **right now**. If the route bends, taking one enormous step along that direction can miss the destination. What if the network instead predicted the average velocity over the entire remaining interval?

**MeanFlow learns that interval-dependent quantity.** Generation becomes noise minus predicted displacement. The harder question is training: how do you supervise an average without first computing a costly trajectory? The paper derives an identity that turns local velocity and a network derivative into a usable target.

This guide follows the supplied **23-page NeurIPS 2025 conference PDF**, whose creation metadata is 14 February 2026. That metadata is not a publication date or an inferred arXiv revision. All seven diagrams are original teaching constructions; only the labeled result bars reproduce measured values. Open a figure to enlarge it. Allow about 20 minutes.

## 1 · Straight pairs do not mean straight generation

[![A curved marginal trajectory contrasts a local tangent step with a chord reaching the endpoint.](/figures/meanflows/problem.svg)](/figures/meanflows/problem.svg)

Read the curved route as a trajectory through a high-dimensional latent space, compressed into two dimensions. The gray tangent gives local motion; the green chord gives net displacement. This is a schematic, not a learned trajectory or an image result.

During training, select a clean data representation $x$, Gaussian noise $\epsilon$, and time $t\in[0,1]$. The paper uses the straight interpolation

$$
z_t=(1-t)x+t\epsilon,\qquad v_t=\epsilon-x.
$$

Here $z_t$ has the same shape as $x$; $v_t$ is the **sample-conditional** velocity for this particular pair. Time 0 is data and time 1 is noise, so generation runs backward. For $x=2$, $\epsilon=6$, and $t=0.75$, the training input is $z_t=5$ and the pair velocity is 4.

The model sees $z_t$ and time, not the hidden pair that produced it. Multiple pairs can yield the same input with different velocities. Under squared-error regression, the optimal instantaneous predictor is their conditional mean (paper Eq. 1):

$$
v(z,t)=\mathbb E[v_t\mid z_t=z].
$$

As the state and time change, the mixture of possible pairs changes. The resulting **marginal** ODE $dz_t/dt=v(z_t,t)$ can curve even though every pair interpolation is straight (§3, Figure 2). This curvature is not merely a neural-network error.

<details><summary>Predict: does training on straight noise–data pairs make a one-step Euler sampler exact?</summary><p>No. The generator follows the marginal field, whose trajectories need not be straight. A large Euler step approximates the whole interval with a local tangent.</p></details>

Our reconstruction of the design logic is: change what the network predicts so that its output already describes a finite interval. This is a teaching interpretation, not a claim about the authors' private reasoning.

## 2 · Represent a displacement using two times

[![One state has different average velocities for a short interval and a long interval; output has the same tensor shape as the state.](/figures/meanflows/interval.svg)](/figures/meanflows/interval.svg)

Read the two branches as two questions asked at the **same** state $z_t$: where should we end at $r=0.5$, or at $r=0$? The output is a velocity tensor, not a time-averaged image. This representation diagram is schematic.

First define the exact quantity before introducing a network (paper Eqs. 3 and 12):

$$
u(z_t,r,t)=\frac{1}{t-r}\int_r^t v(z_\tau,\tau)\,d\tau,
\qquad z_r=z_t-(t-r)u(z_t,r,t).
$$

$u$ is average velocity, $v$ instantaneous velocity, $r$ the destination time, $t$ the current time, and $z_\tau$ the state along the marginal ODE trajectory passing through $z_t$. The integral follows that trajectory; it does not hold the spatial input fixed. Multiplying $u$ by the interval $t-r$ produces displacement. The minus sign moves backward in time.

**Why two times?** The destination matters. A network $u_\theta(z_t,r,t)$ must know which interval it summarizes. At $r=t$, the displayed quotient is undefined, but its continuous limit is $u(z_t,t,t)=v(z_t,t)$. Zero-length sampling leaves the state unchanged. Training can use this diagonal without dividing by zero.

For ImageNet, Appendix A specifies a pretrained VAE that maps $256\times256\times3$ images to $32\times32\times4$ latents. Those 4,096 latent values enter a DiT-style transformer with adaLN-Zero conditioning. In the XL/2 configuration, $2\times2$ latent patches give 256 tokens; Table 4 specifies 28 blocks, hidden dimension 1,152, and 16 heads. The output reconstructs a $32\times32\times4$ average-velocity tensor. These details describe XL/2, not every MeanFlow model.

The implementation embeds current time $t$ and interval $t-r$ separately with positional embeddings and two-layer MLPs, then sums the time embeddings. Class $c$ also conditions the class-conditional model. The mathematical function remains a function of $(z,r,t)$, which matters for the derivative below.

## 3 · Train an average without evaluating its integral

[![Training branches from paired latent and noise to a network prediction and a directional derivative, then constructs a detached target and updates weights.](/figures/meanflows/training.svg)](/figures/meanflows/training.svg)

Read A → B → C as the forward prediction. The lower branch constructs the target; the orange loss compares that target to C. Purple boxes use the same trainable network, not a separate pretrained flow teacher. Green boxes contain supplied or constructed quantities. This is a functional training schematic for the unguided core.

The obstacle is the integral in the definition of $u$. Multiply by the interval and differentiate along the trajectory, **holding $r$ fixed**. The product rule and fundamental theorem of calculus give (paper Eqs. 4–6):

$$
(t-r)u=\int_r^t v\,d\tau
\quad\Longrightarrow\quad
u+(t-r)\frac{du}{dt}=v
\quad\Longrightarrow\quad
u=v-(t-r)\frac{du}{dt}.
$$

The last expression is the **MeanFlow identity**. It says how the average differs from the instantaneous velocity: subtract an interval-scaled correction for how the average changes. Appendix B.3 explains sufficiency as well as necessity: the displacement $(t-r)u$ vanishes at $t=r$, fixing the integration constant, assuming a well-behaved finite field there. This mathematical identity does not guarantee that optimization finds the exact field.

### The derivative follows the moving state

The target branch needs a total derivative, not just the explicit time derivative (paper Eqs. 7–8):

$$
\frac{du}{dt}=J_z u\,v+\partial_tu
=\operatorname{JVP}_{(z,r,t)}u\,[(v,0,1)].
$$

$J_z u$ is the Jacobian with respect to the state; $\partial_tu$ holds the other inputs fixed. The tangent $(v,0,1)$ says: move the state with instantaneous velocity, keep destination time fixed, and advance current time at unit rate. A Jacobian–vector product (JVP) computes this directional derivative without materializing the full Jacobian. Replacing the middle 0 with 1 differentiates along a different path.

Even when the network internally embeds $(t,t-r)$, differentiate the wrapper $u_\theta(z,r,t)=\operatorname{net}_\theta(z,t,t-r)$. With $r$ fixed, both internal time and interval change. Ignoring that chain rule changes the target.

### The practical target and weight update

The available training signal is $v_t=\epsilon-x$. Following the paper's conditional-velocity replacement, Algorithm 1 and Eq. 11 use

$$
D_\theta=\operatorname{JVP}u_\theta[(v_t,0,1)],\qquad
u_{\mathrm{tgt}}=v_t-(t-r)D_\theta,\qquad
\mathcal L=\mathbb E\left[\|u_\theta-\operatorname{sg}(u_{\mathrm{tgt}})\|_2^2\right].
$$

All network quantities here are evaluated at $(z_t,r,t)$. $D_\theta$ is a tensor with the same shape as the predicted velocity. $\operatorname{sg}$ means stop-gradient: the target is treated as constant during this weight update, although it changes as the network changes across updates. The prediction branch remains trainable. This avoids differentiating the optimization loss through the JVP, and hence avoids the higher-order parameter derivatives that branch would require.

This sample target is stochastic; it is not the exact integrated velocity of each chosen noise–data pair. Keep the underlying marginal-field identity distinct from the practical conditional training estimator.

The experiments also use adaptive loss weighting (§4.3, Appendix B.2): with residual $\Delta=u_\theta-u_{\mathrm{tgt}}$, use $w=(\|\Delta\|_2^2+\delta)^{-p}$ and loss $\operatorname{sg}(w)\|\Delta\|_2^2$. Here $\delta>0$ prevents division by zero and $p$ controls downweighting of large residuals. Table 4 uses $p=1$ for ImageNet. The numerical example below deliberately shows the unweighted core loss.

## 4 · Trace one training example through the boxes

[![Scalar training example computes state 5, derivative 2, target 3, and squared loss 0.25.](/figures/meanflows/numbers.svg)](/figures/meanflows/numbers.svg)

This is synthetic arithmetic, not a trained model output. Use A: $x=2$, $\epsilon=6$; B: $t=0.75$, $r=0.25$, giving $z_t=5$, $v_t=4$, and interval 0.5. Suppose C predicts $u_\theta=2.5$ and its local derivatives are $\partial_z u_\theta=0.25$, $\partial_tu_\theta=1$.

The target branch calculates

$$
D_\theta=0.25\times4+1=2,\qquad
u_{\mathrm{tgt}}=4-0.5\times2=3,\qquad
\mathcal L_{\mathrm{example}}=(2.5-3)^2=0.25.
$$

Holding the target fixed, the loss derivative with respect to the prediction is $2(2.5-3)=-1$, encouraging the prediction upward. The chain rule then distributes that signal to the weights. If the same prediction were used for a sampling step, it would give $z_r=5-0.5(2.5)=3.75$. A training target of 3 does not prove the model's endpoint is already correct.

<details><summary>Predict: what changes if we set r = t in this example?</summary><p>The correction is multiplied by zero. The target becomes the pair velocity 4, so this sample uses a Flow Matching target. Its sampling displacement is zero. Diagonal-only training does not teach the finite-interval field needed for one-call generation.</p></details>

Training repeatedly samples pairs and times, evaluates prediction plus JVP, constructs the detached target, and updates weights. The ImageNet default uses unequal times on 25% of samples and $r=t$ on the rest; the two sampled times come from a logit-normal distribution and are ordered. This ratio is an empirical design choice, not a universal constant.

## 5 · Inference: one call, or a sequence of interval jumps

[![Gaussian latent and class enter the trained MeanFlow network once; its velocity is subtracted from the noise and the resulting latent is decoded.](/figures/meanflows/sampling.svg)](/figures/meanflows/sampling.svg)

Read left to right. The weights stay fixed; there is no JVP or loss at inference. Algorithm 2 computes

$$
z_1=\epsilon\sim\mathcal N(0,I),\qquad
\hat z_0=\epsilon-u_\theta(\epsilon,0,1),\qquad
\hat I=\operatorname{VAEdecode}(\hat z_0).
$$

$I$ in the Gaussian covariance is the identity matrix; $\hat I$ denotes the output image. The diagram is schematic. For ImageNet, **1-NFE counts one evaluation of the flow network**, not the separate VAE decoding work or the entire application's latency. “From scratch” refers to the flow model; the ImageNet setup still uses a pretrained VAE.

[![One full jump and two sequential half-interval jumps compare how state must be carried forward.](/figures/meanflows/steps.svg)](/figures/meanflows/steps.svg)

The two-step route first computes $z_s=z_t-(t-s)u_\theta(z_t,s,t)$, then queries the network at that **new** state to reach $r$. For the exact field, interval displacements add (discussion after Eq. 3):

$$
(t-r)u(z_t,r,t)=(t-s)u(z_t,s,t)+(s-r)u(z_s,r,s).
$$

This teaching diagram shows the consistency relation over time, not measured convergence. Crucially, the second call takes $z_s$, not $z_t$. Finite networks need not satisfy this identity perfectly; more steps are supported, but the identity alone does not promise monotonically better samples.

### Guidance is learned into the field

Ordinary classifier-free guidance can require conditional and unconditional network calls at each sampling step. MeanFlow instead trains the average velocity of a guided field (§4.2, Eq. 13):

$$
v^{\mathrm{cfg}}(z,t\mid c)=\omega v(z,t\mid c)+(1-\omega)v(z,t).
$$

$c$ is a class and $\omega$ the guidance scale; $\omega=1$ is the class-conditional unguided case. The training target uses a modified sample velocity $\tilde v_t$, and the JVP's state tangent must use that same modified velocity. In the basic formulation (Eq. 19), $\tilde v_t=\omega v_t+(1-\omega)u_\theta^{\mathrm{cfg}}(z_t,t,t)$, with an unconditional diagonal prediction. Appendix B.1 further mixes conditional and unconditional diagonal outputs using a parameter $\kappa$; the main reported configurations use that extension and selected guidance intervals (Table 4).

After training, one call to the conditional guided-average network suffices: no two-output guidance combination is required during sampling. This moves guidance computation into training. It does not make training free or establish arbitrary post-training guidance adjustment.

## 6 · What does the evidence establish?

[![Reported ImageNet one-evaluation FID bars compare iCT 34.24, Shortcut 10.60, and MeanFlow XL/2 3.43.](/figures/meanflows/results.svg)](/figures/meanflows/results.svg)

These bars reproduce **Table 2's class-conditional ImageNet 256×256 FID-50K**, with guidance where applicable. Lower FID is better; it compares distributions of Inception features from 50,000 generated images and reference images. It is not a percentage of correct images or a per-image quality guarantee. The chart is a transcription of reported evidence, not our benchmark.

| Table 2 model | Flow-network evaluations | FID ↓ | Relevant setting |
| --- | ---: | ---: | --- |
| iCT-XL/2 | 1 | 34.24 | iCT result reported through IMM |
| Shortcut-XL/2 | 1 | 10.60 | Prior one-call baseline |
| MeanFlow-XL/2 | 1 | 3.43 | 676M parameters, 240 epochs |
| IMM-XL/2 | 2 | 7.77 | One step, two calls for guidance |
| MeanFlow-XL/2 | 2 | 2.93 | 240 epochs |
| MeanFlow-XL/2+ | 2 | 2.20 | 1,000 epochs and changed configuration |

The 3.43 versus 10.60 comparison is a 67.6% relative FID reduction, not a 67.6% gain in perceptual quality. The 2.20 result must not be presented as one-call performance or as merely changing the sampler of the 240-epoch model. The PDF's prose on page 8 quotes SiT as 2.15 while Table 2 lists 2.06; use the table's 2.06 if comparing that baseline. Neither discrepancy affects the MeanFlow rows above.

The most direct mechanism tests are Table 1's **B/4, 80-epoch, 1-NFE ImageNet** ablations: diagonal-only training gives FID 328.91, while 25% unequal times gives 61.06. Changing the JVP tangent from $(v,0,1)$ to $(v,0,0)$ worsens FID to 268.06. These support the importance of learning intervals and using the correct derivative under that setup. They are not the large guided XL/2 results.

On **unconditional CIFAR-10 32×32**, Table 3 reports 1-NFE FID 2.92 for MeanFlow versus 2.83 for iCT. Both use approximately 55M-parameter U-nets, but iCT uses EDM preconditioning and MeanFlow does not. MeanFlow is competitive here, not universally best.

### Limits and a useful next experiment

The exact identity is stronger than an empirical network fit: finite capacity, noisy targets, loss weighting, and optimization error still matter. Appendix B.4 measures 0.052 seconds per training iteration versus 0.045 for Flow Matching on B/4 with JAX and v4-8 TPUs—about 16% overhead. This is a specific training benchmark, not an end-to-end inference speed claim.

To isolate one-call versus two-call sampling, hold checkpoint, guidance configuration, evaluation images, and metric implementation fixed. To examine identity accuracy, measure one-interval displacement against two consecutive interval displacements at the correct intermediate state. Repeat across times and seeds. These are proposed experiments, not additional paper results. The reported benchmark tables do not provide uncertainty intervals, and ImageNet/CIFAR-10 evidence alone does not establish text-to-image or higher-resolution performance.

## 7 · The idea to carry forward

**Predict the quantity the sampler needs at its intended time scale.** Instantaneous velocity supports local integration; average velocity supports a finite displacement. MeanFlow's contribution is both the choice of that field and the identity that makes it trainable without computing trajectory integrals as labels.

From memory, redraw the target branch and label $(v_t,0,1)$, stop-gradient, and the interval factor. Then redraw inference with those training operations removed. If you can explain why the target changes across updates while staying detached within an update, and why the output depends on two times, you have the central mechanism.

## Sources and reading map

- [Supplied Mean Flows conference PDF](/papers/meanflows.pdf): §3 / Figure 2 for conditional versus marginal flow; §4.1 / Eqs. 3–12 and Algorithms 1–2 for the field, identity, training and sampling.
- The same PDF: §4.2 / Eqs. 13–19 and Appendix B.1 for guidance; §4.3 and Appendix B.2 for time sampling and loss weighting; Tables 1–3 for evidence; Appendix A / Table 4 for architecture and training; Appendix B.3–B.4 for sufficiency and JVP cost.
- [Authors' code repository](https://github.com/gsunshine/meanflow), linked by the supplied paper, for further implementation study. This guide is grounded in the supplied PDF rather than an inferred current code revision.

All scalar examples and path drawings are pedagogical constructions. No trained model runs on this page.
