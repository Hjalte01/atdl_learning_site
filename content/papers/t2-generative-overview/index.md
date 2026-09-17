---
title: "Deep Generative Models: a reading roadmap"
shortTitle: "Generative models overview"
topic: "topic-2"
order: 1
materialKind: "overview"
description: "Connect GANs, VAEs, diffusion and flow matching, then choose a route into editing, segmentation and the six Topic 2 papers."
status: "ready"
pdf: "/papers/generative-overview.pdf"
---
## The puzzle: generating something new is not retrieving an example

Imagine a small collection of images. Copying an image gives a familiar example, but cannot produce a new configuration on request. A generative model instead learns regularities of the data distribution and uses them to synthesize samples. The difficult questions are whether it covers the distribution, how it learns, how much sampling costs, and how an instruction controls the result.

This guide follows **Mostafa Mehdipour Ghazi’s three-page “Deep Generative Models” course reading overview**, from the Pioneer Centre for AI, University of Copenhagen. The supplied document prints no date or revision. It is a conceptual reading list, with no numbered equations, benchmark tables, or model implementation. All diagrams and arithmetic here are original teaching constructions, not reported experiments. Allow about 12 minutes.

## 1 · Read the topic as three questions

[![Three rows organize the reading around learning distributions, accelerating trajectories, and conditioning outputs.](/figures/generative-overview/map.svg)](/figures/generative-overview/map.svg)

Read each row as a question to carry into a paper. The first row lists alternative model families as separate boxes. The second follows the overview’s shift toward transport; the third follows its expansion of tasks. This is a schematic topic map, not a claim that one family universally replaces another.

The opening paragraph motivates synthesis, creativity, augmentation and cross-modal generation. Those are potential uses, not proof that synthetic data automatically improves a downstream model. A useful experiment would compare held-out performance with and without synthetic data while controlling training budget and preventing leakage.

Before the papers, be comfortable with these prerequisites:

| Concept | Why you need it |
| --- | --- |
| Probability distribution and sampling | A new sample should reflect more than one memorized example. |
| Latent variable | A decoder can generate an image from a hidden representation. |
| Gradient and loss | Training changes network weights to reduce an objective. |
| Gaussian noise | Diffusion constructs supervised corruption tasks. |
| Derivative and numerical integration | A velocity field moves a sample through time. |
| Conditional probability | Context changes which outputs are appropriate. |

You do not need a complete SDE derivation to start. Keep **data**, **network weights**, **time**, and **sampling state** separate.

## 2 · Three ways to learn a generator

[![GAN, VAE and DDPM training paths contrast discriminator feedback, latent reconstruction, and known-noise prediction; a final row lists their generation paths.](/figures/generative-overview/families.svg)](/figures/generative-overview/families.svg)

Read the first three rows separately. Each is a functional mechanism, not a layer specification. The last row lists three independent generation recipes; the boxes are not serial stages of one model.

**GAN:** a generator transforms noise into a sample. A discriminator learns to distinguish generated and real samples, providing the generator’s adversarial learning signal. At generation time, the generator produces the sample; a discriminator is not normally needed for that forward pass. The overview (§1) highlights sharp samples but unstable optimization and possible **mode collapse**: some kinds of data can be missing even when individual outputs look convincing.

**VAE:** during training, an encoder maps an example to a distribution over latent variables; a sampled latent goes through a decoder. Reconstruction learning is combined with regularization toward a prior so that sampling latents can support generation. At generation time, sample the prior and decode, without requiring an input example. The overview emphasizes probabilistic representation and potentially smoother outputs. Its description of “interpretable” latents should not be read as a guarantee that every latent coordinate has a human-readable meaning.

**DDPM:** training adds Gaussian noise to data and learns to reverse corruption. Sampling starts from noise and applies learned reverse updates repeatedly. Section 1 describes the balance of quality, diversity, training stability and sampling cost. These are qualitative comparisons; this source supplies no common dataset, timing protocol or score that would rank all three families.

<details><summary>Predict: could a generator make beautiful images while failing to model the dataset well?</summary><p>Yes. If it produces only a few kinds of examples, it may have poor distribution coverage. Inspecting a few attractive outputs cannot rule out mode collapse.</p></details>

## 3 · Training and sampling do different jobs

[![Training constructs targets and updates network weights; inference keeps weights fixed and repeatedly updates a sample state.](/figures/generative-overview/paths.svg)](/figures/generative-overview/paths.svg)

Follow the top row to the loss, then the bottom row to the next state. The loop in training learns parameters; the loop in sampling constructs one output. This schematic applies to iterative diffusion/flow generation, not the ordinary one-pass GAN recipe.

A minimal teaching expression for Gaussian corruption is

$$
x_t = a_t x + b_t\epsilon.
$$

Here $x$ is a clean example, $\epsilon$ is a drawn Gaussian-noise tensor, $t$ is time, and the scalars $a_t,b_t$ control signal and noise. This equation belongs in **construct noisy input**. For the common variance-preserving choice, $a_t^2+b_t^2=1$; this alone does not specify a schedule or a reverse sampler. The overview does not prescribe these coefficients.

A corresponding noise-prediction teaching loss is

$$
\mathcal L(\theta)=\mathbb E_{x,t,\epsilon}\left[\|\epsilon-\epsilon_\theta(x_t,t)\|^2\right].
$$

$\epsilon_\theta$ is the network with learned weights $\theta$; the expectation averages examples, times and noise draws; the squared norm measures prediction error. This operates at **loss → update weights**. We know the target because we drew the noise. At inference, the original clean example and that training target are unavailable. A sampler uses the learned prediction to update its state; this loss alone is not a sampling algorithm.

## 4 · From denoising steps to transport

Section 2 provides several routes to cheaper generation. Improved DDPMs refine noise schedules and reverse variance. DDIM allows deterministic trajectories and fewer steps. Score-based modeling describes continuous-time dynamics: a score points in the direction of increasing log density at a noise level. A reverse SDE includes stochastic dynamics; a probability-flow ODE is deterministic given its starting state. Matching distributions under the theoretical construction does not mean identical individual trajectories or identical numerical errors.

Flow matching instead learns a velocity field directly. Rectified-flow approaches encourage straighter transport. The pedagogical design logic is: if sampling must traverse a path, make the path and its update rule easier to integrate. This is an interpretation of the overview’s progression, not the authors’ private reasoning.

[![Toy scalar flow travels from noise value two through four to data value six, using a constant target velocity four.](/figures/generative-overview/numeric.svg)](/figures/generative-overview/numeric.svg)

Read the first row as three time points, and the second as one update. **For this example only, time 0 is noise and time 1 is data.** Papers may reverse that convention. Define a straight training path

$$
y_t=(1-t)z+tx,\qquad \frac{dy_t}{dt}=x-z.
$$

$z$ is a sampled starting noise value, $x$ a paired data value, $y_t$ their interpolated state, and $t\in[0,1]$. A velocity network $v_\theta(y_t,t)$ learns from such targets. With $z=2$, $x=6$, and $t=0.5$, the state is $4$ and the target velocity is $4$. This is one artificial pair, not a claim about typical Gaussian draws or image measurements.

In the inference **model + solver update** box, a forward Euler step is

$$
y_{t+h}\approx y_t+h\,v_\theta(y_t,t).
$$

$h$ is the time increment. If the predicted velocity is exactly $4$, a half-step from $2$ gives $2+0.5(4)=4$; another gives $6$. If the prediction is $3$, the first step gives $3.5$. The displayed diagram and arithmetic use the same constant-velocity assumption. In a learned field, velocity can vary with state and time; interpolation paths for training pairs do not guarantee every sampling trajectory is straight. A large Euler step is generally approximate.

<details><summary>Predict: does deterministic sampling mean every run produces the same image?</summary><p>No. A deterministic solver fixes the trajectory given the initial noise and context. Different starting noise can still produce different outputs.</p></details>

For interval-average velocity and the difference between one step and one network evaluation, continue to [Mean Flows](/papers/t2-paper-2). For training-time representation guidance, continue to [REPA](/papers/t2-paper-1).

## 5 · Context changes the task

[![Unconditional generation uses noise alone; conditional generation includes fixed context; structured prediction uses an observed image to guide a mask or explanation.](/figures/generative-overview/conditioning.svg)](/figures/generative-overview/conditioning.svg)

Read the rows as different tasks. This is a functional schematic: it does not imply that editing, mask generation and explanations share one architecture. Ask which input stays available, which state changes, and what the final output represents.

The probability shorthand is $p_\theta(x\mid c)$: a model distribution over output $x$ given context $c$, with learned weights $\theta$. The context may be a class, text or an image. Conditioning restricts the desired distribution; it does not guarantee perfect instruction following or preservation of every unedited pixel.

Section 3 distinguishes several ideas that can be combined:

- **Guidance** steers sampling using additional information. Classifier guidance and classifier-free guidance are different mechanisms; “classifier-free” does not mean “unconditional.”
- **Latent diffusion** changes the representation in which denoising happens, using a compressed space. The latent state must eventually be decoded to pixels; compression can discard details.
- **Cross-attention** connects context to image representations. It is one conditioning mechanism, not a synonym for guidance or latent diffusion.
- **Structured prediction** changes what is generated or inferred: a mask has a different role from an edited image.

The source’s §3 attaches reference [15] to “counterfactual explanations,” but its bibliography names **What the DAAM: Interpreting Stable Diffusion Using Cross Attention**. Treat this as a citation mismatch: this overview alone does not establish a counterfactual algorithm or its validity. Use the separately supplied [DiME guide](/papers/t2-paper-3) for the course’s counterfactual method; it is an additional course connection, not a replacement silently attributed to reference [15].

<details><summary>Predict: does moving diffusion into a latent space tell you how text controls the output?</summary><p>No. Latent space specifies the modeled representation. Conditioning and guidance specify how context influences generation. Those are separate design choices.</p></details>

## 6 · Choose a reading route

[![Three optional reading routes connect transport to Mean Flows and REPA, editing to FLUX and Qwen-Image, and structured tasks to DiffAtlas and DiME.](/figures/generative-overview/route.svg)](/figures/generative-overview/route.svg)

Read each row as an optional study sequence, not a dependency graph. These links connect the overview to the repository’s six supplied Topic 2 papers; the overview itself does not list these six as its bibliography.

| Your question | Read next | What to trace |
| --- | --- | --- |
| Can generation take one step? | [Mean Flows · 2](/papers/t2-paper-2) | Instantaneous versus interval-average velocity. |
| Can a generator learn representations faster? | [REPA · 1](/papers/t2-paper-1) | Teacher features during training; teacher absent during sampling. |
| How does an image guide an edit? | [FLUX.1 Kontext · 5](/papers/flux-kontext) | Fixed reference versus evolving output. |
| How are text rendering and editing combined? | [Qwen-Image · 6](/papers/t2-paper-6) | Training curriculum and dual-reference conditioning. |
| How can a generator segment an image? | [DiffAtlas · 4](/papers/t2-paper-4) | Joint image–mask modeling and image replacement. |
| How can an image explain a classifier decision? | [DiME · 3](/papers/t2-paper-3) | Classifier gradients, image changes and validity limits. |

## 7 · The idea to carry forward

For every paper, redraw four things: **the training example, the learned predictor, the sampling state, and the context**. Put the loss next to the weight update and the solver next to the state update. Then ask whether the paper improves learning, sampling, representation, conditioning, or evaluation. More than one answer can be right.

To assess a claimed improvement, record dataset, metric, training budget, number of model evaluations and baseline. A fast sampler is not automatically a better distribution model; a plausible edit is not automatically a faithful explanation. These are evaluation questions to bring to the papers, not experimental conclusions from this reading overview.

## Sources and reading map

[Open the supplied three-page course overview](/papers/generative-overview.pdf). No printed date/version is available; the repository preserves the imported source unchanged.

| Source location | What it supports here |
| --- | --- |
| Page 1, opening paragraph | Applications and main reading families; references [1]–[4]. |
| Page 1, §1 | GAN/VAE/DDPM mechanisms and qualitative trade-offs. |
| Pages 1–2, §2 | Improved DDPM, DDIM, score/SDE/ODE, flow matching and rectified-flow progression; [5]–[9]. |
| Page 2, §3 | Guidance, latent representation, editing and structured tasks; [10]–[15]. |
| Pages 2–3, bibliography | Source reading identities and the reference [15] mismatch. |

The equations, scalar example, diagrams, proposed evaluation checks and six-paper route are teaching additions. The source provides no quantitative results to redraw and no architecture layers to reproduce. There is no trained model or interactive simulator on this page.
