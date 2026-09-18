---
title: "Deep Generative Models · Lecture 2"
shortTitle: "Generative models · Lecture 2"
topic: "topic-2"
order: 2
materialKind: "slides"
description: "Follow noise into images: diffusion, probability-flow ODEs, guidance, latent representations, attention, and the metrics that test generation."
status: "ready"
pdf: "/papers/generative-lecture-2.pdf"
---
## The puzzle: why generate an image through noise?

A single network could map a random vector to an image. Why instead call a network repeatedly on a noisy image? The lecture’s answer is a useful division of labor: **make corruption easy to sample, learn local corrections, and compose those corrections into generation.** The cost is an iterative sampling loop. Much of this lecture asks which part of that loop we can change without losing useful image structure.

This guide follows Mostafa Mehdipour Ghazi’s supplied **35-slide “ATDL GM Lecture 2” PDF** (`4-5_ATDL GM Lecture 2.pdf`). The title slide has no lecture date; PDF creation metadata says **10 September 2026**, which identifies this export rather than a publication date. Slide numbers below are PDF page numbers. Read alongside the [local lecture PDF](/papers/generative-lecture-2.pdf). Allow about 30 minutes; the [Topic 2 roadmap](/papers/t2-generative-overview) supplies prerequisites.

The diagrams are original teaching schematics. All numerical examples are constructed calculations, not trained-model outputs or experimental results. This is a guided explanation of the deck’s sequence, not a substitute for the full papers cited on its slides.

## 1 · A map of the design choices

[![Four independent design choices: learned target, sampling procedure, image representation, and conditioning.](/figures/generative-lecture-2/map.svg)](/figures/generative-lecture-2/map.svg)

Read each row as a separate choice. A transformer can operate on noisy latent patches; classifier-free guidance can modify its predictions; an ODE solver can then integrate them. These names describe different parts of a system, so they are not mutually exclusive model families. The map is schematic, with no measured speed ranking.

The deck moves from **DDPM and faster sampling** (slides 2–6), through **scores, differential equations and alternative generators** (7–17), to **representations and conditioning** (18–26), **applications and explanations** (27–31), and **evaluation** (32–34).

A plausible design rationale—not a claim about the lecturer’s private reasoning—is to ask four questions: What does the network predict? How is its prediction used to update a sample? In which space does that happen? What evidence says the output is useful?

## 2 · DDPM: learn the noise, then use it to move backward

[![Training sends a clean image through known corruption to a time-conditioned denoiser and noise loss. Sampling repeatedly applies the frozen denoiser to a random state.](/figures/generative-lecture-2/ddpm.svg)](/figures/generative-lecture-2/ddpm.svg)

**Training, top row:** sample a clean image $x_0$, a time index $t$, and independent standard Gaussian noise $\epsilon$. The corruption box produces $x_t$. A time-conditioned U-Net predicts the added noise. The loss updates network parameters $\theta$; the chosen corruption rule is fixed. **Sampling, bottom row:** start with a fresh Gaussian tensor $x_T$ and carry the evolving sample into the next iteration. Keep $\theta$ fixed. No clean target image or target noise is supplied at inference. This is the functional architecture on slides 2–3, not an invented layer specification.

### The corruption box

The following cumulative identity expands the discrete Gaussian chain in slides 2–3:

$$
\alpha_t=1-\beta_t,\qquad \bar\alpha_t=\prod_{s=1}^t\alpha_s,\qquad
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon.
$$

Here $\beta_t$ is the noise variance for one forward step, $\alpha_t$ its retained signal factor, and $\bar\alpha_t$ the product across steps. All $x$ and $\epsilon$ tensors have the same image shape; $t$ is a separate network input. Square roots convert variance weights to amplitude coefficients. This identity lets training jump directly to a sampled time instead of simulating every preceding step.

**Trace one scalar through the boxes:** let $x_0=2$, $\bar\alpha_t=0.64$, and $\epsilon=-1$. Corruption gives $x_t=0.8(2)+0.6(-1)=1$. Suppose the network predicts $\epsilon_\theta(1,t)=-0.5$. Its squared noise error is $(-1+0.5)^2=0.25$.

### The loss and reverse-update boxes

Slide 3’s simplified training loss is

$$
\mathcal L_{\mathrm{simple}}=\mathbb E_{t,x_0,\epsilon}
\left[\|\epsilon-\epsilon_\theta(x_t,t)\|^2\right].
$$

The expectation averages the sampled times, images and noise; the norm sums squared tensor errors. The network learns a predictor across examples. It does not recover the exact hidden noise for every ambiguous noisy image.

At inference the prediction sets the reverse mean:

$$
\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right),
\qquad x_{t-1}=\mu_\theta(x_t,t)+\sigma_t z.
$$

$z$ is fresh standard Gaussian sampling noise, distinct from the training noise $\epsilon$; $\sigma_t$ is the reverse standard deviation. The update box uses the chosen schedule and network output together. The terminal clean-output step omits the random term. The subtraction is a schedule-scaled correction, not “subtract the entire predicted noise tensor.”

Continuing the toy example with $\alpha_t=0.8$, $\beta_t=0.2$, and hence a compatible $\bar\alpha_{t-1}=0.8$, the mean is $(1-(0.2/0.6)(-0.5))/\sqrt{0.8}\approx1.304$. With illustrative $\sigma_t=0.1$ and $z=0.5$, the next state is approximately $1.354$. These chosen values demonstrate one intermediate update, not a complete valid sampler configuration.

<details><summary>Predict: is the next state the model’s estimated clean image?</summary>

No. It is a less noisy state. The separate clean estimate is $\hat x_0=(x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta)/\sqrt{\bar\alpha_t}=1.625$ in this example. Confusing it with $x_{t-1}$ skips the role of the schedule.

</details>

## 3 · Speed is more than a step count

Slides 4–6, 13–14 introduce distinct interventions:

| Intervention | What changes | What must still be checked |
| --- | --- | --- |
| Improved DDPM, slide 4 | Cosine corruption schedule; learned reverse variance supports fewer sampling steps | Quality under the selected schedule and step budget |
| Step-aware model, slide 5 | Network size depends on generative-step importance | Work per call and cost of the complete system |
| DDIM, slide 6 | A sampling construction can remove per-step randomness | Quality and numerical error when skipping steps |
| Consistency models, slide 13 | Map points on one probability-flow trajectory to the same near-data endpoint | Learned consistency and few-step sample quality |
| DDGAN, slide 14 | Adversarially learn richer reverse transitions across larger noise intervals | Adversarial training behavior and retained coverage |

A deterministic sampler still generates diverse outputs by starting from different random seeds. DDIM does not mean “use zero as the initial noise.” Nor does deterministic sampling by itself guarantee accurate large steps.

Consistency models target a shared endpoint along a trajectory, rather than merely reproducing each local DDPM noise prediction. DDGAN changes the learned transition’s expressiveness: its slide shows a noise-conditioned generator and a discriminator judging reverse-transition pairs. Those are training roles; the discriminator is not the image sampler.

A useful speed report must give **network evaluations, work per evaluation, resolution, hardware and quality**. The deck does not supply a controlled numerical leaderboard across these methods, so this guide does not invent one.

## 4 · Scores connect stochastic and deterministic paths

[![A shared learned score feeds either a reverse SDE with fresh noise or a probability-flow ODE with half the score coefficient and no fresh noise.](/figures/generative-lecture-2/score.svg)](/figures/generative-lecture-2/score.svg)

The two rows use the same score box but different update rules. The bottom row is not obtained by deleting the final random term in the top row. This is a schematic of slides 7–12; shared distributions in the ideal theory do not mean identical sample trajectories.

A **score** is an input-space gradient, $s_\theta(x,t)\approx\nabla_x\log p_t(x)$, where $p_t$ is the noisy-data density at time $t$. It points toward locally increasing log density. For a one-dimensional Gaussian $p_t=\mathcal N(m,v)$, its score is $-(x-m)/v$. At $m=0$, $v=4$, and $x=2$, the score is $-0.5$: it points toward the center and weakens when variance is larger.

For the Gaussian corruption above, the conditional score target is $-(x_t-\sqrt{\bar\alpha_t}x_0)/(1-\bar\alpha_t)=-\epsilon/\sqrt{1-\bar\alpha_t}$. Averaging such denoising targets at a noisy input connects noise prediction to the marginal score. A conditional target for a known clean image and the unknown marginal score are not interchangeable pointwise.

Use the precise stochastic notation behind slides 8–10:

$$
 dx=f(x,t)\,dt+g(t)\,dW_t.
$$

$f$ is the forward drift; $g$ controls noise amplitude; $W_t$ is Brownian motion. Brownian increments over a positive duration $h$ have standard deviation $\sqrt h$, not $h$. Slide 8’s white-noise derivative is informal notation: Brownian motion itself is not white noise. Similarly, the ODE integral uses $f(x(s),s)$ along the evolving solution.

For **backward sampling** write $h>0$ and decrease time explicitly, as in slide 10:

$$
 x_{t-h}\approx x_t-\big[f(x_t,t)-g(t)^2s_\theta(x_t,t)\big]h
       +g(t)\sqrt h\,z,\qquad z\sim\mathcal N(0,I).
$$

The reverse SDE update box subtracts its drift because time runs backward. Slide 12’s probability-flow ODE instead gives

$$
 x_{t-h}\approx x_t-\big[f(x_t,t)-\tfrac12g(t)^2s_\theta(x_t,t)\big]h.
$$

With exact scores and the appropriate regularity, these continuous-time processes share the same time marginals. Learned scores and finite-step solvers introduce errors. **Toy update:** $x_t=2$, $f=0$, $g=1$, $s_\theta=-0.5$, $h=0.1$. The reverse SDE gives $1.95+\sqrt{0.1}z$, while the ODE gives $1.975$. Simply dropping SDE noise would leave $1.95$ and the wrong ODE drift.

Slide 11 chooses analytically tractable corruption: VE uses zero drift with growing noise variance; VP uses $f=-\beta(t)x/2$ and $g=\sqrt{\beta(t)}$. VP keeps the unit Gaussian invariant; it does not say every starting data distribution has exactly constant variance. The Gaussian prior is approached as sufficiently strong corruption removes the data signal.

## 5 · Flow matching: supervise movement directly

[![A toy noise-to-data interpolation gives a known velocity target; a trained velocity network is then applied repeatedly by an Euler solver.](/figures/generative-lecture-2/flow.svg)](/figures/generative-lecture-2/flow.svg)

The upper row constructs a training target, while the lower rows trace inference through repeated calls. This is an exact scalar teaching path, not a learned image trajectory. Slide 17 reverses the endpoint convention used in the earlier diffusion slides: **here $x_0$ is noise and $x_1$ is data**.

For independently sampled endpoints in the simple straight-line construction,

$$
 x_t=(1-t)x_0+tx_1,\qquad u=x_1-x_0,\qquad
 \mathcal L(\theta)=\mathbb E\|v_\theta(x_t,t)-u\|^2.
$$

$t\in[0,1]$, $u$ is the pair’s known velocity, and $v_\theta$ is the learned velocity field. This is the straight-interpolation specialization illustrated on slide 17, not every conditional flow-matching path. The training box regresses a vector that says how to move, so inference can use the ODE $dx/dt=v_\theta(x,t)$.

**Trace:** choose noise endpoint $x_0=-2$ and data endpoint $x_1=2$. At $t=0.25$, $x_t=-1$ and the target is $u=4$. An Euler step of size $h=0.25$ with a perfect toy predictor gives $-1+0.25(4)=0$. Starting at $t=0$, four such steps give $-2\to-1\to0\to1\to2$.

Real inference has no paired data endpoint available. Training averages targets consistent with $(x_t,t)$; the learned field can curve even when each training pair uses a straight interpolation. Rectified flow motivates straighter transport and easier numerical integration, but one exact scalar trace does not establish one-step image generation. Continue to [Mean Flows](/papers/t2-paper-2) to see the distinction between instantaneous and interval-average velocity.

## 6 · Change the representation, keep the loop explicit

[![Three representation pipelines compare pixels, invertible wavelet bands, and learned latent tensors. A separate inference path starts from latent noise and ends at the decoder.](/figures/generative-lecture-2/representation.svg)](/figures/generative-lecture-2/representation.svg)

Read the first three rows as alternatives, not consecutive transformations. The final row isolates latent inference. All shapes are symbolic: the deck does not specify a universal compression ratio, patch size or channel count.

Slides 18–20 move costly generation away from full-resolution pixels. A wavelet transform rearranges information into low- and high-frequency bands; inverse DWT reconstructs it. An invertible DWT does **not** automatically reduce the number of coefficients. A learned encoder instead maps $x\in\mathbb R^{H\times W\times C}$ to a latent $z=E(x)\in\mathbb R^{h\times w\times c}$, typically at lower spatial resolution. Its decoder $D$ reconstructs an image. Compression changes both cost and the information available to the generator.

In a staged latent-diffusion system, first learn the image representation, then train a denoiser on noisy encoded training images with the representation held fixed. At text-to-image inference, draw latent noise, run the denoising loop and decode the final latent. **There is no source image to encode** in this unconditional-on-image sampling path. Image-conditioned editing adds an input path of its own.

Slide 26’s DiT changes the denoiser backbone to a transformer on latent patches. Patching turns a spatial tensor into a token sequence; it does not necessarily quantize each patch into a discrete code. In contrast, slide 24’s MaskGIT predicts discrete visual tokens with bidirectional context and repeatedly refines masked positions in parallel. Mask replacement and Gaussian denoising are different state transitions.

<details><summary>Predict: does a faster latent denoiser guarantee a faster complete generator?</summary>

No. Compare the complete path, including condition encoding, all denoiser evaluations and decoding, at matched output resolution and quality. A decoder can also limit fine detail regardless of the denoiser’s quality.

</details>

## 7 · Put the condition into the computation

[![Attention computes weighted values from queries and keys; FiLM rescales features; classifier-free guidance combines unconditional and conditional predictions.](/figures/generative-lecture-2/conditioning.svg)](/figures/generative-lecture-2/conditioning.svg)

These rows act at different locations: attention routes information inside the network; FiLM modulates features; guidance combines predictions at the sampler boundary. The numerical values are separate toy examples, not activations of a trained network.

### Attention and feature modulation

Slides 21–22 compute a weighted combination of values:

$$
 A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right),\qquad O=AV.
$$

$Q\in\mathbb R^{n\times d_k}$ contains query vectors, $K\in\mathbb R^{m\times d_k}$ keys, and $V\in\mathbb R^{m\times d_v}$ values. Row-wise softmax gives $A\in\mathbb R^{n\times m}$; the output is $O\in\mathbb R^{n\times d_v}$. Scaling by $\sqrt{d_k}$ controls dot-product magnitude. For cross-attention, image features can supply queries and text tokens supply keys and values. Each image position receives a weighted text representation, not necessarily a single “best” word.

**Toy:** one query produces scaled logits $(0,\ln3)$, so weights are $(1/4,3/4)$. With scalar values $(2,6)$ the output is $5$. Multiple heads repeat this operation with learned projections, concatenate their outputs and project again. The weights describe information routing; alone they do not prove causal influence.

Slide 23’s FiLM performs a different operation:

$$
 \widetilde F_{i,c}=\gamma_{i,c}(a_i)F_{i,c}+\beta_{i,c}(a_i).
$$

$F_{i,c}$ is sample $i$’s feature channel $c$ (possibly a spatial map), $a_i$ is its condition, and learned functions produce per-channel scale $\gamma$ and shift $\beta$. They broadcast across spatial positions when applied to a feature map. For $F=3$, $\gamma=2$ and $\beta=-1$, the output is $5$. These $\beta$ values are feature shifts, unrelated to the earlier diffusion noise schedule.

### Guidance at sampling time

Slide 15 uses a classifier on noisy inputs. Bayes’ rule at a fixed time gives the conditional score as the unconditional score plus $\nabla_{x_t}\log p(y\mid x_t,t)$, where $y$ is the requested class. The classifier gradient enters the reverse update, so a classifier trained only on clean images may be unsuitable at high noise.

Slide 16 avoids a separate classifier by randomly dropping conditioning during denoiser training. Using its guidance-scale convention,

$$
 s_{\mathrm{guided}}=s_{\mathrm{uncond}}+\gamma
 (s_{\mathrm{cond}}-s_{\mathrm{uncond}}).
$$

$s_{\mathrm{cond}}$ and $s_{\mathrm{uncond}}$ are score predictions at the same noisy input and time; $\gamma$ is the guidance scale, unrelated to FiLM’s per-channel scale. At $\gamma=0$ this is unconditional; at $1$ it is conditional; above $1$ it extrapolates. For scalar predictions $1$ and $3$, $\gamma=2$ gives $5$. This extra strength is not a general guarantee of better quality or diversity. Some implementations shift the parameter convention, so compare formulas before comparing scale numbers.

## 8 · Editing, segmentation and explanations

Slides 25 and 27–31 show why the generated variable and the fixed condition matter:

| Slide | Mechanism to follow | Question to ask |
| --- | --- | --- |
| 25, Prompt-to-Prompt | Control cross-attention while changing prompt words | Which spatial relationships should remain stable? |
| 27, MedSegDiff-V2 | Condition diffusion-based segmentation on the scan | Is the evolving state the segmentation rather than a new patient image? |
| 28, SDSeg | Use latent diffusion for a single-step reverse segmentation path | How were the segmentation representation and conditioning learned? |
| 29, attention attribution | Reshape, upscale and aggregate text-attention maps over layers and steps | Does attention correlate with actual output sensitivity? |
| 30, perturbation explanations | Perturb information and aggregate output changes, with SSIM as one comparison | Does the perturbation test the intended semantic change? |
| 31, conditional reconstruction | Compare reconstruction errors under candidate class conditions | Does the class comparison remain reliable on new data? |

The deck’s single-step SDSeg example does not imply any existing diffusion model can skip straight to a correct segmentation. Likewise, multiple sampled outputs are not automatically calibrated uncertainty estimates. Those are properties to measure on the intended task.

Slide 29 averages spatially aligned attention maps across layers and time. Aggregation can reveal persistent token associations, but hides variation between steps and depends on resizing and normalization. A useful ablation would perturb the highlighted condition and compare against matched random perturbations while holding the seed fixed. This is a **proposed check**, not a result reported in the deck.

Continue with [DiffAtlas](/papers/t2-paper-4) to contrast conditional mask generation with joint image–mask generation, or [DiME](/papers/t2-paper-3) to trace classifier gradients through a denoiser.

## 9 · Measure what the generator actually gets right

[![Evaluation separates classifier confidence, feature-distribution similarity, coverage, and task usefulness. A scalar FID example shows mean and spread mismatch.](/figures/generative-lecture-2/metrics.svg)](/figures/generative-lecture-2/metrics.svg)

The upper rows separate questions that a single score cannot answer. The last row is a calculated scalar FID example, not a measured model comparison. Slides 32–34 introduce metrics but do not provide enough benchmark context for a numerical cross-model ranking here.

**Inception Score (slide 32)** rewards confident class predictions for individual generated images and a diverse marginal predicted class distribution:

$$
 \mathrm{IS}=\exp\!\left(\mathbb E_{x\sim p_g}
 D_{\mathrm{KL}}(p(y\mid x)\|p(y))\right).
$$

$p_g$ is the generated-image distribution; $p(y\mid x)$ is the Inception classifier output and $p(y)$ its average across generated images. This metric uses no real-image reference set. Toy example: two equally frequent, perfectly classified classes yield IS $=2$; always predicting one class yields $1$. Repeating the same example within each class can still score $2$, so this is not proof of within-class diversity.

**FID (slide 32)** compares means and covariances of real and generated feature embeddings under a Gaussian approximation. In one dimension it simplifies to

$$
 \mathrm{FID}_{1D}=(\mu_r-\mu_g)^2+(\sigma_r-\sigma_g)^2.
$$

$\mu_r,\mu_g$ are real/generated feature means and $\sigma_r,\sigma_g$ their standard deviations. With means $0,1$ and standard deviations $1,2$, the value is $1+1=2$. The general metric uses covariance matrices and a matrix square-root term; this scalar derivation explains its two penalties without pretending real features are one-dimensional. Matching two moments does not establish identical image distributions. Finite samples and the chosen feature extractor affect the result.

**Precision and recall (slide 33)** use feature-space neighborhoods. Precision asks what fraction of generated examples falls inside the real-data neighborhood union; recall reverses the roles to ask how much real data the generated neighborhoods cover. High precision with low recall is compatible with attractive but narrow outputs. Downstream scores and human studies answer further questions: visual quality need not imply usable segmentation labels or faithful prompt following.

**CMMD (slide 34)** compares CLIP feature distributions with a kernel discrepancy instead of fitting Gaussians. For real features $a_1,\ldots,a_m$, generated features $b_1,\ldots,b_n$, and kernel $k$, the slide’s unbiased squared-MMD estimator is

$$
 \widehat{\mathrm{MMD}^2}=
 \frac{\sum_{i\ne j}k(a_i,a_j)}{m(m-1)}+
 \frac{\sum_{i\ne j}k(b_i,b_j)}{n(n-1)}-
 \frac{2\sum_{i,j}k(a_i,b_j)}{mn}.
$$

The first two terms measure within-set similarity and the last subtracts cross-set similarity; $m,n>1$. The slide uses a Gaussian kernel $k(a,b)=\exp(-\|a-b\|^2/(2\sigma^2))$, where $\sigma$ is its bandwidth. “Unbiased” describes the estimator of squared MMD, not an absence of semantic bias in CLIP features. Finite-sample estimates can be negative even though population squared MMD is nonnegative. Slide 34’s distortion plot illustrates metric behavior; the deck does not fully specify the dataset and protocol needed to turn it into a reproducible benchmark claim.

<details><summary>Predict: FID improves but recall drops. Is that contradictory?</summary>

No. FID summarizes fitted feature moments; recall measures coverage through local neighborhoods. Report the full setting, real reference set, feature extractor, sample count, generation budget and both results before deciding whether the tradeoff is useful.

</details>

## 10 · Reading map and the idea to carry forward

| Source slides | Revisit for | Continue into a course guide |
| --- | --- | --- |
| 2–6 | Noise prediction, reverse transitions, schedules and DDIM | [DiffAtlas](/papers/t2-paper-4) |
| 7–14 | Score fields, SDE/ODE distinction and faster transitions | [Mean Flows](/papers/t2-paper-2) |
| 15–17 | Guidance and noise-to-data velocity | [FLUX.1 Kontext](/papers/flux-kontext) |
| 18–26 | Wavelets, latents, attention, FiLM, tokens and DiT | [REPA](/papers/t2-paper-1), [Qwen-Image](/papers/t2-paper-6) |
| 27–31 | Segmentation and explanatory signals | [DiME](/papers/t2-paper-3) |
| 32–34 | Metric assumptions and complementary evidence | Compare the evaluation sections of those guides |

**The idea to carry forward:** reconstruct the computation before comparing method names. Write down the evolving state, learned prediction, update rule, fixed condition and decoder. Then separate training supervision from information available at inference. That small diagram explains why changing a schedule, replacing a U-Net, adding guidance and compressing an image solve different problems—and why none alone establishes useful generation.
