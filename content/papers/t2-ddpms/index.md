---
title: "Denoising Diffusion Probabilistic Models · learn small steps from noise"
shortTitle: "DDPMs"
topic: "topic-2"
order: 5
materialKind: "extra"
description: "Connect forward corruption, a shared noise predictor, variational training and stochastic reverse sampling through worked calculations and the original evidence."
status: "ready"
source: "https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html"
pdf: "/papers/ddpms.pdf"
---
## The puzzle: how can removing noise create something new?

A denoiser normally receives a damaged photograph. A generator starts without a photograph at all. How could the same operation do both? Train a network on many noise levels, then use it repeatedly: start from almost pure noise, make a small distribution-aware correction, and repeat until an image emerges.

**DDPM learns the reverse of a fixed corruption process.** Its training targets come from noise we add ourselves. At generation time there is no hidden original image to recover; the learned distribution makes many plausible outcomes possible.

This guide follows Ho, Jain and Abbeel’s **NeurIPS 2020 Denoising Diffusion Probabilistic Models**, supplied as `topic 2/extra reading material/DDPMs.pdf`. The [exact course copy](/papers/ddpms.pdf) has 12 pages and file metadata dated December 16, 2020; that timestamp is not an arXiv revision date. The [official proceedings record](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html) confirms publication identity. Appendices referenced by the paper are absent from this copy, so we do not invent their implementation details.

Allow 25 minutes. All mechanism diagrams and scalar examples are teaching constructions; the final chart redraws reported results. Click any diagram to enlarge it. No trained model runs on this page.

## 1 · Architecture: one network serves many noise levels

[![Fixed corruption produces a noisy image; a time-conditioned U-Net predicts image-shaped noise, which feeds either a training loss or a reverse update.](/figures/ddpms/architecture.svg)](/figures/ddpms/architecture.svg)

Read the diagram as two uses of the same box C. **A** supplies a training image $x_0\in\mathbb R^{H\times W\times C}$, with integer pixel intensities scaled to $[-1,1]$. Here $H,W,C$ denote height, width and image channels. **B** adds known Gaussian noise. **C**, the learned network $\epsilon_\theta(x_t,t)$, receives the noisy tensor and timestep and returns a tensor of the same shape. **D** compares this output with the known training noise. During sampling, **E** instead converts the prediction to a reverse mean and adds fresh noise.

Section 4 specifies a U-Net similar to an unmasked PixelCNN++, group normalization, sinusoidal timestep embeddings, and self-attention at $16\times16$ feature resolution. Weights $\theta$ are shared across times. There are not 1,000 separately trained networks. The diagram is functional: precise channel widths and block counts are not supplied in the main text.

A useful reconstruction of the design logic—not a claim about the authors’ private thoughts—is to replace one difficult leap from noise to an image with many small conditional transitions. Time conditioning tells the shared network which denoising problem it is solving. The fixed forward process supplies examples at every difficulty level.

## 2 · Representation: the latent is still image-shaped

[![Three noise levels retain the same image dimensions while signal shrinks and noise grows; one-step alpha and cumulative alpha-bar are distinguished.](/figures/ddpms/representation.svg)](/figures/ddpms/representation.svg)

Read left to right as increasing corruption, not spatial compression. Each $x_t$ has the dimensionality of $x_0$. These are pixel-space latent states, not compact autoencoder codes.

A forward step keeps part of the current state and adds independent noise (paper Eq. 2):

$$
q(x_t\mid x_{t-1})=\mathcal N(\sqrt{1-\beta_t}\,x_{t-1},\beta_t I).
$$

$\beta_t$ is the step’s noise variance; $I$ is identity covariance, and $q$ denotes the fixed forward distribution. Define $\alpha_t=1-\beta_t$ and $\bar\alpha_t=\prod_{s=1}^t\alpha_s$, with $\bar\alpha_0=1$. Independent Gaussian increments combine into a closed-form marginal (Eq. 4):

$$
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad\epsilon\sim\mathcal N(0,I).
$$

This equation lives in **B**. It allows training to jump directly to a randomly selected time. The aggregate $\epsilon$ need not be the last noise increment of a sequential trajectory. Drawing separate marginals does not automatically produce a correctly coupled forward path.

The square roots are amplitude coefficients: multiplying a random variable by $a$ multiplies its variance by $a^2$. For unit-variance independent signal and noise, $\bar\alpha_t$ and $1-\bar\alpha_t$ are variance contributions, not fractions of pixels replaced. Conditional on a fixed image, only the noise contributes random variance.

The paper uses $T=1000$ steps and a linear variance schedule from $10^{-4}$ to $0.02$ (§4). Small steps support the Gaussian reverse approximation; accumulated noise makes the terminal state close to a standard normal. This is an approximation at finite $T$, not exact erasure of all information.

<details><summary>Predict: does training need 1,000 successive network calls to construct one noisy example?</summary><p>No. The closed-form marginal makes the noisy example directly at a sampled time. Sampling a new image still uses the reverse sequence.</p></details>

## 3 · Training: known corruption gives a supervised target

[![Training samples an image, time and noise, computes one noisy input and updates the network; generation freezes weights and repeats reverse updates.](/figures/ddpms/paths.svg)](/figures/ddpms/paths.svg)

Read the upper row as one stochastic training example and the lower rows as generation. Algorithm 1 samples $x_0$ from data, $t$ uniformly from $\{1,\ldots,T\}$ and independent Gaussian $\epsilon$. It constructs $x_t$ in B, predicts noise in C, and takes a gradient step on the squared error in D. Equation 14 averages these errors:

$$
L_{\mathrm{simple}}(\theta)=\mathbb E_{t,x_0,\epsilon}
\left[\|\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}x_0+
\sqrt{1-\bar\alpha_t}\epsilon,t)\|^2\right].
$$

The norm sums squared coordinate errors; implementations may normalize by coordinate count. $\theta$ is the only learned quantity in this setup: the forward schedule and chosen reverse variances remain fixed. Noise prediction is supervised by synthetic corruption, so semantic class labels are unnecessary for unconditional training.

Why not simply subtract the exact noise at generation time? We do not know it. Even during training the network sees only $(x_t,t)$, not $(x_0,\epsilon)$. Under squared error, the ideal prediction is a conditional average $\mathbb E[\epsilon\mid x_t,t]$. It cannot generally identify the exact random draw behind each ambiguous input.

A teaching identity explains the score connection in §3.2:

$$
\nabla_{x_t}\log q(x_t\mid x_0)=-\frac{\epsilon}{\sqrt{1-\bar\alpha_t}},
\qquad s_\theta(x_t,t)=-\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar\alpha_t}}.
$$

The gradient, called a score, points toward increasing log density. The first expression is the conditional Gaussian score; averaging over plausible clean images gives the marginal noisy-data score that the learned predictor approximates. This is why noise prediction supplies a useful direction in E. The scaling depends on time; the predicted noise itself is not already a score.

## 4 · Why the simple loss is not the original likelihood objective

The likelihood formulation trains reverse conditionals $p_\theta(x_{t-1}\mid x_t)$. Directly knowing the true reverse distribution is hard, but during training the original image is available. The posterior $q(x_{t-1}\mid x_t,x_0)$ is Gaussian with a known mean $\tilde\mu_t$ and variance $\tilde\beta_t$ (Eqs. 6–7). Knowing $x_0$ here is a training advantage, not an inference input.

The variational bound on expected negative log-likelihood is (Eq. 5):

$$
\begin{aligned}
\mathbb E[-\log p_\theta(x_0)]&\leq L,\\
L&=\mathbb E_q\left[L_T+\sum_{t=2}^T L_{t-1}+L_0\right],\\
L_T&=D_{\mathrm{KL}}(q(x_T\mid x_0)\|p(x_T)),\\
L_{t-1}&=D_{\mathrm{KL}}(q(x_{t-1}\mid x_t,x_0)\|p_\theta(x_{t-1}\mid x_t)),\\
L_0&=-\log p_\theta(x_0\mid x_1).
\end{aligned}
$$

$D_{\mathrm{KL}}$ compares distributions; $p(x_T)=\mathcal N(0,I)$ is the generation prior. $L_T$ measures terminal mismatch and is constant with the fixed forward schedule. The middle terms train reverse steps. $L_0$ describes the final discrete pixel decoder: Eq. 13 integrates Gaussian mass over the quantized pixel bins. A density at a point is not a discrete pixel probability.

For fixed reverse covariance $\sigma_t^2I$ and $t>1$, rewriting a middle term with the noise parameterization gives Eq. 12, up to a parameter-independent constant:

$$
L_{t-1}=\mathbb E\left[
\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}
\|\epsilon-\epsilon_\theta(x_t,t)\|^2\right]+C.
$$

Here $C$ is a constant, not the image channel count from §1. The prefactor weights timesteps differently. **The simple loss removes this prefactor.** Its $t=1$ treatment also approximates the decoder term while ignoring edge effects (§3.4). Thus it is not the identical likelihood bound with a cosmetic rewrite. Under the paper’s schedule, this change reduces the relative emphasis on near-clean reconstruction and improves the reported sample quality. The evidence below separates that from likelihood performance.

<details><summary>Predict: if two objectives use the same noise target, must they learn the same finite network?</summary><p>No. Time-dependent weights change which errors receive priority under finite capacity and optimization. Sharing a target does not make the objectives equivalent.</p></details>

## 5 · Sampling: a clean estimate is not a reverse step

[![Reverse sampling carries the current state through the shared denoiser, mean conversion and random increment, then repeats with time decreased; final noise is zero.](/figures/ddpms/sampling.svg)](/figures/ddpms/sampling.svg)

Read the loop as a state update, not a parameter update. Draw $x_T\sim\mathcal N(0,I)$, freeze the trained network, and for $t=T,\ldots,1$ apply the mean in Eq. 11 and Algorithm 2:

$$
\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right),
\qquad x_{t-1}=\mu_\theta(x_t,t)+\sigma_t z.
$$

For $t>1$, $z\sim\mathcal N(0,I)$ is freshly drawn. At $t=1$, Algorithm 2 sets $z=0$ and displays the mean. $\sigma_t$ is a standard deviation; §3.2 discusses fixed variance choices $\sigma_t^2=\beta_t$ or $\tilde\beta_t=\beta_t(1-\bar\alpha_{t-1})/(1-\bar\alpha_t)$ for the intermediate transitions. Do not confuse variance with standard deviation.

Box E first converts C’s noise prediction into a reverse mean; it does not subtract all estimated corruption in one go. The $z$ term represents conditional uncertainty. Removing it at every step changes this sampler; it is not automatically a valid fast DDPM or a derivation of DDIM.

For inspection, one can also predict the final clean image (paper Eq. 15):

$$
\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}.
$$

This estimate is useful for progressive previews and compression analysis. It is **not $x_{t-1}$**. At low signal, dividing by $\sqrt{\bar\alpha_t}$ amplifies prediction errors; the expression is undefined at exactly zero. The finite paper schedule keeps $\bar\alpha_t>0$.

## 6 · Worked example: trace one coordinate through the same boxes

[![A scalar clean pixel 0.8 becomes 0.76 after noising; prediction 0.1 gives loss 0.01, clean estimate 0.875 and a reverse state approximately 0.6823.](/figures/ddpms/worked.svg)](/figures/ddpms/worked.svg)

This is a constructed scalar example, not the paper’s 1,000-step schedule or a trained network. Choose $x_0=0.8$, $\bar\alpha_t=0.64$, $\epsilon=0.2$, and suppose C returns $\hat\epsilon=0.1$.

**A → B:** $x_t=0.8(0.8)+0.6(0.2)=0.76$. **C → D:** the squared error is $(0.2-0.1)^2=0.01$. **Clean preview:** $\hat x_0=(0.76-0.6(0.1))/0.8=0.875$.

For an intermediate reverse step choose $\beta_t=0.1$, so $\alpha_t=0.9$ and the consistent preceding cumulative value is $\bar\alpha_{t-1}=0.64/0.9\approx0.7111$. Use $\sigma_t^2=\beta_t$ and a fresh draw $z=-0.32$:

$$
\mu_\theta=\frac{0.76-(0.1/0.6)(0.1)}{\sqrt{0.9}}
\approx0.7835,
\qquad x_{t-1}=0.7835+\sqrt{0.1}(-0.32)\approx0.6823.
$$

The reverse state can move away from the original pixel on an individual random draw. Sampling does not promise monotonic pixelwise reconstruction error. If the predictor were the exact added noise, the algebraic clean estimate would recover $0.8$; an intermediate stochastic reverse sample would still not be forced to equal it.

## 7 · Over time: coarse information and fine details

[![Progressive decoding first reveals broad structure and later fine detail; the clean preview is derived from each noisy state, with separate coding and generation interpretations.](/figures/ddpms/progression.svg)](/figures/ddpms/progression.svg)

Read this as an interpretation of the paper’s Figures 5–7, not measured snapshots generated here. The authors show broad image features appearing before fine details during reverse sampling. Their progressive coding construction sends information through the same hierarchy; a receiver with a partially decoded $x_t$ can use Eq. 15 to preview the image.

On CIFAR10, §4.3 decomposes the best-quality model’s 3.75 bits/dimension bound into 1.78 rate and 1.97 distortion, with the latter corresponding to RMSE 0.95 on a $[0,255]$ scale. These are different units: 1.97 bits/dimension is not RMSE 1.97. Algorithms 3–4 assume a procedure for communicating distribution samples at approximately KL cost. This is an analysis of progressive coding, not a measured ready-to-use file codec with headers and runtime accounted for.

The analogy to autoregression is about an ordering of information: Gaussian diffusion can refine all coordinates together rather than fill one pixel coordinate at a time. It does not imply that U-Net layers use an autoregressive mask. This connects to the [neural compression guide](/papers/t2-neural-compression), where actual transmitted indices and ideal rate estimates must also be distinguished.

## 8 · Evidence: sample quality and likelihood pull differently

[![Reported unconditional CIFAR10 FID is 13.22 for mean prediction with the bound, 13.51 for noise prediction with the bound, and 3.17 for noise prediction with the simple loss.](/figures/ddpms/results.svg)](/figures/ddpms/results.svg)

The chart redraws selected **Table 2 unconditional CIFAR10** ablations; lower FID is better. Its reference distribution is the training set (§4.1). It is historical evidence for this paper’s setup, not a present-day leaderboard.

| Parameterization and objective | FID ↓ | Inception score ↑ | Test NLL bound, bits/dim ↓ |
| --- | ---: | ---: | ---: |
| Posterior-mean prediction, variational bound, fixed covariance | 13.22 | 8.06 ± 0.09 | Not reported here |
| Noise prediction, variational bound, fixed covariance | 13.51 | 7.67 ± 0.13 | ≤ 3.70 |
| Noise prediction, simple loss | **3.17** | **9.46 ± 0.11** | ≤ 3.75 |

FID and IS are from Table 2; NLL bounds for the two noise-prediction rows are from Table 1. The inequality matters: the reported variational bound is not exact negative log-likelihood. Noise prediction by itself does not beat mean prediction in the fixed-covariance bound comparison. The strongest result uses noise prediction **and** reweighting. FID improves from 13.51 to 3.17 while the NLL bound worsens from 3.70 to 3.75.

Table 1’s unconditional StyleGAN2 + ADA (v1) comparison has FID 3.26 and IS 9.74 ± 0.05: DDPM’s lower FID does not mean it wins both metrics. The paper also reports DDPM FID **5.24 against the CIFAR10 test set**, so 3.17 and 5.24 must not be mixed as if they used the same reference. The main text does not specify the uncertainty protocol for the displayed IS ± values, so we do not call them training-seed confidence intervals.

Figures 3–4 report LSUN Church and Bedroom $256\times256$ FIDs of 7.89 and 4.90. Those are different datasets, not additional CIFAR10 ablations. The supplied copy omits the appendix’s further implementation and evaluation details.

**What remains open?** All paper experiments use 1,000 reverse network evaluations; the results do not establish cheap one-call generation. Learned reverse variances were unstable in some tested configurations (§4.2), not proven universally unsuitable. To isolate weighting effects further, hold architecture, data, compute and sampling fixed, repeat seeds and report both FID and likelihood bounds. To claim acceleration, measure wall-clock cost and quality together. Those are proposed tests, not extra paper findings.

<details><summary>Predict: does the best FID row also have the best likelihood bound?</summary><p>No. The simple loss has FID 3.17 but a 3.75 bits/dimension bound, compared with 13.51 and 3.70 for the original bound objective. Sample quality and codelength are different criteria.</p></details>

## 9 · The idea to carry forward

**Make a hard generative task learnable by defining an easy corruption process, learning its local reversal, and composing those reversals.** Keep four objects separate: the noise target, the clean-image estimate, the reverse transition mean, and the sampled next state.

Now revisit [DiffAtlas](/papers/t2-paper-4): it extends the state to image–mask pairs and constrains part of it during inference. Compare [Mean Flows](/papers/t2-paper-2), which targets large interval updates, and [Flow Matching](/papers/t2-flow-matching), which learns a velocity along a path. Their time conventions and prediction targets need not match DDPM’s.

## Sources and reading map

- [Exact 12-page course PDF](/papers/ddpms.pdf): §2 / Eqs. 1–7 for the two chains and Gaussian posterior; §3.2 / Eqs. 8–12 for the mean/noise connection; Algorithms 1–2 for training and sampling.
- §3.3 / Eq. 13 explains the discrete decoder; §3.4 / Eq. 14 explains the simplified loss. §4 gives the schedule and main-text backbone specification.
- Tables 1–2 and §4.1–4.2 contain the evidence; §4.3 / Eq. 15 and Figures 5–7 explain progressive coding and generation. §4.4 explores interpolation. Referenced appendices are not included in the supplied file.
- [Official NeurIPS 2020 record](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html) confirms source identity. All schematic calculations are labeled teaching examples, separately from reported results.
