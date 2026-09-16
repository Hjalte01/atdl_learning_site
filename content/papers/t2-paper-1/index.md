---
title: "Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think"
shortTitle: "REPA"
topic: "topic-2"
order: 1
description: "Why teach a noisy-image generator with a clean-image encoder? Follow REPA’s training-only alignment through tensors, losses, and evidence."
status: "ready"
pdf: "/papers/repa.pdf"
---
## The puzzle: why must a generator learn recognition from scratch?

Imagine generating a bird from noise. The network must learn both what a bird is and how its feathers should look. Predicting a denoising target can eventually teach both, but a separate visual encoder may already have much better features for recognizing objects. Can we borrow those features without asking that encoder to generate pixels?

**REPA adds a training loss that makes an early hidden representation of a noisy image resemble a frozen encoder’s representation of the corresponding clean image.** The generator still learns its original denoising task. At sampling time, the extra encoder and projection head are unnecessary.

This guide follows the supplied **43-page ICLR 2025 conference PDF**, whose PDF creation metadata is dated **25 March 2025**. That is a file metadata date, not a claimed arXiv version or publication date. All numerical results below come from this course copy. Allow about 20 minutes. Our diagrams and toy calculations are teaching constructions; the results chart transcribes measured values from the paper.

## 1 · Two jobs hiding inside denoising

[![Ordinary training learns semantics and details through the generation loss; REPA adds a clean-feature teacher to an early representation.](/figures/repa/problem.svg)](/figures/repa/problem.svg)

Read the upper route as the original generator and the lower route as the extra supervision. Both routes retain a generation objective. REPA changes how the hidden representation is learned, rather than replacing generation with image classification.

The authors first measure a **semantic gap** (§3.2, Figure 2): a linear classifier on frozen SiT features performs worse than one on DINOv2 features. They also measure representational alignment using CKNNA, a neighborhood-based comparison of representations. These are diagnostic measurements, not the loss used for the main cosine-alignment experiments.

A plausible design argument is: help early layers organize useful visual information, then let later layers turn it into a detailed denoising prediction. This is our interpretation of the design, not the authors’ private reasoning or a proof that semantic learning is the only bottleneck.

## 2 · Training architecture: two views of the same example

[![Training branches from a clean RGB image into a frozen VAE and noisy SiT stream, and into a frozen visual encoder; an early hidden-state projection is compared patch by patch with clean features.](/figures/repa/training.svg)](/figures/repa/training.svg)

Follow boxes A–F. **A:** a clean training image $x_*$. **B:** the pretrained VAE encoder $E$ converts it to latent $z=E(x_*)$. **C:** known Gaussian noise creates $z_t$. **D:** the transformer receives this noisy latent, time $t$, and the class condition $c$. An intermediate state $h_t$ feeds both the remaining transformer blocks and a projection MLP. **E:** the final prediction gets the usual velocity loss. **F:** the projected intermediate features get the alignment loss against a frozen clean-image encoder $f$.

The two encoders have different jobs: $E$ supplies a compact latent for image generation; $f$ supplies the semantic training target $y_*=f(x_*)$. Neither is the generator’s early transformer blocks, which we denote $f_\theta$. The paper writes the generator functionally as $g_\theta\circ f_\theta$: these are two portions of one transformer, not a separately pretrained encoder–decoder pair.

For the ImageNet 256×256 setup (§4.1, Table 1, Appendix D):

| Stage | Shape or documented configuration |
| --- | --- |
| Clean image | $256\times256\times3$ |
| VAE latent | $32\times32\times4$ |
| Transformer patch size | $2\times2$ latent positions |
| Patch sequence | $N=16\times16=256$ tokens (derived from the dimensions above) |
| SiT-XL/2 | 28 blocks, hidden width 1152, 16 attention heads |
| Alignment branch in XL experiments | Hidden state at depth 8 → three-layer MLP with SiLU → teacher feature width $D$ |
| Main XL teacher / loss | DINOv2-B; cosine similarity; $\lambda=0.5$ |

Time and class conditioning modulate the transformer blocks; the blocks contain attention and pointwise feedforward processing (Appendix B, Figure 9). The MLP maps feature widths so comparison is meaningful. Teacher patch layout must also correspond to the generator’s spatial grid; Appendix D describes interpolating teacher positional embeddings to accommodate patch counts. Matching tensor shape alone would not justify comparing unrelated image locations.

These details are setting-specific. For example, Table 7 uses depth 4 for SiT-B and NT-Xent with DINOv2-L for the SiT-L system comparison. “Every REPA experiment aligns at depth 8 with the same teacher” would be incorrect.

## 3 · Where the generation mathematics happens

Before adding a teacher, the model needs a generation task. The **B → C corruption arrow** uses the latent version of paper Eq. 1:

$$
z_t=\alpha_t z+\sigma_t\epsilon,\qquad \epsilon\sim\mathcal N(0,I).
$$

$z$ is the clean VAE latent, $z_t$ the noisy latent, $\epsilon$ a tensor of independent standard Gaussian noise, and $I$ the identity covariance. The SiT experiments use $t\in[0,1]$, $\alpha_t=1-t$, and $\sigma_t=t$. Thus $t=0$ is clean and $t=1$ is noise. These are amplitude coefficients; their squared values do not generally sum to one.

The **E prediction box** learns the rate of change along these constructed paths. With dots denoting time derivatives, Eq. 4 becomes:

$$
\mathcal L_{\mathrm{velocity}}=
\mathbb E_{x_*,\epsilon,t,c}
\left[\left\|v_\theta(z_t,t,c)-(\dot\alpha_t z+\dot\sigma_t\epsilon)\right\|^2\right].
$$

$v_\theta$ is the transformer’s predicted latent velocity and $\theta$ its weights. For the linear path, the target is $\epsilon-z$. We explicitly show class condition $c$, which the paper suppresses in its generic equations. The expectation averages over examples, sampled noise and times, with each example’s condition. The loss compares known manufactured targets with predictions; it does not require knowing the true distribution’s velocity in advance.

Why is this useful? Across many paths, squared-error regression learns their conditional average velocity at a given noisy state (paper Eq. 3). At inference that learned field guides the evolving latent. A single sample’s target $\epsilon-z$ is not a universally correct velocity for every path passing through that state.

REPA also applies to DiT’s diffusion objective. It adds feature supervision to the existing objective; velocity prediction is the SiT case used for this walkthrough.

## 4 · Representation close-up: compare patches, not class labels

[![Two spatial patch grids match corresponding positions, while each projected student vector is compared by direction with a teacher vector.](/figures/repa/patches.svg)](/figures/repa/patches.svg)

Each cell is a feature vector, not an RGB pixel or a predicted class. The diagram shows four toy patches; the example architecture above uses 256. A patch feature can contain context from other patches because attention mixes information.

At **F**, the frozen teacher provides $y_*\in\mathbb R^{N\times D}$. The early generator state $h_t\in\mathbb R^{N\times d}$ has hidden width $d$. A trainable projection $h_\phi$ maps it to $p=h_\phi(h_t)\in\mathbb R^{N\times D}$, where $\phi$ denotes MLP weights. Paper Eq. 8 rewards agreement at matching patch index $n$:

$$
\mathcal L_{\mathrm{REPA}}=-\mathbb E_{x_*,\epsilon,t}
\left[\frac{1}{N}\sum_{n=1}^{N}\operatorname{sim}(y_*^{[n]},p^{[n]})\right].
$$

For the cosine version,

$$
\operatorname{sim}(a,b)=\frac{a^\top b}{\|a\|\,\|b\|}.
$$

Here $a,b$ are nonzero feature vectors, $a^\top b$ their dot product, and $\|\cdot\|$ Euclidean length. Cosine compares direction rather than magnitude. A practical implementation needs safe normalization around zero vectors; the toy vectors below are nonzero. Minimizing the negative similarity maximizes agreement. The loss can therefore be negative without being broken.

Why use the clean target? The intended signal is stable semantic information despite input corruption. The teacher is frozen: gradients through the projection update $\phi$ and the generator blocks upstream of $h_t$, but do not update $f$. Later blocks receive the generation loss; the alignment branch does not directly backpropagate through blocks after its attachment point.

The clean target is available during training only. At high noise, it may be impossible to determine which training image produced a particular input. Alignment is a statistical training pressure, not a guarantee of perfect feature recovery from pure noise.

<details><summary>Predict: does multiplying a projected feature by 10 improve cosine alignment?</summary><p>No, for a positive scale and nonzero vectors the scale cancels between numerator and denominator. Direction must improve. Nor does cosine alignment make the hidden state identical to the teacher: it constrains a learned projection.</p></details>

## 5 · Trace a numerical example through the same boxes

[![Toy training example traces clean latent 2 and noise 6 to noisy value 3, then combines velocity error 1 with cosine alignment loss minus 0.8.](/figures/repa/numbers.svg)](/figures/repa/numbers.svg)

This is one synthetic latent coordinate and one synthetic two-dimensional feature. It is not a trained model or an ImageNet result.

At **B**, take $z=2$. At **C**, choose $t=0.25$ and a deliberately simple noise realization $\epsilon=6$:

$$
z_t=0.75(2)+0.25(6)=3,\qquad v_{\mathrm{target}}=6-2=4.
$$

At **E**, suppose $v_\theta=3$. The scalar squared error is $(3-4)^2=1$. Independently, suppose the **D → F** projection produces $p=(3,4)$ while the teacher supplies $y_*=(0,2)$:

$$
\operatorname{sim}(y_*,p)=\frac{0(3)+2(4)}{2\sqrt{3^2+4^2}}=0.8,
\qquad \mathcal L_{\mathrm{REPA}}=-0.8.
$$

Paper Eq. 9 combines the two pressures:

$$
\mathcal L=\mathcal L_{\mathrm{velocity}}+\lambda\mathcal L_{\mathrm{REPA}}.
$$

$\lambda>0$ weights alignment relative to generation. With $\lambda=0.5$, this toy loss is $1+0.5(-0.8)=0.6$. If the projected direction becomes perfect while velocity stays unchanged, the total becomes $0.5$. That improvement in total loss does **not** mean the velocity prediction became more accurate. Track both components.

Setting $\lambda=0$ recovers the original objective; raising it increases the pressure to match the teacher. Table 5 tests 0.25–1.0 for SiT-XL/2 at 400K iterations and reports FID 8.6, 7.9, 7.8, 7.8 respectively. That is evidence of saturation in this setting, not a rule that a larger coefficient is always better.

## 6 · Sampling: carry the latent forward, leave the teacher behind

[![Inference starts from Gaussian latent noise, repeatedly applies the trained transformer and sampler, and ends with VAE decoding; the teacher and alignment head are absent.](/figures/repa/sampling.svg)](/figures/repa/sampling.svg)

Read the loop from high $t$ toward zero. Initialize $z_1$ with Gaussian noise, supply a desired class $c$, evaluate the trained transformer, and use a numerical sampler to move to the next time. Carry the updated latent into the next iteration. After sampling, the VAE decoder turns the latent into an image. All network weights are fixed during generation.

To understand the time direction, a **teaching-only ODE Euler step** is:

$$
z_{t-\Delta t}\approx z_t-\Delta t\,v_\theta(z_t,t,c),\qquad \Delta t>0.
$$

$\Delta t$ is the positive step magnitude; the minus sign reflects integration toward smaller $t$. With the toy $z_t=3$, $v_\theta=3$, and $\Delta t=0.1$, the next latent is $2.7$. The model must be evaluated again at the new state and time. Repeating a constant velocity is not the actual algorithm.

The paper’s default reported ImageNet results instead use an **SDE Euler–Maruyama sampler**, with 250 function evaluations, diffusion coefficient $w_t=\sigma_t$, and a special last step described in Appendix D. Its update also involves the score and stochastic noise (paper Eqs. 5–7). Our ODE identity explains direction only; it does not reproduce those evaluation results.

<details><summary>Predict: must each generated sample pass through DINOv2?</summary><p>No. DINOv2 supplied training targets. The transformer has learned from them; sampling uses the trained generator and VAE decoder. REPA does not require a clean reference image at inference and does not itself reduce the number of sampling evaluations.</p></details>

## 7 · Why align early instead of everywhere?

[![Depth ablation shows depth 8 has FID 10.0 and probe accuracy 68.1, while depth 16 has worse FID 12.1 despite higher accuracy 71.1.](/figures/repa/depth.svg)](/figures/repa/depth.svg)

The diagram combines a schematic early-to-late network with **measured Table 2 endpoints**. In this controlled ablation, SiT-L/2 trains for 400K iterations with DINOv2-L, NT-Xent and $\lambda=0.5$. Depth 8 gives FID 10.0 and linear-probe accuracy 68.1%; depth 16 gives FID 12.1 and accuracy 71.1%. Lower FID is better; higher accuracy is better.

So stronger linear classification at the aligned layer does not automatically mean better generation. The authors hypothesize that later layers need freedom to represent fine details. The comparison supports an early alignment choice in this setup; it does not prove a universal semantic/detail division or establish layer 8 as optimal for every architecture.

Table 2 also compares alignment objectives at depth 8: NT-Xent gives FID 10.0 and cosine similarity 9.9. NT-Xent is a contrastive alternative; the simple cosine equation above is not its definition. This comparison motivates the cosine choice for later experiments without claiming the losses are identical.

## 8 · What the speedup actually measures

[![Measured ImageNet unguided FID bars compare vanilla SiT-XL/2 at 400K and 7M iterations against REPA at 400K, 1M and 4M.](/figures/repa/results.svg)](/figures/repa/results.svg)

Read bar length as **FID-50K**, not time. FID compares generated and reference image distributions using Inception features; lower is better. It is neither a percent-correct score nor a guarantee about a particular generated image.

The course PDF’s Table 3 reports these class-conditional **ImageNet 256×256, SiT-XL/2, no classifier-free guidance** results. Evaluation uses 50,000 samples and the default 250-evaluation SDE sampler:

| Method | Training iterations | FID ↓ |
| --- | ---: | ---: |
| Vanilla SiT-XL/2 | 400K | 17.2 |
| Vanilla SiT-XL/2 | 7M | 8.3 |
| SiT-XL/2 + REPA | 400K | 7.9 |
| SiT-XL/2 + REPA | 1M | 6.4 |
| SiT-XL/2 + REPA | 4M | 5.9 |

$7{,}000{,}000/400{,}000=17.5$: REPA reaches a better reported FID with 17.5 times fewer training iterations. This is not a measured 17.5× wall-clock or sampling speedup. The training-only encoder has a cost, and its pretraining is external work. Appendix D reports the authors’ compute setup but does not turn this iteration ratio into a hardware-independent time guarantee.

Keep the headline **FID 1.42** separate. Table 4 uses classifier-free guidance plus a guidance interval, at 800 training epochs. The corresponding REPA result with ordinary guidance is 1.80, while vanilla SiT-XL/2 at 1400 epochs is 2.06. These guided results are not interchangeable with the unguided 7.9 row. They were strong historical results in this paper, not a claim about today’s best model.

The version also contains ImageNet 512×512 experiments (Appendix J) and MS-COCO text-to-image experiments (Appendix K). The latter reports MMDiT FID 6.05 → 4.73 with an ODE at NFE 50, and 5.30 → 4.14 with an SDE at NFE 250, both at guidance scale 2.0. Those extensions support broader usefulness without establishing performance on arbitrary datasets or large-scale text-to-image training.

## 9 · Limits and experiments worth doing

The practical dependency is a useful pretrained teacher. Its features reflect its training data and objective; alignment can emphasize what the teacher recognizes and underemphasize what it ignores. Appendix M explicitly leaves alignment-depth theory, other data types, and time-varying alignment weighting as future work.

Useful next experiments, proposed here rather than reported findings:

- **Count total compute:** compare wall-clock and accelerator time at matched FID, including online teacher inference and accounting separately for teacher pretraining.
- **Test the depth explanation:** hold architecture, teacher, seed budget and optimization fixed while changing the attachment point; measure both semantic probes and fine-detail errors.
- **Test noise-dependent weighting:** compare constant $\lambda$ with a time-dependent schedule, especially near pure noise where image-specific information is absent.
- **Test teacher mismatch:** use an out-of-domain dataset and inspect failure cases as well as aggregate FID. Repeat with multiple seeds to assess variability.

## 10 · The idea to carry forward

**A generative objective can benefit from a representation target that is easier to learn from than raw reconstruction alone.** REPA transfers that target through an auxiliary loss into early generator features. The original objective still teaches generation, and later layers still have work to do.

Redraw the two training branches, cross out the teacher and projection for inference, and label which loss updates which weights. If you can explain why better probe accuracy can coexist with worse FID, you understand the key tradeoff.

## Sources and reading map

- [Supplied REPA course PDF, ICLR 2025 conference version, 43 pages](/papers/repa.pdf). PDF metadata creation date: 25 March 2025. The local file is the authority for this guide; no arXiv revision is inferred from that date.
- §§3.1–3.2 and Figures 2–3: representation bottleneck hypothesis and diagnostics.
- §2, Eqs. 1–7: corruption, velocity objective and sampling; §3.3, Eqs. 8–9: alignment and combined loss.
- §4.1, Table 1, Appendices B/D, Figure 9 and Table 7: architecture and implementation.
- Table 2: teacher, depth and objective ablations; Tables 3–5: unguided, guided and coefficient comparisons.
- Appendices J/K/M: resolution and text-to-image extensions, limitations and future work.

All diagrams can be opened at full size. Schematic paths, toy arithmetic, measured results and proposed experiments are explicitly distinguished above.
