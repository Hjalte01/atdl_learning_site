---
title: "Qwen-Image Technical Report"
shortTitle: "Qwen-Image"
topic: "topic-2"
order: 6
description: "Trace text rendering and image editing through dual conditioning, latent flow matching, a text-aware decoder, and a progressive data curriculum."
status: "ready"
source: "https://arxiv.org/abs/2508.02324v1"
pdf: "/papers/qwen-image.pdf"
---
## 1 · The puzzle: a beautiful poster with the wrong words

Ask for a café poster that says **OPEN AT NINE**, then ask to change the lettering while preserving the cup, lighting, and layout. A plausible poster can still spell the text incorrectly. An editor can understand the requested change yet redraw details you wanted to keep. These are different failures: precise symbol rendering, instruction following, and visual preservation.

Qwen-Image tackles the whole route from training examples to decoded pixels. Its condition encoder understands the request; a flow model generates image latents; a specialized VAE decoder reconstructs fine details. For editing, the reference image supplies both semantic features and reconstructive features.

[![Three requirements for a poster: correct symbols, correct requested edit, and preserved visual details.](/figures/qwen-image/problem.svg)](/figures/qwen-image/problem.svg)

Read the three boxes as separate tests of success. This is a teaching scenario, not a generated sample or measured paper result. Click any diagram to enlarge it.

**Source boundary.** This guide follows the supplied 46-page course PDF: arXiv **2508.02324v1**, with an arXiv stamp of **4 August 2025** and a cover date of **5 August 2025**. The [version-specific primary record](https://arxiv.org/abs/2508.02324v1) confirms the submission identity. All results below are historical comparisons from that version, not rankings of current products. [Read the course PDF](/papers/qwen-image.pdf).

**A plausible design rationale, interpreted rather than attributed:** if the data rarely teaches a character, the model cannot reliably learn it; if compression and decoding blur its strokes, understanding the word alone cannot recover them; if image conditioning describes only meaning, an editor may lose the original details. This explains why the report changes several parts of the system together.

## 2 · Architecture: follow the information

The base text-to-image path has three main components (§2, Figure 6, Table 1):

- **A — Qwen2.5-VL condition encoder:** a 7B vision-language model supplies last-layer language-backbone hidden states, denoted $h$. Task-specific system prompts organize the input. These are feature vectors, not a separate generated caption that is then rendered by a font engine.
- **B — VAE image representation:** the encoder compresses spatial dimensions by $8\times8$ with 16 latent channels. The image decoder converts the final latent back to pixels.
- **C — MMDiT velocity predictor:** a 20B model with 60 blocks jointly processes conditioning and image tokens. Table 1 reports 24 attention heads of size 128 and image patch factor 2.

[![Prompt goes through Qwen2.5-VL to condition a 60-block MMDiT. A noisy latent is patched, updated using predicted velocity, and decoded by the VAE.](/figures/qwen-image/architecture.svg)](/figures/qwen-image/architecture.svg)

Follow the green image-state path through C repeatedly, then decode at D. The violet condition path tells C what to generate. This functional schematic omits block-internal normalization and projection details; it does not invent additional layers.

For an illustrative **256 × 256** image, B produces a **32 × 32 × 16** latent. Grouping 2 × 2 latent cells gives **16 × 16 = 256 image tokens**, each containing $2\cdot2\cdot16=64$ latent values before the learned embedding. This is a dimension calculation from Table 1, not a claim that every training image has this size. Actual training supports multiple resolutions and aspect ratios.

The condition stream and image stream can interact inside MMDiT, but the generated object is the image latent. The model is not predicting the poster’s letters as an autoregressive string of output text tokens.

## 3 · Representation: why the decoder matters

A correct instruction can still yield illegible pixels. Before studying generation, ask whether the VAE can reconstruct small text when it receives an encoding of the correct image.

The report uses the Wan-2.1-VAE architecture with a **shared, frozen encoder** and separate image/video decoders. It fine-tunes the **image decoder** on text-rich real documents and synthetic paragraphs (§2.3). It uses reconstruction and perceptual losses, adjusting their relative weight; it reports that adversarial loss stopped providing useful guidance as reconstruction improved. This describes the VAE adaptation stage, not a claim that every component is jointly trained in every later stage.

[![A 256-square image compresses to a 32 by 32 by 16 latent, then becomes 256 image patches. The frozen encoder and adapted image decoder have different training roles.](/figures/qwen-image/representation.svg)](/figures/qwen-image/representation.svg)

The upper row traces the sizes; the lower row identifies the reconstruction bottleneck. Dimensions are a teaching example derived from Table 1. No actual pixels or measured reconstructions are displayed.

Table 2 tests float32 VAE reconstruction at 256 × 256. On the report’s **in-house text-rich corpus**, Wan2.1-VAE has PSNR **26.77 dB** and SSIM **0.9386**, while Qwen-Image-VAE has **36.63 dB** and **0.9839**. ImageNet-1k validation PSNR improves from **31.29** to **33.42 dB**. Higher is better for these reconstruction metrics. These measurements support improved reconstruction; they do not establish spelling accuracy for newly generated prompts. The in-house text corpus also limits independent reproduction.

<details><summary>Predict: does better VAE reconstruction guarantee correct spelling?</summary>
No. The decoder can preserve a correctly formed latent more faithfully, but the generator still has to construct the right latent from the request. A sharp rendering of the wrong character is still wrong.
</details>

## 4 · Where the flow mathematics happens

The training task teaches C the direction from a noisy latent toward a clean training latent. To remove a notation ambiguity in §4.1, call the clean image $I$, its VAE encoding $z=E(I)$, and sampled Gaussian noise $\epsilon$. The paper calls the clean latent $x_0$ and noise $x_1$, **but its Equation 1 places noise at time 0 and data at time 1**:

$$
x_t=t z+(1-t)\epsilon,\qquad v^*=\frac{d x_t}{dt}=z-\epsilon.
$$

Here $t\in[0,1]$ is flow time, $x_t$ is the corrupted image latent entering C, and $v^*$ is its training velocity target. The report samples pre-training times from a logit-normal distribution. Interpolation is elementwise across the entire latent tensor. At $t=0$, $x_t=\epsilon$; at $t=1$, $x_t=z$. This convention reverses the direction used in the [REPA guide](/papers/t2-paper-1).

C receives $x_t$, time $t$, and conditioning $h=\phi(S)$, where $S$ is the user input and $\phi$ is the Qwen2.5-VL feature extractor. Equation 2 fits its predicted velocity:

$$
\mathcal L_{\mathrm{FM}}=\mathbb E_{(z,h),\epsilon,t}\left[\|v_\theta(x_t,t,h)-(z-\epsilon)\|_2^2\right].
$$

$\theta$ denotes the learned velocity-model parameters; the expectation averages training pairs, noises, and times; $\|\cdot\|_2^2$ sums squared component errors. This loss updates the predictor’s weights. It is not an OCR accuracy loss or a pixel-preservation constraint. A low average velocity error does not guarantee every character is correct.

[![Training encodes the target, mixes it with noise, predicts velocity, compares to clean latent minus noise, and updates model weights. Inference holds weights fixed and repeatedly updates the target state.](/figures/qwen-image/training.svg)](/figures/qwen-image/training.svg)

Read the upper row as supervised learning with a known target and the lower row as generation without that target. The diagram is schematic. During inference, model parameters are fixed; only the image state evolves.

For intuition, an Euler step follows the predicted direction:

$$
x_{t+\Delta t}=x_t+\Delta t\,v_\theta(x_t,t,h),\qquad \Delta t>0.
$$

This is a **teaching discretization** of the flow ODE, not the report’s complete production sampler or a specified sampling schedule. The same state is carried forward and re-evaluated. Pairwise training paths are straight, but the learned field can create curved generated trajectories because many training pairs contribute to its predictions.

## 5 · Trace a numerical example

Use one scalar latent coordinate, $z=2$, $\epsilon=10$, and $t=0.25$. The VAE and condition encoder are assumed to have already supplied B and A; no model runs in this example.

The input to C is $x_t=0.25(2)+0.75(10)=8$. Its correct training velocity is $v^*=2-10=-8$. If C predicts $-7$, this coordinate contributes $(-7-(-8))^2=1$ to the squared-error loss.

At inference, an illustrative step of $\Delta t=0.25$ gives $8+0.25(-7)=6.25$. An oracle velocity of $-8$ would instead give $6$, the exact interpolant at $t=0.5$. The difference is a local example of prediction error, not a global error bound.

[![Scalar values through the architecture: clean 2 and noise 10 make state 8 at time 0.25; predicted minus 7 versus target minus 8 gives loss 1 and Euler output 6.25.](/figures/qwen-image/numbers.svg)](/figures/qwen-image/numbers.svg)

The boxes reuse A–C from the architecture and training paths. All values are synthetic. A single coordinate is not a pixel brightness or character probability, and the final VAE decode is not represented by a scalar identity.

[![An oracle illustrative path runs from 10 at time zero through 8, 6, and 4 to 2 at time one, carrying the latent forward while conditioning remains fixed.](/figures/qwen-image/steps.svg)](/figures/qwen-image/steps.svg)

Read left to right: the state progresses toward data while the prompt features remain fixed. Four uniform oracle steps illustrate the sign and endpoints; neither this step count nor the constant velocity is an experimental sampling claim.

<details><summary>Predict: should we subtract the negative velocity to denoise?</summary>
No. Under this report’s time convention we advance time and add Δt times velocity. Here adding a negative velocity moves 10 toward 2. Copying a backward-time update from another guide would move in the wrong direction.
</details>

## 6 · Positions: give text and image tokens a useful geometry

MMDiT needs both image locations and text order. Simply placing text along one image row can give the modalities confusingly similar positional structure. **Multimodal Scalable RoPE (MSRoPE)** centers image positions and gives each text token equal position IDs on both spatial axes (§2.4, Figure 8).

[![Image positions form a centered 3 by 3 grid while successive text token positions lie diagonally at 2,2; 3,3; and 4,4. Editing adds a separate frame coordinate.](/figures/qwen-image/positions.svg)](/figures/qwen-image/positions.svg)

The tiny grid adapts the paper’s Figure 8. Text positions $(2,2),(3,3),(4,4)$ preserve one ordered sequence across both axes; they are **positional labels**, not instructions to draw words diagonally on the output. Image positions retain two-dimensional structure for resolution scaling. The actual latent grid is much larger.

For editing, an additional **frame coordinate** distinguishes reference-image tokens from target-image tokens (§4.3, Figure 14). It identifies which image a token belongs to; it is distinct from the flow time $t$ that describes the target’s noise level. Positional encoding supports the model’s learning—it does not enforce exact lettering or perfect edits by itself.

## 7 · Editing: one reference, two encodings

Suppose the request is to change the lettering on the café poster. A semantic representation can recognize the poster and interpret the edit, but may not preserve the exact cup texture. A reconstructive representation retains appearance information, but does not alone explain what should change.

Qwen’s editing variant supplies both (§4.3): the original image and instruction go into Qwen2.5-VL to produce semantic conditioning; the original image also goes through the VAE encoder, and its clean latent tokens are concatenated with the noisy target along the **sequence dimension** of the image stream.

[![Reference image feeds Qwen2.5-VL with the instruction for semantic conditioning, and separately feeds a VAE encoder for reconstructive tokens. Both condition the evolving noisy target in MMDiT.](/figures/qwen-image/editing.svg)](/figures/qwen-image/editing.svg)

Follow the two branches into C. The orange reference latent is conditioning, while the green target latent is the state being generated. The diagram is functional and omits embedding projections. Sequence concatenation does not mean replacing the target with the reference or adding their pixels together.

During editing training, a clean **target** image provides the flow target; it is separate from the **reference** image that conditions the request. During sampling, the clean target is unknown. Keep reference features and model weights fixed while repeatedly updating the target state, then decode that state. Fixed reference conditioning does not make the output outside the edit region mathematically invariant.

The introduction and abstract describe joint **T2I, I2I reconstruction, and TI2I** learning to align understanding and generation. Section 4.3 explains the dual input paths but does not give a complete per-task mixture schedule or a standalone I2I loss recipe. Treat the alignment explanation as the reported design intent; do not invent task weights or read it as a controlled ablation.

<details><summary>Predict: if the edited image is identical to the reference, is the system successful?</summary>
Only if no change was requested. Preservation must be evaluated together with instruction following. An unchanged poster preserves every detail but fails a lettering-change request.
</details>

## 8 · Training text: coverage before polish

The report’s text capability also depends on what it learns from (§3 and §4.1.3). Its collection spans billions of image–text pairs, with approximate category proportions of 55% nature, 27% design, 13% people, and 5% synthetic material. These describe the collection, not a fixed mixture at every optimization step.

Captions include visible text transcriptions, object attributes, and spatial relationships. Structured metadata helps filter and balance examples. Synthetic text here means **controlled rendering**, rather than blindly using another image generator’s potentially misspelled output.

Three synthesis strategies teach complementary skills: clean-background paragraphs provide character coverage; text composited into scenes teaches appearance in context; structured templates such as slides teach layout. In the pure-rendering pipeline, a paragraph is discarded if any character cannot be rendered correctly. This is a data-quality rule, not a guarantee about model outputs.

[![Training curriculum increases resolution from 256 to 640 to 1328 while text synthesis progresses from simple paragraphs to contextual scenes and structured templates.](/figures/qwen-image/curriculum.svg)](/figures/qwen-image/curriculum.svg)

Read the two rows as complementary progressions. This schematic combines §3.4 and §4.1.3; it does not assert that each synthesis type starts at exactly the resolution aligned above it. Multiple aspect ratios are used. Filtering becomes stricter, distributions become more balanced, and synthetic data fills gaps such as rare characters and long text.

After pre-training, supervised fine-tuning uses selected, human-annotated examples to improve quality. Preference training then refines the model (§4.2). The producer–consumer infrastructure (§4.1.1–2) separates preprocessing such as VAE encoding from distributed model training; it is a scaling mechanism rather than a new generation objective.

## 9 · Preference learning: what the extra objectives change

**DPO** compares a preferred and rejected image for the same conditioning. It changes the velocity model’s relative fit to those examples compared with a reference model. A compact rewrite of paper Equation 3 is:

$$
D_\theta=e_{\theta,w}-e_{\theta,l},\qquad
\ell_{\mathrm{DPO}}=-\log\sigma\bigl(-\beta(D_\theta-D_{\mathrm{ref}})\bigr).
$$

Here $w$ and $l$ denote the preferred and rejected samples; each $e$ is that sample’s squared flow-velocity error; $D_{\mathrm{ref}}$ uses a fixed reference predictor; $\beta>0$ sets the preference scale; and $\sigma$ is the sigmoid. The full loss averages pairs and sampled times. This operates at C’s training loss, not as an inference-time image filter.

For a toy pair, errors $e_{\theta,w}=1$ and $e_{\theta,l}=4$ give $D_\theta=-3$. If $D_{\mathrm{ref}}=-1$ and $\beta=1$, the sigmoid argument is $2$, and the loss is about **0.127**. Lower preferred error relative to rejected error improves the preference objective; it does not ensure every output becomes better by every metric.

**GRPO** then uses groups of generated trajectories and reward-model scores for a smaller refinement stage. Equation 4 normalizes each image’s reward within its prompt group:

$$
A_i=\frac{R_i-\operatorname{mean}(R_1,\ldots,R_G)}{\operatorname{std}(R_1,\ldots,R_G)}.
$$

$G$ is group size, $R_i$ is the reward of sample $i$, and $A_i$ is its relative advantage. With toy rewards $(1,2,3)$ and population standard deviation $\sqrt{2/3}$, advantages are approximately $(-1.225,0,1.225)$. The paper does not specify a zero-variance safeguard in this equation: if all rewards tie, this expression alone is undefined.

Equation 5 combines these advantages with clipped policy probability ratios and a reference KL penalty. Equations 6–8 introduce stochastic trajectory sampling for exploration and its associated KL expression. They describe **post-training**, not an instruction to add fresh reference noise to ordinary editing. This guide does not implement that RL sampler; the explicit $1/t$ terms require endpoint handling beyond substituting $t=0$ into the displayed equations. Preference judgments and reward models also introduce their own evaluation biases.

## 10 · Evidence: strong results with visible boundaries

The following values come from the supplied v1 tables. Separate character-level text, long text, general generation, and editing: a single aggregate does not measure them all.

[![ChineseWord character accuracy bars: Qwen-Image 97.29 percent at Level 1, 40.53 at Level 2, and 6.48 at Level 3. More difficult tiers still show substantial failures.](/figures/qwen-image/results.svg)](/figures/qwen-image/results.svg)

This is a measured chart redrawn from **Table 9**, with a zero-based 0–100% axis. ChineseWord is the authors’ benchmark: Level 1 contains 3,500 characters, Level 2 contains 3,000, and Level 3 contains 1,605, prompted as single-character images. High common-character accuracy coexists with only **6.48%** on Level 3.

| Evaluation and setting | Qwen-Image | Comparison in the same table | What it supports |
| --- | --- | --- | --- |
| ChineseWord overall accuracy, Table 9 | 58.30% | Seedream 3.0: 33.05%; GPT Image 1 [High]: 36.14% | Better reported character rendering, with substantial rare-character failures |
| CVTG-2K average English word accuracy; 2K prompts, 2–5 text regions, Table 8 | 0.8288 | GPT Image 1 [High]: 0.8569 | Strong English rendering, but not the best word accuracy here |
| LongText-Bench EN / ZH; 160 prompts across eight scenarios, Table 10 | 0.943 / 0.946 | GPT Image 1 [High]: 0.956 / 0.619; Seedream 3.0: 0.896 / 0.878 | The relative ranking depends on language |
| GenEval overall, Table 4 | 0.87; RL variant 0.91 | Seedream 3.0 and GPT Image 1 [High]: both 0.84 | Reported improvement after RL; these are distinct Qwen variants |
| GEdit full-set EN / CN overall, GPT-4.1 judge, Table 11 | 7.56 / 7.52 | GPT Image 1 [High]: 7.53 / 7.30 | High judged editing quality under this protocol |
| ImgEdit overall; 734 real-world cases, GPT-4.1 judge, Table 12 | 4.27 | GPT Image 1 [High]: 4.20; FLUX.1 Kontext [Pro]: 4.00 | A small aggregate lead, not superiority in every editing category |

For GEdit, the overall score is the **mean of per-sample geometric means** of semantic consistency and perceptual quality. It is not generally the geometric mean of the two already-averaged table columns. The report gives scores, but these small differences alone do not establish statistical significance or universal user preference.

**What remains uncertain?** The results compare complete systems. They do not independently establish how much of each gain is caused by MSRoPE, synthesis, VAE adaptation, dual conditioning, or task mixing. A proposed ablation would hold data, compute, and model size fixed while removing one path or training intervention, then report both text accuracy and editing preservation. For dual conditioning, compare semantic-only, VAE-only, and both; score instruction compliance as well as unchanged-region fidelity. This is a proposed experiment, not a reported result.

The model also cannot guarantee exact text, arbitrary rare-character coverage, or pixel-perfect preservation. Novel-view and depth experiments (§5.2.3, Tables 13–14) demonstrate additional uses of the multitask formulation; they should not be confused with proof that the model always beats specialized geometry systems. The conclusion explicitly acknowledges the depth gap. The compatible image/video VAE is preparation for future video models, not evidence that this report evaluates a complete video generator.

## 11 · Reading map and idea to carry forward

| Question | Supplied PDF location |
| --- | --- |
| Which models and tensor sizes are used? | §2, Figures 6–8, Table 1, pp. 7–9 |
| Why adapt the decoder? | §2.3 and §5.2.1, Table 2, pp. 8 and 20 |
| How are characters and layouts taught? | §3, Figure 13, §4.1.3, pp. 9–16 |
| Which direction does flow time run? | §4.1, Equations 1–2, p. 15 |
| How do preferences change training? | §4.2, Equations 3–8, pp. 16–18 |
| Where do reference features enter? | §4.3, Figures 14–15, pp. 18–19; I2I motivation in the abstract/introduction |
| What do the numbers establish? | §5.2, Tables 2–14, pp. 20–26 |

**The idea to carry forward:** precise generation needs the desired information to survive the entire pipeline. Qwen-Image combines character-rich supervision, a decoder that preserves fine detail, and two complementary reference representations. Understanding an edit and retaining its source appearance are separate jobs; test both.

Try the study cards below, then compare [FLUX.1 Kontext’s reference conditioning](/papers/flux-kontext) and [REPA’s training-only teacher](/papers/t2-paper-1). Qwen2.5-VL is an inference-time condition encoder here, unlike REPA’s teacher.
