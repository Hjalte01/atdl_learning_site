---
title: "FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space"
shortTitle: "FLUX.1 Kontext"
topic: "topic-2"
order: 5
description: "How reference images and text guide a latent flow transformer."
status: "ready"
source: "https://arxiv.org/abs/2506.15742v2"
pdf: "/papers/flux-kontext.pdf"
---
## The paper in one minute

**The problem:** “Make this character sit in a café” is harder than “Generate a character in a café.” The first instruction asks the model to recognize a particular character, preserve its identity, change the scene, and produce a coherent new image. Repeating edits makes this harder: small unwanted changes can accumulate.

**The idea:** Give a flow-matching transformer both the text instruction and a sequence of tokens representing the reference image. Generate a new image from noise while attending to that reference. The reference tells the model **what to keep**; the instruction helps specify **what to change**. This is a useful intuition, not a hard architectural separation: the network must learn which details matter jointly from both inputs.

**The contribution:** FLUX.1 Kontext combines this simple conditioning design with training on image relationships and distillation for faster generation. The authors also introduce KontextBench to evaluate several kinds of editing. This is a model-and-evaluation report, not a proof that sequence concatenation alone solves image editing. [§§1–4, course PDF](/papers/flux-kontext.pdf#page=4)

**Reading route:** Spend about 10 minutes on the intuition and architecture, 10 on the equations and interactive example, and 10 on evidence and recall. No model installation is required.

## A concrete editing problem

**Toy example — not a result from the paper:** You have a photograph of a red bicycle against a brick wall. You ask: “Make the bicycle blue, keeping everything else the same.”

| Input or output | Its job |
| --- | --- |
| Reference image, y | Specifies this bicycle, wall, camera angle, and composition. |
| Instruction, c | Requests a color change and preservation of the rest. |
| Target image, x | A plausible image of that bicycle in blue. During training this is the desired answer; during generation it is unknown. |

A text-only model does not directly receive the exact arrangement of bricks or the particular bicycle. A reference-conditioned model does. But receiving these details is **not a guarantee of exact pixel preservation**: it still synthesizes a new output.

The same interface covers a much larger change: “Put this bicycle on a mountain trail.” Now the bicycle should stay recognizable while the background changes substantially. The paper calls this broader reuse of visual concepts generative editing. Local editing and generative editing therefore differ in how much should change, rather than requiring completely separate model interfaces. [§§1, 3](/papers/flux-kontext.pdf#page=1)

## Prerequisites in plain English

- **Autoencoder:** An encoder compresses an image into a spatial array of learned features; a decoder turns those features back into an image. Compression loses some information. FLUX operates in this learned latent space instead of updating every RGB pixel directly.
- **Latent token:** A vector made from latent image features, used as one item in the transformer's input sequence. Image tokens still carry spatial position information; they are not words or object labels.
- **Attention:** A mechanism through which a token combines information from other tokens. Here, an evolving output token can use information from the reference and instruction.
- **Conditioning:** Supplying information that changes which outputs the model should generate. We want a distribution of suitable edits given this image and instruction, written p(x | y, c).
- **Velocity field:** A function that says how a noisy latent should change at a particular time. Repeated small updates can move noise toward an image.
- **Distillation:** Training a model to reproduce useful behavior of another model or sampling process more efficiently. The exact recipe matters; not every distilled model uses adversarial training.

These definitions explain the machinery used in §§2–3 and Appendix A. They are learning paraphrases, not quotations. [Source](/papers/flux-kontext.pdf#page=4)

## Architecture: follow the information

### 1. Compress the reference

The frozen FLUX autoencoder encodes the reference image into latent tokens. “Frozen” means its weights are not updated during the Kontext fine-tuning described here. The base FLUX autoencoder uses 16 latent channels. This is a feature dimension, not 16 objects or 16 images. [§§2–3](/papers/flux-kontext.pdf#page=4)

### 2. Create the evolving output tokens

During training, start with a known target latent and mix it with noise. During inference, start the target sequence from noise. **The target sequence evolves; the reference image is conditioning and is not progressively denoised.** The model has an image to consult while it constructs a different image.

### 3. Append reference tokens along the sequence

The visual stream contains target tokens followed by reference tokens. Sequence concatenation means adding more items to the list. Channel concatenation would instead join features at corresponding positions. The paper reports that its initial channel-concatenation experiments performed worse, but does not provide a detailed quantitative ablation establishing a universal advantage.

Because separate image sequences need not have identical spatial grids, sequence concatenation supports different input and output resolutions and aspect ratios. Its formulation also extends to multiple references, **but this report focuses on a single context image**. Architectural extensibility is not evidence of tested multi-image performance. [§3, p. 5–6](/papers/flux-kontext.pdf#page=5)

### 4. Tell the model where tokens belong

FLUX uses three-dimensional rotary positional embeddings, or **3D RoPE**. Target positions are (0, h, w); reference positions are (1, h, w) for one reference, where h and w identify spatial coordinates. Additional references could use 2, 3, and so on in the first coordinate.

That first coordinate acts as an image identifier or “virtual time” offset. **It is not the noise time t in the flow objective.** The notation is easy to mix up: one labels tokens; the other tells the model how noisy the evolving target is.

### 5. Mix image and text information

The base FLUX transformer first uses double-stream blocks: image and text tokens have separate weights, with attention over their concatenation. It then uses 38 single-stream blocks operating on the combined sequence. Kontext adds the reference tokens to the visual input of this backbone. The network predicts the target's velocity; the resulting target latent is decoded into the output image. [§§2–3, Figures 3–4](/papers/flux-kontext.pdf#page=4)

**Mental picture:** reference tokens are a visual document on the desk; text describes the requested edit; target tokens are the evolving draft. Attention lets the draft consult the document. This analogy describes information access, not literal copying or a symbolic understanding of objects.

## Flow matching, step by step

The equations below follow the paper's convention: **t = 0 is a clean target and t = 1 is noise**. Other explanations may reverse these endpoints. Here x denotes the encoded target latent, not raw pixels; y denotes encoded reference conditioning.

### Construct a training example

Choose a target x, its reference y, and instruction c. Draw Gaussian noise ε of the same shape as x, and a time t. Form:

$$
z_t = (1-t)x + t\varepsilon.
$$

zₜ is the noisy target latent. At t = 0 it equals x; at t = 1 it equals ε. The reference y stays available as context at all sampled noise levels. The report samples training times with a resolution-dependent shifted logit-normal schedule, rather than simply claiming uniform time sampling. [§3, Eq. 3; Appendix A.2](/papers/flux-kontext.pdf#page=5)

### Derive what the network should predict

Differentiate the straight interpolation with respect to t:

$$
\frac{d z_t}{dt} = \varepsilon - x.
$$

The training target is therefore a **velocity**, ε − x. The model vθ receives zₜ, time t, reference y, and instruction c. Its parameters θ are trained to minimize squared error:

$$
\mathcal L(\theta) = \mathbb E\left[\left\|v_\theta(z_t,t,y,c)-(\varepsilon-x)\right\|_2^2\right].
$$

The expectation averages over training examples, sampled times, and noise. The squared norm sums squared prediction errors over latent coordinates. This objective asks for a direction and rate of change, not an image class or a word to emit next. [§3, Eq. 3](/papers/flux-kontext.pdf#page=5)

**Subtle point:** Each sampled training pair has a straight interpolation, but the learned field must work across many possible pairs. It does not receive the clean target at inference. Straight training paths do not imply that every generated trajectory is exactly straight or that a single model evaluation is enough.

### Generate by running time backward

Start with z₁ drawn from noise and integrate toward t = 0. A simple illustrative Euler update is:

$$
z_{t-\Delta t} \approx z_t - \Delta t\,v_\theta(z_t,t,y,c), \qquad \Delta t > 0.
$$

The minus sign matters: the predicted velocity follows increasing t toward noise, so generation travels in the opposite time direction. Real sampling and distilled variants need not use this exact toy solver.

**Scalar toy calculation:** Let x = 2 and ε = 10. At t = 0.75, zₜ = 8, and the exact training velocity is 8. A backward step of Δt = 0.25 gives 8 − 0.25 × 8 = 6, exactly the interpolation at t = 0.5. This works exactly because we supplied the true constant velocity for a single one-dimensional pair. A real model estimates a high-dimensional conditional field.

[Try the interactive version below](#flow-explorer).

## What is trained, and why it is fast

Kontext starts from a FLUX.1 text-to-image checkpoint and uses millions of curated relational training examples: reference, instruction, and target. The joint formulation also includes text-only examples, for which the reference tokens are omitted. This teaches one architecture both generation and editing. It does not mean the network learns a new set of weights each time you upload a reference. “In-context” here refers to adapting the output through inputs at inference. [§3](/papers/flux-kontext.pdf#page=5)

| Variant in this report | What the report says | What to remember |
| --- | --- | --- |
| Kontext [pro] | Flow-objective training followed by latent adversarial diffusion distillation (LADD). | Adversarial distillation reduces sampling work; it is not merely the original velocity loss. |
| Kontext [dev] | Guidance distillation into a 12B transformer, focusing exclusively on image-to-image training. | Do not assign the joint text-to-image training recipe or pro's LADD recipe to dev. |
| Kontext [max] | Uses more compute to improve generative performance. | The report gives limited recipe detail; do not invent its architecture or parameter count. |

The authors describe ordinary flow sampling as requiring many network evaluations, and distillation as a way to reduce that cost. They report approximately 3–5 seconds for 1024 × 1024 image synthesis and compare API median latencies in Figure 7. Those are **reported deployment measurements**, not a promise about a laptop, a different GPU, or today's APIs. [§§1, 3, 4.2](/papers/flux-kontext.pdf#page=6)

## Evidence: what the results actually support

### KontextBench covers five editing tasks

The benchmark contains **1,026 image–prompt pairs from 108 base images**. These are not 1,026 independent source images. Several instructions can share an image. [§4.1](/papers/flux-kontext.pdf#page=7)

| Category | Pairs | Question it tests |
| --- | ---: | --- |
| Local instruction editing | 416 | Can it make a limited change while retaining context? |
| Global instruction editing | 262 | Can it transform the overall image as requested? |
| Text editing | 92 | Can it change written content within an image? |
| Style reference | 63 | Can it reuse appearance or rendering style in a new scene? |
| Character reference | 193 | Can it preserve a particular subject in a new setting? |

**Reported result:** Human evaluations place pro and max strongly in local editing, text editing, and character reference. The paper reports that global editing trails gpt-image-1 and style reference trails Gen-4 References. Its claim is not “wins every task.” Figure 8 presents five task categories plus an AuraFace similarity measurement; that extra panel is not a sixth dataset category. [§4.2, Figure 8](/papers/flux-kontext.pdf#page=8)

### Identity and multi-turn editing

The authors compare facial embeddings from AuraFace using cosine similarity and examine repeated edit sequences. Higher similarity means the face representations are more alike according to this recognition model. Together with human judgments and example sequences, this supports improved identity retention in the evaluated setting. [§§4.2–4.3, Figure 12](/papers/flux-kontext.pdf#page=9)

**Critical reading:** A face score is not a complete measure of whether the requested edit is correct, the background is unchanged, or a product's logo is accurate. An editor that does nothing could preserve identity perfectly and still fail the task. Identity, instruction following, and image quality must be considered together. Occlusion, such as adding sunglasses, can also reduce the face score without necessarily indicating a new identity; the paper notes this in Figure 12.

### Autoencoder evidence is a separate result

Table 1 compares reconstruction on 4,096 ImageNet images. FLUX-VAE reports SSIM 0.896 ± 0.004 and PSNR 31.1 ± 0.08, with the table specifying standard errors. This concerns encoding and reconstructing an image. It supports the quality of the latent representation's reconstruction, **not a direct measurement of instruction-following ability**. [Table 1](/papers/flux-kontext.pdf#page=5)

### What would make the claim stronger?

These are study questions and methodological critiques, not additional reported findings:

- How much improvement comes from the relational data, sequence concatenation, base model, and distillation separately? Controlled ablations would help isolate causes.
- Do gains hold on independently collected images and instructions? A benchmark introduced by the model authors is useful, but independent evaluation would broaden the evidence.
- How do results change when categories are equally weighted? The benchmark contains many more local edits than style-reference examples.
- How much of the API speed difference is model design versus serving hardware and infrastructure? End-to-end latency cannot isolate this by itself.

## Failure cases and boundaries

The authors explicitly report instruction-following failures, identity changes, and artifacts from excessive multi-turn editing or distillation. Figure 15 includes an object-movement request that instead changes the coffee's appearance, and visible degradation after six edits. **Better consistency means less drift in the reported comparisons, not no drift.** [§5, Figure 15](/papers/flux-kontext.pdf#page=12)

Multi-image inputs and video editing are discussed as extensions. This report does not establish those extensions as evaluated capabilities. Likewise, a natural-language request to preserve everything else does not impose a pixel-level constraint or guarantee that every unmentioned detail remains fixed.

**Practical interpretation:** For a hypothetical product-editing workflow, check both the requested change and invariants such as the logo, geometry, and text after every edit. A good-looking image alone does not establish a faithful edit.

## Common misunderstandings

| Misunderstanding | Correction |
| --- | --- |
| “It starts by adding noise to the reference and undoing it.” | In the described formulation, the target is generated from noise while the reference is separately supplied as context. |
| “The reference must match the target resolution.” | Sequence concatenation supports separate token grids and aspect ratios. |
| “In-context means training a LoRA for this person.” | The reference conditions the output without per-reference parameter updates. The model itself was trained beforehand. |
| “The RoPE time coordinate is the noise level.” | Image indexing and flow time serve different purposes. |
| “Flow matching makes generation one step.” | A velocity objective alone does not establish one-step generation. The paper uses distillation for speed. |
| “All three variants use the same training recipe.” | The report distinguishes pro's LADD and dev's guidance distillation and editing-only focus. |
| “A high face score proves the edit succeeded.” | It measures one aspect of preservation, not instruction compliance or overall fidelity. |

## Close the paper and explain it

Use the five-question routine from your ATDL notes. Try answering aloud before opening the suggested answers.

<details><summary>1. What problem does the paper address?</summary><p>How to combine text instructions and a reference image to make local or generative edits, while preserving relevant identity and context across repeated edits at interactive speed.</p></details>

<details><summary>2. What is the proposed method?</summary><p>Encode the reference into frozen-autoencoder latents, append those tokens to the evolving target sequence, distinguish their positions with 3D RoPE, and train a conditional flow transformer on reference–instruction–target relationships. Use variant-specific distillation to improve inference efficiency.</p></details>

<details><summary>3. What is the strongest evidence?</summary><p>The combination of task-specific human evaluations on KontextBench, identity-similarity measurements and iterative examples, and measured API latency. Each tests a different part of the claim. No single selected image or face score establishes the entire claim.</p></details>

<details><summary>4. Which three concepts should you be able to teach?</summary><p>Sequence concatenation: reference and target contribute separate tokens. Conditional flow matching: predict the velocity of a noisy target given context. Distillation: learn a more efficient generation procedure, with different recipes for pro and dev.</p></details>

<details><summary>5. What remains unclear?</summary><p>One good question is how much of the gain is attributable to the conditioning architecture versus the relational data and distillation. Another is how robust preservation is over long edit chains on new kinds of subjects. Give your own question and describe an experiment that could answer it.</p></details>

**Explain it in 60 seconds:** “Kontext turns image editing into conditional generation. It gives a latent flow transformer a reference image as extra tokens and asks it to generate a target from noise under a text instruction. Position information separates reference from target. Relational training teaches which visual information to preserve, and distillation helps make inference fast. The report finds strong editing and identity performance, but drift and instruction failures remain.”

## Source and reading map

**Primary source:** Black Forest Labs (2025). *FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space.* arXiv:2506.15742v2, 24 June 2025. This guide was checked against your local `T2_5_FLUX.1 Kontext.pdf`, which carries that version. [arXiv record](https://arxiv.org/abs/2506.15742v2) · [Local course PDF](/papers/flux-kontext.pdf)

| Read | Look for |
| --- | --- |
| §2, p. 4 | Autoencoder, double/single-stream blocks, positional embeddings. |
| §3, pp. 5–6 | Token concatenation, flow objective, conditioning, variant training recipes. |
| §4.1, p. 7 | Dataset composition and five task counts. |
| §4.2, pp. 8–9 | Human evaluation, facial similarity, speed and text-to-image evaluation. |
| §4.3 and Figure 12, pp. 9–11 | Iterative editing and identity drift. |
| §5 and Figure 15, pp. 12–13 | Failures and future work. |
| Appendix A, p. 14 | Flow-matching background and time schedules. |

The bicycle story, scalar calculation, interactive schematic, and methodological questions are teaching material created for this guide. They are not experiments from the paper. Model comparisons describe the 2025 report, not a current leaderboard.
