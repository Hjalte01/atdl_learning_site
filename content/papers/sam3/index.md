---
title: "SAM 3: Segment Anything with Concepts"
shortTitle: "SAM 3"
topic: "topic-3"
order: 3
description: "How a concept becomes a set of masks—and how global recognition, local object queries and persistent video memory divide the work."
status: "ready"
source: "https://proceedings.iclr.cc/paper_files/paper/2026/hash/e0982cbc81401df3430ee1ff780dc7a2-Abstract-Conference.html"
pdf: "/papers/sam3.pdf"
---
## The puzzle: you clicked an apple, but wanted every apple

A click can tell a segmentation model exactly which object you mean. But imagine a video of a fruit stall: you want **every red apple**, including apples that enter later. Clicking one fruit does not enumerate the others, recognize the concept, or preserve their identities through occlusion.

**SAM 3 adds concept detection to promptable segmentation and couples it to a memory-based tracker.** A global presence head asks whether the concept exists; object queries propose its instances; a tracker carries each instance through video. These are different jobs, even though they share visual features.

This is Topic 3, paper 3. We follow the supplied **78-page ICLR 2026 conference PDF**, whose PDF creation metadata is **6 March 2026**. That is an export timestamp, not a claimed publication date. The [official proceedings record](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e0982cbc81401df3430ee1ff780dc7a2-Abstract-Conference.html) verifies the title and venue. We use this course copy throughout, without substituting later checkpoint behavior. Eight original teaching diagrams are schematic except the explicitly measured Table 1 chart. Toy scores are invented for arithmetic; no SAM model runs here. Allow about 25 minutes.

## 1 · A concept is a set, a click targets an instance

[![A point prompt outlines one apple; the concept red apple outlines all three red apples while excluding a yellow one.](/figures/sam3/problem.svg)](/figures/sam3/problem.svg)

Read the cyan outlines as desired outputs, not measured predictions. **Promptable Visual Segmentation (PVS)** uses points, boxes or masks to segment one indicated object. **Promptable Concept Segmentation (PCS)** takes a short noun phrase, image exemplars, or both, and finds every matching instance. For video, it also returns persistent identities. §2 defines the PCS task for images and short videos of at most 30 seconds.

An **exemplar box** has a position, visual content and a positive/negative label. A positive example of a dog can help detect *all dogs*; a negative exemplar can reject a mistaken concept interpretation. A PVS box or click instead guides one instance's mask. Identical geometric shapes can serve different prompting roles.

Concepts must stay consistent across refinement. “Fish” plus an exemplar of only its tail is an inconsistent category definition (§2); update the concept rather than expect the model to reconcile it. A phrase such as “mouse” also has several valid meanings. The task therefore needs ambiguity-aware annotation and evaluation.

A plausible design interpretation is: reuse visual grounding, add language-conditioned set prediction, split global recognition from localization, and add identity-specific memory. This is a reconstruction of the design logic, not the authors' private reasoning.

<details><summary>Predict: does a positive exemplar always mean “segment only this box”?</summary><p>No. In PCS it specifies an example of the concept and may change predictions for many instances. PVS prompts refine an individual instance.</p></details>

## 2 · Architecture: one backbone, two paths

[![Parallel visual and prompt encoders feed a fusion encoder and query decoder; unconditioned visual features also feed per-object memory tracking, followed by detection association and recurrent memory updates.](/figures/sam3/architecture.svg)](/figures/sam3/architecture.svg)

Follow the image detector **A → B → C → D**. Read the video row as a second branch starting at **A**, with a loop back from updated memory on the next frame:

- **A: encode the frame and prompt.** The aligned Perception Encoder (PE) provides image features and text tokens. A geometry/exemplar encoder combines box position, positive/negative label and ROI-pooled visual features. Text and exemplar tokens form the prompt sequence.
- **B: condition the image.** A six-block fusion encoder cross-attends from image features to prompt tokens. The spatial grid stays a grid; its values become concept-conditioned.
- **C: propose instances and recognize presence.** A six-block DETR-style decoder processes **200 learned object queries** and an additional global presence token. Queries self-attend and cross-attend to prompt tokens and conditioned image features. Box-relative position bias helps focus attention; boxes are refined across decoder layers.
- **D: output boxes, masks and scores.** MLP heads predict boxes and scores. A MaskFormer-adapted segmentation head uses SimpleFPN features; instance masks additionally use output object queries. A semantic head predicts the pixels matching the concept without separating instance identities. The shared presence factor gates instance and semantic confidence.

The **tracker** uses *unconditioned* PE frame features through a separate neck and memory attention. Current-frame tokens attend to spatial memories and object pointers. A SAM 2-style mask decoder predicts the propagated mask, occlusion and confidence; a memory encoder combines mask predictions with frame features for later frames (§D.3). Inference keeps confident references in memory. The highest-confidence choice among three tracked-object masks supplies the current mask.

Detection is identity agnostic: any query may discover a new apple. Tracking is identity specific: ID 7 must stay attached to the same apple. Shared visual features do not make their objectives identical. PE is run once per frame and its features can be cached across interactions, including forward/backward propagation passes through video.

§D.1 reports approximately **850M parameters**: 450M vision, 300M text, and 100M detector/tracker components. The vision transformer has 32 layers, four with global attention; the usual 1008-pixel input is divided into nine 336-pixel attention windows. The causal text encoder has context length 32. These documented counts do not imply that this guide specifies every implementation dimension.

## 3 · Representation: prompts, queries, pixels and identities

[![Image features have N spatial tokens, prompts have L tokens, and 200 learned query slots produce instances; instance masks, a semantic map and video memory serve different output roles.](/figures/sam3/representation.svg)](/figures/sam3/representation.svg)

The rows compare representations; they are not one concatenated sequence. In teaching notation, image features are $F\in\mathbb R^{N\times d}$, prompt tokens are $P\in\mathbb R^{L\times d}$ after compatible projections, and decoder query states are $Z\in\mathbb R^{Q\times d}$ with $Q=200$. Here $N$ is the spatial-token count, $L$ the number of prompt tokens, and $d$ a symbolic common feature width. We do not infer an undocumented numeric width.

A **query is a learned output slot**, not an input noun phrase or one supplied annotation. Predictions can be unmatched or duplicates; Q is not a promise of 200 valid masks. A spatial mask has one value per output pixel; an instance mask separates objects, while a semantic mask merges matching pixels. A video masklet is a linked sequence of masks for one identity.

An attention operation explains the **A/P → B** arrow. The following is a standard teaching expression, not a numbered SAM 3 equation:

$$
\operatorname{Attn}(U,V,W)
=\operatorname{softmax}\!\left(\frac{UV^\top}{\sqrt{d_k}}\right)W.
$$

$U$ contains projected image queries, $V$ projected prompt keys, $W$ prompt value vectors, and $d_k$ the key width. Softmax acts over prompt positions for each image query. If two prompt attention weights are 0.75 and 0.25 and their toy scalar values are 2 and 0, the resulting value is 1.5. This conditions a feature; it does not immediately produce a binary pixel mask. The real fusion blocks also contain self-attention, residual paths and MLPs.

Unlike [DINOv3](/papers/dinov3) and [V-JEPA 2.1](/papers/vjepa-21), this task is not merely to expose reusable frozen features to a downstream probe. SAM 3 learns prompt-conditioned output machinery for supervised detection and segmentation, including language grounding and video memory.

## 4 · Where the math happens: one global gate, many local scores

[![A global presence probability multiplies every conditional object score; toy presence 0.8 gives scores 0.72, 0.56 and 0.32, and absent concepts supervise the presence branch.](/figures/sam3/presence.svg)](/figures/sam3/presence.svg)

An apple-shaped patch may be a red apple, a toy, or a printed picture. Global context helps decide whether the concept exists, but a query must also focus on a local boundary. §D.2 decomposes the **C → D score arrow**:

$$
s_i=a_i r,
\qquad r=p(\text{NP present in image}),
$$

$$
a_i=p(q_i\text{ matches NP}\mid\text{NP present}).
$$

NP is the noun phrase, $q_i$ is object query $i$, $r$ the shared presence probability, $a_i$ its conditional match score, and $s_i$ the final score. The presence token and its MLP supply $r$; each object query supplies $a_i$. All queries share the same $r$. This factorization is a modeling choice, not proof of calibrated neural probabilities. If the presence head misses a real concept, every local candidate is penalized.

At $r=0.8$, local scores $(0.9,0.7,0.4)$ become $(0.72,0.56,0.32)$. A strict confidence cutoff above 0.5 keeps the first two. At $r=0.2$, even the best local score becomes 0.18. A local score measures a candidate's conditional match, not the exact IoU of its mask.

**Supervision follows the division of labor.** The presence head learns both present and absent concepts. Per-query classification is computed only when the concept is present. On a positive image, Hungarian matching still supplies negative labels to unmatched local queries; “positive concepts only” does not mean every query gets a positive label (§B.1). The semantic binary-mask head is likewise trained on positive concept examples.

A teaching form of the presence BCE at **C** is

$$
\mathcal L_{\rm pres}=-y\log r-(1-y)\log(1-r).
$$

$y\in\{0,1\}$ is ground-truth concept presence; logs are natural. For $y=1,r=0.8$, this is 0.2231. For a hard negative with $y=0,r=0.8$, it is 1.6094: the same prediction deserves a large penalty. Stage 3 weights this term by 20. With logit $z$ and $r=\sigma(z)$, the unweighted derivative is $r-y$; at the positive toy example it is −0.2, so gradient descent raises the logit. This explains a learning signal, not the full detector loss.

The box path uses L1 and generalized-IoU losses, masks use focal and Dice terms, and classification uses Align loss with matching. Their weighted combination is broader than presence BCE (§D.4.1). We do not replace Align with an invented ordinary cross-entropy detector objective.

If presence is externally known, the paper permits setting $r=1$, as in particular counting or PVS settings. That removes the global recognition gate; it does not guarantee correct counting or localization.

<details><summary>Predict: on a negative image, are all 200 local query scores trained as negatives?</summary><p>Not in the selected decoupled supervision strategy. The presence head is trained on the absent concept; per-query classification is skipped for that concept-image pair. Unmatched queries still get negative supervision on positive images.</p></details>

## 5 · Work one example through the same boxes

[![A red-apple prompt conditions image features, assumed head outputs produce two retained instances, and a positive presence label gives a weighted BCE contribution of 4.4629.](/figures/sam3/numeric.svg)](/figures/sam3/numeric.svg)

At **A**, take an image with two red apples and encode it once. At **B**, fuse the “red apple” prompt with the grid. We cannot compute learned features without weights, so this toy explicitly assumes the **C** head outputs $r=0.8$ and local scores 0.9, 0.7 and 0.4. At **D**, multiplication and a cutoff above 0.5 retain two instances. The third proposal is filtered; mask overlap and quality remain separate questions.

If this were a training pair labeled present, the presence branch would contribute

$$
20\mathcal L_{\rm pres}=20[-\log(0.8)]\approx4.4629.
$$

This is only one term of a training loss; it is not an inference cost or measured segmentation error. Matched-query box/mask terms need ground truth and actual predicted masks. A good score alone does not establish a good contour.

For a toy propagated mask $A$ of 10 pixels and detection $B$ of 8 pixels with 6 shared pixels, the association arrow uses

$$
\operatorname{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}
=\frac{6}{10+8-6}=0.5.
$$

$|\cdot|$ counts pixels. This exceeds the default 0.1 association threshold in §D.3, but not the stricter 0.8 overlap required for periodic re-prompting. It illustrates one overlap calculation, not the complete crowded-scene matching implementation. With two competing tracks, overlap alone cannot establish physical identity.

## 6 · Training and inference carry different state

[![Four training stages learn aligned encoders, detection, presence and interaction, then tracking on frozen PE; inference uses cached features, prompt state and per-object memory; the data engine supplies future training labels.](/figures/sam3/training.svg)](/figures/sam3/training.svg)

Read the upper rows as **successive learning stages**, the inference row as a separate execution path, and the data-engine row as a dataset feedback loop:

| Stage | What is learned or fixed | Source setting |
| --- | --- | --- |
| 1: PE | Aligned image/text encoders | 5.4B image-text pairs; no stage-1 distillation or video fine-tuning |
| 2: detector pretraining | Detector and image/text encoders | Approximately 95k iterations, batch 896; pseudo and human masks, auxiliary PVS |
| 3: refinement | High-quality image training, presence and interaction | Approximately 5k iterations, batch 512; presence BCE weight 20; learning rates multiplied by 0.025 |
| 4: tracker | Learn tracker on frozen shared PE | Approximately 190k iterations, batch 512; then 60k longer-context fine-tuning steps with 16/32-frame videos |

Stage 3 samples positive boxes from false negatives and negative boxes from high-confidence false positives. Prompts accumulate over five PCS refinement iterations; PVS increases to seven interaction steps. Presence is introduced during this stage, not assumed present throughout pretraining.

Stage 4 uses focal/Dice mask losses, IoU-prediction MAE and occlusion cross-entropy with weights **20:1:1:1**. Only the lowest-segmentation-loss mask receives multi-mask supervision. If a ground-truth frame has no mask, mask outputs are not supervised there, but occlusion is always supervised. No teacher EMA is specified for this pipeline; do not transfer the V-JEPA training architecture here.

At **inference**, learned weights are fixed. Image PCS encodes the image, applies the concept prompt and returns a set. Exemplar refinement updates concept-level predictions; clicks can switch to instance-level refinement. Video adds cached frame features, IDs, prompt/conditioning frames and per-object memories. There is no user-label optimizer update in this path.

The **data engine** mines media and phrases, proposes masks and hard negatives, verifies mask quality (**MV**) and completeness (**EV**), then sends failures to human correction. A correct mask on one apple can pass MV while missing three other apples and failing EV. Collected labels train a later model; the verifiers are not mandatory inference modules in ordinary SAM 3. Table 9d's verifier-assisted results must therefore stay separate.

§5 reports 5.2M HQ images with 4M distinct phrases; §4 reports 52.5K videos with 467K masklets. The introduction also describes a large synthetic dataset of 38M phrases and 1.4B masks. Different units—images, pairs, unique phrases and masks—cannot be added as if they were the same count. The paper reports roughly doubled image annotation throughput, while its Table 16 video all-human engine is **40% slower** and yields **9% more masklets** than manual noun-phrase-input annotation. Efficiency depends on the task and verification policy.

## 7 · Video: discover, propagate, associate, confirm

First propagate already-known identities; simultaneously detect the concept in the current frame. §3 gives an unnumbered functional algorithm:

$$
\hat M_t=\operatorname{propagate}(M_{t-1}),
\qquad O_t=\operatorname{detect}(I_t,P),
$$

$$
M_t=\operatorname{matchAndUpdate}(\hat M_t,O_t).
$$

$I_t$ is frame $t$, $P$ the concept prompt, $M_{t-1}$ the previous masklet state, $\hat M_t$ propagated masks, $O_t$ current detections, and $M_t$ updated identity-linked masks. The propagation function is shorthand for the memory-conditioned path; it does not see only a previous binary mask. Unmatched detections spawn new masklets. Matched detections support existing ones and can recondition tracking. This lets an apple enter after frame 1.

[![A toy match sequence plus one, plus one, minus one, minus one, minus one, plus one, plus one yields cumulative masklet scores 1, 2, 1, 0, minus 1, 0, 1.](/figures/sam3/timeline.svg)](/figures/sam3/timeline.svg)

Read the curve as **toy arithmetic for a confirmed track**, not measured accuracy or the complete confirmation window. §D.3 defines detector agreement

$$
\Delta_i(t)=\begin{cases}
+1 & \text{if some detection overlaps track }i\text{ above the IoU threshold},\\
-1 & \text{otherwise},
\end{cases}
$$

$$
S_i(t_0,t_1)=\sum_{t=t_0}^{t_1}\Delta_i(t).
$$

$i$ is an identity, $t_0,t_1$ inclusive frame limits, and $S_i$ its masklet detection score (MDS). The default association threshold is 0.1. Each match adds evidence; each failure removes evidence. The seven-frame toy gives cumulative scores **1, 2, 1, 0, −1, 0, 1**. A confirmed track with negative lifetime score has its output mask zeroed, while its internal state is retained so later agreement can recover it. This differs from removing an unconfirmed candidate from state.

The actual policy delays output by **T=15 frames**, about half a second at 30 fps. Newly appearing tracks whose confirmation-window score is below zero are removed. If two tracks consistently overlap the same detection for at least $\lceil T/2\rceil$ frames, the newer one is removed when it began inside the window. The paper describes confirmation as matching at least half the frames, but writes the sum over the inclusive interval $[t,t+T]$; that formula has $T+1$ terms. An exact reproduction should resolve the endpoint convention rather than silently change it. Our toy does not simulate that window.

Every **16th frame**, periodic re-prompting can replace a propagated reference with a detection when overlap is at least 0.8 and both confidences exceed 0.8. A separate detection-guided rule reconditions drifting tracks when bounding-box IoU with the highest-matching detection is below 0.85. Matching, periodic reliable refresh and drift repair are distinct operations. Their thresholds are inference policies, not learned loss weights.

<details><summary>Predict: does suppressing a confirmed masklet delete its identity?</summary><p>No. Negative lifetime MDS suppresses its displayed mask but retains tracker state. Removing an unconfirmed masklet is a separate operation.</p></details>

## 8 · Evidence: recognition, boundaries and association

[![Reported Table 1 SA-Co Gold instance cgF1 bars compare OWLv2 17.3, partially LVIS-trained OWLv2 24.6, LLMDet-L 6.5, DINO-X 21.3, SAM 3 54.1 and human oracle 72.8.](/figures/sam3/results.svg)](/figures/sam3/results.svg)

These are **conference-copy Table 1 measurements** for text-prompted image PCS, not synthetic performance. Bars start at zero; no uncertainty intervals are supplied here. OWLv2* is partially trained on LVIS, DINO-X is evaluated through its API, and baseline box detectors use SAM 1 to obtain masks. The human number is an ambiguity-aware oracle agreement reference, not perfect ground truth.

SAM 3's 54.1 is **2.20×** the strongest listed Gold baseline's 24.6, and about **74.3%** of 72.8. That particular comparison supports the paper's “double” claim on this benchmark; it does not imply every metric doubles. Table 1's **LVIS instance mask AP is 48.5**, versus DINO-X 38.5 from its original paper. The introduction says 48.8; we retain the table value and flag the mismatch. The gray 53.0 APE-D* AP is marked as using closed-set training data, so it is not an interchangeable zero-shot comparison.

### What cgF1 measures

A model that draws excellent masks only when given a guaranteed-positive image may still hallucinate on absent concepts. §F.3 therefore separates localization from recognition. It filters masks at confidence above 0.5, then computes

$$
\mathrm{cgF1}=100\,\mathrm{pmF1}\,\mathrm{IL\ MCC}.
$$

This is **paper Eq. 3**, operating in evaluation, not the training loss. pmF1 is positive micro F1: accumulate matched-mask TP/FP/FN on positive image-phrase pairs, compute $2TP/(2TP+FP+FN)$ at each mask-IoU threshold 0.50, 0.55, …, 0.95, and average the ten values (paper Eq. 1). IL MCC is image-level Matthews correlation for whether any mask is emitted; it ignores contour quality (paper Eq. 2).

A teaching metric example uses image-level counts $TP=8,TN=8,FP=2,FN=2$, giving

$$
\mathrm{IL\ MCC}=\frac{8\cdot8-2\cdot2}{\sqrt{10\cdot10\cdot10\cdot10}}=0.6.
$$

If averaged pmF1 is 0.75, cgF1 is $100(0.75)(0.6)=45$. Image-level counts here are distinct from mask-level matching counts. Denominator-zero MCC cases require the evaluation implementation's convention; this toy avoids them. MCC can be negative, so cgF1 is not simply another percentage of correct masks.

Gold selects the best local-F1 match among **three independent annotations per pair**. That accommodates legitimate interpretations; it is an evaluation oracle, not an annotation available to the inference model. A fixed-threshold metric also depends on calibration, negative prevalence and prompt distribution.

### Ablations and video results stay separate

| Conference-copy evidence | Without / alternative | With / full | What it supports |
| --- | --- | --- | --- |
| Table 9a, presence head | cgF1 50.7; IL MCC 0.77; pmF1 65.4 | 52.2; 0.82; 63.4 | Better recognition with a localization tradeoff |
| Table 9b, 0 → 30 hard negatives/image | cgF1 28.3; IL MCC 0.44 | 43.0; 0.68 | Negative phrases improve recognition in this run |
| Table 5, VEval SA-V test: detector + T-by-D → full SAM 3 | cgF1 25.7; pHOTA 55.7 | 30.3; 58.0 | Memory-based tracker improves this setting |
| Table 5, VEval SmartGlasses test: same comparison | cgF1 29.7; pHOTA 60.0 | 36.4; 63.6 | Gains also appear in this egocentric subset |

Table 9 rows come from different, shorter training runs and are **not directly comparable across sub-tables or to Table 1**. Presence improves cgF1 by 1.5 while pmF1 falls by 2.0; this is useful evidence for separating recognition and localization. Table 10's product-supervision variant scores slightly higher (54.9 versus 54.0), but the authors favor separately supervised scores for flexibility. The selected strategy is not the numerical winner of every ablation.

pHOTA measures detection and identity association on positive video-phrase pairs. It is not an image-mask AP or a click-prompted VOS J&F score. In Table 5, SA-V human pHOTA is 70.5, leaving a visible gap to SAM 3's 58.0. GLEE and LLMDet were not trained on SA-Co; swapping those detectors changes concept coverage as well as module structure. These comparisons do not isolate architecture under matched data exposure.

A stronger proposed ablation would hold backbone, data, compute and prompting fixed while crossing presence/no-presence with hard-negative/no-hard-negative training, and report both recognition and mask quality. For video, measure confirmation-delay/refresh interventions with both association quality and latency, rather than accuracy alone.

## 9 · Ambiguity and the limits of “anything”

§D.2 adds a post-trained ambiguity head on a frozen model. Two experts adjust **classification logits**, leaving masks, boxes and presence unchanged. With expert losses $\mathcal L_k$, the paper's unnumbered winner-takes-all rule is

$$
k^*=\arg\min_k\mathcal L_k,
\qquad \mathcal L_{\rm WTA}=\mathcal L_{k^*}.
$$

$k$ indexes an interpretation expert; $k^*$ is the lowest-loss expert. Only that expert receives its expert-specific gradient. For toy losses 0.2 and 0.7, expert 1 is selected. A separate classifier learns to predict the winning expert for inference—ground-truth losses are unavailable then. This encourages specialization but is not proof that every ambiguity has exactly two meanings. The paper reports difficulty with more than three experts and mode collapse. This image ambiguity mechanism differs from the tracker's three mask alternatives.

The main limitations in §C are concrete:

- Fine-grained unfamiliar concepts and niche domains, such as thermal imagery, can defeat zero-shot PCS. Class-agnostic PVS transfer does not imply concept-recognition transfer.
- Short noun phrases with one or two attributes are the intended language input. Long referring expressions and reasoning require an additional MLLM/agent; Table 8's agent results are not base SAM 3 alone.
- Per-object tracking cost grows approximately linearly with object count. The paper's 30 ms image inference claim is on an H200; video real-time examples use multiple H200s. Half-second confirmation delay is another latency dimension.
- This version lacks shared object-level contextual memory for resolving crowded multi-object ambiguities. It describes shared global memory as future work; no later shared-memory checkpoint is silently included here.
- A hard switch between concept and instance modes protects individual edits but makes interleaving them less seamless. Consistent exemplars cannot remove every subjective interpretation.

## The idea to carry forward

A useful segmentation system must answer **whether**, **where**, **how many**, and **which identity over time**. SAM 3 distributes those questions across a shared global presence score, local queries, masks and memory. Read a gain through the measurement that supports it: recognition, boundary quality and temporal association can move differently.

## Source reading map

| Guide question | Supplied conference PDF |
| --- | --- |
| PCS versus PVS, prompt consistency | §2, p. 3; Figures 1 and 3 |
| Functional detector/tracker paths | §3, pp. 3–5; Figure 4 |
| 200 queries, fusion blocks, score decomposition | §D.1–D.2, pp. 19–20; Figure 10 |
| Ambiguity experts and inference selection | §D.2, p. 21; Figure 11 |
| MDS, delay, suppression and re-prompting | §D.3, pp. 22–23 |
| Four training stages, supervision and schedules | §D.4.1, pp. 23–24; Table 18 |
| Data engine, quality versus completeness | §4–5, pp. 5–7; §E; Tables 15–16 |
| Main image/video results | Tables 1 and 5, pp. 7 and 9; §G.1, §G.5 |
| Presence and hard-negative ablations | Tables 9–10, pp. 10 and 12; §B.1 |
| Evaluation equations and annotation oracle | §F.3–F.4, pp. 36–38, Eqs. 1–3 |
| Domains, language, object count and mode switch | §C, p. 18 |

All page numbers refer to the supplied printed pages, not to a newer online version. [Open the course PDF](/papers/sam3.pdf), return to [Topic 3](/topics/topic-3), or practice with the cards below.
