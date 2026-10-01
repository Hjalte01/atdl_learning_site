---
title: "V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning"
shortTitle: "V-JEPA 2.1"
topic: "topic-3"
order: 2
description: "Why recognizing a video does not locate its objects—and how weighted context prediction and deep self-supervision improve local features without abandoning dynamics."
status: "ready"
source: "https://arxiv.org/abs/2603.14482v3"
pdf: "/papers/vjepa-21.pdf"
---
## The puzzle: understand the grasp, miss the cup

A video model can recognize “grasping a cup” while its patch features give a poor outline of the cup. That is a problem if a robot needs depth or a tracker needs to follow the cup through time. What would make each patch useful locally, rather than merely useful for recognizing the whole scene?

**V-JEPA 2.1 adds a prediction target at visible positions as well as hidden ones, and supervises several encoder layers.** It learns in representation space: the target is a clean-input teacher feature, not an RGB pixel or a human segmentation label.

This is Topic 3, paper 2. We follow the supplied **37-page arXiv:2603.14482v3 PDF**: the margin says **11 June 2026**, while the title page says **June 12, 2026**. The [primary version record](https://arxiv.org/abs/2603.14482v3) confirms the June 11 submission. Eight original teaching graphics below are schematic except the Table 1 result chart. Toy numbers are invented for arithmetic, not measured features. No model runs on this page. Allow about 25 minutes.

## 1 · Why a visible token needs a target

[![Masked-only prediction can use visible tokens as global aggregators; dense prediction also compares visible outputs to clean local targets.](/figures/vjepa21/problem.svg)](/figures/vjepa21/problem.svg)

Read the top row as the previous V-JEPA objective. Remove some patch tokens, encode what remains, and predict features at the hidden positions. Visible tokens receive gradients indirectly through the predictor, but their own outputs have no position-specific matching loss. They can become convenient scene aggregators without exposing the local structure needed by a simple dense head.

The authors hypothesize that this explains fragmented feature maps (§2.2, Figures 1 and 3). The evidence includes PCA visualizations and frozen-backbone linear probes. It is a hypothesis supported by interventions, not a theorem that context tokens contain no local information.

The bottom row adds clean-input targets at visible positions. A plausible design interpretation is: keep the predictive task that learns scene dynamics, ground the context locally, then provide intermediate supervision so locality survives into the final layer. This reconstructs the design logic; it does not claim access to the authors’ private reasoning.

Connect this to [DINOv3](/papers/dinov3): both papers diagnose a gap between global and dense features. DINOv3 preserves patch relationships with a Gram target; V-JEPA 2.1 directly predicts local teacher features across space, time and depth in the network.

<details><summary>Predict: were visible tokens completely untrained in V-JEPA 2?</summary><p>No. Their features helped predict hidden tokens and received gradients through that path. The missing ingredient was a direct loss at their own positions.</p></details>

## 2 · Architecture: two encoders, one multi-level predictor

[![Student visible tokens pass through four encoder levels and MLP fusion to a predictor; the clean EMA teacher supplies four-level targets for masked and context losses.](/figures/vjepa21/architecture.svg)](/figures/vjepa21/architecture.svg)

Follow **A → B → C → D**, then return through the update loop:

- **A, tokenization and corruption:** use a native image or video patch embedding, modality information and 3D rotary positional encoding (RoPE). Drop masked patch tokens before the student encoder. The teacher retains the full clean sequence.
- **B, x-encoder:** a vision transformer with trainable parameters $\theta$ processes visible tokens. Collect normalized outputs from three intermediate blocks and the final block. Concatenate their channels; a lightweight MLP fuses them and reduces width.
- **C, predictor:** concatenate fused context tokens with learned mask tokens carrying the hidden positions. A 24-block transformer of embedding width 384 during pretraining predicts four feature levels at both visible and masked positions.
- **D, comparison:** the y-encoder, with moving-average parameters $\bar\theta$, supplies matching clean-input targets. The masked and weighted context L1 losses are applied at each of the four levels. Gradients update the student path and predictor; targets are stop-gradient.

The flagship encoder is **ViT-G, approximately 2B parameters**; ViT-g is approximately 1B, ViT-L 300M and ViT-B 80M. Appendix A gives ViT-G supervision blocks **[12, 24, 36, 48]**. Do not confuse the predictor's width 384 with the encoder's feature width. The figure is functional, not an invented full layer specification.

For the teacher update, a teaching expression of the paper’s EMA rule is

$$
\bar\theta \leftarrow \tau\bar\theta+(1-\tau)\theta,
\qquad \tau=0.99925.
$$

Here $\theta$ is the updated student encoder, $\bar\theta$ is the previous teacher, and $\tau$ is the old-teacher weight. This operates on the **update arrow into the y-encoder**, not on individual token features. For a toy scalar old teacher 2 and student 3, the new teacher is 2.00075. Stop-gradient and EMA give an asymmetric, slowly changing target; they are not a general mathematical guarantee against every collapse configuration.

## 3 · Representation: native grids, four depths

[![A 256-pixel image has 256 spatial patch tokens; a 16-frame video has 2048 tubelet tokens, and four encoder levels concatenate along channels.](/figures/vjepa21/representation.svg)](/figures/vjepa21/representation.svg)

Read each row as a different axis of the representation. An image tokenizer uses a **16×16 2D convolution**. A video tokenizer uses a **16×16×2 3D convolution**: each tubelet covers two frames. For the primary-phase inputs, the patch grids are therefore

$$
N_{\rm image}=16\cdot16=256,
\qquad N_{\rm video}=8\cdot16\cdot16=2048.
$$

These are derived token counts, excluding any modality-token convention. If $N_C$ positions remain visible and each layer has width $d$, four layer outputs have concatenated shape $N_C\times4d$. This is **channel concatenation**, followed by fusion. Adding mask tokens later is **sequence concatenation**. Those two operations serve different purposes.

V-JEPA 2 treated an image as a repeated static video; V-JEPA 2.1 handles the image directly. Both modalities use a shared transformer after their own tokenizer. “Multi-modal” here means **images and videos**, not language during base SSL. §2.3.4 calls the modality signal a learnable token; Figure 4 depicts an added modality embedding. The guide does not infer a precise extra-token count from this wording.

RoPE encodes positional relationships in time and space. A mask token says **where to predict**; it does not reveal the missing patch content. Each output is a contextual vector, not an independent RGB reconstruction or a semantic class label.

## 4 · The dense loss: predict hidden and visible features

First understand the arrows: **C → D** compares a predicted feature with the clean teacher at the same patch location. The predictor sees only the corrupted student context, while the teacher sees the full input.

Let $x$ be the masked view, $y$ the clean view, $E_\theta$ the student encoder, $E_{\bar\theta}$ the teacher, $P_\phi$ the predictor with trainable parameters $\phi$, and $\Delta_y$ the positional mask tokens. Write $\hat z_i=P_\phi(E_\theta(x),\Delta_y)_i$ and $z_i=E_{\bar\theta}(y)_i$ for one supervision level. With masked positions $M$, paper Eq. 1 is

$$
\mathcal L_{\rm predict}
=\frac{1}{|M|}\sum_{i\in M}
\left\|\hat z_i-\operatorname{sg}(z_i)\right\|_1.
$$

$|M|$ is the number of masked positions, $\operatorname{sg}$ stops target gradients, and $\|v\|_1$ sums absolute coordinate differences. Unlike a pixel loss, this asks the model to predict learned abstractions of the clean scene. It does not specify a decoder that can reconstruct every visual detail.

For visible context positions $C$, paper Eq. 2 adds

$$
\mathcal L_{\rm ctx}
=\frac{1}{|C|}\sum_{i\in C}\lambda_i
\left\|\hat z_i-\operatorname{sg}(z_i)\right\|_1.
$$

$$
\mathcal L_{\rm dense}=\mathcal L_{\rm predict}+\mathcal L_{\rm ctx}.
$$

$|C|$ counts visible positions and $\lambda_i$ controls their loss strength. **Each branch has its own count denominator.** This is not a single mean over all patches, and the context denominator is not the sum of weights. Empty $C$ or $M$ would make these expressions undefined; the teaching example uses both nonempty sets.

Why not strongly supervise every visible output equally? Copying easy context features can weaken global understanding. Table 2 shows that a constant coefficient of 0.5 gives ADE20K 27.5 mIoU but SSv2 only 53.8 accuracy, versus baseline 22.2 and 72.8. The useful design question is how to add local grounding while preserving the harder predictive task.

## 5 · Weight by distance, supervise through depth

[![Context weights decrease as inverse square root distance to a masked site; masked and context errors are averaged over their own position counts.](/figures/vjepa21/weighting.svg)](/figures/vjepa21/weighting.svg)

The top row is a **toy one-dimensional grid**, not the full training geometry. A visible patch bordering a hole is important for connecting the visible and hidden regions. Farther context receives a smaller weight. Paper Eq. 3 sets

$$
\lambda_i=\frac{\lambda}{\sqrt{d_{\min}(i,M)}}.
$$

$d_{\min}(i,M)$ is the minimum spatio-temporal distance, in token blocks, from visible position $i$ to a masked position. $\lambda$ is the base context coefficient. At $\lambda=0.5$, distances 1, 4 and 9 produce weights 0.5, 0.25 and approximately 0.167. These multiply the **context loss arrow at D**, not the token vector entering the encoder. The prose does not fully specify the distance norm or axis scaling, so the toy uses explicit one-dimensional distances. Distinct positions have positive distance; this rule is not evaluated on a masked position at distance zero.

The paper introduces a coefficient warmup across **epochs 50–100**. Its main training duration is stated in iterations; a complete reproduction needs the exact epoch-to-iteration convention rather than an invented conversion. Appendix A gives base coefficients **0.5 for videos and 0.7 for images**.

Weighting helps, but does not by itself restore the original recognition score. Table 2's weighted/warmup row is 33.8 mIoU and 62.5 SSv2 accuracy. The next ingredient, **deep self-supervision**, applies both loss branches at four encoder depths. This makes local information accessible throughout the network and improves the final-layer features.

A teaching summary is $\mathcal L_{\rm multi}\propto\sum_{\ell\in\{12,24,36,48\}}\mathcal L_{\rm dense}^{(\ell)}$ for ViT-G. The superscript $\ell$ identifies corresponding student predictions and teacher targets. This is not a numbered paper equation: §2.3.2 says both losses are used at four levels but does not give the exact cross-level reduction constant. Matching intermediate **features** is different from supervising only intermediate predictor blocks, and different from concatenating layers only at downstream evaluation.

<details><summary>Predict: does deep supervision require every downstream probe to use four layers?</summary><p>No. Table 8 uses final-layer encoder features. Appendix D.1 shows that deep-supervised features are stronger at the last layer; four-layer evaluation can still help a little.</p></details>

## 6 · Work one example through the architecture

[![One masked L1 error of 1 and two weighted context errors yield context loss 0.175 and dense loss 1.175 at one toy feature level.](/figures/vjepa21/numeric.svg)](/figures/vjepa21/numeric.svg)

This toy has **one masked token, two visible tokens, one feature level and two coordinates per vector**. At **A**, remove the masked patch. At **B**, encode the visible patches and fuse the selected layers. At **C**, add its mask token and predict all three positions. The **teacher branch** supplies these illustrative targets at **D**:

| Position | Teacher target $z_i$ | Predictor $\hat z_i$ | L1 error | Distance / weight |
| --- | --- | --- | --- | --- |
| Masked | $(2,1)$ | $(1.5,1.5)$ | $0.5+0.5=1$ | Masked branch, weight 1 |
| Context 1 | $(1,0)$ | $(1.2,0.2)$ | $0.2+0.2=0.4$ | $d=1$, weight 0.5 |
| Context 2 | $(0,1)$ | $(0.4,0.8)$ | $0.4+0.2=0.6$ | $d=4$, weight 0.25 |

Thus

$$
\mathcal L_{\rm predict}=1,
\qquad
\mathcal L_{\rm ctx}=\frac{0.5(0.4)+0.25(0.6)}{2}=0.175,
$$

$$
\mathcal L_{\rm dense}=1+0.175=1.175.
$$

The nearer context contributes 0.20 before averaging; the farther contributes 0.15 despite a larger raw error. Dividing by the weight sum 0.75 instead would give approximately 0.467 for the context branch, a different objective. The real model repeats the matching at four layers; this example deliberately isolates one. Its vectors are arbitrary and do not simulate learned normalization or feature semantics.

<details><summary>Predict: if both context errors double, does the masked loss change?</summary><p>No. The masked branch remains 1. The context branch doubles to 0.35 and the total becomes 1.35, with all weights and positions fixed.</p></details>

## 7 · The training loop and the inference branches

[![Training repeats masked student prediction, clean teacher comparison, gradient updates and EMA, followed by a high-resolution cooldown phase.](/figures/vjepa21/timeline.svg)](/figures/vjepa21/timeline.svg)

The first row is a repeated **learning algorithm**, not a temporal forecast rollout. Sample inputs and masks, run student and teacher, compute the four-level losses, backpropagate through the student/predictor, then update the teacher by EMA. Resample on the next iteration. Teacher targets move slowly; they are not an offline fixed feature dataset.

The primary phase runs **135k iterations**, with a **12k learning-rate warmup** and then a constant rate. Inputs are 256×256 images or 16-frame 256×256 clips sampled at 4 fps. Image-only and video-only workers aggregate their gradients before updating shared parameters; global batch sizes are 2304 images and 128 clips. VisionMix163M replaces ImageNet's 1M-image subset with LVD-142M and changes video sampling weights, including more SSv2 and YT-1B. This is a combined data-distribution intervention, not merely adding more copies of the old data.

The additional **12k-iteration cooldown** decays learning rate to $10^{-6}$ and raises image resolution to 512×512 and video inputs to 64 frames at 384×384. Appendix A uses patch size 16, tubelet size 2, weight decay 0.04 and EMA 0.99925. Most component ablations use the primary phase; final downstream tables use the full schedule.

[![Dense inference uses an encoder plus trained head; action anticipation keeps the JEPA predictor; robotics trains a new action-conditioned predictor; distillation uses a frozen teacher.](/figures/vjepa21/paths.svg)](/figures/vjepa21/paths.svg)

Read the rows as **separate task pathways**, not one serial pipeline:

**Dense depth or segmentation:** process a full image with the 2D tokenizer and frozen final-layer encoder; fit a supervised linear dense head. Depth additionally uses a learned BatchNorm layer. Training labels are needed for that head even though pretraining was self-supervised. Inference uses the learned head on new images, without SSL masks or four-level loss branches.

**Video object tracking:** Appendix C.3 uses frozen patch similarities to propagate a supplied first-frame object mask to later frames via spatially constrained weighted k-nearest neighbors. The first-frame mask is external information. No supervised tracker network is learned in this evaluation, but hyperparameters are selected on DAVIS training data.

**Action anticipation:** §3.2 keeps the frozen encoder **and JEPA predictor**. Observed video context and mask tokens for a future frame produce predicted future features. Concatenate observed and predicted features along the token dimension and train an attentive probe with separate action, verb and noun outputs. EK100 validation uses context ending one second before the action, with 32 frames at 8 fps, 384×384; report mean-class Recall@5. Appendix C.6 describes probe and context settings without fully restating the future-feature branch; the guide follows the explicit main-text pathway.

**Robot manipulation:** train a **new frame-causal, action-conditioned 24-layer predictor**, approximately 300M parameters, on Droid robot data with teacher-forcing and two-step rollout losses. At inference, model-predictive control (MPC) uses the cross-entropy method (CEM) to search candidate action sequences toward a visual goal, executes actions, observes again and replans. Randomly masking a training clip is not itself a robot policy. “Zero-shot” describes deployment in the lab environment after this action-data training.

**Navigation:** train a conditional diffusion transformer on features, predicting clean representations, and sample with DDIM. CEM searches 480 candidate 2-second trajectories at 4 fps in a reduced translation/yaw action space. Eight denoising steps versus at least 128 for the SD-VAE baseline support the reported speed comparison. This is distinct from both base SSL and arm-control planning.

**Distillation:** replace the loss teacher by a frozen ViT-G, use only last-layer matching and a newly trained **12-block** predictor. A separate student EMA is maintained for the released model but does **not** supply loss targets. Appendix B describes low-resolution teacher/student distillation then a high-resolution stage initialized from the first-stage student EMA. Do not transfer the pretraining teacher's role onto that student EMA.

## 8 · Evidence: first the tradeoff, then the final models

[![Sequential Table 1 ablation bars show ADE20K rising from 22.2 to 47.9, while SSv2 falls with context loss then recovers with multi-level supervision and scaling.](/figures/vjepa21/results.svg)](/figures/vjepa21/results.svg)

These are **reported Table 1 measurements**, not synthetic curves. The bars use a zero origin and show ADE20K linear-probe mIoU; the right column is SSv2 attentive-probe top-1 accuracy. The experiment starts with ViT-L, then scales to ViT-G and adds cooldown. No uncertainty intervals are supplied here.

| Sequential recipe | ADE20K mIoU ↑ | NYUv2 RMSE ↓ | SSv2 accuracy ↑ |
| --- | --- | --- | --- |
| V-JEPA 2 baseline | 22.2 | 0.682 | 72.8 |
| + Context loss | 33.8 | 0.474 | 62.5 |
| + Multi-level prediction | 38.6 | 0.463 | 72.1 |
| + VisionMix | 40.8 | 0.418 | 72.6 |
| + Multi-modal tokenizer | 41.4 | 0.415 | 72.6 |
| + Model scaling | 47.1 | 0.365 | 76.1 |
| + Cooldown | 47.9 | 0.307 | 77.7 |

The context row establishes a concrete tradeoff. Multi-level supervision recovers **9.6 accuracy points** while improving segmentation another 4.8 points; it does not exactly restore the original 72.8. Table 1 supports an incremental recipe, not independent causal contributions of every component under all possible orderings. A stronger ablation would hold data, compute and size fixed and test a factorial interaction between weighted context loss and deep supervision. Proposed ablations here are study ideas, not completed experiments.

**Final-model comparison, Table 8:** V-JEPA 2 ViT-g gives NYUv2 0.642 RMSE and ADE20K 24.4 mIoU; V-JEPA 2.1 ViT-G gives 0.307 and 47.9. Those are different settings from the initial ViT-L baseline above. The change from 1B to 2B also prevents assigning this whole improvement to the loss alone. DINOv3 7B has NYUv2 0.309, KITTI 2.346 and ADE20K 55.9; V-JEPA 2.1 ViT-G has KITTI 2.461. The new model is competitive across tasks, not uniformly superior. Segmentation uses frozen final-layer features with a trained linear head; Appendix C.2 gives 512-pixel ADE20K/VOC and 1024-pixel-height Cityscapes evaluation.

Other reported results and their scope:

| Evaluation | Reported V-JEPA 2.1 result | Setting and comparison |
| --- | --- | --- |
| Ego4D STA v2, Table 4 | 7.71 Top-5 mAP All | Frozen ViT-G; trained localization/anticipation head; V-JEPA 2 is 6.02; the prior task-specific STAformer is 5.67. “All” requires noun, verb, box overlap and contact-time criteria, not simple classification accuracy. |
| EK100 anticipation, Table 5 | 40.8 action Recall@5 | Validation, one-second anticipation, ViT-G; V-JEPA 2 ViT-g is 39.7. V-JEPA 2.1 ViT-g is 38.4. |
| SSv2 recognition, Table 9 | 77.7% top-1 | Frozen ViT-G with trained attentive head; V-JEPA 2 is 77.3. On Diving-48, 89.2 is below V-JEPA 2's 90.2. |
| NYUv2 depth, Table 8 | 0.307 RMSE | Frozen ViT-G image features, supervised linear probe; lower is better. |
| Tartan Drive navigation, Table 7 | 5.687 normalized ATE | Open-loop planning; average planning time 10.6 s vs NWM 103.2 s. Other datasets have exceptions, e.g. ViT-G Sacson ATE 4.054 vs NWM 4.037. |

For **robot grasping**, Table 6 averages ten tasks per skill. Baseline V-JEPA 2: 60% grasp success at 800 samples, 10 CEM iterations, horizon 1 and 3 seconds planning time. V-JEPA 2.1 at the same listed settings: **70%**, a 10 percentage-point gain. Changing V-JEPA 2.1 to 300 samples, 15 iterations, horizon 8 and **14 seconds** yields **80%**, a 20-point gain over baseline. That is not a 20% relative improvement, nor a fixed-planner comparison. Reach stays 100% and pick-and-place 80%. A larger controlled trial is needed before treating the small task average as a robust general success rate.

Table 11's distilled ViT-L improves ADE20K from 42.0 to 46.7 and SSv2 from 74.2 to 76.5 relative to training it from scratch, but still trails ViT-G on those metrics. Its teacher is not a free source of performance: producing the teacher incurs the large pretraining cost.

## 9 · Limits and source details worth noticing

**Features are not guaranteed physical state.** Latent matching helps geometry and temporal structure, but does not prove that all needed task variables are present, that forecasts resolve multimodal futures, or that a planner will safely act. Robot failures in §3.3 include premature closing and opening of the gripper. A good spatial representation does not fix every control problem.

**PCA is qualitative evidence.** §3.9 maps three feature components to RGB and picks the most visually appealing of six channel permutations. Smooth matching colors are not measured segmentation accuracy or calibrated object identities; compare quantitative probes too.

**Protocols and data matter.** Recognition uses attentive supervised probes, depth/segmentation linear heads, tracking first-frame labels and selected propagation settings, manipulation action data, and VideoQA extra language alignment. None is an annotation-free turnkey consequence of one encoder checkpoint. VisionMix and 2B-scale training limit lightweight reproduction. The paper’s appendix omits some exact distance/reduction conventions needed for a standalone implementation.

**The supplied version has small internal discrepancies.** We retain table values and expose exceptions instead of silently harmonizing them:

- §2.2/§2.3.1 prose reports context-only 33.9 mIoU and 0.473 RMSE; Table 1 gives **33.8 and 0.474**, used in the chart.
- §2.3.2 prose says 72.0 SSv2 after multi-level supervision; Table 1 gives **72.1**.
- Table 9 caption lists SSv2 16-frame evaluation clips, while Appendix C.5 prints a **64×2×3** input but then describes 16-frame clips. The exact clip setting needs reproduction-code confirmation; the 77.7 score is retained as reported, without inventing a reconciliation.
- Table 8 and Table 14 disagree on some tracking entries, e.g. ViT-G YouTube-VOS **72.7 vs 72.6**. §3.10 prose cites **67.0** as the ViT-G DAVIS score, while Table 11 gives **69.0** (67.0 is its distilled ViT-B row).
- Appendix D.2 says higher cooldown resolution helps every task; Table 14 has exceptions such as ViT-g DAVIS **68.0 → 67.4** and ViT-G Diving-48 **89.5 → 89.2**.

These do not negate the dense-feature gains, but they constrain precise comparative claims. Report the version, table and setting when quoting a number.

## 10 · Reading map and experiments to try

| Guide mechanism | Supplied v3 source |
| --- | --- |
| Masked prediction and stop-gradient teacher | §2.1, Eq. 1, pp. 3–4 |
| Visible-token hypothesis and context target | §2.2, Figure 3, Eq. 2, pp. 4–5 |
| Full training architecture | Figure 4, p. 5; architecture description, p. 6 |
| Distance weighting and warmup | §2.3.1, Eq. 3, Table 2, p. 7 |
| Four-layer supervision | §2.3.2, pp. 7–8; Appendix A, Table 12, p. 33 |
| Data/tokenizers/scaling | §§2.3.3–2.3.5, Table 3, pp. 8–9 |
| Distillation teacher and student EMA | §2.3.6, p. 9; Appendix B, p. 34 |
| Anticipation, manipulation, navigation | §§3.1–3.4, Tables 4–7, pp. 10–14 |
| Frozen dense/global probes | §§3.5–3.7, Tables 8–9, pp. 15–19; Appendix C |
| Feature visualization and compact models | §§3.9–3.10, Figures 14–15, Table 11, pp. 20–23 |
| Layer/resolution evaluation ablations | Appendix D, Tables 13–14, p. 37 |

Three useful **proposed** experiments: hold model/data/compute fixed while comparing no context loss, constant context weights and distance weights; compare last-layer and four-layer probes for encoders trained with/without deep supervision; hold CEM horizon, sample count and runtime fixed when testing robot encoders. Each isolates a different claim. No such additional trials have been run for this guide.

Further reading: the [authors’ implementation](https://github.com/facebookresearch/vjepa2), the [matching arXiv version](https://arxiv.org/abs/2603.14482v3), and the [Topic 3 reading library](/materials?topic=3). The local course PDF is the authority for this guide’s equations and result numbers.

## The idea to carry forward

**A feature can be useful for prediction elsewhere without being useful at its own location.** V-JEPA 2.1 supplies local targets at visible and masked sites, controls their relative influence, and carries the signal through several network depths. The payoff is a stronger basis for dense probes and learned world models; the remaining task heads, action predictors and planning choices still matter.
