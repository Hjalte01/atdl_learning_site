---
title: "DINOv3"
shortTitle: "DINOv3"
topic: "topic-3"
order: 1
description: "Why better image recognition can mean worse patch features—and how Gram anchoring preserves their relationships. A visual architecture-to-math guide."
status: "ready"
source: "https://arxiv.org/abs/2508.10104v1"
pdf: "/papers/dinov3.pdf"
---
## The puzzle: recognize the bird, lose its outline

Imagine a bird perched on a branch. An image classifier only needs evidence that a bird is present. A segmentation system needs to distinguish bird patches from branch and sky patches. Could training a better image representation make the second job harder?

**DINOv3 observes exactly this tension during long self-supervised training. Its central intervention preserves relationships between patch features while allowing the feature coordinates themselves to evolve.** The intervention is called **Gram anchoring**.

This is Topic 3, paper 1. We follow the supplied **67-page arXiv:2508.10104v1 report, 13 August 2025**. The [matching primary record](https://arxiv.org/abs/2508.10104v1) confirms the version. All eight diagrams below are original teaching graphics; only the result bars reproduce reported measurements. No trained model runs on this page. Allow about 25 minutes.

## 1 · Global success does not guarantee local coherence

[![Global classification asks whether a bird exists, while dense prediction needs coherent patch features to locate it.](/figures/dinov3/problem.svg)](/figures/dinov3/problem.svg)

Read the two rows as different uses of the same encoder. The **CLS token** summarizes an image. **Patch tokens** retain a spatial grid of learned vectors. A patch vector is contextual: transformer attention lets it use information from elsewhere in the image. It is neither a raw pixel nor a semantic label.

In §4.1 / Figures 5–6, ImageNet-1k linear classification improves over training while Pascal VOC linear segmentation declines after roughly 200k iterations. Similarity maps become less localized: irrelevant patches become highly similar to a reference patch. With four register tokens, patch norms remain stable, so this is distinct from the older high-norm outlier problem. Increasing CLS–patch cosine similarity is another observed sign of lost locality.

A plausible reconstruction of the design logic is: keep the useful global learning, add an explicit constraint on patch relationships, and choose a target from a checkpoint whose local geometry was better. This is a teaching interpretation, not the authors’ private reasoning or a proof of the cause of every artifact.

<details><summary>Predict: would checking only ImageNet accuracy reveal this problem?</summary><p>No. The image-level score can keep improving while dense prediction becomes worse. Monitor both global and local tasks, along with patch-similarity maps.</p></details>

## 2 · Architecture: two views, two token roles, three losses

[![Student crops pass through a ViT to CLS and patch heads; an EMA teacher supplies targets, and three losses update the student.](/figures/dinov3/architecture.svg)](/figures/dinov3/architecture.svg)

**A → B:** form two global crops of 256×256 pixels and eight local crops of 112×112 from each training image. The student sees all ten; the moving-average teacher sees only the two unmasked global crops. Student global patch tokens are masked for the iBOT branch. Appendix C specifies a random masked proportion in [0.1, 0.5] with 50% probability.

**B:** the flagship ViT has **6.7B parameters**, rounded to “7B,” 40 blocks, width $d=4096$, patch size 16, 32 attention heads and four registers (Table 2). It uses SwiGLU and rotary positional embeddings (RoPE). Coordinate-box jittering scales normalized coordinates by a random factor in [0.5, 2], helping robustness to scale and aspect ratio. Registers are extra learned tokens; they are not four image patches. The diagram shows functional branches rather than every layer.

**B → C:** dedicated heads produce distributions over learned prototypes. The DINO head acts on CLS; the iBOT head acts on patch features. These prototype indices are learned categories, not human annotation labels. Sinkhorn–Knopp balancing replaces DINO’s original centering in both objectives.

- **DINO:** match teacher and student image-level distributions across views. Local student crops match global teacher crops; global-to-global pairs use different views.
- **iBOT:** predict the teacher’s visible patch distribution at student-masked positions of corresponding global crops. This is latent-target reconstruction, not pixel reconstruction.
- **Distributed KoLeo (DKoleo):** encourage spread of student CLS features; the regularizer uses small groups of 16 samples, potentially across GPUs.

Paper Eq. 1 combines those arrows at the student optimizer:

$$
\mathcal L_{\rm Pre}=\mathcal L_{\rm DINO}+\mathcal L_{\rm iBOT}+0.1\mathcal L_{\rm DKoleo}.
$$

Each $\mathcal L$ is a scalar loss; 0.1 is the initial KoLeo weight. The global and masked-local objectives supply complementary learning signals; spread regularization discourages crowded representations. None alone guarantees spatial coherence throughout an extended run.

To understand a distribution-matching head, a **teaching form** of cross-entropy is

$$
H(q_T,p_S)=-\sum_{k=1}^{K}q_T(k)\log p_S(k).
$$

$K$ is the head’s prototype count; $q_T$ is a balanced teacher target distribution; $p_S$ is the student distribution. This operation lives after the appropriate CLS or patch head. For toy $q_T=(1,0)$ and $p_S=(0.8,0.2)$, the loss is $-\log 0.8\approx0.223$. Predicting $(0.2,0.8)$ gives 1.609. The real targets need not be one-hot, and the full algorithm includes crop pairing, temperatures, balancing and averaging. Table 2 gives different prototype counts for DINO (256k) and iBOT (96k).

Only the student receives gradients. The ordinary teacher follows an exponential moving average, illustrated by

$$
\theta_T\leftarrow m\theta_T+(1-m)\theta_S,\qquad m=0.999.
$$

$\theta_S,\theta_T$ are student and teacher parameter tensors, not activations. For old teacher weight 2 and student weight 3, the new teacher weight is 2.001. This stabilizes targets while letting them evolve. It is a parameter update, not an extra prediction loss.

## 3 · What is actually scaled?

Section 3.1 starts with approximately 17 billion public Instagram images. Hierarchical clustering using DINOv2 embeddings creates **LVD-1689M**, a balanced sampled subset of 1,689 million images. The final recipe also uses retrieval around seed datasets and public vision datasets. LVD-1689M is one component of the mixture, not the total raw pool or a statement that all images have annotations.

Ten percent of training iterations use homogeneous ImageNet-1k batches; other iterations mix the other components. Table 1’s shorter 200k-iteration ablation shows no single curation strategy is best across every benchmark. Thus “more data” includes decisions about which visual content is sampled.

The web flagship pretraining uses 4096 images per batch across 256 GPUs for 1M iterations (§3.2, Appendix C). After warmup, learning rate, weight decay and EMA momentum are constant. Appendix C specifies a 100k-step learning-rate warmup, learning rate 0.0004, weight decay 0.04, layerwise learning-rate decay 0.98 and stochastic depth 0.4. Later distillation uses a cooldown; “constant schedule” should not be extended to every stage.

Self-supervised here means the core objectives do not require manual per-image task targets or captions. It does not mean the entire data pipeline is uncurated, downstream probes need no labels, or the optional text-alignment stage is caption-free.

## 4 · Representation close-up: a matrix of patch relationships

[![A 256-patch grid produces a 256 by 4096 feature matrix and a 256 by 256 cosine-similarity Gram matrix.](/figures/dinov3/representation.svg)](/figures/dinov3/representation.svg)

At 256×256 with patch size 16, the patch grid is 16×16, so $P=256$. Let $X\in\mathbb R^{P\times d}$ contain **row-normalized patch vectors**. CLS and registers are excluded. The Gram matrix is

$$
G=XX^\top,\qquad G_{ij}=x_i^\top x_j.
$$

Here $x_i$ is a unit-length feature vector at patch position $i$. $G_{ij}$ is cosine similarity between two patch features, in [−1, 1]. For nonzero normalized rows, $G_{ii}=1$. This matrix compares all patch pairs, not merely adjacent pixels. It has shape $P\times P$, unlike the channel-correlation matrix $X^\top X$.

**Why these relationships?** Patch matching, object discovery and dense prediction benefit when visually meaningful regions have coherent features. But equal Gram matrices do not guarantee correct semantic labels: the anchor can preserve a mistaken relationship too. At higher resolution, $P$ increases and storing every pair scales quadratically in patch count.

## 5 · Gram anchoring: stabilize geometry, keep coordinate freedom

[![Student and stable Gram-teacher patch branches form matching Gram matrices; higher-resolution teacher features are downsampled to the student grid.](/figures/dinov3/gram.svg)](/figures/dinov3/gram.svg)

Read the student and Gram-teacher rows as **parallel paths for the same global crop content**. The target teacher is initially an early checkpoint with good dense features, distinct from the ordinary moving-average teacher. The student remains trainable. Paper Eq. 2 compares the two relationship matrices:

$$
\mathcal L_{\rm Gram}=\left\|X_SX_S^\top-X_GX_G^\top\right\|_F^2.
$$

$X_S$ and $X_G$ contain normalized student and Gram-teacher patch features, with matching spatial rows. $\|A\|_F^2=\sum_{i,j}A_{ij}^2$ sums squared entry errors. This loss operates **between the two Gram boxes**, on global crops only. Teacher targets receive no gradient.

In §4.3, the Gram teacher sees a crop at twice the normal resolution. Its feature map is bicubically downsampled by 2 to match the student’s grid; the resulting patch vectors define $X_G$. That procedure brings finer teacher information and smoothing into the target relationship matrix. It is not downsampling a predicted label mask. Figure 9(b) tests teacher checkpoints and resolution; the 1M checkpoint is worse than earlier anchors in the reported dense metrics.

Why compare Gram matrices instead of forcing identical vectors? For an orthogonal coordinate transform $Q$, $QQ^\top=I$:

$$
(X_SQ)(X_SQ)^\top=X_SQQ^\top X_S^\top=X_SX_S^\top.
$$

This is our derivation. A shared rotation or reflection preserves relationships, so it pays no additional Gram penalty. Independent arbitrary per-patch changes generally do not. “Free to move” therefore means constrained freedom, not an absence of gradient on features.

The refinement loss keeps the original tasks (paper Eq. 3):

$$
\mathcal L_{\rm Ref}=w_D\mathcal L_{\rm DINO}+\mathcal L_{\rm iBOT}
+w_{DK}\mathcal L_{\rm DKoleo}+w_{\rm Gram}\mathcal L_{\rm Gram}.
$$

$w_D,w_{DK},w_{\rm Gram}$ are weights balancing the global, spread and relationship terms; iBOT’s displayed coefficient is 1. Appendix C states $w_{\rm Gram}=2$. We do not infer unlisted refinement weights from the pretraining coefficient. In Figure 7, Gram refinement lowers iBOT loss more sharply while the DINO loss changes relatively little. That is reported evidence of partially separable roles, not proof that the objectives never interact.

<details><summary>Predict: if student and Gram-teacher vectors differ by the same rotation, must Gram loss be positive?</summary><p>No. A shared orthogonal rotation preserves every pairwise dot product. Gram anchoring preserves relational structure rather than requiring identical feature coordinates.</p></details>

## 6 · Follow two patches through the exact boxes

[![Two normalized student patch vectors produce off-diagonal similarity 0.6, compared with zero in the teacher, giving squared Gram loss 0.72.](/figures/dinov3/numeric.svg)](/figures/dinov3/numeric.svg)

Use a deliberately tiny $P=d=2$ example, unrelated to an actual trained image. The target vectors are orthogonal; the student makes the two patches too similar:

$$
X_G=\begin{bmatrix}1&0\\0&1\end{bmatrix},\qquad
X_S=\begin{bmatrix}1&0\\0.6&0.8\end{bmatrix}.
$$

Both student rows are unit length because $0.6^2+0.8^2=1$. The Gram boxes produce

$$
G_G=\begin{bmatrix}1&0\\0&1\end{bmatrix},\qquad
G_S=\begin{bmatrix}1&0.6\\0.6&1\end{bmatrix}.
$$

Subtracting and summing squared entries gives $\mathcal L_{\rm Gram}=0.6^2+0.6^2=0.72$. Both symmetric off-diagonal entries count; using only one gives the wrong value for the displayed Frobenius norm. With weight 2, this contributes **1.44** to the refinement loss, before its other terms. Moving the second student vector toward $(0,1)$ reduces the unwanted similarity.

The paper equation is an unnormalized squared norm. This toy calculation follows it literally; averaging matrix entries would change the numerical scale to $0.72/4=0.18$. Do not silently switch reduction conventions or mistake these loss units for an mIoU score.

## 7 · Follow the training state over time

[![Pretraining updates student and EMA teacher; refinement adds a fixed-between-refreshes Gram teacher; high-resolution adaptation is a separate later phase.](/figures/dinov3/timeline.svg)](/figures/dinov3/timeline.svg)

The web-model sequence is:

1. **Pretrain for 1M iterations:** update student by the original three losses and update ordinary teacher by EMA; preserve an early checkpoint for anchoring.
2. **Refine:** add Gram loss to repair degraded dense features. The Gram teacher stays fixed within each 10k-step interval.
3. **Refresh the Gram target:** copy the current ordinary EMA teacher into the Gram teacher every 10k steps, **at most three updates** in Appendix C. This carries improved geometry forward; it does not freeze one early target forever.
4. **Adapt resolution for 10k additional iterations:** §5.1 uses larger student global crops (512/768) and Gram anchoring with the 7B teacher. Appendix C lists specific global/local/Gram crop mixtures; those Gram sizes are 768/1152, so the earlier refinement’s “2×” rule is not a universal ratio for every later stage.

Figure 8 reports dense-task gains within the first 10k Gram iterations. High-resolution adaptation improves local performance as image resolution increases, but §5.1 also notes slightly worse low-resolution ObjectNet transfer. Successful adaptation is not a guarantee that arbitrarily increasing resolution improves every task.

<details><summary>Predict: is the Gram teacher just another name for the ordinary EMA teacher?</summary><p>No. EMA follows the student every step. The Gram teacher begins as an early snapshot and is fixed between discrete refreshes. At a refresh it is copied from EMA; the two then diverge again.</p></details>

## 8 · Inference, smaller models and text are separate paths

[![Fixed-teacher distillation trains smaller students; frozen-backbone dense probing trains a labeled head; text alignment is a separate caption-supervised path.](/figures/dinov3/paths.svg)](/figures/dinov3/paths.svg)

**Ordinary downstream inference:** an image enters a trained encoder once, producing CLS and patch features. A task-specific head or non-parametric procedure consumes them. There is no EMA update, Gram loss or iterative anchoring at test time. Segmentation requires a decoding/head procedure; the base encoder is not itself a class-label mask generator.

**Dense linear probing (§6.1.2, Appendix D):** keep the backbone frozen, train a linear transform on labeled segmentation/depth training data. The protocol also uses trained batch normalization after layer-normalized features and a learning-rate/weight-decay sweep. “Without fine-tuning” refers to the backbone, not to zero training or zero task annotations.

**Distillation (§5.2):** a fixed flagship 7B teacher trains smaller students with the original pretraining objectives. The report does not observe patch-consistency problems here and **does not apply Gram anchoring**, including in their high-resolution stage. Students train for 1M iterations plus 250k cosine cooldown. The family spans S (21M), S+ (29M), B (86M), L (0.3B) and H+ (0.8B). Shared teacher inference across student groups reduces repeated teacher computation; it does not imply all student architectures have identical costs.

**Text alignment (§5.3):** freeze the vision backbone, add two transformer layers and learn a text encoder with image–caption contrastive training. Concatenate CLS with mean-pooled patch embeddings before matching text. These extra training signals add open-vocabulary capabilities; they are absent from base image-only SSL. Table 16’s text-aligned ViT-L excels in dense alignment but trails SigLIP 2 and PE on several global alignment metrics.

**Domain transfer (§8):** the report separately trains a satellite model on SAT-493M RGB images with a shorter schedule. Its task-dependent strengths differ from the web model. Do not substitute satellite training counts for the web flagship schedule above.

## 9 · Evidence: frozen features, learned heads

[![Selected Table 3 bars show ADE20k mIoU of 49.5 for DINOv2, 42.7 for Web-DINO, 53.0 for AM-RADIOv2.5 and 55.9 for DINOv3.](/figures/dinov3/results.svg)](/figures/dinov3/results.svg)

These are measured values from **v1 Table 3**, not outputs of this website. Segmentation uses 1024 patch tokens for all models: 448×448 images for patch 14 and 512×512 for patch 16. Each has a frozen backbone and separately trained linear predictor. The chart selects four baselines; the source table has eight rows.

| Frozen backbone | ADE20k mIoU ↑ | Cityscapes mIoU ↑ | VOC mIoU ↑ | NYUv2 RMSE ↓ | KITTI RMSE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| DINOv2 g/14 | 49.5 | 75.6 | 83.1 | 0.372 | 2.624 |
| Web-DINO 7B/14 | 42.7 | 68.3 | 76.1 | 0.466 | 3.158 |
| AM-RADIOv2.5 g/14 | 53.0 | 78.4 | 85.4 | 0.340 | 2.918 |
| DINOv3 7B/16 | **55.9** | **81.1** | **86.6** | **0.309** | **2.346** |

mIoU averages class intersection-over-union. For one toy class with 80 pixels overlapping out of a 100-pixel union, IoU is 80%; that is not the same as Dice. RMSE penalizes squared depth error and then takes a square root; lower is better. DINOv3 improves ADE20k by **6.4 percentage points** over DINOv2 in this protocol. This comparison changes capacity, data and training recipe together; it cannot assign the entire improvement to Gram anchoring.

A closer mechanism test is **Figure 9(b)**:

| Refinement setting | IN1k linear accuracy | ADE20k mIoU | NYUv2 RMSE ↓ |
| --- | ---: | ---: | ---: |
| Baseline | 88.2 | 50.3 | 0.307 |
| Gram: 200k teacher, ×1 resolution | 88.0 | 53.6 | 0.285 |
| Gram: 200k teacher, ×2 resolution | 88.0 | 55.7 | 0.281 |
| Gram: 100k teacher, ×2 resolution | 87.9 | 55.7 | 0.284 |
| Gram: 1M teacher, ×2 resolution | 88.1 | 54.9 | 0.290 |

Higher-resolution 200k anchoring improves ADE20k by 2.1 points over ×1 and 5.4 over the baseline, while image classification is slightly lower. The 1M anchor still improves on baseline but is weaker than early ×2 anchors. These ablation values precede the final Table 3 setting: **55.7 and 55.9 are not interchangeable**. No confidence intervals are supplied in these rows.

Beyond the chart, Table 4 reports NAVI geometric correspondence recall 64.4 versus DINOv2’s 60.1, and SPair semantic recall 58.7 versus 56.1. Section 6 includes object discovery, tracking and video probes; §6.3 adds trained detection/segmentation/depth systems. These use different protocols and supervision; do not combine their metrics into one universal “quality percentage.”

## 10 · What remains limited, and what would you test?

- **No universal superiority:** Appendix B / Table 25 shows OCR-heavy datasets where weakly supervised PE-core is much stronger, such as Logo-2K+ (93.2 versus DINOv3’s 86.0). Figure 9 also shows a small global-score tradeoff.
- **Coverage is not fairness:** Table 26 reports low-income accuracy 69.6 versus high-income 90.9, and Africa 76.7 versus Europe 90.7. Improvements over DINOv2 do not eliminate gaps. Balanced embedding clusters do not prove social or geographic balance.
- **Visual evidence needs context:** §6.1.1 chooses the visually most compelling PCA sign/color permutation. PCA colors are not class labels; qualitative appeal alone cannot establish prediction accuracy.
- **Targets can carry errors:** the anchor protects its own similarity geometry. An ablation should vary anchor quality and examine boundary failures, not just aggregate dense scores. Hold model size, data, optimization and evaluation fixed to isolate Gram loss, resolution and refresh timing.
- **Compute and reproducibility:** the huge curated training mixture is not reproduced here. Table 20 estimates 47 MWh and 18 tCO₂-equivalent for a flagship pretraining run under stated power/grid assumptions. Section 9’s rough whole-project estimate is about 2600 tCO₂-equivalent; these are different scopes. Appendix A discusses additional outliers, so Gram anchoring is not a claim that every representation artifact disappears.

The proposed tests above are our research questions. This guide explains the report; it neither reproduces its benchmarks nor promises transfer to a new application.

## 11 · The idea to carry forward

**When one representation serves several tasks, monitor the properties each task actually needs. If relationships matter more than feature coordinates, regularize the relationships directly.**

Redraw three weight states from memory: trainable student, per-step EMA teacher, and periodically refreshed Gram teacher. Then mark which head uses CLS, which predicts masked patch targets, and where the $P\times P$ comparison happens. Finally erase every training-only branch to draw inference. That connects what happens, where the math happens, and why this design makes sense.

## Sources and reading map

- [Supplied course PDF: DINOv3, arXiv v1, 13 August 2025](/papers/dinov3.pdf), preserved byte-for-byte.
- [Matching primary arXiv record](https://arxiv.org/abs/2508.10104v1) and [authors’ implementation](https://github.com/facebookresearch/dinov3) for further reading. No later repository result substitutes for the course version.
- §3.1 / Table 1: curation and sampling. §3.2 / Eq. 1 / Table 2 / Appendix C: initial architecture, crop pairing, optimization.
- §4 / Figures 5–10 / Eqs. 2–3: degradation, anchoring, high-resolution target and teacher ablation.
- §5 / Figure 12: resolution adaptation, distillation and text alignment. §6.1.2 / Table 3 / Appendix D: dense probe evidence. §8: separate satellite model. §9 / Table 20: compute estimates. Appendices A–B / Tables 25–26: artifacts, OCR and fairness caveats.

All examples and architecture graphics are teaching constructions; reported numbers retain their source version and protocol.
