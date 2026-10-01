---
title: "Active Prompt Learning with Vision-Language Model Priors"
shortTitle: "Active Prompt · CB + SQ"
topic: "topic-4"
order: 3
description: "Use soft class evidence to shape clusters, then spend human annotation budget only where class-specific confidence thresholds call for it."
status: "ready"
source: "https://openreview.net/forum?id=qBeGCzD3Ij"
pdf: "/papers/active-prompt.pdf"
---
## 1 · The puzzle: must every useful image cost a human label?

You have a pretrained CLIP model, an unlabeled image pool and the names of the classes you want to recognize. A limited annotation budget must cover different kinds of images. Random sampling may miss important regions; ranking only by uncertainty may repeatedly choose unusual or similar examples. But selecting diverse examples creates a second problem: some representatives are already easy for the model. Must you pay a human to label those too?

**Active Prompt combines class-guided, cluster-balanced acquisition (CB) with selective querying (SQ).** It first picks representatives in a feature space informed by both images and class descriptions. It then decides whether each candidate needs an oracle label or can receive the model’s own pseudo-label. A new prompt learns from the accumulated dataset while the pretrained encoders stay frozen.

This is Topic 4 paper 3 by Hoyoung Kim, Seokhee Jin, Changhwan Sung, Jaechang Kim and Jungseul Ok. We follow the supplied **24-page paper headed “Published in Transactions on Machine Learning Research (10/2025)”**, including its appendices. “Active Prompt” is the course filename shorthand; the full title appears above. Allow about 20 minutes. The colored figures are original teaching diagrams; only the results chart contains paper measurements. No trained CLIP model runs on this page. Open figures to enlarge them.

[![Random sampling queries every selected image; class-guided clustering chooses representatives and selective querying splits human and pseudo targets, with a risk of incorrect pseudo-labels.](/figures/active-prompt/problem.svg)](/figures/active-prompt/problem.svg)

**Read selection and payment as two separate decisions.** Diversity says which images enter the training set; confidence says which require human labels. The diagram is schematic, and its inequality describes annotation counts rather than accuracy.

A plausible design rationale is: use pretrained semantic knowledge to avoid a random cold start, cover the task’s feature regions, then reserve human effort for representatives the model cannot confidently label. This is a teaching interpretation of the mechanism, not a reconstruction of the authors’ private thoughts.

## 2 · Architecture: predictions shape geometry before labels are revealed

[![Parallel frozen image and text encoders yield probabilities; probabilities weight text features, concatenation feeds clustering, a threshold gate supplies targets, and cross-entropy trains reset prompts.](/figures/active-prompt/architecture.svg)](/figures/active-prompt/architecture.svg)

**A predicts, B represents and selects, C assigns targets, and D learns.** Image and text encoders are parallel branches. The probability vector from A has one component per target class; B’s concatenated feature has an image block and a text-mixture block. C uses the previous round’s data and prompt. D resets the prompt parameters, then returns a learned prompt for the next round. Arrows between rows are described by this A → B → C → D reading sequence.

Let $\mathcal C$ be the known class set, $C=|\mathcal C|$ its size, $x$ an image, $\theta_{\mathrm{img}}$ the frozen image encoder, and $\theta_{\mathrm{txt}}$ the frozen text encoder. A text prompt $t_c$ contains context and class $c$’s name. Write their shared-space outputs as $I(x),T_c\in\mathbb R^d$; $d$ is left symbolic. Paper Eq. (1), at A’s encoder-to-probabilities arrow, is

$$
p_c(x;t)=\frac{\exp(\cos(\theta_{\mathrm{img}}(x),\theta_{\mathrm{txt}}(t_c))/\tau)}{\sum_{k\in\mathcal C}\exp(\cos(\theta_{\mathrm{img}}(x),\theta_{\mathrm{txt}}(t_k))/\tau)}.
$$

$\cos$ compares embedding directions and $\tau>0$ is temperature. Softmax makes the scores nonnegative and sum to one; temperature changes their concentration. No true image label enters this calculation. Source Eq. (2) takes the highest-probability class:

$$
\widetilde y(x;t)=\operatorname*{arg\,max}_{c\in\mathcal C}p_c(x;t).
$$

This is a **prediction**, not verified truth. High softmax confidence can accompany an incorrect answer, especially under shift. Class names and prompt wording change the probabilities.

In round one, the context is the hand-crafted “a photo of a” template. Later acquisition uses the preceding learned prompt $t_{r-1}$. Source Eq. (3) represents a learned context as $[V_1]\cdots[V_M][\mathrm{CLASS}]$, with trainable vectors $V_m$. Section 5.1 specifies CLIP **ViT-B/32**, 16 context vectors, Gaussian reinitialization with mean zero and standard deviation 0.02, 200 training epochs per round, SGD starting at 0.002 and cosine learning-rate decay. The guide does not invent transformer layer widths or additional training settings.

## 3 · Representation: keep the image, add soft class evidence

A hard pseudo-label discards almost all competing-class information. B instead uses the whole probability vector to blend class text representations. Source Eqs. (5)–(7) give

$$
I(x)=\theta_{\mathrm{img}}(x),\qquad
\widetilde T_{\mathcal C}(x;t)=\sum_{c\in\mathcal C}p_c(x;t)\,\theta_{\mathrm{txt}}(t_c),
$$

$$
F_{\mathcal C}(x;t)=[I(x),\widetilde T_{\mathcal C}(x;t)]\in\mathbb R^{2d}.
$$

$\widetilde T$ is the probability-weighted text feature. Square brackets here mean concatenation, not addition. The class set determines which semantic directions are available. The image block retains visual information; the mixture block tells clustering what the VLM believes about the target classes. This happens **before acquisition**, not as an extra supervised prediction head.

[![A two-dimensional image feature is concatenated with a probability-weighted two-class text mixture, producing a four-dimensional feature; a second mixture changes squared distance by 0.72.](/figures/active-prompt/representation.svg)](/figures/active-prompt/representation.svg)

**Trace the toy through A’s probabilities and B’s mixture and concatenation boxes.** Choose illustrative encoder outputs $I=(1,0)$, $T_A=(1,0)$ and $T_B=(0,1)$. Suppose A produces $p=(0.8,0.2)$; these probabilities are a stated toy input, not a claim about a particular CLIP temperature. Then $\widetilde T=(0.8,0.2)$ and $F=(1,0,0.8,0.2)$. With the same image block and mixture $(0.2,0.8)$, squared Euclidean distance is $0.6^2+(-0.6)^2=0.72$.

The last comparison isolates the mixture’s contribution mathematically; identical images processed by the same deterministic model would not have different probabilities. In practice, images and their predictions change together. The paper does not specify an additional block-rescaling rule in Eqs. (5)–(7); we do not insert one. Concatenation can preserve useful details, but also carries encoder biases and errors into the clustering geometry.

The paper visualizes this representation with t-SNE and **GradFAM** (§4.1, Appendix A). GradFAM differentiates a target-feature cosine similarity with respect to activation maps, averages those gradients and forms a ReLU-weighted, upsampled activation map (Eqs. 20–21). For the concatenated feature target it duplicates the image vector to match dimensions. These are representation-analysis tools, not steps needed to select labels. A heatmap is not a correctness proof; a two-dimensional t-SNE projection is not the actual clustering space.

<details><summary>Predict: does the text mixture require knowing the image’s correct class?</summary>

No. Eq. (6) uses predicted probabilities and known class-name prompts. Replacing the mixture with the text feature of the true label creates an oracle-assisted comparison, such as CB* in Table 1, not ordinary unlabeled acquisition.

</details>

## 4 · Acquisition: choose real representatives, not synthetic centroids

Now apply K-means to class-guided features. In round one, set $K=B$, where $B$ is the maximum number of candidate images per round. For cluster $\mathcal C_i$, source Eqs. (8)–(10) compute

$$
c_i=\frac{1}{|\mathcal C_i|}\sum_{x\in\mathcal C_i}F_{\mathcal C}(x;t),\qquad
x_i^*=\operatorname*{arg\,min}_{x\in\mathcal C_i}\|F_{\mathcal C}(x;t)-c_i\|_2,
\qquad Q=\{x_1^*,\ldots,x_B^*\}.
$$

$\mathcal C_i$ denotes a feature cluster, whereas $\mathcal C$ denotes the target **class set**. The centroid $c_i$ is an average feature vector. $x_i^*$ is an actual image closest to that mean; $Q$ is the candidate set. Selecting representatives covers clusters instead of concentrating the batch around a single uncertainty peak. This rule does not maximize entropy within each cluster.

[![K equals B in round one, centroid-nearest images become candidates, and later rounds increase K to r times B while seeking uncovered clusters.](/figures/active-prompt/selection.svg)](/figures/active-prompt/selection.svg)

**Read the middle row as a projection of one cluster, not a learned CLIP embedding.** Features at scalar positions $0,2,3$ have centroid $5/3$. Their distances to it are $5/3,1/3,4/3$, so position 2 supplies the candidate. The centroid itself has no image to annotate.

Later rounds use $K=rB$ to increase coverage as the accumulated dataset grows. Section 4.2 describes avoiding previously covered clusters and prioritizing larger unlabeled clusters if re-clustering merges earlier selections. **A source ambiguity remains:** that paragraph says to choose “top-K clusters,” while §4 and Eq. (12) require $B$ candidates per round and $|D_r|=rB$. These statements do not fully specify an implementation when $K>B$. We retain the intended $B$-candidate protocol and explain the ambiguity rather than supplying unreported tie-breaking or fallback rules.

Cluster balance does not imply exactly one image from each true class—even when the experiment sets $B=C$. Wrong priors, class overlap and nuisance features can produce clusters that cross class boundaries. The source’s claim that increasing $K$ leaves enough unlabeled clusters relies on its described accounting; changing assignments and dataset sizes still require care in an implementation.

## 5 · The gate: compare with the predicted class’s own threshold

An image can be a useful representative and still be easy to classify. C decides whether to ask a human by comparing its confidence with a class-specific reference from the **previous accumulated training dataset**.

Let $D_{r-1,c}=\{(x,y)\in D_{r-1}:y=c\}$. Paper Eq. (11) computes

$$
\epsilon_{r,c}=\frac{1}{|D_{r-1,c}|}\sum_{(x,y=c)\in D_{r-1,c}}p_c(x;t_{r-1}).
$$

$\epsilon_{r,c}$ is the threshold for class $c$ in round $r$. Each term is the probability of the example’s **assigned training label**, not necessarily its maximum probability. Earlier pseudo-labeled examples are carried in $D_{r-1}$ too. The threshold is a mean training confidence, not a calibrated error probability or a statistical lower bound.

For new candidate $x$, predict $\widetilde y=\arg\max_c p_c(x;t_{r-1})$ and set $q=p_{\widetilde y}(x;t_{r-1})$. Eq. (12)’s gate is

$$
y_{\mathrm{train}}(x)=
\begin{cases}
y_{\mathrm{oracle}}(x),&q<\epsilon_{r,\widetilde y},\\
\widetilde y,&q\geq\epsilon_{r,\widetilde y}.
\end{cases}
$$

This case expression is a teaching rewrite of Eq. (12)’s dataset union. It attaches to C’s **target-assignment arrow**. The oracle reveals truth only on the first branch; the second branch uses the model’s predicted label. Equality follows the pseudo-label branch.

[![Class A has threshold 0.85 and B has 0.60; confidence 0.80 triggers a human query for predicted A but a pseudo-label for predicted B; equality uses a pseudo-label.](/figures/active-prompt/thresholds.svg)](/figures/active-prompt/thresholds.svg)

**Worked gate:** previous class-A assigned-label probabilities $0.8,0.9$ give $\epsilon_A=0.85$. Class-B probabilities $0.5,0.7$ give $\epsilon_B=0.60$. A new image predicted A with confidence 0.80 needs a human; another predicted B with 0.80 gets pseudo-label B. One global threshold would erase this distinction. At exactly 0.85, predicted A gets a pseudo-label.

There are no previous labels in round one, so the paper queries **all** $B$ initial candidates. A second source gap is important: Eq. (11) is undefined if a class has no examples in $D_{r-1}$. Cluster balance does not rule this out, and the supplied PDF specifies no empty-class threshold fallback. An implementation would need to resolve this explicitly; the guide does not invent a policy.

<details><summary>Predict: does passing the threshold prove the pseudo-label is correct?</summary>

No. It says the model is at least as confident as its mean assigned-label confidence for that class. A confident mistake can enter training and later threshold calculations. Annotation savings and correct-label ratio must be evaluated separately.

</details>

## 6 · Training and inference: carry the dataset, reset the prompt

[![Selection uses the previous fixed prompt, training resets prompts and minimizes cross-entropy on mixed targets, and inference uses the final prompt without clusters or annotation queries.](/figures/active-prompt/paths.svg)](/figures/active-prompt/paths.svg)

**Follow the information available on each row.** Acquisition sees unlabeled images and the old prompt. Training sees assigned targets in the accumulated dataset. Inference sees a new image and the final prompt; it needs neither an oracle nor K-means. Frozen text-encoder parameters do not prevent gradients from flowing through that encoder to the trainable context inputs.

In D, source Eq. (13) minimizes cross-entropy over all collected examples:

$$
\mathcal L_r(t_r)=\frac1{|D_r|}\sum_{(x,y)\in D_r}-\log p_y(x;t_r).
$$

$\mathcal L_r$ is the training objective, $t_r$ the newly initialized prompt, and $y$ the stored human or pseudo target. This is the finite-dataset form of the paper’s empirical expectation. The paper’s loss gives both target types the same role; it does not introduce confidence-based loss weights. If the model assigns a stored target probability 0.8, that example contributes $-\ln0.8\approx0.2231$. A wrong pseudo target still produces a gradient encouraging its assigned class.

The **dataset persists**: $D_r$ contains $D_{r-1}$ plus $B$ newly selected targets. The **prompt does not persist as the training initialization**: the paper resets it randomly each round to avoid bias from the previous round. The old learned prompt remains useful for selecting the new data before reset. Eq. (12) carries earlier targets forward; it does not describe re-querying or relabeling all old pseudo targets.

To reduce overfitting of class-specific prompts, Eq. (14) combines shared and class-wise prompt similarities:

$$
s_c(x)=\tfrac12\left[\cos(I(x),\theta_{\mathrm{txt}}(t_{u,c}))+
\cos(I(x),\theta_{\mathrm{txt}}(t_c))\right].
$$

$t_u$ is context shared across classes, $t_{u,c}$ appends class $c$, and $t_c$ uses class-wise context. This average replaces the similarity used at A before temperature and softmax. **Averaging similarities is not averaging probabilities**: exponential normalization is nonlinear. Figures 5–6 associate the unified component with less extreme seen-example confidence and better SQ savings. That supports its usefulness in the reported setting, not perfect calibration. The guide leaves unreported details of combining prompt variants in the clustering text mixture unspecified.

## 7 · Trace two rounds and count what was actually purchased

[![With B equal to two, the first round buys two labels; the second adds one human and one pseudo target, yielding four training examples but three human annotations.](/figures/active-prompt/loop.svg)](/figures/active-prompt/loop.svg)

**This is a fixed toy trace through A → B → C → D, not an experiment.** Assume round-one clustering selects one A and one B and both are oracle-labeled. Then $|D_1|=2$ and two annotations have been purchased. After prompt training, use the thresholds from the worked gate: 0.85 and 0.60. In round two, $K=4$ and the intended candidate count stays $B=2$. A candidate predicted A at 0.80 receives an oracle label; one predicted B at 0.80 receives a pseudo-label. Now $|D_2|=4$ but the human bill is three.

If $a_r$ is the count of human queries in round $r$, then the paid annotation total after $R$ rounds is $A_R=\sum_{r=1}^R a_r$, while the intended training-set size is $RB$ for distinct new candidates. The toy saving against querying all four is $1-3/4=25\%$. This is teaching arithmetic; it is **not** the paper’s measured saving. Retraining, embedding extraction and clustering still cost compute, and real annotation difficulty may vary by image.

<details><summary>Predict: if the prompt is reset in round two, does the learner forget round-one annotations?</summary>

No. It trains from a fresh prompt initialization on the accumulated $D_2$, which still contains round-one examples. Resetting parameters and discarding data are different operations.

</details>

## 8 · Evidence: what feature guidance establishes

[![Source Table 2 compares initial-round mean accuracy across seven datasets: image only 52.72, weighted text only 48.24, combined soft features 55.92, oracle label-balanced 48.10, and image with true-label text 58.27 percent.](/figures/active-prompt/results.svg)](/figures/active-prompt/results.svg)

**Read the first three bars as feature-space ablations, and the last two as oracle-assisted comparisons.** These reproduce Table 2 in the supplied October 2025 version: initial acquisition, average classification accuracy across seven datasets, CLIP ViT-B/32, reported mean ± standard deviation over three trials. Zero-based bars use percentage accuracy. This is not an SQ budget curve—there is no selective querying in round one.

| Acquisition representation or dataset construction | Average accuracy (%) | Uses ground-truth information to construct the dataset? |
| --- | --- | --- |
| Image features only | 52.72 ± 0.37 | No |
| Weighted text features only | 48.24 ± 0.21 | No |
| Image + soft weighted text features | 55.92 ± 0.38 | No |
| Conventional label-balanced few-shot dataset | 48.10 ± 0.41 | Yes |
| Image + true-label text feature | 58.27 ± 0.43 | Yes |

The combined soft representation gains **3.20 percentage points** over image only and **7.68 points** over text only in this average. The final oracle row shows a useful reference for better class guidance; it is unavailable when true pool labels are hidden. Standard deviations are not confidence intervals, and average improvement does not promise improvement on every dataset.

The main seven datasets are OxfordPets, FGVCAircraft, Caltech101, Flowers102, DTD, StanfordCars and EuroSAT. Section 5.2 uses eight rounds with $B=C$ candidates per round. The maximum human annotation schedule is $8C$; 100% on the baseline budget axis denotes that maximum, **not 100% of the image pool**. Experiments report mean accuracy over three trials with standard-deviation shading. Baselines include Random, Entropy, CoreSet, BADGE and PCB; the reproduction follows PCB’s class-description and CoOp setting.

Section 5.2 reports a 19.5 percentage-point first-round gain and 17.6% annotation-budget reduction for its main comparison. These are the authors’ aggregate claims tied to that protocol, not guarantees or numbers inferred from our toy. Because SQ saves labels after round one, equal rounds need not mean equal paid annotations. Appendix B/Figure 11 additionally evaluates using conserved budgets. Figure 7 separates paid-budget ratio from correct-label ratio; pseudo-label quality cannot be assumed from the budget reduction.

Table 1’s **CB*** replaces predicted soft class evidence with known true-label text to make an oracle-assisted comparison with perfectly class-balanced one-shot data. Do not use those rows as results attainable from a hidden-label pool. Even ordinary CB is not universally beneficial: MaPle’s DTD result drops from 48.77 to 46.47 in Table 1 while its average rises. Table 3 reports base-to-novel initial-round results over nine datasets, a different setting from Table 2’s seven-dataset average. ImageNet and larger-backbone experiments further broaden the evidence, but they do not establish every downstream application.

## 9 · Limitations, a useful test, and the idea to carry forward

The paper relies on strong VLM priors and evaluates classification. Its limitations section leaves detection, segmentation and real human-in-the-loop evaluation to future work. Domain mismatch can damage both the geometry and pseudo-label confidence. Appendix D’s ISIC and KaoKore experiments test CB in less typical domains; they do not supply a general SQ reliability guarantee.

Representation visualizations are explanatory evidence. They do not prove an object-localization mechanism, causal feature use or universally perfect clustering. The reported ARI values measure cluster-label agreement in particular analyses; they are not classification accuracy. The method also leaves the later-round top-K/top-B wording, empty-class thresholds and some prompt-variant details unresolved in the supplied text. A faithful implementation must settle those from code or author clarification rather than guessing.

A useful **proposed experiment** would hold the CLIP backbone, prompt design, training schedule and random seeds fixed, then compare CB with and without SQ at both matched rounds and matched **paid** annotation counts. Record accuracy, total targets, human queries, pseudo-label precision by class and wall-clock acquisition/training cost. Include an oracle evaluation of accepted pseudo-labels only for measuring quality, without exposing that truth to the acquisition policy. Separately compare image-only and class-guided representations. This would separate gains from geometry, extra pseudo-supervision and saved annotations; it is a proposal, not an experiment run here.

The idea to carry forward is to separate **which examples provide coverage** from **which examples need human truth**. A pretrained semantic model can guide both decisions, while errors in its priors can propagate through both. Keep candidate count, training-set size, annotation bill and label quality distinct.

Continue from [LabelBench](/papers/labelbench), which reduces acquisition-training cost using a proxy, or [CEC](/papers/cec), which calibrates and weights uncertainty before clustering. CEC’s entropy-weighted geometry and Active Prompt’s appended text mixture are different mechanisms; this guide does not substitute one for the other.

## 10 · Reading map to the supplied course version

| Guide mechanism | Source location in the 24-page TMLR 10/2025 copy |
| --- | --- |
| Full pipeline and round state | Figure 1, p. 2; Algorithm 1, p. 4; §4 introduction, p. 5 |
| CLIP probabilities and learnable inputs | §§3.1–3.2, p. 4; Eqs. (1)–(4) |
| Weighted text and concatenation | §4.1, pp. 5–6; Eqs. (5)–(7); Figure 2 |
| Centroids, representatives and increasing K | §4.2, p. 6; Eqs. (8)–(10); Appendix C/Table 6, pp. 22–23 |
| Thresholds, mixed targets, reset and loss | §4.3, p. 7; Eqs. (11)–(13) |
| Shared plus class-wise similarity | Eq. (14), p. 7; Figures 5–6, pp. 10–11 |
| Main seven-dataset protocol and aggregate claims | §§5.1–5.2, pp. 7–9; Figure 3 |
| Source measurements in our chart | Table 2, p. 11; §5.3 feature-space analysis |
| Oracle-assisted CB*, dataset quality, novel classes | Table 1, p. 9; Figure 7 and Table 3, p. 12 |
| Visualization and further comparisons | Appendix A, pp. 18–21; Appendix B/Figure 11, pp. 19, 21; Appendices C–D, pp. 22–24 |
| Scope and future work | Limitations, p. 13 |

These are printed page numbers. The [OpenReview record named in the paper](https://openreview.net/forum?id=qBeGCzD3Ij) is linked for provenance; its web page required a browser challenge during authoring. The local course PDF is the authority for this guide’s version and measurements.
