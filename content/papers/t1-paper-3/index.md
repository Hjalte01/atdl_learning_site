---
title: "A Bayesian Perspective on Generalization and Stochastic Gradient Descent"
shortTitle: "A Bayesian perspective on SGD"
topic: "topic-1"
order: 3
description: "Why fitting labels is not enough: connect Bayesian evidence, prior-relative width, and the noise scale of mini-batch SGD."
status: "ready"
pdf: "/papers/bayesian-sgd.pdf"
---
## 1 · Why can a better gradient give a worse predictor?

A full-batch gradient removes the randomness caused by selecting a few training examples. Surely this should improve learning? It improves the estimate of the training gradient. That alone does not determine which solution predicts new data well.

Smith and Le connect two puzzles: models can fit both meaningful and random labels, and larger training batches can reduce test accuracy when other hyperparameters stay fixed. Their proposed bridge is **Bayesian evidence: good fit must occupy appreciable probability mass under a specified prior.** Mini-batch noise may influence which regions training visits.

This guide follows the supplied **13-page ICLR 2018 conference PDF**, whose PDF creation metadata is **14 February 2018**. This is a file date, not an inferred arXiv revision. All diagrams are original teaching schematics; numerical examples are marked as toys. Open a diagram to enlarge it. No trained model runs on this page.

[![The same 800 MNIST images permit fitting informative or randomized labels, with different test behavior.](/figures/bayesian-sgd/problem.svg)](/figures/bayesian-sgd/problem.svg)

**Read across each branch.** Section 3 uses logistic regression with 784 pixel weights and one bias, trained on 800 balanced images of zeros and ones. Its balanced test set has 10,000 images. Random labels are assigned to both training and test sets in the random-label task. This small model can memorize training labels yet predict random test labels at chance. The picture is conceptual, not a plot of measured accuracies.

Accuracy also hides overconfidence. A correct prediction with probability 0.99 and one with probability 0.60 receive the same accuracy score; a wrong prediction made with extreme confidence incurs much larger cross-entropy. Figures 1–2 show why the two metrics tell different stories.

**A useful reconstruction of the reasoning, rather than the authors’ private thought process:** compare entire neighborhoods of plausible predictors, then ask how noisy optimization explores them. The paper’s “generalization gap” means the **test-accuracy difference between small- and large-batch training**, not the usual train–test gap.

## 2 · Architecture: train a network, analyze its solutions

[![MNIST images flow through an 800-unit ReLU hidden layer and ten-class softmax, with loss and gradient arrows leading to a weight update.](/figures/bayesian-sgd/architecture.svg)](/figures/bayesian-sgd/architecture.svg)

**Follow A → B → C for prediction, then D → E → F for learning.** The nonconvex experiment in §§4–5 is a shallow MNIST network with 800 ReLU hidden units and a softmax output. The functional tensor view uses $B$ images, each flattened to 784 pixels, 800 hidden activations per image, and ten class probabilities. The source does not require an invented convolutional backbone.

The standard setting uses 1,000 randomly selected training images, no L2 regularization, momentum $m=0.9$, and constant learning rate $\epsilon=1$. Batch size and other settings change in the explicitly identified sweeps. The logistic regression in §3 is a separate two-class experiment.

At D, let $p_\omega(y_i\mid x_i)$ be the probability assigned to the observed label of example $i$, and let $\omega\in\mathbb R^p$ collect the model’s $p$ parameters. For the Bayesian analysis in §2, define

$$
H(\omega)=-\sum_{i=1}^{N}\log p_\omega(y_i\mid x_i),\qquad
C(\omega)=H(\omega)+\frac{\lambda}{2}\|\omega\|^2.
$$

Here $N$ is training-set size and $\lambda>0$ is the precision of an isotropic Gaussian prior. $H$ is **summed**, not mean, cross-entropy. $C$ is the regularized cost. The SGD update below divides its gradient by $N$. Confusing these conventions creates spurious learning-rate factors.

The prior $P(\omega)=(\lambda/2\pi)^{p/2}\exp(-\lambda\|\omega\|^2/2)$ gives posterior density proportional to $\exp[-C(\omega)]$ (paper Eq. 1). Minimizing $C$ finds a maximum a posteriori point. It does not compute the posterior’s mass, nor does it optimize the evidence as an extra loss. The unregularized network experiment must not be substituted into a proper Gaussian-prior evidence formula by casually setting $\lambda=0$.

## 3 · Representation close-up: density is not mass

[![Two schematic equal-height peaks have different widths; the wider peak contains more integrated mass.](/figures/bayesian-sgd/representation.svg)](/figures/bayesian-sgd/representation.svg)

**Read height as local fit and area as integrated contribution.** The plotted curves schematically represent $e^{-C}$ in one parameter coordinate. They are not measured loss landscapes. Evidence includes the prior normalization and integrates over every parameter, not just the highest point:

$$
Z=P(y\mid x;M)=\int P(y\mid\omega,x;M)P(\omega;M)\,d\omega.
$$

This is paper Eq. 5. $M$ denotes the model, including its prior, and $Z$ is the marginal likelihood of the observed training labels given the inputs. In Eq. 4, posterior odds between models equal evidence odds times prior model odds. Evidence is not test accuracy and is not itself a numerical generalization-error certificate.

To connect mass to curvature, approximate the cost near a minimum $\omega_0$ by a quadratic. Let $\mathcal H=\nabla^2 C(\omega_0)$ have positive eigenvalues $h_1,\ldots,h_p$. The local Gaussian integral gives paper Eq. 9 in renamed notation:

$$
Z\approx e^{-C(\omega_0)}\frac{\lambda^{p/2}}{\sqrt{\det\mathcal H}}
=\exp\!\left[-C(\omega_0)-\frac12\sum_{j=1}^{p}\log\frac{h_j}{\lambda}\right].
$$

This operates in an **analysis box after a solution is found**, not in the network forward pass. The determinant multiplies all principal curvatures, so one large eigenvalue alone cannot determine evidence. The ratio compares posterior width to prior width; it penalizes having to specify parameters precisely. The approximation needs a well-described quadratic neighborhood; multimodality, symmetries and non-Gaussian regions complicate it. The paper proposes excluding eigenvalues below $\lambda$ when extending the approximation away from minima; that is a heuristic, not the general evidence integral.

**Computed one-dimensional toy.** Two regions with $C(\omega_0)=10$ and prior precision $\lambda=1$ have curvatures 4 and 100. Their local contributions are $e^{-10}/2$ and $e^{-10}/10$: the broader region gets five times the mass. If the sharp region instead has cost 8, its deeper fit multiplies its contribution by $e^2$; it then beats the broad region by $e^2/5\approx1.48$. Width alone cannot rank solutions.

For $n$ equally likely classes the null model has evidence $Z_{\rm null}=e^{-N\log n}$. Paper Eq. 10 writes

$$
\frac{Z}{Z_{\rm null}}=e^{-E},\qquad
E=C(\omega_0)+\frac12\sum_j\log(h_j/\lambda)-N\log n.
$$

$E<0$ favors the learned model. This sign is easy to reverse: $E$ is the log evidence ratio **in favor of the null**. For example, $E=-2$ means odds $e^2\approx7.39$ in favor of the learned model before model-prior odds. The source’s Figures 1–2 report that random labels are not supported over the null, while informative labels can be supported at suitable regularization.

<details><summary>Predict: does doubling a parameter coordinate make the model worse?</summary>

No, if the likelihood and prior measure are transformed consistently. Under $u=a\omega$, both scalar curvature and Gaussian prior precision divide by $a^2$, preserving their ratio. The exact integral also includes the density’s Jacobian. Keeping the same numerical prior after a coordinate change defines a different model. The source explicitly warns that a general nonlinear reparameterization can still alter the Laplace approximation.

</details>

## 4 · Training is not posterior prediction

[![Training refreshes batches and carries weights and momentum; inference freezes weights and runs the predictor, while Bayesian averaging is a separate ideal.](/figures/bayesian-sgd/training.svg)](/figures/bayesian-sgd/training.svg)

**Read the upper row as a repeated process; the lower row is a frozen predictor.** During SGD, the data remain fixed while batches are selected again, gradients are recomputed, and weights and momentum are carried forward. At point prediction, a new image goes through the frozen network and the largest class probability gives the predicted label.

The Bayesian ideal in paper Eq. 2 instead averages class probabilities:

$$
P(y_*\mid x_*,D)=\int P(y_*\mid x_*,\omega)P(\omega\mid D)\,d\omega.
$$

Here $D=(x,y)$ is the training data, and $(x_*,y_*)$ denotes a new example and its unknown label. This is neither an average of weights nor simply the prediction of the highest-density parameter vector. The paper motivates using a near-mode point approximation, but its SGD experiment does not explicitly evaluate this integral.

Appendix A supplies the bridge to noisy dynamics: ideal isotropic Langevin noise with covariance $2T I\delta(t-t')$ has stationary density proportional to $e^{-C/T}$ under the stated limiting assumptions. $T$ is temperature, $I$ the identity and $\delta$ the Dirac delta; $T=1$ targets the posterior. The probability of occupying a basin then depends on its integrated posterior mass. Ordinary SGD noise is generally anisotropic and changes with $\omega$, so this idealized sampler motivates the hypothesis; it does not prove exact posterior sampling by SGD.

## 5 · The noise scale lives at the update arrow

The randomness is in E → F: a mini-batch provides an estimate of the full gradient. Using the summed-cost convention, paper Eq. 11 becomes

$$
\Delta\omega=-\frac{\epsilon}{N}(\nabla C+\alpha),\qquad
\alpha=\nabla\widehat C-\nabla C,\qquad
\nabla\widehat C=\frac NB\sum_{i\in\mathcal B}\nabla C_i.
$$

$\mathcal B$ is the sampled batch of size $B$, $C_i$ its per-example cost contribution, $\epsilon$ the learning rate, and $\alpha$ the error in the estimated **summed** gradient. An unbiased batch gives $\mathbb E[\alpha]=0$. Making finite-population normalization explicit with covariance $F(\omega)$ defined using denominator $N-1$, the covariance takes the form used in the paper:

$$
\operatorname{Cov}(\alpha)=N\left(\frac NB-1\right)F(\omega).
$$

$F$ is a matrix of per-example gradient covariances, not a scalar or automatically the identity. The finite-batch correction models sampling without replacement and vanishes when $B=N$. Using covariance with denominator $N-1$ gives this expression; a denominator-$N$ convention moves a factor $N/(N-1)$ into the formula. Neither convention changes the large-$N$, small-$B/N$ interpretation.

[![A computed scalar batch update separates full-gradient drift from batch deviation and distinguishes the realized error from the covariance scale g.](/figures/bayesian-sgd/numbers.svg)](/figures/bayesian-sgd/numbers.svg)

**Follow the top row, then split the update using the bottom row.** In this no-momentum toy, one coordinate has weight 1, full mean gradient 0.2 and batch mean gradient 0.3. At learning rate 0.1 its next value is $1-0.1(0.3)=0.97$. The full-gradient contribution is $-0.02$ and this batch’s random deviation is $-0.01$. For $N=1000$ and $B=100$, the summed-gradient error is $\alpha=100$, consistent with $-(\epsilon/N)\alpha=-0.01$.

Now approximate successive updates by continuous dynamics (paper Eqs. 12–13):

$$
\frac{d\omega}{dt}=-\nabla C+\eta(t),\qquad
\mathbb E[\eta(t)\eta(t')^\top]=gF(\omega)\delta(t-t').
$$

$t$ is continuous optimization time, not epoch number. Integrate over one step $\Delta t=\epsilon/N$. The discrete update’s noise covariance is $(\epsilon/N)^2\operatorname{Cov}(\alpha)$, while the continuous increment has covariance $(\epsilon/N)gF$. Equating them yields

$$
\boxed{g=\epsilon\left(\frac NB-1\right)\approx\frac{\epsilon N}{B}\quad(B\ll N).}
$$

In the toy, $g=0.9$. This is **not** the realized deviation $-0.01$, nor a coordinate’s standard deviation. It multiplies the covariance matrix in the continuous-time model. Gaussian gradient-noise and small-discretization-error assumptions matter. Appendix C checks one softmax parameter at initialization: a batch of 30 gives a more Gaussian-looking distribution than individual examples, not proof for every parameter throughout training.

## 6 · Scaling batches, learning rates and momentum

[![A numerical scaling example contrasts approximate linear batch scaling, the exact finite-batch correction, full-batch limits, and the momentum rule.](/figures/bayesian-sgd/iteration.svg)](/figures/bayesian-sgd/iteration.svg)

**Read left to right: approximate scaling can miss the finite-batch correction.** Starting from $N=1000$, $B=100$, $\epsilon=0.1$ gives $g=0.9$. Doubling both $B$ and $\epsilon$ gives $g=0.8$, not exactly 0.9. Solving the no-momentum expression for batch size gives the teaching identity

$$
B=\frac{N}{1+g/\epsilon}.
$$

Holding $g=0.9$ at $\epsilon=0.2$ asks for $B\approx181.8$, rounded to an integer batch size. At $B=N$ the exact batch noise disappears, while $\epsilon N/B$ misleadingly gives $\epsilon$. The linear rule is a small-batch approximation, not an endpoint formula.

For momentum, a familiar equivalent update convention is $v_{k+1}=m v_k+\widehat{\nabla C}_k/N$ and $\omega_{k+1}=\omega_k-\epsilon v_{k+1}$. The iteration index is $k$, and $v$ is the accumulated mean gradient. Appendix D derives, for small batches,

$$
g\approx\frac{\epsilon N}{B(1-m)},\qquad
B_{\rm opt}\propto\frac{\epsilon N}{1-m}.
$$

This explains why momentum is part of the noise story. Raising $m$ from 0.9 to 0.95 halves $1-m$, so the approximation asks for twice the batch size at fixed $\epsilon,N,g$. The relation assumes this momentum convention, $m<1$, and a regime where the continuous approximation is useful. It does not license taking $m$ arbitrarily close to one or increasing $\epsilon$ without stability limits.

Over repeated updates, weights change, so $F(\omega)$ changes too. Keeping scalar $g$ fixed need not preserve the entire covariance, the trajectory, or the test result. Increasing batch size also changes examples processed per step; equal steps are not equal epochs or equal compute.

<details><summary>Predict: at fixed learning rate, is the smallest possible batch always best?</summary>

No. The paper predicts a balance: excessive noise interferes with fitting, while too little may lose beneficial exploration. In its default MNIST setup, batches around ten or fewer fail to train. An interior optimum is the claim, not “smaller is always better.”

</details>

## 7 · What the experiments establish

[![Source-reported experimental settings and quantitative ranges for the batch-size, learning-rate and dataset-size sweeps, without invented accuracy values.](/figures/bayesian-sgd/evidence.svg)](/figures/bayesian-sgd/evidence.svg)

**Read each row as a protocol paired with its reported observation.** This evidence panel transcribes numerical settings and reported ranges from the source; it does not digitize curves or invent precise accuracy percentages. The [course PDF, pages 6–8](/papers/bayesian-sgd.pdf#page=6) contains the measured result plots.

| Source | Setting and comparison | Reported evidence and its limit |
| --- | --- | --- |
| Figures 1–2, §3 | Binary MNIST logistic regression; 800 train, 10,000 test; true versus randomized labels; sweep L2 precision | Both can fit training labels. Evidence and test cross-entropy track each other across regularization. This is a small-model calculation, not a deep-network evidence computation. |
| Figures 3–4, §4 | Ten-class MNIST, 800-unit hidden layer, 1,000 training images, $\epsilon=1$, $m=0.9$, no L2; batches from 1 to 1,000 | At 10,000 updates, accuracy has an interior batch-size optimum. Small batches around ten or below fail to train. Full batch and batch 30 have different test-accuracy trajectories. |
| Figure 5, §5 | Same network; evaluate after $10000/\epsilon$ updates while sweeping learning rate and batch size | Best observed batch scales with learning rate across two orders of magnitude. Peak accuracy falls around $\epsilon\gtrsim3$, limiting the rule. Error bars represent spacing to the next tested batch size, not confidence intervals. |
| Figure 6, §5 | Dataset-size sweep; $\epsilon=1$, 10,000 updates; each curve averaged over five experiments | Best batch is approximately proportional to $N$ once $N\gtrsim20{,}000$, as the caption qualifies. More data also improves accuracy and reduces the batch-size gap. |
| Figure 7, §5 | Momentum sweep; $\epsilon=1$, 10,000 updates | Best batch increases with momentum and agrees with a fitted $1/(1-m)$ curve in the tested range. This is empirical support, not a universal optimum formula. |
| Figure 8, Appendix B | Add L2 coefficient 0.1 to the Figure 3 setting | The gap shrinks substantially; regularization changes the story. Full-batch training takes longer to converge here. |

Test cross-entropy can worsen while accuracy stabilizes (Figure 3), so the later experiments deliberately focus on accuracy. Their test-set sweeps explain a phenomenon; a practitioner should select hyperparameters on validation data, as Appendix E recommends, then reserve the test set for final evaluation.

**What would a stronger ablation need?** A proposed follow-up would compare matched noise scales across seeds, report both accuracy and cross-entropy, and separately match update count and examples processed. It would vary regularization and measure gradient covariance, rather than treating scalar $g$ as a complete description. It would also separate a Bayesian sampling claim from an optimization result. These are suggested experiments, not results reported in this paper.

The paper does not compute exact evidence for its deep-network minima. Hessian cost and equivalent hidden-unit permutations make that difficult. The evidence intuition is strongest in the tractable example; the SGD interpretation adds modeling assumptions and experimental support.

## 8 · Idea to carry forward

Ask two questions about a learned solution: **how well does it fit, and how much prior probability supports similarly good fits?** Then ask how the optimizer’s noise changes which solutions it reaches. Evidence connects fit and prior-relative width; SGD’s approximate noise scale connects batch size, learning rate, data size and momentum. Neither connection turns curvature alone into a generalization guarantee.

Continue from [the information bottleneck](/papers/t1-paper-1), which studies representations, and [nonvacuous PAC-Bayes bounds](/papers/t1-paper-2), which explicitly certifies a randomized predictor. This paper offers an evidence-and-dynamics explanation rather than computing that same certificate.

### Source reading map

| Guide mechanism | Supplied course PDF |
| --- | --- |
| Posterior, prediction and evidence | §2, Eqs. 1–10, pp. 2–3 |
| Informative versus random labels | §3, Figures 1–2, pp. 3–5 |
| Shallow-network setup and batch optimum | §4, Figures 3–4, pp. 5–6 |
| Gradient noise and finite-batch correction | §5, Eqs. 11–13, pp. 6–7 |
| Learning-rate, data-size and momentum sweeps | §5, Figures 5–7, pp. 6–8 |
| Langevin posterior and local basin mass | Appendix A, Eqs. 14–21, pp. 10–11 |
| Regularization and Gaussian-noise caveats | Appendices B–C, Figures 8–9, pp. 11–12 |
| Momentum derivation and tuning heuristic | Appendices D–E, Eqs. 22–32, pp. 12–13 |

The source’s introductory footnote describes an optimal learning rate proportional to training-set size, but solving its derived $g\approx\epsilon N/B$ at fixed $g,B$ gives **inverse** dependence on $N$. This guide follows the derivation. Appendix C’s reference to “figure 7b” for its gradient histogram appears to mean Figure 9b. Neither discrepancy changes the calculations above.
