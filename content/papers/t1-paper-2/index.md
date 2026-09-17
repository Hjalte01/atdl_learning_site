---
title: "Computing Nonvacuous Generalization Bounds for Deep (Stochastic) Neural Networks with Many More Parameters than Training Data"
shortTitle: "Nonvacuous generalization bounds"
topic: "topic-1"
order: 2
description: "Turn a trained network into a distribution of networks, then follow its noise tolerance through a PAC-Bayes certificate."
status: "ready"
source: "https://arxiv.org/abs/1703.11008v2"
pdf: "/papers/nonvacuous-bounds.pdf"
---
## The puzzle: a good test score is not yet an explanation

A network has 2.4 million parameters and only 55,000 training examples. It fits the training labels almost perfectly. Is it learning a useful rule, or merely memorizing? Parameter counting cannot distinguish the two: the same architecture can also fit random labels.

**Dziugaite and Roy build a quantitative witness to generalization: a distribution of perturbed networks that remains accurate and is not too expensive relative to a reference distribution.** They optimize a PAC-Bayes bound to find that distribution. The result is a certificate for a stochastic classifier, not a proof that every network found by SGD generalizes.

This is Topic 1, paper 2. We follow the supplied **19 October 2017 arXiv v2** course PDF, whose identity matches the [version-specific source record](https://arxiv.org/abs/1703.11008v2). Allow about 25 minutes. The eight local figures are original teaching diagrams; only the results chart reproduces reported measurements. Click a figure to enlarge it. No trained model runs on this page.

## 1 · What would count as a useful bound?

[![A large network can fit true or random labels; a useful certificate must distinguish these cases rather than count parameters alone.](/figures/nonvacuous-bounds/problem.svg)](/figures/nonvacuous-bounds/problem.svg)

Read the two branches as a controlled contrast. Both networks can fit their training labels, but only real labels carry a predictable relationship to the images. A **generalization bound** upper-bounds the population error on new examples from the same data distribution, under stated assumptions and confidence.

An error rate is always at most 1. A bound of 1.4 adds nothing: it is **vacuous**, and can be clipped to the trivial bound 1. A bound of 0.2 is nonvacuous. On this binary task, fair random guessing has expected error 0.5, so 0.2 also certifies performance better than that reference. A bound of 0.8 would be nonvacuous but would not establish that comparison.

The paper's hypothesis is that good SGD solutions have a substantial neighborhood of other good solutions. Our reconstruction of the design logic is: replace the question “How big is the entire model class?” with “Can we describe an accurate distribution near this solution at a manageable information cost?” This is an interpretation of the method, not access to the authors' private reasoning.

<details><summary>Predict: does zero training error alone imply a small generalization bound?</summary><p>No. The random-label network also fits its training set. A certificate must pay for the learned distribution's complexity and the uncertainty from finite data.</p></details>

## 2 · Architecture: a network inside a distribution

[![Input images enter a sampled fully connected ReLU network; its scores feed training loss or sign prediction, while posterior and prior distributions supply a KL cost.](/figures/nonvacuous-bounds/architecture.svg)](/figures/nonvacuous-bounds/architecture.svg)

The predictor is a fully connected multilayer perceptron (§4.2). A batch of flattened MNIST images is $X\in\mathbb R^{B\times784}$. Each hidden layer produces $B\times N$ ReLU activations; the final linear layer produces $B$ scalar scores. The label is the sign of a score. Table 1 varies hidden width $N$ and hidden-layer count $L$; the diagram's $784\to600\to1$ is the T-600 configuration, not every experiment.

For one image, let $f_v(x)$ be the real score using weights and biases collected in $v\in\mathbb R^d$. Then $h_v(x)=\operatorname{sign}(f_v(x))$. This notation separates the score used by the logistic loss from the discrete prediction used to count errors.

The learned object is a diagonal Gaussian **posterior** over those parameters:

$$
Q=\mathcal N(w,\operatorname{diag}(s)),\qquad
v=w+\sqrt{s}\odot\xi,\qquad \xi\sim\mathcal N(0,I_d).
$$

Here $w\in\mathbb R^d$ is the learned mean, $s\in\mathbb R_{>0}^d$ contains **variances**, $\sqrt{s}$ contains standard deviations, and $\odot$ multiplies coordinates. The sampling arrow in the diagram produces one complete network. “Posterior” is PAC-Bayes terminology: this distribution is optimized for a bound and need not be a Bayesian posterior from a likelihood.

The reference, or **prior**, is $P_\lambda=\mathcal N(w_0,\lambda I_d)$. Its center $w_0$ is the original random initialization, chosen independently of the training data. Its scalar variance $\lambda$ is selected from a predeclared family with an explicit selection penalty (§4 below).

**Inference:** freeze $w$ and $s$, sample a fresh $v$ for a forward evaluation, and take the sign of its output. Labels, the KL term, and the optimizer are absent. The theorem controls error averaged over examples and sampled networks. It does not automatically certify the deterministic mean network $h_w$, a majority vote, or each individual draw.

## 3 · Representation close-up: width has a price and a benefit

[![A diagonal Gaussian has a separate spread along each weight coordinate; increasing spread changes both noise sensitivity and its divergence from the fixed prior.](/figures/nonvacuous-bounds/representation.svg)](/figures/nonvacuous-bounds/representation.svg)

The ellipse is a two-coordinate schematic of an axis-aligned covariance. It is not a measured loss contour or a hard boundary: a Gaussian has unbounded support. A wide coordinate means the classifier tolerates uncertainty in that weight. A narrow coordinate preserves a sensitive weight more precisely. Diagonal covariance cannot follow an arbitrarily rotated valley of jointly varying weights.

The information cost is computed in the **KL box**, using the Gaussian formula in paper Eq. 1 specialized in §3.1:

$$
\mathrm{KL}(Q\Vert P_\lambda)
=\frac12\sum_{i=1}^d\left[
\frac{s_i}{\lambda}+\frac{(w_i-w_{0,i})^2}{\lambda}-1+\log\frac{\lambda}{s_i}
\right].
$$

All logarithms here are natural, so KL is measured in nats. The first and last terms compare posterior and prior variances; the squared-distance term charges for moving the mean from initialization. For $w=w_0$ and every $s_i=\lambda$, KL is zero. Making $s_i$ extremely small sends the log term upward. Making it much larger than $\lambda$ increases the first term. **Broader is not always cheaper.** Broadening an overly narrow posterior toward the prior scale can reduce KL, but may increase prediction errors.

This is why a visually flat valley alone is insufficient. Its location relative to the prior and its spread in the chosen parameterization also matter. The sum includes every parameter, yet parameter count alone does not determine its value.

## 4 · The theorem and the cost of choosing a prior

Write $S_m=\{(x_i,y_i)\}_{i=1}^m$ for independent, identically distributed training examples, and $\mu$ for their unknown population distribution. The two errors are

$$
\widehat e(Q,S_m)=\mathbb E_{v\sim Q}\frac1m\sum_{i=1}^m
\mathbf1[h_v(x_i)\ne y_i],\qquad
e(Q)=\mathbb E_{v\sim Q,(x,y)\sim\mu}\mathbf1[h_v(x)\ne y].
$$

The first is expected error on the observed sample; the second is expected error on new data. **Source notation caveat:** the supplied PDF's §2 prints equality inside the 0–1 loss indicator. That counts correct predictions, contradicting its classification-error discussion and logistic upper bound. We explicitly use inequality, the intended misclassification loss, throughout this guide.

Theorem 2.3 states that for a fixed data-independent prior $P$, with probability at least $1-\delta$ over the training sample, **simultaneously for all $Q$**,

$$
\operatorname{kl}(\widehat e(Q,S_m)\Vert e(Q))
\le \frac{\mathrm{KL}(Q\Vert P)+\log(m/\delta)}{m-1}.
$$

Lowercase $\operatorname{kl}(q\Vert p)=q\log(q/p)+(1-q)\log((1-q)/(1-p))$ compares two Bernoulli error probabilities. Uppercase $\mathrm{KL}(Q\Vert P)$ compares distributions over network parameters. They serve different roles. The uniform quantifier permits learning $Q$ from the same training data: no separate holdout set is required for this certificate. The i.i.d. assumption and valid prior choice still matter.

**Why not center the prior on the trained weights?** Choosing that center after seeing the training data would evade a cost that this theorem requires. The random initialization is allowed because it precedes learning and is independent of the sample; sharing that initialization with the training procedure is not a violation.

The paper does adapt the prior *variance*, but pays for choosing it. Before observing the data, define the countable family

$$
\lambda_j=c\exp(-j/b),\qquad
\delta_j=\frac{6\delta}{\pi^2j^2},\qquad j=1,2,\ldots.
$$

Here $b=100$ sets grid resolution, $c=0.1$ sets its upper scale, and $\delta=0.025$ is the total failure budget. Because $\sum_j\delta_j=\delta$, a union bound covers all candidates at once. Substituting $j=b\log(c/\lambda)$ gives paper Eq. 5:

$$
B_{\mathrm{RE}}(w,s,\lambda;\delta)=
\frac{\mathrm{KL}(Q\Vert P_\lambda)+2\log\!\left(b\log\frac c\lambda\right)
+\log\frac{\pi^2m}{6\delta}}{m-1}.
$$

$B_{\mathrm{RE}}$ is the relative-entropy budget used in the certificate. Continuous $\lambda$ is a relaxation for optimization; the guarantee uses the discrete grid. Final evaluation tries the adjacent grid values and takes the better bound. The union bound already accounts for that selection.

<details><summary>Predict: can we search over prior variances and report the best fixed-prior bound without a penalty?</summary><p>Not with the fixed-prior theorem used here. Selecting a variance using the training data must be covered by the predeclared family and its confidence allocation. The posterior itself can depend on the data because the theorem already holds for all posteriors simultaneously.</p></details>

## 5 · Training has two stages, then certification

[![Three stages show ordinary SGD, optimization of posterior mean and variances with fresh Gaussian noise, and separate final certification with grid rounding and Monte Carlo error correction.](/figures/nonvacuous-bounds/training.svg)](/figures/nonvacuous-bounds/training.svg)

Read downward. First train a deterministic network to obtain $w_{\mathrm{SGD}}$. Then initialize $w$ there and optimize the distribution. The prior center $w_0$ stays fixed, but the posterior mean **can move away** from $w_{\mathrm{SGD}}$. Finally freeze the learned distribution and compute the classification-error certificate.

The differentiable loss box uses the normalized logistic loss (§2):

$$
\breve\ell(f_v(x),y)=\frac{\log(1+\exp(-y f_v(x)))}{\log 2}.
$$

The label $y$ is $+1$ or $-1$; $y f_v(x)$ is the signed margin. A nonpositive margin gives loss at least 1, making this a convenient upper bound on misclassification. It is differentiable, unlike the sign-based error, but it is unbounded; the final bounded-loss theorem is applied to classification error, not directly to this surrogate.

The paper minimizes the smooth upper-bound objective (Eq. 4):

$$
J(w,s,\lambda)=\mathbb E_{v\sim Q}\frac1m\sum_{i=1}^m
\breve\ell(f_v(x_i),y_i)+\sqrt{\frac12 B_{\mathrm{RE}}(w,s,\lambda;\delta)}.
$$

The expectation rewards sampled networks that predict well; the square-root term regularizes their distribution. Its form comes from the inequality $\operatorname{kl}^{-1}(q\mid C)\le q+\sqrt{C/2}$ (paper Eq. 2). This is a convenient optimization surrogate, generally looser than the inverse-KL certificate evaluated later.

[![An iterative loop samples noise, forms weights, computes logistic loss plus the complexity penalty, and updates mean and log scales while retaining the original prior center.](/figures/nonvacuous-bounds/iteration.svg)](/figures/nonvacuous-bounds/iteration.svg)

At iteration $t$, draw fresh $\xi_t$, form $v_t=w_t+\sqrt{s_t}\odot\xi_t$, evaluate the loss plus analytic KL penalty, and backpropagate through the sampling expression. Carry the updated **distribution parameters** to the next step; do not carry the last noisy weight draw as the new mean. The paper uses $\rho=\tfrac12\log s$ and $\eta=\tfrac12\log\lambda$ so exponentiation produces positive scales. The continuous prior search remains in $0<\lambda<c$; exponentiation alone enforces positivity, not the upper limit.

The source's Algorithm 1 is vanilla-SGD pseudocode; §4.3 reports RMSprop experiments. Its initialization line writes $\rho\leftarrow|w|$ while also defining $s=\exp(2\rho)$; this is inconsistent with §4.3's stated $s=|w|$ initialization. We follow the prose for the experimental description, without treating the pseudocode as a verified executable specification.

For true labels, initial network training uses 20 epochs, batch size 100, learning rate 0.01 and momentum 0.9. Bound optimization then uses the full training set for each sampled perturbation, RMSprop decay 0.9, 150,000 iterations at 0.001 and 50,000 at 0.0001. Random-label training uses 120 epochs and bound optimization uses 500,000 iterations at 0.0001. This is substantial offline computation, not a cheap consequence of an ordinary training run.

## 6 · Follow one noisy weight through the same boxes

[![A one-weight toy example samples v=0.5, computes score 1 and loss 0.452, and adds a KL-derived penalty to obtain objective 0.509.](/figures/nonvacuous-bounds/numbers.svg)](/figures/nonvacuous-bounds/numbers.svg)

This scalar example is a teaching derivation, not a reproduced MNIST run. Replace the MLP with $f_v(x)=vx$. Set $w_0=0$, $w=0.4$, $s=0.01$, and a **fixed, data-independent** prior variance $\lambda=0.04$. Draw $\xi=1$ and use $x=2,y=+1$:

$$
v=0.4+\sqrt{0.01}\cdot1=0.5,\qquad f_v(2)=1,
\qquad \breve\ell=\frac{\log(1+e^{-1})}{\log2}\approx0.452.
$$

Its classification error is zero, but its surrogate loss is positive. The corresponding KL box gives

$$
\mathrm{KL}(Q\Vert P)=\frac12\left[0.25+4-1+\log4\right]\approx2.318.
$$

For arithmetic only, take a toy dataset of $m=2000$ identical $(2,+1)$ observations and $\delta=0.05$. The one sampled network has average logistic loss 0.452 on it. With a fixed prior, there is no prior-selection penalty:

$$
C=\frac{2.318+\log(2000/0.05)}{1999}\approx0.00646,
\qquad \widehat J\approx0.452+\sqrt{C/2}\approx0.509.
$$

The sampled objective $\widehat J$ is not a final certificate or an exact posterior expectation. Its gradient is useful for optimizing one. For example, the loss gradient with respect to $v$ is $-yx/[\log2(1+e^{yvx})]\approx-0.776$. Holding the draw fixed, $\partial v/\partial w=1$ and $\partial v/\partial\rho=\xi e^\rho=0.1$, so the loss contributions to those gradients are about $-0.776$ and $-0.0776$. The analytic penalty adds its own gradients: an update must include both paths.

<details><summary>Predict: if we enlarge s while holding w fixed, must the full objective improve?</summary><p>No. The KL change depends on the relation to the prior variance, and a larger perturbation can flip predictions. Optimization balances those effects rather than maximizing noise without limit.</p></details>

## 7 · Certification: two uncertainties, two inverse-KL steps

[![A frozen posterior is sampled to estimate training error; one inverse-KL step corrects Monte Carlo uncertainty and another turns the corrected training error into a population-error bound.](/figures/nonvacuous-bounds/certificate.svg)](/figures/nonvacuous-bounds/certificate.svg)

The first uncertainty is **finite Monte Carlo estimation of the posterior's training error**. The second is **generalization from a finite training sample to the population**. They need separate budgets.

Define the upper inverse

$$
\operatorname{kl}^{-1}(q\mid C)=\sup\{p\in[0,1]:\operatorname{kl}(q\Vert p)\le C\}.
$$

It returns the largest error consistent with a Bernoulli KL budget. For fixed $q<1$, solve on $p\in[q,1)$; $C=0$ returns $q$, and $q=1$ returns 1. The special case $q=0$ gives $1-e^{-C}$. A square-root upper bound at $q=0,C=0.1$ gives 0.224, whereas inverse KL gives 0.0952: the choice matters near low empirical error.

Freeze $Q$ and draw $n$ independent networks. Let $\bar e_n$ be their average training classification error. Using the sample-convergence step in §3.3, first compute

$$
q_{\mathrm{upper}}=\operatorname{kl}^{-1}\!\left(\bar e_n\mid\frac{\log(2/\delta')}{n}\right).
$$

Each sampled network is evaluated over the training sample. Do not count all $nm$ predictions as independent posterior draws: predictions from one sampled network share its parameters. This estimate uses $n$, not $nm$, as its Monte Carlo sample count.

Next apply the discrete-prior PAC-Bayes budget (paper Eq. 6):

$$
e(Q)\le\operatorname{kl}^{-1}\!\left(q_{\mathrm{upper}}\mid B_{\mathrm{RE}}(w,s,\lambda;\delta)\right).
$$

The combined confidence is at least $1-\delta-\delta'$. The experiments use $n=150{,}000$, $\delta'=0.01$ and $\delta=0.025$, giving **0.965 confidence over the training sample and certification draws**. This does not mean every individual network is 96.5% accurate.

For a separate illustrative calculation, take $\bar e_n=0.028$ and those same $n,\delta'$ values. The first inverse gives $q_{\mathrm{upper}}\approx0.02941$. If a valid final complexity budget were $B_{\mathrm{RE}}=0.1$, the second inverse gives an upper error of about **0.1697**. These are computed toy numbers, not a reconstruction of a rounded Table 1 row. The source uses Newton iterations (Appendix A); reproducible teaching numbers in our figure script use bisection.

## 8 · What the evidence actually establishes

[![Paired bars compare stochastic-network test-error upper estimates around 3.2 to 3.5 percent with PAC-Bayes bounds from 16.1 to 22.3 percent across six true-label architectures.](/figures/nonvacuous-bounds/results.svg)](/figures/nonvacuous-bounds/results.svg)

This chart redraws **Table 1 of the supplied v2 PDF**. The task is binary MNIST: digits 0–4 map to $+1$ and 5–9 to $-1$. There are 55,000 training and 10,000 test images; the validation split is unused. All entries below are classification-error percentages, lower being better. T-$N^L$ means true labels with $L$ hidden layers of $N$ units each.

| Architecture | Parameters (reported) | SGD test error | SNN test error upper estimate | PAC-Bayes bound |
| --- | ---: | ---: | ---: | ---: |
| T-600 | 471k | 1.8% | 3.4% | 16.1% |
| T-1200 | 943k | 1.8% | 3.5% | 17.9% |
| T-300² | 326k | 1.5% | 3.4% | 17.0% |
| T-600² | 832k | 1.6% | 3.3% | 18.6% |
| T-1200² | 2384k | 1.5% | 3.5% | 22.3% |
| T-600³ | 1193k | 1.3% | 3.2% | 20.1% |

**Read the gap, not just the smaller bar.** The deterministic SGD baseline is most accurate here, the stochastic network is slightly worse, and the training-derived certificate is much looser. The SNN columns are already upper estimates that account for finite posterior sampling (§4.4); they are not raw deterministic test fractions. The PAC-Bayes theorem bounds population error. Held-out test evaluation supplies a comparison, not an ingredient of the training-derived certificate.

The random-label R-600 experiment has SGD training error 0.7% and test error 50.8%. Its SNN train/test upper estimates are 11.2%/50.3%, and KL rises to 201,131 compared with 5,144 for T-600. **Its table entry 1.352 is specially defined in §4.4 as $\sqrt{B_{\mathrm{RE}}/2}$**, a vacuity diagnostic, not an inverse-KL probability of 135.2%. The inverse-KL output itself lies in $[0,1]$. We exclude that differently defined number from the percentage-bound chart.

The result establishes that a numerical certificate can remain useful even when the number of parameters greatly exceeds the number of examples. It does not fully explain the much lower test error, prove SGD always finds such regions, or establish a result for ten-class MNIST, convolutional networks, or modern large models. Those are outside this paper's experiments.

## 9 · Limitations, useful ablations, and the idea to carry forward

The mean is optimized as well as the variances. Appendix C reports that the original SGD solution is close to the learned mean in the covariance-scaled geometry for true-label runs, but proximity is not a deterministic error guarantee. The bound is still about $Q$.

A diagonal Gaussian cannot express correlated perturbations. Appendix B discusses network symmetries and possible KL reductions from averaging equivalent parameterizations, but does not make exhaustive symmetry accounting practical. Appendix D finds the evaluated path-norm/Rademacher bounds vacuous for the unregularized setup; that does not rule out all possible norm-based generalization results.

**Proposed experiments, not reported findings:** freeze $w$ during posterior optimization to isolate how much mean movement helps; compare diagonal and correlated posteriors while retaining valid KL calculations; compare zero-centered and initialization-centered priors using a predeclared selection rule; and report optimization compute alongside certificate tightness. These would distinguish distribution shape, location, and optimization effort as explanations for improvement. Any data-dependent prior comparison must retain its statistical accounting.

<details><summary>Predict: does a 16.1% bound certify that the mean network makes at most 16.1% errors?</summary><p>No. It certifies the expected error of the randomized classifier under the theorem's assumptions and confidence. A guarantee for the mean network, a vote, or a particular sampled network requires additional reasoning.</p></details>

**The idea to carry forward:** noise tolerance becomes evidence about generalization only when it is combined with a valid reference distribution, an information cost, and finite-sample accounting. This paper turns that combination into a computed certificate, while leaving a visible gap between what the network achieves and what the proof guarantees.

### Reading map

| Guide question | Course PDF location |
| --- | --- |
| Why parameter count is insufficient | §§1.1–1.2 |
| Error definitions, Gaussian KL, inverse KL | §§2–2.2, Eq. 1–2; loss typo noted above |
| Why the posterior can use training data | Theorem 2.3 and Eq. 3 |
| Prior family and optimization objective | §§3–3.2, Eq. 4–5 |
| Monte Carlo correction and final certificate | §3.3, Eq. 6, Appendix A |
| Dataset, architecture and optimization details | §§4.1–4.3, Algorithm 1; initialization mismatch noted above |
| Results and the special random-label entry | §4.4, Table 1, §5 |
| Symmetries, mean movement and alternative bounds | Appendices B–D |

Continue with the recall cards below, revisit [Information bottleneck](/papers/t1-paper-1), or return to [Topic 1](/topics/topic-1). The [local course PDF](/papers/nonvacuous-bounds.pdf) preserves the supplied version.
