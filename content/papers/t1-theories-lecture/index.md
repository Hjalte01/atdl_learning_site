---
title: "Theories for Deep Learning · Lecture 1"
shortTitle: "Theories for Deep Learning"
topic: "topic-1"
order: 1
materialKind: "slides"
description: "Why can a network fit every training label and still generalize? Trace fit, selected solutions, information, PAC-Bayes, evidence, double descent and grokking."
status: "ready"
pdf: "/papers/theories-lecture.pdf"
---
## The puzzle: two perfect fits, two different predictions

You observe just two pairs: $(-1,-1)$ and $(1,1)$. Both $h_1(x)=x$ and $h_2(x)=x^3$ fit perfectly. At $x=1/2$, however, they predict $1/2$ and $1/8$. **Training error cannot choose between them. What makes a learning procedure select a useful continuation?**

This guide follows Raghavendra Selvan’s **Theories for Deep Learning**, the supplied 22-slide Topic 1 lecture dated **September 1, 2026** (`slides/1_T1L1_raghav.pdf`). [Open the course deck](/papers/theories-lecture.pdf). Slide numbers refer to PDF pages. Allow 25–30 minutes; familiarity with gradients, train/test splits and basic probability is enough. The guide follows the lecture’s sequence and connects its six papers, rather than proposing a new generalization theorem.

Every colored figure and numerical example below is a teaching construction, not measured model performance. The lecture deliberately uses provocative claims to motivate discussion. We distinguish those claims from statements that need assumptions: perfect training accuracy is not zero cross-entropy, interpolation does not remove variance, and no single complexity measure is established here as the explanation of deep learning.

## 1 · Fit and complexity ask different questions

[![Two functions agree at the training endpoints but disagree at a new input; fit alone cannot choose the continuation.](/figures/theories-lecture/interpolation.svg)](/figures/theories-lecture/interpolation.svg)

Read the top row as observed information and the lower rows as alternative continuations. Neither model has seen the middle target. If the hidden rule is $y=x$, the linear choice is better there; if it is $y=x^3$, the cubic choice is better. A preference is useful relative to a task, not automatically correct.

Slides 2–4 introduce the fit–complexity lens. A representative regularized empirical-risk objective is

$$
J(w)=\widehat R_S(w)+\lambda\Omega(w),\qquad
\widehat R_S(w)=\frac1n\sum_{i=1}^n\ell(h_w(x_i),y_i).
$$

$S$ is the $n$-example training sample; $h_w$ is the predictor with parameters $w$; $\ell$ measures one prediction’s loss; $\Omega$ penalizes a chosen property; and $\lambda\geq0$ sets its strength. The loss answers “does it fit these observations?” The penalty answers “which fits do we prefer?” This equation operates at the **objective box** in the next figure. It is a representative formulation, not a claim that every method explicitly optimizes a visible penalty.

For a toy polynomial representation $h_w(x)=w_1x+w_3x^3$, use $\Omega(w)=w_1^2+4w_3^2$. The two exact fits have penalties 1 and 4. With $\lambda=0.1$, their objectives are 0.1 and 0.4. This compares two candidates; it does not prove that either minimizes the full regularized objective. Shrinking the coefficients might reduce the penalty enough to accept some training loss.

The lecture’s classical U-shaped curve is a schematic. Statistical learning theory does not universally require nonzero training error, and fit/complexity terms need not vary monotonically with parameter count. These qualifications matter when reading the “do not fit perfectly” slogan on slide 4.

<details><summary>Predict: does selecting the lower-penalty fit prove it has lower test error?</summary><p>No. The preference and the data-generating task must align. Training observations alone do not identify the unseen middle target in this example.</p></details>

## 2 · Follow training, prediction and diagnosis separately

[![Training updates weights using data and an objective; prediction freezes weights; diagnostic branches inspect representations, weight distributions and curvature.](/figures/theories-lecture/pipeline.svg)](/figures/theories-lecture/pipeline.svg)

Read each row separately. **Training** changes $w$ using the objective and optimizer. **Inference** freezes the selected weights and maps a fresh input to a prediction. **Diagnosis** asks what changed in the learned solution; a diagnostic is not necessarily a training loss.

A functional neural architecture is $x\in\mathbb R^d\to T=f_w(x)\in\mathbb R^m\to z=g_w(T)\in\mathbb R^K\to\operatorname{softmax}(z)$. Here $T$ is a hidden representation, $z$ the class logits and $K$ the number of classes. The deck covers multiple architectures, so $d,m,K$ are symbolic; no undocumented layer widths are implied. For true class $y$, cross-entropy is $-\ln p_y$, where $p_y$ is that class’s softmax probability. Even a correct prediction with $p_y=0.9$ has loss about 0.105, not zero.

The optimizer arrow can be written

$$
w_{t+1}=w_t-\eta_t\nabla_w\left[\widehat R_{B_t}(w_t)+\lambda\Omega(w_t)\right].
$$

$t$ counts updates, $B_t$ is the sampled mini-batch, $\widehat R_{B_t}$ its average loss and $\eta_t$ the learning rate. Backpropagation computes the derivative; the optimizer applies it. With no explicit penalty ($\lambda=0$), initialization, architecture, mini-batch sampling and optimization can still favor particular solutions: this is **implicit bias**, not proof of a known universal hidden regularizer.

A small optimizer trace makes the distinction concrete. Let $h_w(x)=wx$, one pair be $(1,1)$, and use $\ell=\tfrac12(w-1)^2$. From $w_0=0$ with $\eta=1/2$, the derivative is $-1$ and $w_1=0.5$. The next derivative is $-0.5$, giving $w_2=0.75$. At inference with $x=2$, the current predictor outputs $1.5$. No label or gradient is needed for that prediction. These are illustrative updates, not the lecture’s experiments.

## 3 · Interpolation shifts the question to selection

[![A schematic double-descent curve varies capacity while a grokking timeline varies training time; they have distinct horizontal axes.](/figures/theories-lecture/phenomena.svg)](/figures/theories-lecture/phenomena.svg)

The upper panel varies a procedure’s capacity; the lower panel varies updates for a fixed task. Neither panel contains measured values. Slides 5–10 juxtapose three observations:

- **Random-label fitting:** the same architecture can fit meaningful and shuffled labels. A capacity measure that only describes the available hypothesis class cannot by itself distinguish these two fitted solutions. This does not mean every data-dependent bound is useless.
- **Double descent:** test error can rise near the interpolation threshold and fall again beyond it. The threshold depends on architecture, data, optimizer and training budget; $p\approx n$ is an illustrative linear-model picture, not a universal neural-network rule.
- **Grokking:** training accuracy can saturate long before held-out accuracy rises. The post-fit interval can contain meaningful representation change. The slide’s delay range is a motivating report, not a fixed prediction for every task.

For squared prediction error at a fixed input $x$, the familiar decomposition remains

$$
\mathbb E_{S,Y\mid x}\!\left[(Y-\widehat h_S(x))^2\right]
=\sigma^2(x)+\left(\mathbb E_S\widehat h_S(x)-f_*(x)\right)^2
+\operatorname{Var}_S[\widehat h_S(x)].
$$

Here $f_*(x)=\mathbb E[Y\mid x]$, $\sigma^2(x)=\operatorname{Var}(Y\mid x)$ is irreducible label noise, and $\widehat h_S$ is trained on random sample $S$, independently of the fresh test label. The terms are noise, squared bias and variance across training samples. Interpolating each sample does **not** force the last term to zero. The question behind slides 10 and 19 is better phrased as: *why need variance not increase monotonically with nominal size?*

In underdetermined linear regression, $X\in\mathbb R^{n\times p}$ is the design matrix and $y\in\mathbb R^n$ the targets. The minimum Euclidean-norm least-squares solution is $\widehat\beta=X^+y$, with $X^+$ the Moore–Penrose pseudoinverse. It interpolates when $y$ lies in the column space of $X$. Which solution is selected depends on the algorithm and initialization. This slide-7 example is not a formula for arbitrary neural networks.

<details><summary>Predict: does zero training error identify the best stopping time?</summary><p>No. Held-out behavior may still improve or degrade after fitting. A validation set can guide stopping; the final test set should remain reserved for evaluation.</p></details>

## 4 · Information measures the representation

[![Two independent input bits become a representation that retains the label bit and discards the nuisance bit.](/figures/theories-lecture/information.svg)](/figures/theories-lecture/information.svg)

Slide 12 moves the diagnostic into the hidden-state box $T$. In this toy, $X=(A,B)$ contains two independent fair bits, the label is $Y=A$, and $T=A$. Read the arrows as information retained, not a prescribed network architecture.

For finite discrete variables $U,V$, mutual information measures how much knowing one reduces uncertainty about the other:

$$
I(U;V)=\sum_{u,v}p(u,v)\log_2\frac{p(u,v)}{p(u)p(v)}.
$$

$p(u,v)$ is the joint probability and $p(u),p(v)$ its marginals; zero-probability contributions are interpreted as zero. Base-2 logarithms give bits. The information-bottleneck objective is often written, for positive tradeoff $\beta$,

$$
I(X;T)-\beta I(T;Y).
$$

The first term penalizes retained input information; the negative second term rewards retained label information. In the toy, $T=X$ gives $(I(X;T),I(T;Y))=(2,1)$ bits, while $T=A$ gives $(1,1)$. With $\beta=2$, the objectives are 0 and $-1$. A constant representation gives $(0,0)$: maximal discarding also discards the task.

This objective explains the theoretical tradeoff. It is **not** the explicit loss used to train the classifier in [Topic 1 paper 1](/papers/t1-paper-1). That guide separates cross-entropy training from later information-plane measurements. Continuous deterministic representations and discretized estimates require care; changing quantization can change the measured compression story. The lecture’s two-phase account is a claim to examine, not a universal property of SGD.

## 5 · PAC-Bayes measures a distribution relative to a prior

[![A fixed prior and learned distribution feed KL and empirical randomized risk into a bound; inference samples a network from that distribution.](/figures/theories-lecture/pacbayes.svg)](/figures/theories-lecture/pacbayes.svg)

Slide 13 changes the object of study from a single weight vector to a distribution $Q$ of predictors. Read the first row as certificate construction and the second as randomized prediction. A sample-independent prior $P$ supplies the reference; the trained distribution $Q$ may depend on $S$.

For i.i.d. examples, a loss in $[0,1]$, and a suitable fixed prior, the lecture displays the additive PAC-Bayes form

$$
R(Q)\leq\widehat R_S(Q)+
\sqrt{\frac{\mathrm{KL}(Q\Vert P)+\ln(2\sqrt n/\delta)}{2n}}.
$$

$R(Q)$ is expected fresh-example loss averaged over a predictor drawn from $Q$; $\widehat R_S(Q)$ is the analogous training average; $n$ is sample size; and $\delta\in(0,1)$ is the allowed probability that the simultaneous guarantee fails over drawing $S$. The logarithms here are natural, so $\mathrm{KL}$ is in **nats**, not bits. In a finite model family, $\mathrm{KL}(Q\Vert P)=\sum_hQ(h)\ln[Q(h)/P(h)]$, requiring positive prior mass wherever $Q$ places mass.

Toy certificate: $n=1000$, $\delta=0.05$, $\widehat R_S(Q)=0.05$ and $\mathrm{KL}=10$ give a penalty about 0.0926 and upper bound **0.1426**. This is computed from assumed inputs, not a result reported for the lecture’s network. Computing a certificate for a real stochastic network also requires accounting for estimation of its randomized error; see [paper 2](/papers/t1-paper-2).

Why use KL rather than parameter count? It measures how far the selected distribution moved from the reference. A large network can in principle have a modest prior-relative cost. The guarantee concerns randomized risk; it does not directly certify the unperturbed mean network or a majority-vote classifier. A bound above one is uninformative for this loss, not evidence that the model must fail.

<details><summary>Predict: can we choose P equal to the learned Q afterward to make KL zero?</summary><p>Not under this sample-independent-prior guarantee. That would hide the data-dependent selection inside the reference. Different prior-selection mechanisms need their own valid accounting.</p></details>

## 6 · Evidence weighs width together with prior density

[![Two local posterior peaks have different curvature; their integrated evidence depends on peak height, width and the prior, not width alone.](/figures/theories-lecture/evidence.svg)](/figures/theories-lecture/evidence.svg)

Slides 14 and 22 ask about the **mass around a fit**, rather than only its loss value. Bayesian evidence averages likelihood over the prior: $p(S)=\int p(S\mid w)p(w)\,dw$. Under a local Gaussian approximation around an isolated mode $\widehat w$,

$$
p(S)\approx p(S\mid\widehat w)p(\widehat w)
(2\pi)^{p/2}(\det H)^{-1/2}.
$$

Here $p$ is the number of weight coordinates (distinct from the probability-density symbol $p(\cdot)$), and $H$ is the positive-definite Hessian of the negative log joint density at the mode. Its eigenvalues describe local curvature. The formula multiplies height by effective width at the **evidence box**, not the SGD update arrow. It needs a sufficiently accurate local approximation; singular modes, broad non-Gaussian regions and multiple modes complicate the picture.

Taking negative logs gives

$$
-\ln p(S)\approx-\ln p(S\mid\widehat w)-\ln p(\widehat w)
+\tfrac12\ln\det H-\tfrac p2\ln(2\pi).
$$

The last term is omitted in the slide’s abbreviated negative-log expression; it cancels for comparisons at fixed $p$ but should not be silently discarded across dimensions. In a one-dimensional toy with equal likelihood and prior height, $H=4$ gives width factor $1/2$ while $H=1$ gives 1. The latter has twice the approximate evidence, corresponding to $\ln2\approx0.693$ less negative log evidence. Equal peak heights are an assumption of this example.

“Flatness alone” is insufficient: rescaling coordinates changes curvature and prior density. Their combined mass is the relevant object. See [paper 3](/papers/t1-paper-3) for the Bayesian interpretation and its limitations.

The lecture also sketches SGD noise scale as $g\approx\epsilon N/B$, using learning rate $\epsilon$, dataset size $N$ and batch size $B$. In the paper’s convention, the more informative finite-batch expression is $g=\epsilon(N/B-1)$; the sketch assumes $B\ll N$. For $N=1000$, $B=100$, $\epsilon=0.1$, these are 1 and 0.9; at full batch the finite-batch expression is zero. This scale is not a proof that SGD samples an exact Bayesian posterior or always selects the broadest basin.

## 7 · Six lenses, six different objects

[![Six theory lenses are mapped to representations, weight distributions, local evidence, training procedures, representation structure and weight descriptions.](/figures/theories-lecture/lenses.svg)](/figures/theories-lecture/lenses.svg)

Read the boxes as parallel questions, not stages in one algorithm. Slides 15–18 complete the route through Topic 1. The differences prevent us from treating all six quantities as interchangeable complexity scores.

| Lecture / guide | What is measured or modeled? | What to question |
| --- | --- | --- |
| Slide 12 · [Information bottleneck](/papers/t1-paper-1) | Information retained by hidden representations | Does the estimated compression survive changing the estimator? |
| Slide 13 · [Nonvacuous bounds](/papers/t1-paper-2) | Randomized error and prior-relative KL | Which predictor and confidence accounting does the certificate cover? |
| Slide 14 · [Bayesian SGD](/papers/t1-paper-3) | Prior-weighted local volume and optimizer noise | Are local and stochastic approximations justified? |
| Slide 15 · [Double descent](/papers/t1-paper-4) | Fitting capacity of the full training procedure | Are architecture, budget and data distribution controlled? |
| Slide 16 · [Grokking](/papers/t1-paper-5) | Emergence of structured representations | Does the effective toy theory transfer to the actual network? |
| Slide 17 · [Binarized compression](/papers/t1-paper-6) | BDM approximation to weight-description complexity | Is a loss correlation mistaken for causal or exact compression? |

Effective model complexity (EMC) on slide 15 refers to the largest sample size the **procedure** can fit to a small prescribed error. Its compact maximum notation suppresses how samples are drawn and how fitting is evaluated. It is an operational measurement, not merely the number of parameters or an established universal solution-complexity penalty.

For grokking, the useful object is representation structure while training fit has nearly stopped changing. For binarized networks, BDM is a computable approximation inspired by algorithmic complexity, not exact Kolmogorov complexity and not a measured full two-part data code. The supplied paper uses Adam with cross-entropy and straight-through gradient estimation, so the slide’s phrase “SGD drifts” should not be read as its exact optimizer specification. Neither correlation establishes that the training objective directly minimizes BDM.

## 8 · Turn the open questions into discriminating experiments

Slides 19–21 invite a critical reading and two exercises: random-label fitting/model-wise double descent and modular-addition grokking. The deck supplies schematic curves and references, but no common, fully specified six-method benchmark. A numerical leaderboard would be misleading; use the individual paper guides for their source-specific datasets, metrics and settings.

For the first exercise, hold the data split and training protocol fixed, compare true versus shuffled training labels, and sweep width with enough training to identify fitting failure. Record training and held-out error separately, plus one candidate diagnostic. This is a **proposed experiment**, not an execution or a result. If two fitted solutions have the same diagnostic but very different test error, that challenges the diagnostic’s sufficiency in this setting.

For the grokking exercise, log training accuracy, validation accuracy, loss and a prespecified representation measure over a long run. Repeat seeds. A diagnostic moving before validation accuracy is suggestive; it does not establish causation. An intervention that changes the proposed mechanism while controlling competing changes would provide stronger evidence. Use validation for choices and reserve test results for the final comparison.

The reasoning path reconstructed here is: fit alone leaves ambiguity; the training procedure selects one solution; alternative theories inspect different aspects of that selection. This is a teaching interpretation, not access to the lecturer’s private reasoning. The slide’s strongest “entirely complexity” language is a research framing, not a theorem that ignores the task, data distribution or noise.

## The idea to carry forward

Ask **what object a theory describes, where that object enters computation, and which assumptions connect it to unseen data**. A representation statistic, a randomized-risk certificate and an empirical capacity curve can all be useful without being the same explanation. The lecture’s “theories, plural” is an invitation to test their scope.

Try redrawing the training/prediction/diagnosis diagram and placing all six papers on it. Then explain why perfect training accuracy neither identifies the selected function nor proves that test variance vanished.

## Source and slide reading map

The source is the [local 22-slide course PDF, September 1, 2026](/papers/theories-lecture.pdf), with its original references retained. No results from a different deck version are merged here.

- **Slides 1–4:** lecture identity and fit–complexity framing; guide §§1–2 add the polynomial and gradient calculations.
- **Slides 5–10:** motivating scaling, interpolation, double descent and grokking; §3 distinguishes the schematic trends from general identities. The slide-5 compute-growth figures are historical motivations, not current estimates verified by this guide.
- **Slides 11–18:** six readings and candidate complexity measures; §§4–7 connect their mathematical objects and the existing detailed guides.
- **Slides 19–21:** open questions, exercises and takeaways; §8 proposes controlled checks rather than reporting new experiments.
- **Slide 22:** Laplace approximation; §6 restores the dimension-dependent Gaussian constant and states the approximation’s conditions.

All equations with numeric inputs are teaching examples unless explicitly identified as a displayed lecture formula. The guide adds no trained-model simulator; prediction disclosures and the eight study cards below provide practice.
