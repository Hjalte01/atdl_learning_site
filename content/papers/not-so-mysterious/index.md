---
title: "Deep Learning is Not So Mysterious or Different"
shortTitle: "Not so mysterious"
topic: "general"
order: 3
materialKind: "overview"
description: "How soft inductive biases connect noise fitting, overparametrization and double descent—and what remains unexplained."
status: "ready"
pdf: "/papers/not-so-mysterious.pdf"
---
## The puzzle: does the ability to memorize explain what gets learned?

Two learners can fit every training label, including random labels. Must both fail on new examples? That conclusion confuses **what a model can represent** with **which solution its learning procedure prefers**.

Andrew Gordon Wilson’s position paper argues that benign overfitting, successful overparametrization and double descent are compatible with established generalization frameworks and occur outside neural networks. The unifying idea is a flexible hypothesis space with a preference for simple solutions. This is the author’s position, not a claim that deep learning is completely understood.

This general course overview follows the supplied **arXiv:2503.02113v2, July 10, 2025**, a 21-page ICML 2025 paper. It remains unassigned to a numbered topic in the library. Allow about 20 minutes. The original teaching diagrams and worked calculations below are schematic; the source Figure 5 is explicitly identified as reported evidence. No trained model runs here. Eight foundation cards are grouped with Topic 1 for study and printing.

## 1 · Replace a prohibition with a preference

[![Restriction excludes high-order polynomials; a soft penalty keeps them available but increasingly costly.](/figures/not-so-mysterious/bias.svg)](/figures/not-so-mysterious/bias.svg)

Read the two rows as alternative choices, not sequential stages. A degree-two model excludes every higher-order term. A degree-150 polynomial with an order-dependent penalty can still use those terms, but pays more for doing so. This is the distinction in §2 and conceptual Figures 3–4.

A plausible teaching reconstruction is: preserve flexibility because the data may be complicated; then prefer economical explanations among competing fits. That reconstructs the argument, not the author’s private reasoning. A soft preference can come from explicit regularization, a prior, or the interaction of architecture and training. It is broader than simply adding weight decay.

Prerequisites: a weighted sum, squared error, probability, and the difference between training and held-out data. For the bound, remember that a probability distribution assigns a total mass of one. You do not need a neural-network backbone to follow the central example.

<details><summary>Predict: can a strong soft preference still be wrong?</summary>

Yes. A preference for smooth, low-order functions may be poorly aligned with the task, or so strong that fitting suffers. “Soft” means the alternatives are available; it does not guarantee useful learning from finite data.

</details>

## 2 · Follow the polynomial through training and prediction

[![Inputs become fixed polynomial features, learned coefficients produce predictions, and training combines fit with an order-dependent penalty.](/figures/not-so-mysterious/architecture.svg)](/figures/not-so-mysterious/architecture.svg)

Read left to right. Each scalar input $x_i$ becomes a fixed feature vector $\phi(x_i)=(1,x_i,\ldots,x_i^J)^\top\in\mathbb R^{J+1}$. The learned vector $w\in\mathbb R^{J+1}$ produces $f_w(x_i)=w^\top\phi(x_i)$. Across $n$ examples, the design matrix $\Phi\in\mathbb R^{n\times(J+1)}$ gives predictions $\Phi w$. The feature map is nonlinear in the input but the predictor is linear in its parameters.

For Gaussian observation noise, a teaching version of the §2 objective is

$$
\mathcal L(w)=\frac{1}{2\sigma^2}\sum_{i=1}^n(y_i-w^\top\phi(x_i))^2+\sum_{j=0}^J\gamma^j w_j^2,\qquad \gamma>1.
$$

Here $y_i$ is an observed target, $\sigma^2>0$ is the assumed noise variance and $\gamma$ controls the increasing cost of higher-order coefficients. Constants independent of $w$ are omitted. The first term lives at the prediction-to-target comparison; the second reads the coefficients directly. The source prose describes division by $1/(2\sigma^2)$, but the Gaussian negative log-likelihood **multiplies squared error by $1/(2\sigma^2)$**, as written here and consistent with §4’s small-noise argument.

Why this objective? Decreasing $\sigma$ makes residual error more expensive, while the penalty ranks alternative fits. Finite noise and regularization do not in general produce exact interpolation. The source’s interpolation argument concerns prioritizing fit strongly enough; the formula alone does not promise zero training loss.

[![Training repeatedly updates coefficients from targets and loss; inference uses the same fixed features with learned coefficients and no targets.](/figures/not-so-mysterious/paths.svg)](/figures/not-so-mysterious/paths.svg)

This schematic separates two paths. Training learns $w$; inference holds $w$ fixed and evaluates a new $x$. The degree, feature map, $\sigma$ and $\gamma$ are fixed within this toy fit. Selecting them from data is an additional model-selection step.

To see where optimization operates, a teaching gradient step is

$$
w^{(t+1)}=w^{(t)}-\eta\left[\sigma^{-2}\Phi^\top(\Phi w^{(t)}-y)+2Dw^{(t)}\right],\quad
D=\operatorname{diag}(\gamma^0,\ldots,\gamma^J).
$$

$t$ counts updates, $\eta>0$ is a learning rate, and $y$ stacks targets. The residual arrow supplies the first gradient; the coefficient penalty supplies the second. Recompute both after each step. This is an illustrative optimizer, not a claimed implementation of the source experiments. Convergence depends on step size and conditioning; high polynomial powers can be numerically troublesome.

## 3 · A numerical preference between identical training fits

[![Two cubic coefficient vectors fit the same endpoint observations but have penalties 2 and 8 and make different midpoint predictions.](/figures/not-so-mysterious/example.svg)](/figures/not-so-mysterious/example.svg)

Take two observations, $(x,y)=(-1,-1)$ and $(1,1)$, with $J=3$ and $\gamma=2$. Candidate A is $f_A(x)=x$, so $w_A=(0,1,0,0)$. Candidate B is $f_B(x)=x^3$, so $w_B=(0,0,0,1)$. Both have zero residual on those two observations, but their penalties are $2^1=2$ and $2^3=8$. At a new input $x=0.5$, their predictions are $0.5$ and $0.125$.

Trace the diagram through the same feature and weighted-sum boxes as §2: $\phi(0.5)=(1,0.5,0.25,0.125)$. The preference picks A **between these two candidates**. This is not proof that A is the global minimizer: combinations of coefficients can have lower penalty, and a finite-noise optimum may accept residual error. Nor do the two training points reveal which midpoint prediction is correct.

A one-step calculation makes the learning loop concrete. Start with $w=0$, take $\sigma^2=1$ and $\eta=0.1$. For these two examples, $\Phi^\top(\Phi w-y)=(0,-2,0,-2)$ and the regularization gradient is zero. The next vector is $(0,0.2,0,0.2)$. Its endpoint predictions are $\pm0.4$. At this new state, the penalty-gradient coordinates for the linear and cubic terms are $0.8$ and $3.2$: subsequent updates feel their different costs. These are computed toy values, not results from Figure 1.

## 4 · Bound the selected solution, not just the available space

[![A data-independent prior assigns a complexity cost, empirical loss measures fit, and both enter a high-probability risk certificate.](/figures/not-so-mysterious/bound.svg)](/figures/not-so-mysterious/bound.svg)

The countable-hypothesis bound in source Eq. (1), proved in Appendix C, formalizes the distinction. A highly expressive class can contain a particular hypothesis with low empirical risk and substantial prior mass:

$$
R(h)\leq\widehat R(h)+\Delta\sqrt{\frac{\log(1/P(h))+\log(1/\delta)}{2n}}.
$$

$h$ is a hypothesis from a countable set; $P(h)>0$ is its probability under a prior fixed independently of the training sample; $\widehat R(h)$ is its average training loss; $R(h)$ is expected loss on the same data-generating distribution. The per-example loss lies in an interval of width $\Delta$, $n$ is sample size and $1-\delta$ is confidence. Logs are natural. Assume independent, identically distributed examples for this teaching application. Unbounded squared error from §2 cannot simply be inserted without a different bound or justified bounded-loss construction.

The diagram is a **post-training certification path**, not the polynomial’s training objective. First fit a predictor, then measure its empirical risk and its cost under a pre-specified prior. The prior used for certification need not be the training regularizer. It cannot be freely chosen after inspecting the same sample without accounting for that dependence.

Why does a data-selected $h$ qualify? For a fixed $h$, a concentration inequality gives a failure budget $\delta P(h)$. Summing these budgets across all hypotheses gives at most $\delta$. Thus the event controls all hypotheses together, including the one training selects. This is the key step of Appendix C.

**Worked certificate:** for a bounded classification loss with $\Delta=1$, $n=1000$, $\delta=0.05$, empirical error $0.02$ and prior mass $P(h)=2^{-10}$, the penalty is approximately $0.07045$. The bound is $R(h)\leq0.09045$ at 95% confidence. With mass $2^{-100}$ and the same empirical error, it becomes approximately $0.21015$. These are hypothetical valid prior masses, not measured networks or a claim that any arbitrary ten-bit file defines a certificate.

Source Eqs. (2)–(4) connect this cost to description length. Prefix-free descriptions allow a prior proportional to $2^{-K(h\mid A)}$, where $K$ is shortest program length conditional on fixed background information $A$. We cannot compute that shortest length; a valid explicit encoding upper-bounds it. Source Eq. (4) includes a $2\log C(h)$ overhead when moving to a standard code length $C(h)$. A compressed model’s risk must be measured for the predictor actually encoded; data-dependent architectures, decoders or codebooks cannot be silently free.

PAC-Bayes extends this perspective to distributions $Q$ over predictors through a divergence $\mathrm{KL}(Q\|P)$. Do not substitute that divergence into the displayed bound and assume every constant and risk interpretation is unchanged. A point mass under a continuous prior can have infinite KL. For a detailed distributional certificate, continue to [Nonvacuous generalization bounds](/papers/t1-paper-2).

<details><summary>Predict: does a loose upper bound prove that a model generalizes poorly?</summary>

No. An upper bound can be uninformative while the model performs well. Tightening the certificate and improving the predictor are related but distinct goals.

</details>

## 5 · Connect three phenomena without erasing their conditions

**Benign overfitting (§4):** the same model class can fit random labels and learn useful structure on natural data. The paper uses this broad meaning and also discusses mixed signal and noise. It does not claim that fitting independent random labels makes unseen random labels predictable. A preference for simple solutions can distinguish structured from randomized datasets even when a whole-class capacity measure cannot.

**Overparametrization (§5):** more coefficients need not mean a more complicated fitted function. Increasing flexibility can also alter the bias toward compressible solutions. The paper offers supporting examples and intuitions; why scaling produces this bias so effectively in neural networks remains an open question. “Larger always generalizes better” is not a theorem here.

**Double descent (§6):** test loss can fall, rise around the fitting threshold, then fall again while training loss stays low. Once several models fit training data equally well, their differing test performance requires something beyond training fit. The source connects the second descent to changes in the solutions selected, compressibility and effective dimension.

[![Large and small eigenvalues contribute fractions to effective dimension; three example contributions sum to 1.5 despite three available coordinates.](/figures/not-so-mysterious/dimension.svg)](/figures/not-so-mysterious/dimension.svg)

For a positive semidefinite matrix $A$ with eigenvalues $\lambda_i\geq0$ and regularization scale $\alpha>0$, §3.2 defines

$$
N_{\mathrm{eff}}(A)=\sum_i\frac{\lambda_i}{\lambda_i+\alpha}.
$$

This is a diagnostic box after training, not a parameter update or stand-alone bound. For eigenvalues $(9,1,1/9)$ and $\alpha=1$, contributions are $(0.9,0.5,0.1)$, giving $1.5$. The schematic shows a soft count of influential directions. Changing $\alpha$ changes that count. Hessian-based flatness also depends on parametrization and cannot universally rank generalization.

The source’s Figure 1(f) uses a Hessian diagnostic with $\alpha=1$; Figure 1(g) instead uses a parameter-covariance diagnostic with $\alpha=10$. Their numerical scales are not directly interchangeable. There is also an algebraic caveat in §6: when $X\in\mathbb R^{n\times d}$ and $d>n$, $X^\top X$ is singular, so the printed ordinary inverse cannot provide the interpolating solution. Use $w^*=X^+y$, or $X^\top(XX^\top)^{-1}y$ when $X$ has full row rank. This correction preserves the intended minimum-norm argument; it does not supply missing experimental implementation details.

## 6 · Read the evidence at the scale it supports

[![Source Figure 5 compares test RMSE of degree-two, degree-fifteen and regularized degree-fifteen polynomial fits across three synthetic target functions and training sample sizes.](/figures/not-so-mysterious/source-figure-5.png)](/figures/not-so-mysterious/source-figure-5.png)

**Reported result, not a teaching simulation:** Figure 5 of the supplied v2 PDF, page 5. Read each panel as a different regression problem; the vertical axis is logarithmic test RMSE and the horizontal axis is training sample count. The caption reports means and one-standard-deviation shading over 100 fits with 100 test samples. Appendix D specifies Gaussian input locations, 10–100 training samples, and an order-dependent penalty $\sum_j 0.01^2j^2w_j^2$ for the regularized degree-15 model. This is different from the $2^j$ penalty used for Figure 1 and our toy example.

The targets are a degree-two polynomial, a degree-fifteen polynomial and $\cos(3\pi x/2)$. The baselines are unregularized degree-two and degree-fifteen fits. The author reports that the regularized flexible model matches or improves on the alternatives across these displayed settings. This supports a synthetic demonstration of a useful preference, not a universal ranking of model families. No exact RMSE values are transcribed from the plotted curves.

Other evidence has different roles: Figure 1(d–e) compares increasing label corruption for an RBF Gaussian process and a ResNet on CIFAR-10; Figure 1(f) reports converged ResNet-18 cross-entropy on CIFAR-100 across widths; Figure 1(g) gives a linear random-feature MSE example. Appendix D calls the CIFAR-10 network PreResNet-20 although the panel says ResNet-20. Much evidence is adapted from earlier studies. Conceptual landscape pictures should not be read as measured volumes.

A useful proposed ablation would hold the basis, training samples and selection budget fixed while comparing isotropic and order-dependent penalties, including tuning each fairly on validation data. Repeat across target families and noise levels, reporting held-out error and fit. That would test whether the *shape* of the preference helps beyond simply regularizing. This is a proposed experiment, not an additional reported result.

## 7 · What remains distinctive—and a route through the course

The final sections explicitly preserve open questions. In §8.1, a fixed-basis predictor $w^\top\phi(x)$ becomes $w^\top\phi(x;v)$, where $v$ learns the representation itself. This changes which inputs are treated as similar. Kernel learning can also learn similarities; the claim is about the effectiveness of neural representations, especially in high dimensions, rather than exclusive ownership of the idea.

Section 8.2 discusses broad transfer and in-context learning. Fixed parameters do not imply fixed activations: context can change the computation without gradient updates. No-free-lunch results about averaging over all tasks do not forbid good performance on a structured distribution of real tasks.

Section 8.3 discusses low-loss paths connecting independently trained solutions. Such a path does not imply that straight-line interpolation is low loss, or that connected predictors agree on every test input. Figure 7 concerns ResNet-164 on CIFAR-100 and is adapted from prior work. The discussion also distinguishes evidence that stochasticity is not necessary in some settings from the practical efficiency of stochastic optimizers.

Use this overview to connect [Bayesian SGD](/papers/t1-paper-3), [Deep double descent](/papers/t1-paper-4), and [Binarized networks and compression](/papers/t1-paper-6). Do not equate a BDM diagnostic with the valid prefix-free code used in a certificate. Then compare fixed polynomial features with learned features in [REPA](/papers/t2-paper-1). The [representation textbook roadmap](/papers/representation-overview) develops the broader representation-learning perspective.

## Source map and idea to carry forward

| Question | Supplied source location |
| --- | --- |
| What is the position, and what is not claimed? | Abstract, §1, §§7–9 |
| What distinguishes a soft preference? | §2, Figures 3–5 |
| What assumptions make the certificate valid? | §3.1, Eqs. (1)–(4), Appendices A and C |
| What does effective dimension measure? | §3.2; Figure 1(f–g); Appendix D |
| How do noise fitting, scale and double descent connect? | §§4–6, Figures 1 and 6 |
| What remains distinctive or unresolved? | §8, Figure 7, §9 |
| How were the displayed experiments configured? | Appendix D, page 21 |

Read the [supplied course PDF](/papers/not-so-mysterious.pdf) for the complete position and references. This guide corrects two algebraic/prose pitfalls explicitly and distinguishes synthetic evidence, conceptual intuition and rigorous assumptions.

**The idea to carry forward:** when a flexible learner generalizes, ask which fitted solution it prefers, how that preference is expressed, and what evidence or bound actually supports the claim. Parameter count and the ability to fit noise leave those questions unanswered.
