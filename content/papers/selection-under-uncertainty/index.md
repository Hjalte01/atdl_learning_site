---
title: "Machine Learning: The Science of Selection under Uncertainty"
shortTitle: "Selection under uncertainty"
topic: "general"
order: 4
materialKind: "overview"
description: "Why picking the best observed predictor needs an uncertainty budget: a roadmap from validation to PAC-Bayes and online decisions."
status: "ready"
pdf: "/papers/selection-under-uncertainty.pdf"
---
## The puzzle: did you select skill or a lucky result?

You compare many classifiers and keep the one with the smallest measured error. More candidates make it easier to find a useful rule—but also easier to find a rule that got lucky on these particular examples. How can data support selection without turning the winning score into an unjustified promise?

Yevgeny Seldin’s textbook treats learning as **selection under uncertainty**. Design supplies candidates, computation searches them, and statistics accounts for the gap between observed and future performance. This is a selective reading roadmap through those connections, not an exhaustive treatment of its proofs or exercises.

We follow the supplied **arXiv:2509.21547v2**, whose margin says **29 January 2026** and whose title page says **January 30, 2026**. Both dates describe this same course copy. The material remains general/unassigned in the library; eight foundation cards are grouped with Topic 1 for study and printing. Allow about 25 minutes. All diagrams and numerical examples below are original teaching constructions, not benchmark results. No trained model or simulation runs on this page.

## 1 · A map of the selection problem

[![Three roles in learning and three reading routes: design candidates, compute a selection, quantify uncertainty; foundations branch toward PAC-Bayes and online learning.](/figures/selection-under-uncertainty/map.svg)](/figures/selection-under-uncertainty/map.svg)

Read the top row as cooperating roles, not a neural network. The lower rows are reading routes. This schematic follows Chapter 1 and the preface’s reading guide: Chapters 1–2, §§3.1–3.3 and §§4.1–4.4 form the foundation; the remaining concentration/generalization material and Chapter 7 can then be studied as separate branches. Chapter 5’s regression route needs Chapters 1–2 but is otherwise independent.

Prerequisites are averages, probability, logarithms and the idea of a function. For PAC-Bayes, add probability distributions over functions; for online learning, add a sequence of decisions. Appendices A–D supply set, probability, linear-algebra and calculus reminders.

A plausible reconstruction of the book’s teaching logic is: first make a prediction rule concrete, then separate its true quality from its noisy estimate, then pay for searching among rules. Finally, ask what changes when decisions determine which feedback becomes visible. This is our interpretation of its organization, not a claim about the author’s private reasoning.

## 2 · Follow a nearest-neighbor classifier through selection and use

[![Training stores labeled examples, validation compares K values, and a frozen classifier sorts distances and votes on a new input.](/figures/selection-under-uncertainty/paths.svg)](/figures/selection-under-uncertainty/paths.svg)

The **candidate construction** box stores a training set $S_{\mathrm{train}}=\{(x_i,y_i)\}_{i=1}^{n_{\mathrm{train}}}$ and a distance rule $d$. For binary labels $y_i\in\{-1,+1\}$, each candidate uses a different neighbor count $K$. The **inference** row implements source Algorithm 1 (§2.2):

$$
h_K(x)=\operatorname{sign}\!\left(\sum_{j=1}^{K}y_{\sigma_x(j)}\right).
$$

Here $x$ is a new input, $\sigma_x$ orders training indices by increasing $d(x_i,x)$, and $K$ determines how many labels enter the vote. Odd $K$ avoids a tied binary vote; a deterministic rule is still needed for equal distances. The book leaves distance selection outside its scope. We do not invent a feature extractor or learned metric.

The **selection** row compares the already constructed candidates on separate validation data and freezes the chosen $K$. Prediction carries forward the stored examples, distance rule and chosen $K$; it needs no new target label or optimization step. A final test set evaluates that frozen procedure. Refitting after selection creates a changed predictor and must occur before the untouched final evaluation.

[![One-dimensional query at 0.2: sorted neighbors have labels plus, minus, minus; one neighbor predicts plus while three predict minus.](/figures/selection-under-uncertainty/neighbors.svg)](/figures/selection-under-uncertainty/neighbors.svg)

Read this schematic as a trace through the same distance → sort → vote boxes. Let training pairs be $(0,+1),(1,-1),(2,-1)$ and query $x=0.2$, using absolute distance. Distances are $0.2,0.8,1.8$. One neighbor votes $+1$; three neighbors sum to $-1$. The representation is the stored labeled point set, not a vector of fitted neural weights. This toy cannot establish which $K$ generalizes better; it exposes how $K$ changes the prediction.

On distinct training inputs, querying a 1-NN classifier with its own stored examples gives zero training error. That is a property of lookup, not evidence of zero future error. Duplicate inputs with conflicting labels complicate this statement, as the source’s §2.2.1 footnote notes.

<details><summary>Predict: if you repeatedly tune K using the “test” score, is it still a test?</summary>

Its name does not protect it. Once the score influences selection, the set plays a validation role. Section 2.3.1 makes this operational distinction. Cross-validation reuses data through held-out folds (§2.3.2), but choosing among candidates still creates a selection problem.

</details>

## 3 · Put uncertainty at the score-to-selection arrow

A loss function says which errors matter (§2.1.2). For zero-one classification loss, a mistake costs one and a correct prediction costs zero. Other applications can weight false positives and false negatives differently. Accuracy alone need not match the objective.

For a fixed rule $h$, the evaluation box computes an empirical risk, while the quantity we care about remains an expectation:

$$
\widehat L(h,S)=\frac1n\sum_{i=1}^{n}\ell(h(X_i),Y_i),
\qquad L(h)=\mathbb E_{(X,Y)\sim D}[\ell(h(X),Y)].
$$

$S$ is the evaluation sample of $n$ independently and identically distributed (i.i.d.) pairs from $D$; $\ell$ is loss; $L$ is future average loss under that same distribution. In this section, $n$ counts evaluation examples, not stored K-NN training points. For a rule chosen independently of $S$, the average is an unbiased estimate of $L(h)$. Replacing $h$ with the winner selected using $S$ breaks that simple argument.

Source Eq. (4.1), derived from Hoeffding’s inequality, gives a one-sided certificate for a fixed $h$ and losses in $[0,1]$:

$$
L(h)\leq\widehat L(h,S)+\sqrt{\frac{\ln(1/\delta)}{2n}}
\quad\text{with probability at least }1-\delta.
$$

$\delta\in(0,1)$ is the allowed failure probability over repeated draws of $S$, and $\ln$ is the natural logarithm. The statement does not assign a posterior probability to an unknown fixed $L(h)$. It describes how often the sampling procedure produces a valid bound. The certificate attaches to the **score** box; it does not train the classifier.

Why a square root? Hoeffding’s tail decays as $e^{-2n\epsilon^2}$ for an error margin $\epsilon$. Setting that tail to $\delta$ and solving yields the margin above. Quadrupling $n$ halves it. The assumptions matter: distribution shift or dependent observations need other analysis; an unbounded squared loss cannot simply be inserted into this $[0,1]$ formula.

## 4 · Pay for the candidates you compare

[![For 1000 evaluation examples and delta 0.05, the one-sided error margin grows from 0.0387 for one candidate to 0.0616 for 100 and 0.0917 for a million.](/figures/selection-under-uncertainty/bounds.svg)](/figures/selection-under-uncertainty/bounds.svg)

This is a calculated theoretical comparison, not a measured learning curve. The plotted quantities are additive error margins at $n=1000$, $\delta=0.05$. They follow source Eq. (4.4): for a finite set $H$ of $M$ rules fixed independently of $S$, simultaneously for every $h\in H$,

$$
L(h)\leq\widehat L(h,S)+\sqrt{\frac{\ln(M/\delta)}{2n}}
\quad\text{with probability at least }1-\delta.
$$

Each rule gets failure budget $\delta/M$. A union bound adds the $M$ failure probabilities, giving total at most $\delta$; their errors do **not** need to be independent. Because the event covers all candidates, it covers the selected winner too. For validation, condition on the independent training set that produced the candidates. An adaptive search that invents new candidates after seeing validation results is not automatically covered by counting only the final few candidates.

**Worked certificate:** a candidate selected among 100 fixed candidates makes 50 mistakes in 1000 validation examples. Its empirical loss is $0.05$, so the simultaneous bound gives $0.05+\sqrt{\ln(2000)/2000}\approx0.1116$. Treating that winner as the only candidate would instead give $0.0887$, without the justification needed for this selection process. These are upper bounds, not predicted error rates. Values above one are uninformative for zero-one loss and can be capped at one.

Source Theorem 4.3 extends this bookkeeping to a countable set with fixed nonnegative weights $\pi(h)$ summing to at most one:

$$
L(h)\leq\widehat L(h,S)+\sqrt{\frac{\ln(1/[\pi(h)\delta])}{2n}}.
$$

The **prior-weight** box allocates failure budget $\pi(h)\delta$ before inspecting $S$. A larger positive weight gives a smaller penalty; zero weight gives no finite certificate. Uniform $\pi(h)=1/M$ recovers the finite-class result. At $n=1000$, $\delta=0.05$, a rule with $\pi(h)=1/2$ gets margin $0.0429$, versus $0.0728$ for $\pi(h)=1/2000$. You cannot award the winning rule a large weight afterward and retain this theorem’s guarantee.

<details><summary>Predict: does a larger candidate set necessarily make actual test performance worse?</summary>

No. It may contain much better rules. The generic bound becomes less favorable, while approximation and search can improve. Chapter 1 separates design, computation and statistics; a bound is not a universal empirical curve.

</details>

## 5 · Replace a single winner with a distribution

[![A sample-independent prior and observed losses determine a posterior distribution; a randomized predictor samples one classifier per prediction, while a separate majority vote requires its own analysis.](/figures/selection-under-uncertainty/posterior.svg)](/figures/selection-under-uncertainty/posterior.svg)

Read the diagram as functional boxes for §4.8, not a prescribed neural architecture. The prior $\pi$ is fixed independently of the certification sample. Learning selects a distribution $\rho$ using the sample. Definition 4.25’s randomized classifier draws $h\sim\rho$ at each prediction and returns $h(x)$. The distribution can be carried forward, while the sampled classifier can change. A weighted majority vote is a different predictor, analyzed in §4.9.

Source Eq. (4.20), a relaxation of the PAC-Bayes-kl bound in Eq. (4.19), gives this interpretable form for bounded losses and an i.i.d. sample:

$$
\mathbb E_{h\sim\rho}L(h)
\leq \mathbb E_{h\sim\rho}\widehat L(h,S)
+\sqrt{\frac{\mathrm{KL}(\rho\Vert\pi)+\ln(2\sqrt n/\delta)}{2n}}.
$$

With probability at least $1-\delta$, it holds simultaneously for all eligible posteriors $\rho$, so $\rho$ may depend on $S$. For a discrete class,

$$
\mathrm{KL}(\rho\Vert\pi)=\sum_h\rho(h)\ln\frac{\rho(h)}{\pi(h)}.
$$

KL measures a change of distribution, not a distance between parameter vectors. It is infinite if $\rho$ assigns positive mass where $\pi$ has none. The first term is the **posterior-average error** box; the second charges the **change from prior to posterior**. A training algorithm may optimize a bound, but the theorem also evaluates other posteriors. “Posterior” here does not require a Bayesian likelihood update.

**Toy representation close-up:** two classifiers have prior $(1/2,1/2)$. Keeping posterior $(1/2,1/2)$ costs zero KL. Selecting $(1,0)$ costs $\ln 2$. If both have empirical error $0.1$, either posterior has empirical average $0.1$, but hard selection increases the penalty. At $n=1000$, $\delta=0.05$, the displayed bound is approximately $0.1598$ for the unchanged distribution and $0.1626$ for hard selection. This illustrates the price of unnecessary commitment, not a guarantee that mixing always helps. A sufficiently better classifier can justify extra KL.

For a continuous hypothesis space, concentrating on one point can have infinite KL against a continuous prior. The discrete two-rule example must not be used to assert a finite deterministic-neural-network certificate. The square-root relaxation can also be looser than inverting the original binary-kl inequality.

## 6 · When selection changes what you get to observe

[![Full information reveals every action loss after a choice; bandit feedback reveals only the chosen action loss. A round cycles from probabilities through action and observation to updated cumulative losses.](/figures/selection-under-uncertainty/online.svg)](/figures/selection-under-uncertainty/online.svg)

Chapter 7 moves from a fixed training sample to repeated interaction. Read the first two rows as alternative feedback protocols; read the final row as the state carried from round $t$ to $t+1$. In **full information**, all action losses become visible after the choice. Under **bandit feedback**, only the selected action’s loss is observed. Separately, the environment may be stochastic or adversarial. The basic adversarial game in §7.2 fixes its loss matrix before play—an oblivious adversary. Planning over changing states is discussed as a further dimension, but left outside the book’s detailed treatment.

For losses $\ell_{t,a}\in[0,1]$, action $A_t$ at round $t$, $K$ available actions and horizon $T$, the comparison in §7.4 is

$$
R_T=\sum_{t=1}^T\ell_{t,A_t}-\min_{a\in\{1,\ldots,K\}}\sum_{t=1}^T\ell_{t,a}.
$$

Regret compares your accumulated loss with the best **single fixed action in hindsight**, not a clairvoyant action that switches every round. This metric operates at the end of the interaction trace, and differs from i.i.d. test error.

For full information, a constant-learning-rate specialization of source Algorithm 4 (Hedge) maintains cumulative losses $C_{t-1,a}=\sum_{s<t}\ell_{s,a}$ and selects using

$$
p_t(a)=\frac{\exp(-\eta C_{t-1,a})}{\sum_{b=1}^K\exp(-\eta C_{t-1,b})},\qquad A_t\sim p_t.
$$

$\eta>0$ is a learning rate; $C$ is teaching notation for the book’s cumulative loss $L_{t-1}(a)$. Lower accumulated loss gets more probability. After observing the full loss vector, add it to $C$ and repeat. This is the **state update** box, not a final frozen-model inference path. A large $\eta$ reacts strongly; it can also overreact to recent differences. Theorem 7.3 gives $\mathbb E[R_T]\leq\ln K/\eta+\eta T/2$, minimized at $\eta=\sqrt{2\ln K/T}$ for known $T$ and $K>1$. This expected guarantee is not a per-run promise.

[![A three-round Hedge example starts uniformly, shifts to probability two-thirds for A after losses zero and one, and returns to uniform after reversed losses.](/figures/selection-under-uncertainty/trace.svg)](/figures/selection-under-uncertainty/trace.svg)

This is an exact toy trace with $K=2$ and $\eta=\ln2$, chosen for readable arithmetic, not as an optimal schedule. Round losses are $(0,1),(1,0),(0,1)$. Start at $C_0=(0,0)$ and $p_1=(1/2,1/2)$. Then $C_1=(0,1)$ yields $p_2=(2/3,1/3)$; after round two, $C_2=(1,1)$ yields $p_3=(1/2,1/2)$. Expected accumulated loss is $1/2+2/3+1/2=5/3$. Action A’s total loss is one and B’s is two, so expected regret is $2/3$. Actual sampled trajectories can differ; no particular random draw is depicted.

Under bandit feedback you cannot make the full-vector update. Section 7.5 uses an importance-weighted estimate:

$$
\widetilde\ell_{t,a}=\frac{\mathbf1\{A_t=a\}\ell_{t,a}}{p_t(a)}.
$$

Here $\mathbf1\{A_t=a\}$ is one when the action is selected and zero otherwise. The source’s separate indicator definition on printed page 101 mistakenly repeats the importance-weighted fraction in its selected branch; we use the ordinary indicator consistent with the surrounding estimator and Algorithm 5.

For positive $p_t(a)$ and a loss fixed before the action draw, its conditional expectation equals $\ell_{t,a}$: probability $p_t(a)$ cancels the denominator. If $p_t(a)=0.2$ and the selected loss is $0.6$, the estimate is $3$ when selected and zero otherwise; its mean is $0.6$. Zero is a bookkeeping estimate for an unobserved arm, not evidence that its true loss was zero. Rare actions create large estimates and high variance. EXP3 substitutes these estimates into exponential weighting; the full-information proof cannot simply assume these estimates stay in $[0,1]$.

## 7 · What to read next, and what the evidence establishes

This overview teaches theoretical guarantees and algorithms. It does not present the toy values as a dataset benchmark or claim empirical superiority of one learner. A useful proposed experiment would fix train/validation/test splits, vary a predeclared candidate count, and repeat the entire selection procedure across independent samples. Report both winning validation error and untouched test error, with variation across repeats. That tests selection optimism; it does not prove a distribution-free theorem.

For an online comparison, declare the loss sequence, feedback protocol, comparator and randomness. Comparing Hedge with EXP3 without disclosing their different observations confounds algorithm choice with information access. Chapter 6 also warns that more data alone is no cure for unrestricted selection, that i.i.d. assumptions can fail, and that prediction does not establish causation.

| Reading goal | Source route in this supplied version | Connection to this site |
| --- | --- | --- |
| Build and evaluate a predictor | §§2.1–2.3, pp. 3–8; Algorithm 1 | Separate fitting, validation and final evaluation before interpreting any paper’s metric. |
| Derive the confidence margin | §3.3, pp. 15–18; §§4.2–4.4, pp. 39–42; Eqs. (4.1), (4.4), Theorem 4.3 | [Soft inductive biases](/papers/not-so-mysterious) connects selection preferences to generalization. |
| Go beyond counting rules | §4.5, pp. 43–50; §4.8, pp. 54–61; Eqs. (4.19)–(4.20) | [Nonvacuous bounds](/papers/t1-paper-2) turns a distribution of networks into a certificate. |
| Distinguish randomized prediction from voting | §4.9, pp. 61–66 | Error correlation matters for an ensemble; average individual risk is not vote risk. |
| Learn regression and identify assumptions | Chapter 5, pp. 81–84; Chapter 6, pp. 85–86 | Squared-loss fitting needs loss-appropriate analysis. |
| Study sequential decisions | §§7.1–7.5, pp. 88–104; Algorithms 3–5 | UCB uses stochastic reward optimism; Hedge and EXP3 address adversarial loss sequences with different feedback. |

Page references are printed page numbers, not PDF viewer indices. The preface supplies the reading dependencies. This roadmap leaves VC proofs, refined concentration inequalities, ensemble bounds, recursive PAC-Bayes and most exercises for direct reading; it does not imply that their details were covered here. Open the [supplied textbook](/papers/selection-under-uncertainty.pdf) for those arguments.

## The idea to carry forward

A good observed score answers only part of the question. Ask **what was selected, what information drove that selection, what uncertainty budget pays for it, and what feedback will exist at the next decision**. These questions connect a nearest-neighbor vote, a PAC-Bayes certificate and an online action without pretending they share the same guarantee.
