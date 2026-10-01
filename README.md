# ATDL Learning

A static Astro learning site modeled on `study/thesis/learning_site`.

## Location

The repository lives at `/home/hjalte/documents/atdl_learning_site`. The source study vault is separate.

## Run

Install Git LFS before cloning. For an existing checkout, run `git lfs install`
and `git lfs pull` before building; the large textbook PDF uses LFS. Normal
`git push` uploads its LFS content automatically. The Pages workflow downloads
LFS files before building.

```sh
cd '/home/hjalte/documents/atdl_learning_site'
npm ci
npm run dev -- --host 127.0.0.1 --port 4322
```

Open http://localhost:4322. For production verification, run `npm run build`; `npm run preview -- --host 127.0.0.1 --port 4322` serves the static build. Node 22.12 or newer is required by this Astro version.

## Scope

Topic 3 paper 2, *V-JEPA 2.1*, has a guide at `/papers/vjepa-21`, eight reproducible SVGs and eight study cards based on the supplied 37-page June 2026 arXiv v3. It explains weighted context prediction, four-layer supervision, native image/video tokenization, EMA and task-specific inference. Sequential ablations and scaled-model results stay separate; robot planning changes and internal source discrepancies are explicit. Exactly one guide is completed; deployment remains with the queue.

Topic 3 paper 1, *DINOv3*, now has a guide at `/papers/dinov3`, eight reproducible SVGs and eight study cards based on the supplied August 13, 2025 arXiv v1. It connects student/EMA/Gram teacher roles, global and masked-patch objectives, relational anchoring, resolution adaptation and downstream probes. Final-result and refinement-ablation settings are kept separate; OCR, fairness and compute limitations are explicit. This run completes one guide; deployment remains with the queue.

Topic 4 paper 6, *CSCS*, has a guide at `/papers/cscs`, eight reproducible SVGs and eight study cards based on the supplied 20-page June 18, 2026 arXiv v1. It connects pool-specific SSL signals, Spearman DCR, budget pacing, clustered selection and separate nnU-Net training. DIANE budget-label discrepancies, small-pool KNN gaps and ablation-table inconsistencies are explicit. All six supplied Topic 4 papers now have guides. This run completes one guide; deployment remains with the queue.

Topic 4 paper 5, *SUGFW+*, has a guide at `/papers/sugfw-plus`, eight reproducible SVGs and eight study cards based on the supplied 17 August 2026 arXiv v1 PDF. It connects cold-start SAM region entropy, uncertainty-weighted feature pooling, cluster/uncertainty coverage and uncertainty-prompted adaptation with a learned inference prompt. Table exceptions and source ambiguities are explicit. This automatic request completes one guide; deployment remains with the queue.

Topic 4 paper 4, *nnActive: A Framework for Evaluation of Active Learning in 3D Biomedical Segmentation*, now has a guide at `/papers/nnactive`, nine reproducible SVGs and eight study cards based on the supplied 62-page TMLR 08/2025 paper. It connects partial-label 3D nnU-Net training, ensemble uncertainty, non-overlapping patch queries and foreground-efficiency metrics, and records the printed Algorithm 1 budget-step omission. The newer repository follow-up is kept separate. This automatic request completes one guide; deployment remains with the queue.

Topic 4 paper 3, *Active Prompt Learning with Vision-Language Model Priors*, now has a guide at `/papers/active-prompt`, eight reproducible SVGs and eight study cards based on the supplied 24-page TMLR 10/2025 paper. It separates class-guided cluster coverage, selective oracle queries, pseudo targets and reset-prompt training, and records the later-round cluster-count wording and empty-class threshold gaps. This automatic request completes one guide; deployment remains with the queue.

Topic 4 paper 2, *Active Learning for Vision-Language Models* (CEC), now has a guide at `/papers/cec`, eight reproducible SVGs and eight study cards based on the supplied 11-page WACV 2025 paper. It connects pool-based class correction, additive neighbor uncertainty and weighted clustering to context-only CoOp training, and flags budget-label ambiguity and missing supplementary details. This automatic request completes one guide; deployment remains with the queue.

Topic 4 paper 1, *LabelBench*, now has a source-grounded guide at `/papers/labelbench`, seven reproducible SVGs and eight study cards based on the supplied 43-page March 2024 PDF. It separates proxy acquisition from final full fine-tuning, test from pool accuracy, and records the compute-cost table inconsistency. The live vault sync on October 1, 2026 imported and verified all 48 currently enumerated PDFs, including Topics 3 and 4. All six supplied Topic 4 papers now have guides; the other newly available paper readings remain unfinished. Topic 4’s descriptive grouping is provisional.

Topic 2 extra reading *Flow Matching* now has a focused guide at `/papers/t2-flow-matching`, based on Holderrieth and Erives’s supplied 84-page MIT 6.S184 notes (2026; PDF export March 18). Seven reproducible SVG diagrams and eight study cards explain conditional-to-marginal regression, sampling, score/SDE connections, CFG and latent architectures. It is a selective guide to the notes, not the original Flow Matching research paper or a benchmark reproduction.

Six topic routes; the twelve available papers for Topics 1 and 2 are catalogued. All six Topic 2 papers—FLUX.1 Kontext, DiffAtlas, REPA, Mean Flows, DiME and Qwen-Image—have full visual learning guides, equations and recall cards; Flux and DiffAtlas also include interactive illustrations. Qwen-Image adds ten reproducible SVG diagrams and eight cards based on the supplied 46-page arXiv v1 report, including dual conditioning, flow time direction, and text-rendering evidence. DiME adds eight reproducible SVG diagrams and eight cards based on the supplied 19-page ACCV 2022 paper, with explicit gradient-approximation and evaluation caveats. Mean Flows adds seven reproducible SVG diagrams, eight cards, and an architecture-to-math walkthrough based on the supplied 23-page NeurIPS 2025 conference PDF. REPA adds seven reproducible SVG diagrams and eight cards, based on the supplied 43-page ICLR 2025 PDF. DiffAtlas includes seven original diagrams and eight multiple-choice cards. Topics 3 and 4 now have supplied paper readings in the library; Additional foundation-model lecture slides are imported; Topics 5–6 have no assigned paper readings. Topic 1 now includes the Information Bottleneck guide (paper 1), with seven reproducible SVG diagrams and eight cards based on the supplied April 2017 v3 PDF; it distinguishes SGD training from IB analysis and exact from quantized information. Topic 1's grouping label remains provisional. Topic 1 also includes Nonvacuous Generalization Bounds (paper 2), with eight reproducible diagrams and eight cards based on the supplied October 2017 v2 PDF; it explains prior selection, posterior optimization, and Monte Carlo certification, and records source notation inconsistencies. Topic 1 paper 3, A Bayesian Perspective on SGD, adds seven reproducible diagrams and eight cards based on the supplied ICLR 2018 PDF; it connects evidence, coordinate invariance, finite-batch noise and momentum scaling. Topic 1 paper 4, Deep Double Descent, adds eight reproducible teaching diagrams, the supplied journal Figure 3 and eight cards, connecting effective model complexity to width, time and sample-size effects. It follows the December 2021 journal update and records source metric and budget inconsistencies. Topic 1 paper 5, Understanding Grokking, adds seven reproducible teaching diagrams, source Figure 4 and eight cards based on the supplied 13-page NeurIPS 2022 paper. It explains embedding transfer, constraint rank and decoder competition, with explicit normalization and missing-appendix caveats. Topic 1 paper 6, Binarized Networks and Compression, adds seven reproducible diagrams (including Table 3 correlation intervals) and eight cards based on the supplied May 2026 paper. It distinguishes diagnostic BDM from the training objective and records missing supplementary and implementation details. All twelve imported Topic 1–2 papers now have guides. The Topic 2 Deep Generative Models reading overview also has a roadmap guide with six reproducible diagrams and eight cards, separating source claims from teaching equations and connecting all six Topic 2 papers.

The general *Principles and Practice of Deep Representation Learning* textbook (Version 2.0, March 1, 2026) now has a reading roadmap at `/papers/representation-overview`, with seven reproducible diagrams, worked PCA/denoising/coding-rate examples, selected Table 8.5 evidence and eight cards grouped under Topic 1. It remains unassigned to a numbered topic in the catalog; the guide header and return link support general material. This roadmap covers selected mechanisms and reading routes, not every chapter or experiment.

The Flux explanation uses the supplied arXiv v2 PDF (24 June 2025), copied into `public/papers/flux-kontext.pdf` for local reading. Toy examples are labeled. Results are historical claims from the report, not current product comparisons.

## Add learning material

Follow [GPT.md](GPT.md) for architecture-to-math visual authoring. “Create sites for topic 2-4” selects Topic 2, paper 4.

- Topic metadata and overview: `content/topics/topic-N/index.md`.
- Paper metadata and explanation: `content/papers/<stable-slug>/index.md`.
- Cards: `content/topics/topic-N/cards.json`; `relatedTopicUrl` points to the relevant paper section.
- Shared presentation: `src/`; validation: `src/content.config.ts`.

Add a paper directory with frontmatter `title`, `shortTitle`, `topic`, `order`, `description`, and `status` (`queued` or `ready`); optionally `source` and `pdf`. It is automatically listed in its topic and receives a stable `/papers/<slug>` route. No component changes are needed for ordinary papers. Flux and DiffAtlas include paper-specific interactive components.

Use level-two headings; explain intuition before equations; define symbols; cite paper sections; distinguish reported evidence from critique. Equations use remark-math and KaTeX, with fonts hosted locally. Sources and optional visual collections inherited from the thesis layout remain available.

`/study` provides practice and `/study/print` provides printable cards. Answers are temporary browser state. The site contains no generation API, account system, or paid service dependency. Theme preference uses local browser storage.

## Checks

Run `npm run build` after changes. Check topic-to-paper links, mathematical rendering, slider endpoints, quiz feedback, and mobile layout when editing application behavior.

## GitHub Pages

Live site: https://hjalte01.github.io/atdl_learning_site/

Pushes to `main` build and deploy through `.github/workflows/deploy.yml`.
The workflow runs `npm ci` and `npm run build` before publishing `dist/`.
Astro's `base` is `/atdl_learning_site`; use `withBase` from `src/lib/url.ts`
for site-local links and assets. Markdown URLs receive the base automatically.

## Reading library and automatic guide selection

`/materials` provides text, topic, and reading-type filters, with core overviews/slides first, followed by papers and extras. Every imported PDF is readable even before its explanation is started. Topic pages also list their PDFs. The original snapshot contained 27 PDFs imported on 16 September 2026. It is not the current vault inventory: Topic 3 and other additions have since been released. The latest sync status is recorded in the catalog and shown in the library. Slide topics and the Topic 2 reading overview were manually classified after reading their contents; general overview texts remain unassigned.

Every automatic guide selection now refreshes directly from the live vault:

```sh
python3 scripts/materials.py next
```

The source path is configured in `content/materials/source.json`, currently `/srv/obsidian-webdav/study/ATDL`. Override it with `ATDL_VAULT` or `--vault /path/to/study/ATDL`. To refresh without selecting, run `python3 scripts/materials.py sync`; the existing `import /path/to/study/ATDL` command remains supported. Review filename-inferred titles, topics and categories after discovery.

The dependency-free importer stores portable assets and SHA-256 hashes, preserves manual metadata and existing guides, and retains records when sources disappear. Changed PDF bytes receive a new asset, preserving the version used by an existing guide, and flag that guide for review. Once the guide and its PDF link match the new source, clear its catalog `reviewRequired` flag. A renamed source is retained separately until its identity is reviewed.

An unreadable directory/PDF, missing vault, or empty scan returns exit code 2, marks inventory incomplete, and blocks automatic selection; readable files can still be imported. Earlier access failures are recorded in `docs/VALIDATION.md`. The October 1, 2026 live scan succeeded with 48 sources and no errors. Source-version review flags on other guides remain pending; inventory availability does not imply guide readiness.

`python3 scripts/materials.py next --offline` explicitly inspects the cached snapshot without changing it. Its `freshness: cached` result never proves the live vault is up to date. Builds deliberately use repository-owned assets without reading the vault, so CI and Nix remain reproducible. This is sync on each authoring selection, not a background watcher; vault changes reach the static website through the normal review/build/deployment flow.

“Goal create site” selects one unfinished guide: papers from the latest supplied topic first, ascending paper order; then overviews, slides and extras. See AGENTS.md and GPT.md. No guide is selected from an incomplete scan. Test discovery and failure handling with `python3 scripts/test-materials.py`.

The registered VPS deployment serves `/atdl/`; deployment is handled by the queue after review. Editing this repository does not update the running site.

The general *Understanding Deep Learning* textbook now has a selective reading roadmap at `/papers/understanding-deep-learning`, following the supplied November 21, 2024 copy. Seven reproducible SVG diagrams, a worked classifier and gradient trace, attention and diffusion calculations, and eight Topic 1 foundation cards connect the textbook to course readings. This is a roadmap, not an exhaustive textbook review; the catalog classification remains general.

The general *Deep Learning is Not So Mysterious or Different* overview now has a guide at `/papers/not-so-mysterious`, following Wilson’s supplied July 10, 2025 arXiv v2. Six reproducible teaching diagrams, source Figure 5, worked polynomial and risk-bound examples, and eight Topic 1 foundation cards connect soft inductive biases with generalization. The guide flags the Gaussian-loss wording and singular least-squares inverse in the source; catalog classification remains general.

The general *Machine Learning: The Science of Selection under Uncertainty* overview now has a selective roadmap at `/papers/selection-under-uncertainty`, following Seldin’s supplied arXiv v2 (margin January 29, title page January 30, 2026). Seven reproducible SVGs, worked validation/PAC-Bayes/Hedge examples and eight Topic 1 foundation cards connect statistical selection to online feedback. It remains general material; this is not an exhaustive textbook review.

Topic 2’s *Deep Generative Models · Lecture 2* slide guide is complete at `/papers/t2-generative-lecture-2`. It follows the supplied 35-slide PDF (export metadata September 10, 2026), with seven reproducible SVG diagrams, worked DDPM/SDE/ODE/flow/attention/metric examples and eight Topic 2 cards. It distinguishes the lecture’s two time conventions and clarifies informal SDE notation. No cross-model benchmark results are invented from the survey slides.

The Topic 1 **Theories for Deep Learning** lecture (September 1, 2026) now has a slides guide at `/papers/t1-theories-lecture`, with seven reproducible teaching diagrams and eight cards. It connects all six Topic 1 papers, traces training versus diagnosis, and makes the assumptions behind PAC-Bayes, Laplace evidence and interpolation claims explicit.

Topic 1’s **Learning through the lens of Complexity · Lecture 2** now has a guide at `/papers/t1-complexity-lecture`, following the supplied 43-slide September 3, 2026 deck. Seven reproducible SVG diagrams, the original FashionMNIST evidence slide and eight cards explain MDL, continuous KL, CTM/BDM, QuBD and the displayed MoMos loop. It explicitly qualifies the source’s continuous-point KL limit, semicomputability wording and objective-versus-bound shorthand.

The Topic 1 [Mosaic-of-Motifs lecture guide](content/papers/t1-momos-lecture/index.md) follows the supplied 42-page build-expanded deck (title date September 2, 2026). Seven reproducible diagrams, the original result slide and eight cards cover bank/mosaic reconstruction, projected updates, storage arithmetic and evidence limitations. PDF page references avoid ambiguity from repeated slide numbers.

Topic 2 extra reading *Deep Generative Modeling* now has a selective textbook roadmap at `/papers/t2-deep-generative-modeling`, based on Tomczak’s supplied second edition (2024). Seven reproducible SVGs and eight study cards connect model-family objectives to separate training and generation paths. It flags score/noise notation and scaling inconsistencies in the supplied edition; it is not an exhaustive textbook summary or benchmark reproduction.

The separately cataloged Topic 2 copy of *Deep Generative Modeling* now has a complementary Chapter 10 guide at `/papers/t2-neural-compression`. Seven reproducible teaching SVGs, source Figure 10.5 and eight study cards explain neural compression from latent representations to actual messages. The guide corrects the source’s uniform-sequence bit-count expression and distinguishes soft training, hard coding, generation and measured bitstream rates. Both source copies remain preserved; this is a focused chapter guide, not a second exhaustive textbook summary.

Topic 2’s original **Generative Adversarial Nets** extra reading now has a guide at `/papers/t2-gans`, based on the supplied nine-page NIPS 2014 paper. Seven reproducible SVGs and eight cards cover the training/sampling split, implicit distributions, a worked gradient update, non-saturating loss, ideal equilibrium and Table 1 evidence limitations.

Topic 2’s **Denoising Diffusion Probabilistic Models** extra reading now has a guide at `/papers/t2-ddpms`, following the supplied 12-page NeurIPS 2020 copy. Seven reproducible SVGs and eight cards cover image-shaped states, shared U-Net conditioning, separate training and sampling, loss weighting, a worked reverse step and original CIFAR10 evidence. Referenced appendices are absent from the supplied PDF; no appendix-only implementation details are asserted.

Topic 2’s **Auto-Encoding Variational Bayes** extra reading now has a guide at `/papers/t2-vaes`, based on the supplied 14-page arXiv v11 (December 10, 2022). Seven reproducible teaching SVGs, the original Figure 2 page and eight cards explain encoder/decoder roles, prior versus posterior KL, reparameterized gradients, a scalar training trace and separate generation. It distinguishes sampled objectives from guaranteed bounds and preserves the source’s experimental evaluation limits.
