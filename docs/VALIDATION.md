# Validation — 2026-09-15

## DiffAtlas guide addition

- Final `npm run build`: passed, 23 routes, zero errors/warnings/hints.
- Seven local SVG diagrams loaded and each links to its full-size asset. Training and inference diagrams visually inspected at full size; desktop/mobile page captures and dark-mode lab reviewed.
- Chromium checks: no JavaScript errors; no horizontal overflow at 390px; equations rendered without KaTeX errors.
- Lab checked at ᾱ = 0, 0.50, 0.64, 1.00: toy pixel values 0.2000, 0.7071, 0.7600, 0.8000.
- All DiffAtlas local links, table-of-contents anchors, and eight card section anchors resolve. Correct and incorrect quiz feedback verified.
- Printable deck now contains 18 cards (10 Flux + 8 DiffAtlas). Topic 2 links to DiffAtlas.
- Local DiffAtlas PDF SHA-256 matches the supplied T2_4_DiffAtlas.pdf: `96b623485471742c79c730fb9c8a66a72b014fdae50f6169dac3574417c8947b`. Content and selected results checked against that arXiv v1 PDF.
- GPT.md and AGENTS.md document paper shorthand, source fidelity, visual authoring, architecture/math mapping, and verification.
- Scope: the requested first installment, Topic 2 paper 4, is complete. Other queued papers remain for future requests. No deployment performed. The lab demonstrates Eq. 5 on synthetic data, not a trained segmentation model.

## Earlier Flux validation


- `npm run build`: passed; 23 static pages, zero Astro diagnostics.
- `npm audit --omit=dev`: zero vulnerabilities.
- Chromium desktop (1440px) and mobile (390px) checks passed with no JavaScript errors.
- Verified six topic entries and six Topic 2 paper entries.
- All 23 generated pages and 263 internal link occurrences checked; linked HTML anchors exist.
- Four display equations rendered with KaTeX and no math errors.
- Scalar slider checked at both endpoints and midpoint.
- Correct and incorrect quiz feedback, next-card navigation, recall disclosure, and theme switching checked.
- Ten practice cards and ten printable cards; answer-key toggle checked.
- No horizontal page overflow on the mobile Flux page. Desktop and mobile screenshots visually reviewed.
- Local Flux PDF checksum matches the original course PDF.
- Resolved project directory is outside the study Obsidian vault.

Scope at the time of the earlier checks: only Flux had a completed learning guide. The six-topic structure and twelve known paper records are present; other explanations and Topics 3–6 course material are intentionally pending.


## Reading library and next-guide workflow — 2026-09-16

- `npm run build`: passed, 24 pages, zero errors/warnings/hints. No project dependencies or lockfiles changed; no Nix dependency hash update needed.
- `python3 scripts/test-materials.py`: passed isolated PDF hash/copy checks, nested topic numbering, repeated imports, manual metadata preservation, paper-before-overview priority, newest-topic selection, and linked ready-guide exclusion.
- Chromium at 1440px and 390px: library search, topic filter, core-material filter and empty results passed; no horizontal overflow or browser errors; theme toggle checked. All displayed PDF and guide links returned HTTP 200. Mobile dark-mode screenshot visually inspected.
- Running local preview verified HTTP 200 at `http://127.0.0.1:4322/atdl_learning_site/materials`. No deployment, queue restart, push, or nxb performed.
- Incomplete requirement: the study/ATDL vault is unavailable to this session. The old `/home/hjalte/Documents/study/ATDL` path does not exist; `/srv/obsidian-webdav` denies reads, and sudo is unavailable under the no-new-privileges constraint. Only the two existing PDFs could be included. The catalog and UI explicitly mark the inventory incomplete. Full-vault import and verification remain pending a readable source path; the tracked goal is not complete.
- Assumed priority: all papers first, descending topic and ascending reading order; then overviews, slides, and extras with the same topic ordering. Selection against the current incomplete inventory yields Topic 2, paper 1 (REPA). No new explanation was requested for this setup task.

## REPA guide and recovered vault inventory — 2026-09-16

- Completed one automatically selected guide: Topic 2 paper 1, REPA, at `/papers/t2-paper-1`. Source is the supplied 43-page ICLR 2025 conference PDF (creation metadata 25 March 2025; no arXiv revision inferred). Read method, experiments and appendices; visually compared the source results table. Local `/papers/repa.pdf` matches the supplied file by SHA-256.
- Added seven original SVG teaching diagrams with reproducible `scripts/repa-figures.py`, architecture-linked equations, a numerical loss example, prediction disclosures, evidence and limitations. No trained model or custom interactive lab is presented. Training iteration ratios are distinguished from wall-clock and inference speed; guided and unguided results remain separate.
- Appended eight REPA cards. Verified all 18 existing cards are semantically unchanged and all 26 cards appear in printable material. Chromium exercised both correct and incorrect feedback for each new card on separate page loads (answers intentionally lock after submission).
- Final `npm run build`: passed, 24 routes, zero errors/warnings/hints. `python3 scripts/test-materials.py`: passed. `git diff --check`: passed. No dependencies, lockfiles or Nix hashes changed.
- Chromium: guide and every linked local resource returned HTTP 200; all new heading/card anchors resolve; seven images loaded; no KaTeX errors or page JavaScript errors. No horizontal page overflow at 390, 768 or 1440 pixels. Desktop and mobile dark-mode screenshots visually inspected, along with all seven enlarged diagrams and math. Corrected an inference arrow that crossed its label, rebuilt, and visually rechecked. SVG text bounds checked. Figures link to full-size SVGs for enlargement.
- The vault at `/srv/obsidian-webdav/study/ATDL` is now readable, resolving the earlier missing-inventory limitation. Imported all 27 present PDFs, verified every source-relative path and both source/copy SHA-256 hashes. Reviewed paper identities, assigned the four slide decks to their topics, and classified the generative-model reading overview as an overview. General overview books remain unassigned; topic textbooks remain extra reading. Separate source copies of the same textbook are preserved. No Topic 3–6 readings are present. All 27 PDF URLs returned HTTP 200.
- Inventory is complete for this vault snapshot; only REPA was newly authored. Both its paper and catalog status are ready. `python3 scripts/materials.py next` now selects Topic 2 paper 2, Mean Flows.
- Running static preview: `http://127.0.0.1:4322/atdl_learning_site/papers/t2-paper-1`, verified HTTP 200. This is a host-local preview, not deployment. Production at the registered `/atdl/` route awaits queue review/commit/deployment. No Git push, deployment, queue restart or nxb performed.
- Inventory import adds large repository-owned PDF assets, including a 137,568,865-byte textbook. The original bytes are retained for source fidelity; this is material repository growth for review and a consideration for the separate GitHub publishing workflow.

## Mean Flows guide — 2026-09-16

- Completed one automatically selected guide: Topic 2 paper 2, Mean Flows, at `/papers/t2-paper-2`. Followed the supplied 23-page NeurIPS 2025 conference PDF (creation metadata 14 February 2026; no publication date/arXiv revision inferred). Read method, algorithms, experiments, and technical appendices; visually checked source Table 2. The guide notes the source's SiT prose/table discrepancy and uses Table 2's values.
- Added seven reproducible SVG teaching diagrams (`scripts/meanflows-figures.py`), architecture-linked equations, a scalar JVP/target/loss example, prediction disclosures, evidence and limitations. No trained model or custom control lab is presented. Distinguishes marginal trajectories from straight training interpolants, exact identity from the stochastic training target, one step from one NFE, and flow-model training from the pretrained VAE.
- Appended eight cards; verified all 26 prior cards are semantically unchanged. All 34 cards appear in printable material. Chromium exercised correct and incorrect feedback for every new card and resolved each card's section anchor.
- Final `npm run build`: passed, 24 routes, zero errors/warnings/hints. `python3 scripts/test-materials.py`: passed. `git diff --check`: passed. No dependencies or lockfiles changed; no Nix dependency hash changes required.
- Chromium: guide and all linked local resources HTTP 200; all heading/TOC anchors resolve; seven images load; no KaTeX errors or JavaScript errors; no horizontal page overflow at 390, 768, or 1440 pixels. Enlarged diagrams visually inspected and crowded labels corrected. Desktop, mobile dark mode, and math screenshots reviewed. Figures link directly to full-size SVGs.
- Local `/papers/meanflows.pdf` SHA-256 matches the vault source and catalog: `c0b560a099c17966bbb0bb293f6c8d295f4a5a5f47b5698b4f26bd2d0f793caa`. Existing portable material copy retained. Guide and catalog are ready; automatic selection now yields Topic 2 paper 3, DiME. Topic overview and README scope updated.
- Running host-local preview verified reachable (HTTP 200): `http://127.0.0.1:4322/atdl_learning_site/papers/t2-paper-2`. Deployment at the registered `/atdl/` service remains pending queue review/commit/deployment. No push, deployment, queue restart, or nxb performed.

## DiME guide — 2026-09-16

- Completed one automatically selected guide: Topic 2 paper 3, DiME, at `/papers/t2-paper-3`. Read the supplied 19-page ACCV 2022 paper (proceedings pages 858–876), verified its identity against the official CVF conference record, and visually checked Table 1. No exact publication day or arXiv revision is inferred from the supplied PDF.
- Added eight reproducible SVG teaching diagrams (`scripts/dime-figures.py`), architecture-linked equations, a worked scalar guidance/sampling example, prediction disclosures, results, ablations and limitations. Distinguishes fixed weights from input gradients, noisy outer states from clean inner rollouts, and the approximate gradient transfer from an exact rollout Jacobian. Explicitly notes the paper's inconsistent clean-image subscripts and Eq. 11 diversity normalization; does not silently correct or recompute reported results. No trained model or custom interactive lab is presented.
- Appended eight four-choice cards; all 34 prior cards remain semantically unchanged. Chromium exercised correct and incorrect feedback for each new card and checked its section anchor. All 42 cards appear in printable study material.
- Final `npm run build`: passed, 24 pages, zero errors/warnings/hints. `python3 scripts/test-materials.py`: passed. `git diff --check`: passed. No dependency, lockfile or Nix dependency hash changes.
- Chromium: guide and linked local resources returned HTTP 200; all guide/TOC/card anchors resolve; eight images load; no KaTeX errors or JavaScript errors. No horizontal page overflow at 390, 768 or 1440 pixels. Desktop, mobile dark mode, rendered equations and all eight enlarged diagrams visually inspected. SVG text bounds checked; all figures link to full-size assets.
- `/papers/dime.pdf`, the catalog's existing portable copy, and the vault source have identical SHA-256: `c5ebf7da74d2b0ee4a19b876a57372dd8855bcd2da97db6998205a0cb41003fb`. Original portable material copy retained. Guide/catalog marked ready; topic overview and README updated. Automatic selection now yields Topic 2 paper 6, Qwen-Image.
- Changed the shared source-button label from “Paper on arXiv” to “Original paper” so the official conference link is labeled accurately.
- Running host-local static preview verified reachable (HTTP 200): `http://127.0.0.1:4322/atdl_learning_site/papers/t2-paper-3`. This is a preview of the built guide, not production deployment. Registered `/atdl/` deployment remains pending the queue's review/commit/deployment flow. No Git push, deployment, queue restart or nxb performed.
