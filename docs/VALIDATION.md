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
