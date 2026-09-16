# ATDL Learning

A static Astro learning site modeled on `study/thesis/learning_site`.

## Location

The repository lives at `/home/hjalte/documents/atdl_learning_site`. The source study vault is separate.

## Run

```sh
cd '/home/hjalte/documents/atdl_learning_site'
npm ci
npm run dev -- --host 127.0.0.1 --port 4322
```

Open http://localhost:4322. For production verification, run `npm run build`; `npm run preview -- --host 127.0.0.1 --port 4322` serves the static build. Node 22.12 or newer is required by this Astro version.

## Scope

Six topic routes; the twelve available papers for Topics 1 and 2 are catalogued. All six Topic 2 papers—FLUX.1 Kontext, DiffAtlas, REPA, Mean Flows, DiME and Qwen-Image—have full visual learning guides, equations and recall cards; Flux and DiffAtlas also include interactive illustrations. Qwen-Image adds ten reproducible SVG diagrams and eight cards based on the supplied 46-page arXiv v1 report, including dual conditioning, flow time direction, and text-rendering evidence. DiME adds eight reproducible SVG diagrams and eight cards based on the supplied 19-page ACCV 2022 paper, with explicit gradient-approximation and evaluation caveats. Mean Flows adds seven reproducible SVG diagrams, eight cards, and an architecture-to-math walkthrough based on the supplied 23-page NeurIPS 2025 conference PDF. REPA adds seven reproducible SVG diagrams and eight cards, based on the supplied 43-page ICLR 2025 PDF. DiffAtlas includes seven original diagrams and eight multiple-choice cards. Topics 3–6 await their actual titles and reading lists. Topic 1 now includes the Information Bottleneck guide (paper 1), with seven reproducible SVG diagrams and eight cards based on the supplied April 2017 v3 PDF; it distinguishes SGD training from IB analysis and exact from quantized information. Topic 1's grouping label remains provisional. Its other five papers are explicitly queued.

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

`/materials` provides text, topic, and reading-type filters, with core overviews/slides first, followed by papers and extras. Every imported PDF is readable even before its explanation is started. Topic pages also list their PDFs. The inventory contains all 27 PDFs present in the supplied ATDL vault as of 16 September 2026, with source and portable-copy hashes verified. Slide topics and the Topic 2 reading overview were manually classified after reading their contents. The two copies of Deep Generative Modeling are retained as separate source entries. General overview texts remain unassigned to a topic. No Topic 3–6 readings were present.

Import all vault PDFs, then review filename-inferred titles, topics and categories:

```sh
python3 scripts/materials.py import /path/to/study/ATDL
python3 scripts/materials.py next
```

The dependency-free importer stores portable assets and SHA-256 hashes in `content/materials/catalog.json`. It excludes embedded site copies, preserves prior records and manual metadata, and does not delete existing guides. Unrecognized topic names are listed as general/unassigned for review. Subsequent imports must be run when vault material changes; builds do not require vault access.

“Goal create site” now selects one unfinished guide: papers from the latest supplied topic first, ascending paper order; then overviews, slides and extras. See AGENTS.md and GPT.md. Missing source PDFs must be supplied before authoring.

The registered VPS deployment serves `/atdl/`; deployment is handled by the queue after review. Editing this repository does not update the running site.
