# Paper learning-site authoring

For requests to create or expand paper guides (including “create sites for topic 2-4”), read and follow [GPT.md](GPT.md). Topic-paper shorthand means one paper unless the user explicitly requests a range. Preserve existing work and verify against the supplied course paper.

## Automatic next-guide requests

“Goal create site”, “create a site”, or “create the next missing site” without a specific target means complete **one** unfinished learning guide. Read [GPT.md](GPT.md), especially automatic selection. Run `python3 scripts/materials.py next` to inspect the next candidate. `queued` paper metadata is the same as the UI label **Not started**. Never interpret an unspecified request as permission to generate every guide.

Priority: unfinished papers first, highest available topic number first (Topic 5 before 4, 3, 2, 1; Topic 6 first when supplied), then ascending paper number within that topic. After all papers, do overviews, then slides, then extra material; within each category use descending topic and ascending reading order. Explicit user targets override this default. Topics without supplied readings are not candidates. If the selected source is unavailable, report the missing source rather than inventing content or silently choosing a lower-priority guide.

The complete PDF inventory belongs in `content/materials/catalog.json`. Import with `python3 scripts/materials.py import /path/to/study/ATDL`; review inferred classifications against the vault. Preserve existing guides, source PDFs, and metadata. Never claim the whole vault is included until its inventory has actually been imported and verified.
