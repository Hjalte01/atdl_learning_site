import { getCollection } from 'astro:content';
import { z } from 'astro/zod';
import catalog from '../../content/materials/catalog.json';
const material = z.object({
  id: z.string(), sourcePath: z.string(), title: z.string(),
  topic: z.number().int().min(1).max(6).nullable(), order: z.number().int().positive().nullable(),
  kind: z.enum(['paper', 'overview', 'slides', 'extra']),
  status: z.enum(['not-started', 'ready']),
  guide: z.string().startsWith('/').nullable(), pdf: z.string().startsWith('/'), sha256: z.string().length(64),
});
export const materialCatalog = z.object({ inventoryComplete: z.boolean(), materials: z.array(material) }).parse(catalog);
export const materialKinds = { overview: 'Overviews', slides: 'Slides', paper: 'Papers', extra: 'Extra material' };

// Resolve progress from the actual guide, rather than stale catalog metadata.
export async function getMaterialsWithProgress() {
  const papers = await getCollection('papers');
  return materialCatalog.materials.map(material => {
    const guide = papers.find(p => material.guide === `/papers/${p.id.replace(/\/index$/, '')}`);
    return { ...material, ready: Boolean(guide && guide.data.status === 'ready') };
  });
}
