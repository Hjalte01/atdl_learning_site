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
