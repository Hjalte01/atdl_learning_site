/** Resolve site-local URLs under Astro's configured deployment base. */
export function withBase(path: string): string {
  if (!path.startsWith("/") || path.startsWith("//")) return path;
  return `${import.meta.env.BASE_URL.replace(/\/$/, "")}${path}`;
}
