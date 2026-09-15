/** Prefix root-relative Markdown links and images for project hosting. */
export default function remarkBase({ base }) {
  return (tree) => {
    function visit(node) {
      if (typeof node.url === "string" && node.url.startsWith("/") && !node.url.startsWith("//")) {
        node.url = base.replace(/\/$/, "") + node.url;
      }
      node.children?.forEach(visit);
    }
    visit(tree);
  };
}
