import { defineConfig } from "astro/config";
import { unified } from "@astrojs/markdown-remark";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import remarkBase from "./scripts/remark-base.mjs";
const base = "/atdl_learning_site";
export default defineConfig({site: "https://hjalte01.github.io", base,output:"static", trailingSlash:"never", markdown:{processor: unified({remarkPlugins:[remarkMath, [remarkBase, { base }]], rehypePlugins:[rehypeKatex]})}});
