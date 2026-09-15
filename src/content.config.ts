import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

const topics = defineCollection({
  loader: glob({ pattern: "*/index.md", base: "./content/topics" }),
  schema: z.object({
    title: z.string(),
    shortTitle: z.string(),
    description: z.string(),
    order: z.number().int().positive(),
    status: z.enum(["not-started", "exploring", "active", "paused"]),
    tags: z.array(z.string()).default([]),
    locale: z.enum(["en", "da"]).default("en"),
  }),
});

const sources = defineCollection({
  loader: glob({ pattern: "*/sources.md", base: "./content/topics" }),
  schema: z.object({
    title: z.string(),
    topic: z.string(),
    locale: z.enum(["en", "da"]).default("en"),
  }),
});

const visualItem = z.object({
  id: z.string().regex(/^[a-z0-9]+(?:-[a-z0-9]+)*$/),
  topic: z.string().regex(/^[a-z0-9]+(?:-[a-z0-9]+)*$/),
  title: z.string(),
  type: z.enum(["diagram", "plot", "workflow", "architecture", "concept-figure"]),
  caption: z.string(),
  explanation: z.string(),
  image: z.string().startsWith("/"),
  alt: z.string().min(1),
  tags: z.array(z.string()).default([]),
  relatedSections: z.array(z.string()).default([]),
  sourceNote: z.string().optional(),
});

const studyCard = z.object({
  id: z.string().regex(/^[a-z0-9]+(?:-[a-z0-9]+)*$/),
  topic: z.string().regex(/^[a-z0-9]+(?:-[a-z0-9]+)*$/),
  title: z.string(),
  question: z.string(),
  image: z.string().startsWith("/").optional(),
  imageAlt: z.string().min(1).optional(),
  example: z.string().optional(),
  choices: z.array(z.string()).length(4),
  correctChoiceIndex: z.number().int().min(0).max(3),
  explanation: z.string(),
  difficulty: z.enum(["easy", "medium", "hard"]).optional(),
  tags: z.array(z.string()).default([]),
  relatedTopicUrl: z.string().startsWith("/").optional(),
});

const visuals = defineCollection({
  loader: glob({ pattern: "*/visuals.json", base: "./content/topics" }),
  schema: z.array(visualItem),
});

const cards = defineCollection({
  loader: glob({ pattern: "*/cards.json", base: "./content/topics" }),
  schema: z.array(studyCard),
});

const papers = defineCollection({
  loader: glob({ pattern: "*/index.md", base: "./content/papers" }),
  schema: z.object({
    title: z.string(), shortTitle: z.string(), topic: z.string(), order: z.number().int().positive(),
    description: z.string(), status: z.enum(["ready", "queued"]),
    source: z.string().optional(), pdf: z.string().optional(),
  }),
});
export const collections = { topics, sources, visuals, cards, papers };
