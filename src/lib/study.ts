import { getCollection, type CollectionEntry } from "astro:content";

export type VisualItem = CollectionEntry<"visuals">["data"][number];
export type StudyCard = CollectionEntry<"cards">["data"][number];

export async function getVisuals(): Promise<VisualItem[]> {
  const entries = await getCollection("visuals");
  return entries.flatMap((entry) => entry.data);
}

export async function getStudyCards(): Promise<StudyCard[]> {
  const entries = await getCollection("cards");
  return entries.flatMap((entry) => entry.data);
}

export async function getVisualsForTopic(topic: string): Promise<VisualItem[]> {
  return (await getVisuals()).filter((visual) => visual.topic === topic);
}

export async function getCardsForTopic(topic: string): Promise<StudyCard[]> {
  return (await getStudyCards()).filter((card) => card.topic === topic);
}
