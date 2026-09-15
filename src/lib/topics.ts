import { getCollection, type CollectionEntry } from "astro:content";

export type Topic = CollectionEntry<"topics">;

export function topicSlug(topic: Topic): string {
  return topic.id.replace(/\/index$/, "");
}

export async function getTopics(): Promise<Topic[]> {
  const topics = await getCollection("topics");
  return topics.sort((a, b) => a.data.order - b.data.order);
}

export function topicHref(topic: Topic): string {
  return `/topics/${topicSlug(topic)}`;
}
