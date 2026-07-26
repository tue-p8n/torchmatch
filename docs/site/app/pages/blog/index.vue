<script setup lang="ts">
// Local override of the theme's own pages/blog/index.vue: this project's
// content.config.ts defines a single "docs" collection (not the theme's
// conventional "content" one -- see content.config.ts), so the inherited
// page's queryCollection("content") call would always come back empty here.
// Mirrors DocyardBlogCard's own BlogPostMeta shape (not imported -- the
// theme is consumed as a filesystem layer via DOCYARD_THEME_PATH, not an
// npm package, so there's no module specifier to import the type from).
interface BlogPostMeta {
  path: string;
  title: string;
  description?: string;
  date?: string;
  author?: string | { name: string; avatar?: string };
  tags?: string[];
  cover?: string;
}

const route = useRoute();
const appConfig = useAppConfig();

const { data: docs, error } = await useAsyncData("blog-index", () =>
  queryCollection("docs").where("path", "LIKE", "/blog/%").all(),
);

const allPosts = computed<BlogPostMeta[]>(() => {
  return (docs.value ?? [])
    .map((d) => {
      const meta = (d as { meta?: Record<string, unknown> }).meta ?? {};
      return {
        path: d.path,
        title: d.title ?? "",
        description: d.description,
        date: meta.date as string | undefined,
        author: meta.author as BlogPostMeta["author"],
        tags: meta.tags as string[] | undefined,
        cover: meta.cover as string | undefined,
      };
    })
    .sort((a, b) => (b.date ?? "").localeCompare(a.date ?? ""));
});

const activeTag = computed(() => route.query.tag as string | undefined);
const filteredPosts = computed(() =>
  activeTag.value
    ? allPosts.value.filter((p) => p.tags?.includes(activeTag.value!))
    : allPosts.value,
);

const postsPerPage = computed(
  () => appConfig.docyard?.blog?.postsPerPage ?? 10,
);
const currentPage = computed(() => {
  const page = Number(route.query.page);
  return Number.isInteger(page) && page > 0 ? page : 1;
});
const totalPages = computed(() =>
  Math.max(1, Math.ceil(filteredPosts.value.length / postsPerPage.value)),
);
const pagedPosts = computed(() => {
  const start = (currentPage.value - 1) * postsPerPage.value;
  return filteredPosts.value.slice(start, start + postsPerPage.value);
});

useSeoMeta({ title: "Blog" });
</script>

<template>
  <UContainer class="blog-index">
    <h1 class="blog-title">Blog</h1>
    <p v-if="activeTag" class="blog-filter">
      Tagged &ldquo;{{ activeTag }}&rdquo; &middot;
      <NuxtLink to="/blog">Clear filter</NuxtLink>
    </p>

    <p v-if="error" class="blog-empty">Couldn't load posts.</p>
    <p v-else-if="!allPosts.length" class="blog-empty">No posts yet.</p>

    <div v-else class="blog-list">
      <DocyardBlogCard
        v-for="post in pagedPosts"
        :key="post.path"
        v-bind="post"
      />
    </div>

    <nav v-if="totalPages > 1" class="blog-pagination">
      <NuxtLink
        v-for="p in totalPages"
        :key="p"
        :to="{ query: { ...route.query, page: p } }"
        class="blog-page-link"
        :class="{ 'blog-page-active': p === currentPage }"
      >
        {{ p }}
      </NuxtLink>
    </nav>
  </UContainer>
</template>

<style scoped>
.blog-index {
  padding-block: clamp(2rem, 5vw, 4rem);
}
.blog-title {
  font-family: var(--font-display);
  font-weight: 600;
  font-size: clamp(1.8rem, 3.2vw, 2.4rem);
  margin: 0 0 1rem;
  color: var(--ui-text-highlighted);
}
.blog-empty {
  font-size: 0.9rem;
  color: var(--ui-text-muted);
}
.blog-filter {
  font-size: 0.85rem;
  color: var(--ui-text-muted);
  margin-bottom: 1.5rem;
}
.blog-pagination {
  display: flex;
  gap: 0.75rem;
  margin-top: 2rem;
}
.blog-page-link {
  font-family: var(--font-mono);
  font-size: 0.85rem;
  color: var(--ui-text-muted);
  text-decoration: none;
}
.blog-page-active {
  color: var(--ui-primary);
  font-weight: 600;
}
</style>
