<script setup lang="ts">
import type { ContentNavigationItem } from "@nuxt/content";

definePageMeta({ layout: "docs" });

const route = useRoute();
const { toc } = useAppConfig();
const navigation = inject<Ref<ContentNavigationItem[]>>("navigation");

const { data: page } = await useAsyncData(route.path, () =>
  queryCollection("docs").path(route.path).first(),
);
if (!page.value) {
  throw createError({
    statusCode: 404,
    statusMessage: "Page not found",
    fatal: true,
  });
}

const { data: surround } = await useAsyncData(`${route.path}-surround`, () => {
  return queryCollectionItemSurroundings("docs", route.path, {
    fields: ["description"],
  });
});

const title = page.value.seo?.title || page.value.title;
const description = page.value.seo?.description || page.value.description;

useSeoMeta({
  title,
  ogTitle: title,
  description,
  ogDescription: description,
});

// First path segment as the section eyebrow: /tutorials/basic → "Tutorials",
// /reference/ops → "Reference". Top-level pages get an empty section.
const section = computed(() => {
  const segments = route.path.split("/").filter(Boolean);
  if (segments.length < 2) return "";
  const first = segments[0]!;
  return first.charAt(0).toUpperCase() + first.slice(1);
});

const links = computed(() => {
  const out = [];
  if (toc?.bottom?.edit) {
    out.push({
      icon: "i-lucide-external-link",
      label: "Edit this page",
      to: `${toc.bottom.edit}/${page?.value?.stem}.${page?.value?.extension}`,
      target: "_blank",
    });
  }

  return [...out, ...(toc?.bottom?.links || [])].filter(Boolean);
});
</script>

<template>
  <UPage v-if="page">
    <DocsPageHeader
      :title="page.title!"
      :description="page.description"
      :section="section"
    >
      <template v-if="page.links?.length" #links>
        <UButton
          v-for="(link, index) in page.links"
          :key="index"
          v-bind="link"
        />
      </template>
    </DocsPageHeader>

    <UPageBody class="docs-prose">
      <ApiBanner :api="page.api" />

      <ContentRenderer v-if="page" :value="page" />

      <USeparator v-if="surround?.length" />

      <UContentSurround :surround="surround" />
    </UPageBody>

    <template v-if="page?.body?.toc?.links?.length" #right>
      <UContentToc :title="toc?.title" :links="page.body?.toc?.links">
        <template v-if="toc?.bottom" #bottom>
          <div
            class="hidden lg:block space-y-6"
            :class="{ 'mt-6!': page.body?.toc?.links?.length }"
          >
            <USeparator v-if="page.body?.toc?.links?.length" type="dashed" />

            <UPageLinks :title="toc.bottom.title" :links="links" />
          </div>
        </template>
      </UContentToc>
    </template>
  </UPage>
</template>
