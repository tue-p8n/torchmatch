<script setup lang="ts">
const { seo } = useAppConfig();

// Blog posts stay in the "docs" collection (so individual post pages still
// resolve through the ordinary path lookup) but are excluded from the nav
// tree -- otherwise they'd nest as ordinary children under whatever
// alphabetically-first section, instead of only appearing through the
// dedicated /blog listing (see app.config.ts's header.extraLinks entry).
const { data: navigation } = await useAsyncData("navigation", () =>
  queryCollectionNavigation("docs").where("path", "NOT LIKE", "/blog/%"),
);
const { data: files } = useLazyAsyncData(
  "search",
  () => queryCollectionSearchSections("docs"),
  { server: false },
);

useHead({
  meta: [{ name: "viewport", content: "width=device-width, initial-scale=1" }],
  link: [{ rel: "icon", type: "image/svg+xml", href: "/favicon.svg" }],
  htmlAttrs: { lang: "en" },
});

useSeoMeta({
  titleTemplate: `%s · ${seo?.siteName}`,
  ogSiteName: seo?.siteName,
  twitterCard: "summary_large_image",
});

provide("navigation", navigation);
</script>

<template>
  <UApp>
    <NuxtLoadingIndicator color="var(--ui-color-primary-500)" />

    <AppHeader />

    <UMain>
      <NuxtLayout>
        <NuxtPage />
      </NuxtLayout>
    </UMain>

    <AppFooter />

    <ClientOnly>
      <LazyUContentSearch :files="files" :navigation="navigation" />
    </ClientOnly>
  </UApp>
</template>
