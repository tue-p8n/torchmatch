<script setup lang="ts">
const props = defineProps<{ api?: "match" | "direct" | "both" }>();
const show = computed(() => props.api === "match" || props.api === "direct");
</script>

<template>
  <div v-if="show" :class="['ab', `ab--${api}`]" role="note">
    <span class="ab-icon" aria-hidden="true">
      <svg
        viewBox="0 0 16 16"
        width="14"
        height="14"
        fill="none"
        stroke="currentColor"
        stroke-width="1.5"
      >
        <circle cx="8" cy="8" r="6.5" />
        <line x1="8" y1="5" x2="8" y2="9" />
        <circle cx="8" cy="11.5" r="0.5" fill="currentColor" />
      </svg>
    </span>
    <span class="ab-text">
      <template v-if="api === 'direct'">
        <strong>Direct-op guide.</strong>
        <code>solve()</code> makes these choices automatically. Read this page
        when benchmarking specific ops or overriding the default.
      </template>
      <template v-else-if="api === 'match'">
        <strong>Dispatcher guide.</strong>
        The underlying ops are documented in the
        <NuxtLink to="/algorithms/assignment/reference"
          >assignment reference</NuxtLink
        >
        and
        <NuxtLink to="/algorithms/transport/reference"
          >transport reference</NuxtLink
        >.
      </template>
    </span>
  </div>
</template>

<style scoped>
.ab {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  padding: 0.55rem 0.85rem;
  margin: 0 0 1.25rem;
  border: 1px solid var(--ui-border);
  border-radius: 3px;
  background: color-mix(
    in oklab,
    var(--ui-bg-muted, transparent) 40%,
    transparent
  );
  color: var(--ui-text-muted);
  font-size: 0.85rem;
  line-height: 1.4;
}

.ab-icon {
  color: var(--ui-text-dimmed);
  display: inline-flex;
}
.ab-text strong {
  color: var(--ui-text);
  font-weight: 600;
  margin-right: 0.2rem;
}
.ab-text :deep(code) {
  font-family: var(--font-mono);
  font-size: 0.95em;
  background: transparent;
  padding: 0;
}
.ab-text :deep(a) {
  color: var(--color-primary-500);
  text-decoration: underline;
}
</style>
