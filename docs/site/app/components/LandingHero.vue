<script setup lang="ts">
// Rendered with v-html so source whitespace is isolated from any Vue/HTML
// formatter. Lines must stay flush-left here; the <pre> preserves them as-is.
const terminal = `<code><span class="lh-prompt">&gt;&gt;&gt;</span> <span class="lh-kw">import</span> torch, torchmatch
<span class="lh-prompt">&gt;&gt;&gt;</span> cost = torch.rand(<span class="lh-num">8</span>, <span class="lh-num">8</span>)
<span class="lh-prompt">&gt;&gt;&gt;</span> torchmatch.assignment.solve(cost)
<span class="lh-out">tensor([5, 7, 0, 4, 1, 6, 2, 3])</span>
<span class="lh-prompt">&gt;&gt;&gt;</span> costs = torch.rand(<span class="lh-num">64</span>, <span class="lh-num">32</span>, <span class="lh-num">32</span>, device=<span class="lh-str">"cuda"</span>)
<span class="lh-prompt">&gt;&gt;&gt;</span> torchmatch.assignment.solve(costs).shape
<span class="lh-out">torch.Size([64, 32])</span>
<span class="lh-prompt">&gt;&gt;&gt;</span> x = torch.randn(<span class="lh-num">512</span>, <span class="lh-num">3</span>, device=<span class="lh-str">"cuda"</span>)
<span class="lh-prompt">&gt;&gt;&gt;</span> y = torch.randn(<span class="lh-num">512</span>, <span class="lh-num">3</span>, device=<span class="lh-str">"cuda"</span>)
<span class="lh-prompt">&gt;&gt;&gt;</span> torchmatch.transport.samples.loss(x, y)
<span class="lh-out">tensor(0.0342, device='cuda:0')</span></code>`;
</script>

<template>
  <section class="lh">
    <LandingBg />

    <div class="lh-body">
      <div class="lh-left">
        <p class="lh-eyebrow">
          <span class="lh-eyebrow-mark">§</span> torch.ops.assignment &middot;
          torch.ops.transport
        </p>

        <h1 class="lh-title">torchmatch</h1>

        <p class="lh-desc">
          LAP solvers (JV, Munkres, Greedy) and optimal transport (Sinkhorn,
          EMD) registered as PyTorch custom ops. Batch-native.
          torch.compile-ready.
        </p>

        <div class="lh-actions">
          <a class="lh-btn lh-btn-primary" href="/getting-started">
            Quickstart <span aria-hidden="true">→</span>
          </a>
          <a
            class="lh-btn lh-btn-secondary"
            href="https://github.com/khwstolle/torchmatch"
            target="_blank"
            rel="noopener"
          >
            Source <span aria-hidden="true">↗</span>
          </a>
        </div>
      </div>

      <div class="lh-right">
        <div
          class="lh-term"
          role="figure"
          aria-label="Terminal session showing torchmatch usage"
        >
          <div class="lh-term-head">
            <span class="lh-term-dot" />
            <span class="lh-term-dot" />
            <span class="lh-term-dot" />
            <span class="lh-term-title">python · cu128</span>
          </div>
          <!-- eslint-disable vue/no-v-html -- static author-controlled markup -->
          <pre class="lh-term-body" v-html="terminal" />
          <!-- eslint-enable vue/no-v-html -->
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.lh {
  position: relative;
  isolation: isolate;
  padding-block: clamp(2.5rem, 4vw, 4rem);
  padding-inline: clamp(1rem, 4vw, 3rem);
  overflow: hidden;
  border-bottom: 1px solid
    var(--ui-border, color-mix(in oklab, currentColor 14%, transparent));
}

/* Two-column: text left (narrower), terminal right (wider). The
 * terminal is the product — it gets the majority of the width. */
.lh-body {
  display: grid;
  grid-template-columns: 1fr;
  gap: clamp(3rem, 6vw, 5rem);
}

@media (min-width: 960px) {
  .lh-body {
    grid-template-columns: minmax(0, 5fr) minmax(0, 7fr);
    align-items: stretch;
  }
}

.lh-left {
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
}

/* "torchmatch" in monospace weight — the font family mirrors the terminal,
 * tying the name to the thing it names. */
.lh-title {
  font-family: var(--font-mono);
  font-weight: 800;
  font-size: clamp(3rem, 9vw, 6.5rem);
  line-height: 1;
  letter-spacing: -0.04em;
  margin: 0 0 1.25rem 0;
  color: var(--ui-text-highlighted);
}

.lh-eyebrow {
  font-family: var(--font-mono);
  font-size: 0.68rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--ui-text-muted);
  margin: 0 0 1.5rem 0;
}

.lh-eyebrow-mark {
  color: var(--color-primary-500);
}

.lh-desc {
  font-family: var(--font-sans);
  font-size: 1.02rem;
  line-height: 1.65;
  color: var(--ui-text-muted);
  margin: 0 0 2.25rem 0;
  max-width: 34em;
}

.lh-actions {
  display: flex;
  gap: 0.6rem;
  flex-wrap: wrap;
}

.lh-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.6rem;
  padding: 0.8rem 1.15rem;
  font-family: var(--font-mono);
  font-size: 0.72rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  text-decoration: none;
  border: 1px solid transparent;
  transition:
    background 120ms ease,
    color 120ms ease,
    border-color 120ms ease;
}

.lh-btn-primary {
  background: var(--color-primary-500);
  color: white;
  border-color: var(--color-primary-500);
}

.lh-btn-primary:hover {
  background: var(--color-primary-600);
  border-color: var(--color-primary-600);
}

.lh-btn-secondary {
  background: transparent;
  color: var(--ui-text);
  border-color: var(
    --ui-border-accented,
    color-mix(in oklab, currentColor 25%, transparent)
  );
}

.lh-btn-secondary:hover {
  border-color: var(--color-primary-500);
  color: var(--color-primary-500);
}

/* Terminal panel. The bloom behind it is already positioned via
 * .lh-bloom; here we only style the box itself. */
.lh-right {
  width: 100%;
}

.lh-term {
  border: 1px solid
    var(
      --ui-border-accented,
      color-mix(in oklab, currentColor 22%, transparent)
    );
  background: var(--ui-bg-elevated, white);
  border-radius: 2px;
  box-shadow:
    0 0 0 1px color-mix(in oklab, currentColor 3%, transparent),
    0 24px 48px -16px
      color-mix(in oklab, var(--color-primary-500) 22%, transparent);
  overflow: hidden;
}

.lh-term-head {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.55rem 0.85rem;
  border-bottom: 1px solid var(--ui-border);
  font-family: var(--font-mono);
  font-size: 0.65rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--ui-text-dimmed);
  background: color-mix(
    in oklab,
    var(--ui-bg-elevated, white) 92%,
    var(--ui-text) 8%
  );
}

.lh-term-dot {
  width: 0.55rem;
  height: 0.55rem;
  border-radius: 999px;
  background: color-mix(in oklab, currentColor 20%, transparent);
}

.lh-term-dot:nth-child(1) {
  background: color-mix(in oklab, var(--color-primary-500) 70%, transparent);
}

.lh-term-dot:nth-child(2) {
  background: color-mix(in oklab, currentColor 30%, transparent);
}

.lh-term-dot:nth-child(3) {
  background: color-mix(in oklab, currentColor 20%, transparent);
}

.lh-term-title {
  margin-left: 0.6rem;
}

.lh-term-body {
  margin: 0;
  padding: 1.1rem 1.25rem;
  font-family: var(--font-mono);
  font-size: 0.83rem;
  line-height: 1.6;
  white-space: pre;
  overflow-x: auto;
  color: var(--ui-text);
  background: var(--ui-bg-elevated, white);
}

.lh-prompt {
  color: var(--color-primary-500);
  user-select: none;
  margin-right: 0.4rem;
}

.lh-kw {
  color: var(--color-primary-500);
}

.lh-num {
  color: var(--ui-text);
  font-weight: 500;
}

.lh-str {
  color: var(--ui-text);
  font-weight: 500;
}

.lh-out {
  color: var(--ui-text-muted);
}
</style>
