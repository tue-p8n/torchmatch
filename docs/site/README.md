# torchmatch documentation site

A Nuxt 4 + Nuxt UI + Nuxt Content documentation site for torchmatch.

## Architecture

```
docs/site/
  app/             Nuxt source: layouts, pages, components, design tokens
  content/         Handwritten Markdown pages
  public/          Static assets (logos, favicon)
  .output/         (gitignored) Nuxt build output
  nuxt.config.ts   Nitro static preset; deploys as plain HTML to any host
```

## Local development

```bash
cd docs/site
bun install
bun run dev
```

Site at <http://localhost:3000>. Edits to `content/**/*.md` and `app/**`
hot-reload.

## Build the production bundle

```bash
bun run build
nix run .#docs-build     # from repo root
```

Static output lands in `.output/public/` and `dist/`, ready for upload
to GitHub Pages, Cloudflare Pages, Netlify, or any static host.

## Authoring conventions

### Page ordering

Pages are ordered by filename prefix:

```
content/1.getting-started.md       → /getting-started
content/2.tutorials/1.basic.md     → /tutorials/basic
content/2.tutorials/index.md       → /tutorials
```

The router strips the numeric prefix from the URL. Keep prefixes
gap-free per section so reordering takes a single rename.

### Frontmatter

```yaml
---
title: Required: appears in <title> and navigation
description: Optional: meta description shown in cards and TOC
---
```

### Embedding components in Markdown

The site uses MDC (Markdown Components) from `@nuxt/content`. Any
component in `app/components/` and Nuxt UI's `u-*` components can be
embedded:

```mdc
::u-page-section
#title
Section title

#description
Section description.
::
```

See `content/index.md` for the landing page layout.
