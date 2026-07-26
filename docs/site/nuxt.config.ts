const repoUrl = "https://github.com/khwstolle/torchmatch";
const siteUrl = process.env.NUXT_PUBLIC_SITE_URL ?? "https://torchmatch.khws.io";

export default defineNuxtConfig({
  extends: process.env.DOCYARD_THEME_PATH ? [process.env.DOCYARD_THEME_PATH] : [],

  modules: [
    "@nuxt/eslint",
    "@nuxt/fonts",
    "@nuxt/image",
    "@nuxt/ui",
    "@nuxt/content",
    "nuxt-llms",
  ],

  devtools: { enabled: true },

  app: {
    head: {
      htmlAttrs: { lang: "en" },
      meta: [
        { name: "viewport", content: "width=device-width, initial-scale=1" },
        { name: "theme-color", content: "#0b0b0f" },
        {
          name: "description",
          content:
            "torchmatch — linear assignment problem solvers for PyTorch. Munkres / Jonker-Volgenant ops registered under torch.ops.matching, with CPU SIMD kernels and CUDA backends.",
        },
      ],
      link: [{ rel: "icon", type: "image/svg+xml", href: "/favicon.svg" }],
    },
  },

  css: ["~/assets/css/main.css"],

  content: {
    build: {
      markdown: {
        highlight: {
          theme: {
            default: "github-light",
            dark: "github-dark",
          },
          langs: [
            "python",
            "cpp",
            "bash",
            "shell",
            "toml",
            "yaml",
            "json",
            "diff",
            "vue",
            "ts",
            "mermaid",
          ],
        },
        toc: { depth: 3, searchDepth: 3 },
      },
    },
    experimental: {
      sqliteConnector: "native",
    },
  },

  runtimeConfig: {
    public: {
      siteUrl,
      repoUrl,
      releaseRef: process.env.GITHUB_SHA?.slice(0, 7) ?? "dev",
    },
  },

  srcDir: "app/",
  future: { compatibilityVersion: 4 },

  experimental: {
    asyncContext: true,
  },
  compatibilityDate: "2025-01-01",

  nitro: {
    // `static` produces plain prerendered HTML in `.output/public/`,
    // suitable for GitHub Pages, Cloudflare Pages, Netlify, or any
    // static host. Content is baked into HTML at build time via
    // better-sqlite3.
    preset: "static",
    prerender: {
      crawlLinks: true,
      routes: ["/"],
      autoSubfolderIndex: false,
      failOnError: false,
    },
  },

  typescript: {
    strict: true,
    typeCheck: false,
  },

  eslint: {
    config: {
      stylistic: {
        commaDangle: "always-multiline",
        braceStyle: "1tbs",
      },
    },
  },

  fonts: {
    // Three-family stack tuned for an engineering-datasheet aesthetic.
    //   Recursive       : variable display (Stephen Nixon / Arrow Type).
    //                     Designed for code editors and technical docs;
    //                     axes wght / CASL / MONO / slnt. We lock
    //                     CASL=0 (linear/formal) and MONO=0 (sans) so
    //                     headings read as a clean engineered grotesque
    //                     rather than a humanist text face.
    //   Manrope         : variable body. Geometric semi-sans; more open
    //                     than Plex Sans, less generic than Inter.
    //   JetBrains Mono  : variable monospace. Sharp terminals, slashed
    //                     zero, hooked l. Engineered for readability in
    //                     datasheet-density text.
    families: [
      {
        name: "Recursive",
        provider: "google",
        weights: ["300 1000"],
        styles: ["normal"],
        subsets: ["latin", "latin-ext"],
      },
      {
        name: "Manrope",
        provider: "google",
        weights: ["200 800"],
        styles: ["normal"],
        subsets: ["latin", "latin-ext"],
      },
      {
        name: "JetBrains Mono",
        provider: "google",
        weights: ["100 800"],
        styles: ["normal", "italic"],
        subsets: ["latin", "latin-ext"],
      },
    ],
    defaults: {
      weights: [400, 500, 600, 700],
      styles: ["normal", "italic"],
      subsets: ["latin", "latin-ext"],
    },
  },

  icon: {
    serverBundle: "local",
  },

  llms: {
    domain: siteUrl,
    title: "torchmatch",
    description:
      "torchmatch ships PyTorch C++/CUDA extensions for the linear assignment problem (LAP). Ops register under torch.ops.matching: Munkres (classical / hybrid / tree, CUDA only) and Jonker-Volgenant (scalar / dense / compact, CPU; dense_batch also CUDA). FakeTensor kernels for every op; cudagraph_unsafe tag on the Munkres family.",
    full: {
      title: "torchmatch — full documentation",
      description:
        "Complete reference for torchmatch: tutorials, concepts, operations, building, and benchmarks.",
    },
    sections: [
      {
        title: "Tutorials",
        contentCollection: "docs",
        contentFilters: [
          { field: "path", operator: "LIKE", value: "/tutorials%" },
        ],
      },
      {
        title: "Concepts",
        contentCollection: "docs",
        contentFilters: [
          { field: "path", operator: "LIKE", value: "/concepts%" },
        ],
      },
      {
        title: "Reference",
        contentCollection: "docs",
        contentFilters: [
          { field: "path", operator: "LIKE", value: "/reference%" },
        ],
      },
      {
        title: "Benchmarks",
        contentCollection: "docs",
        contentFilters: [
          { field: "path", operator: "LIKE", value: "/benchmarks%" },
        ],
      },
    ],
  },
});
