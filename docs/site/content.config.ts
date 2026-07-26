import { defineCollection, defineContentConfig, z } from "@nuxt/content";

export default defineContentConfig({
  collections: {
    landing: defineCollection({
      type: "page",
      source: "index.md",
    }),
    docs: defineCollection({
      type: "page",
      source: {
        include: "**",
        exclude: ["index.md"],
      },
      schema: z.object({
        api: z.enum(["match", "direct", "both"]).optional(),
        links: z
          .array(
            z.object({
              label: z.string(),
              icon: z.string(),
              to: z.string(),
              target: z.string().optional(),
            }),
          )
          .optional(),
        navigation: z
          .object({
            title: z.string().optional(),
          })
          .optional(),
      }),
    }),
  },
});
