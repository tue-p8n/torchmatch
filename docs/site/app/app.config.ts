export default defineAppConfig({
  ui: {
    colors: {
      primary: "primary",
      neutral: "zinc",
    },
  },
  seo: {
    siteName: "torchmatch",
  },
  header: {
    title: "",
    to: "/",
    logo: {
      alt: "torchmatch",
      light: "/logo.svg",
      dark: "/logo-dark.svg",
    },
    search: true,
    colorMode: true,
    links: [
      {
        icon: "i-simple-icons-github",
        to: "https://github.com/khwstolle/torchmatch",
        target: "_blank",
        "aria-label": "GitHub",
      },
    ],
  },
  footer: {
    credits: `MIT · © ${new Date().getFullYear()} Kurt H. W. Stolle`,
    colorMode: false,
    links: [
      {
        icon: "i-simple-icons-github",
        to: "https://github.com/khwstolle/torchmatch",
        target: "_blank",
        "aria-label": "torchmatch on GitHub",
      },
    ],
  },
  toc: {
    title: "On this page",
    bottom: {
      title: "Project",
      edit: "https://github.com/khwstolle/torchmatch/edit/master/docs/site/content",
      links: [
        {
          icon: "i-lucide-star",
          label: "Star on GitHub",
          to: "https://github.com/khwstolle/torchmatch",
          target: "_blank",
        },
        {
          icon: "i-lucide-package",
          label: "PyPI",
          to: "https://pypi.org/project/torchmatch/",
          target: "_blank",
        },
      ],
    },
  },
});
