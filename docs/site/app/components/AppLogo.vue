<template>
  <svg
    viewBox="0 0 168 32"
    xmlns="http://www.w3.org/2000/svg"
    aria-label="torchmatch"
    role="img"
  >
    <title>torchmatch</title>
    <line
      x1="5"
      y1="25"
      x2="20"
      y2="7"
      :stroke="'var(--ui-primary)'"
      stroke-width="1.1"
      stroke-linecap="round"
      opacity="0.22"
    />
    <line
      x1="5"
      y1="16"
      x2="20"
      y2="16"
      :stroke="'var(--ui-primary)'"
      stroke-width="1.3"
      stroke-linecap="round"
      opacity="0.30"
    />
    <line
      x1="5"
      y1="7"
      x2="20"
      y2="25"
      :stroke="'var(--ui-primary)'"
      stroke-width="2.0"
      stroke-linecap="round"
      opacity="0.46"
    />
    <circle cx="5" cy="25" r="2.5" fill="var(--ui-primary)" opacity="0.28" />
    <circle cx="20" cy="7" r="2.5" fill="var(--ui-primary)" opacity="0.28" />
    <circle cx="5" cy="16" r="2.8" fill="var(--ui-primary)" opacity="0.42" />
    <circle cx="20" cy="16" r="2.8" fill="var(--ui-primary)" opacity="0.42" />
    <circle cx="20" cy="25" r="3.5" fill="var(--ui-primary)" opacity="0.55" />
    <circle cx="5" cy="7" r="4" fill="var(--ui-primary)" />
    <path d="M5,7 C2,4 3,1 5,0 C7,1 8,4 5,7Z" fill="#FF8040" opacity="0.92" />
    <text
      x="30"
      y="23"
      font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace"
      font-size="18"
      font-weight="700"
      fill="currentColor"
    >
      torchmatch
    </text>
  </svg>
</template>
