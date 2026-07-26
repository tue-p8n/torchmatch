<script setup lang="ts">
import { onMounted, onUnmounted, ref } from "vue";

const canvas = ref<HTMLCanvasElement | null>(null);

let raf = 0;
let ro: ResizeObserver | null = null;

onMounted(() => {
  const el = canvas.value;
  if (!el) return;
  const ctx = el.getContext("2d");
  if (!ctx) return;

  // Seeded xorshift RNG — deterministic so the pattern is identical on
  // every render and every viewport width.
  let s = 0xd3adb33f;
  const rand = () => {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return (s >>> 0) / 0xffffffff;
  };

  // A "matching problem" is a small bipartite graph placed at (cx, cy)
  // in normalised canvas space. `angle` rotates the axis that separates
  // source from target; `spread` sets the inter-cluster distance.
  // `depth` scales opacity and node size to suggest visual depth.
  //
  // The problems below cover the four use-case families:
  //   – Tracking: sparse, few nodes, near-horizontal (few large objects)
  //   – Set-prediction: medium density, moderate n
  //   – Cluster evaluation: tight, often square-ish angle
  //   – Point-cloud OT: small, many problems, denser distribution
  interface ProblemSpec {
    cx: number;
    cy: number;
    n: number;
    spread: number;
    angle: number;
    phase: number;
    depth: number;
  }
  const SPECS: ProblemSpec[] = [
    // tracking-scale: sparse, few large nodes, wide spread
    {
      cx: 0.08,
      cy: 0.18,
      n: 4,
      spread: 0.1,
      angle: 0.2,
      phase: 0.0,
      depth: 0.88,
    },
    {
      cx: 0.82,
      cy: 0.78,
      n: 4,
      spread: 0.09,
      angle: -0.12,
      phase: 2.5,
      depth: 0.82,
    },
    // set-prediction scale: medium, angled
    {
      cx: 0.38,
      cy: 0.14,
      n: 5,
      spread: 0.12,
      angle: 0.42,
      phase: 1.0,
      depth: 0.9,
    },
    {
      cx: 0.62,
      cy: 0.72,
      n: 5,
      spread: 0.11,
      angle: -0.38,
      phase: 3.8,
      depth: 0.86,
    },
    // cluster evaluation: tighter spread, steeper angle
    {
      cx: 0.16,
      cy: 0.55,
      n: 4,
      spread: 0.08,
      angle: 1.05,
      phase: 4.2,
      depth: 0.76,
    },
    {
      cx: 0.86,
      cy: 0.38,
      n: 3,
      spread: 0.07,
      angle: -0.95,
      phase: 0.8,
      depth: 0.7,
    },
    // centre-piece: largest, most visible
    {
      cx: 0.5,
      cy: 0.45,
      n: 6,
      spread: 0.13,
      angle: 0.06,
      phase: 0.6,
      depth: 0.92,
    },
    // point-cloud scale: small, numerous, scattered
    {
      cx: 0.28,
      cy: 0.82,
      n: 4,
      spread: 0.07,
      angle: -0.22,
      phase: 2.9,
      depth: 0.73,
    },
    {
      cx: 0.68,
      cy: 0.22,
      n: 3,
      spread: 0.06,
      angle: 0.72,
      phase: 5.1,
      depth: 0.67,
    },
    {
      cx: 0.94,
      cy: 0.58,
      n: 3,
      spread: 0.06,
      angle: 0.85,
      phase: 1.6,
      depth: 0.63,
    },
    {
      cx: 0.04,
      cy: 0.72,
      n: 3,
      spread: 0.06,
      angle: -0.65,
      phase: 3.5,
      depth: 0.61,
    },
    {
      cx: 0.54,
      cy: 0.88,
      n: 4,
      spread: 0.08,
      angle: 1.25,
      phase: 4.8,
      depth: 0.7,
    },
  ];

  // Per-node state: base position + slow organic drift
  interface Pt {
    bx: number;
    by: number;
    r: number;
    phase: number;
    dx: number;
    dy: number;
    ds: number;
  }

  // For each problem, generate source and target node arrays.
  // Source nodes sit at -perp offset; target nodes at +perp offset.
  // Targets are ordered in reverse so source[i]→target[i] edges cross.
  interface Problem {
    sources: Pt[];
    targets: Pt[];
    phase: number;
    depth: number;
  }

  const problems: Problem[] = SPECS.map((spec) => {
    const { cx, cy, n, spread, angle, depth } = spec;
    const perpX = Math.cos(angle + Math.PI / 2);
    const perpY = Math.sin(angle + Math.PI / 2);
    const paraX = Math.cos(angle);
    const paraY = Math.sin(angle);

    const sep = spread * 0.9; // source/target centre offset
    const fan = spread * 1.7; // fan width (along the parallel axis)
    const jit = spread * 0.12; // per-node position jitter

    const makeCluster = (sign: number, reverse: boolean): Pt[] =>
      Array.from({ length: n }, (_, i) => {
        const idx = reverse ? n - 1 - i : i;
        const t = (idx / Math.max(n - 1, 1) - 0.5) * fan;
        return {
          bx: cx + sign * perpX * sep + paraX * t + (rand() - 0.5) * jit,
          by: cy + sign * perpY * sep + paraY * t + (rand() - 0.5) * jit,
          r: rand() * 1.4 + 1.1,
          phase: rand() * Math.PI * 2,
          dx: (rand() - 0.5) * 0.012,
          dy: (rand() - 0.5) * 0.018,
          ds: rand() * 0.14 + 0.07,
        };
      });

    return {
      sources: makeCluster(-1, false),
      targets: makeCluster(+1, true), // reversed → crossing edges
      phase: spec.phase,
      depth,
    };
  });

  const resize = () => {
    el.width = el.offsetWidth;
    el.height = el.offsetHeight;
  };
  resize();
  ro = new ResizeObserver(resize);
  ro.observe(el);

  const PRIMARY = "#E03520";
  const start = performance.now();

  const nodePos = (pt: Pt, t: number, W: number, H: number) => ({
    x: (pt.bx + Math.sin(t * pt.ds + pt.phase) * pt.dx) * W,
    y: (pt.by + Math.sin(t * pt.ds * 1.4 + pt.phase + 1.2) * pt.dy) * H,
  });

  const draw = (now: number) => {
    const W = el.width;
    const H = el.height;
    if (!W || !H) {
      raf = requestAnimationFrame(draw);
      return;
    }

    ctx.clearRect(0, 0, W, H);
    const t = (now - start) / 1000;

    for (const prob of problems) {
      const { sources, targets, phase, depth } = prob;
      const n = sources.length;

      // Matching edges — each pulses at a distinct phase within the problem
      for (let i = 0; i < n; i++) {
        const sp = nodePos(sources[i], t, W, H);
        const tp = nodePos(targets[i], t, W, H);
        const pulse =
          0.4 + 0.6 * Math.sin(t * 0.28 + phase + i * ((Math.PI * 2) / n));
        ctx.beginPath();
        ctx.moveTo(sp.x, sp.y);
        ctx.lineTo(tp.x, tp.y);
        ctx.strokeStyle = PRIMARY;
        ctx.globalAlpha = pulse * 0.1 * depth;
        ctx.lineWidth = 0.75;
        ctx.stroke();
      }

      // Source nodes (slightly brighter)
      for (const pt of sources) {
        const { x, y } = nodePos(pt, t, W, H);
        const pulse = 0.8 + 0.2 * Math.sin(t * 0.6 + pt.phase);
        ctx.beginPath();
        ctx.arc(x, y, pt.r * pulse, 0, Math.PI * 2);
        ctx.fillStyle = PRIMARY;
        ctx.globalAlpha = 0.28 * pulse * depth;
        ctx.fill();
      }

      // Target nodes (slightly dimmer)
      for (const pt of targets) {
        const { x, y } = nodePos(pt, t, W, H);
        const pulse = 0.8 + 0.2 * Math.sin(t * 0.55 + pt.phase);
        ctx.beginPath();
        ctx.arc(x, y, pt.r * pulse, 0, Math.PI * 2);
        ctx.fillStyle = PRIMARY;
        ctx.globalAlpha = 0.19 * pulse * depth;
        ctx.fill();
      }
    }

    ctx.globalAlpha = 1;
    raf = requestAnimationFrame(draw);
  };

  raf = requestAnimationFrame(draw);
});

onUnmounted(() => {
  cancelAnimationFrame(raf);
  ro?.disconnect();
});
</script>

<template>
  <canvas ref="canvas" class="lh-bg" aria-hidden="true" />
</template>

<style scoped>
.lh-bg {
  position: absolute;
  inset: 0;
  z-index: -1;
  width: 100%;
  height: 100%;
  display: block;
  /* Vignette: fade the pattern toward all four edges so it blends into
   * the page background rather than cutting hard at the section borders. */
  mask-image: radial-gradient(
    ellipse 90% 85% at 50% 45%,
    black 30%,
    transparent 78%
  );
  -webkit-mask-image: radial-gradient(
    ellipse 90% 85% at 50% 45%,
    black 30%,
    transparent 78%
  );
}
</style>
