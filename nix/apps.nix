# `nix run .#<task>` surface for torchmatch.
#
# Python-touching apps default to the cu128 variant venv. Per-variant
# suffixed forms (`test-<variant>`, `lint-<variant>`, `format-<variant>`)
# are emitted for every variant so contributors can pin to a specific
# torch ABI; the `*-cu128` forms are redundant aliases of the unsuffixed
# ones.
#
# Apps run from the repository root (caller's CWD when invoked via
# `nix run .#<app>`). Docs apps (docs-serve, docs-build, docs-preview) are
# provided by the docyard flake module and are not defined here.
{
  pkgs,
  lib,
  variants,
  defaultVariantName ? "cu128",
}:
let
  defaultVariant = variants.${defaultVariantName};

  # Mirrors devshells.nix's hostGpuHook: `nix run` apps don't source the
  # devShell's shellHook, so Triton (transport.samples, benchmarks) falls
  # back to `ldconfig -p` for libcuda.so discovery -- absent on NixOS.
  hostGpuHook = ''
    if [ -d "/run/opengl-driver/lib" ]; then
      export LD_LIBRARY_PATH="/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      export TRITON_LIBCUDA_PATH="/run/opengl-driver/lib"
    fi
  '';

  mkVenvApp =
    {
      name,
      variant ? defaultVariant,
      runtimeInputs ? [ ],
      text,
    }:
    pkgs.writeShellApplication {
      inherit name;
      runtimeInputs = [ variant.venv ] ++ runtimeInputs;
      text = ''
        set -euo pipefail
        ${hostGpuHook}
        ${text}
      '';
    };

  mkStdlibPythonApp =
    {
      name,
      text,
    }:
    pkgs.writeShellApplication {
      inherit name;
      runtimeInputs = [ pkgs.python313 ];
      text = ''
        set -euo pipefail
        ${text}
      '';
    };

  toApp = drv: {
    type = "app";
    program = "${drv}/bin/${drv.meta.mainProgram or drv.pname or drv.name}";
  };

  benchmark-init = mkVenvApp {
    name = "benchmark-init";
    text = ''python -m torchmatch.bench init-machine "$@"'';
  };

  benchmark-collect = mkVenvApp {
    name = "benchmark-collect";
    text = ''python -m torchmatch.bench collect "$@"'';
  };

  benchmark-aggregate = mkStdlibPythonApp {
    name = "benchmark-aggregate";
    text = ''python3 scripts/benchmark_aggregate.py "$@"'';
  };

  benchmark-validate = mkStdlibPythonApp {
    name = "benchmark-validate";
    text = ''python3 scripts/benchmark_validate.py benchmarks/results "$@"'';
  };

  perVariantTests = lib.mapAttrs' (
    vname: variant:
    lib.nameValuePair "test-${vname}" (mkVenvApp {
      name = "test-${vname}";
      variant = variant;
      text = ''pytest tests/ "$@"'';
    })
  ) variants;

  perVariantLints = lib.mapAttrs' (
    vname: variant:
    lib.nameValuePair "lint-${vname}" (mkVenvApp {
      name = "lint-${vname}";
      variant = variant;
      text = ''ruff check . "$@"'';
    })
  ) variants;

  test = mkVenvApp {
    name = "test";
    text = ''pytest tests/ "$@"'';
  };

  lint = mkVenvApp {
    name = "lint";
    text = ''ruff check . "$@"'';
  };

  # Execute notebooks and export them as markdown pages for the docs site.
  # Prerequisites: uv sync --group notebooks  (adds matplotlib, jupyter, nbconvert, jupytext)
  nb-render = mkVenvApp {
    name = "nb-render";
    text = ''python scripts/nb_render.py "$@"'';
  };

  flatApps = {
    inherit
      benchmark-init
      benchmark-collect
      benchmark-aggregate
      benchmark-validate
      ;
    inherit test lint;
    inherit nb-render;
  }
  // perVariantTests
  // perVariantLints;
in
lib.mapAttrs (_n: drv: toApp drv) flatApps
