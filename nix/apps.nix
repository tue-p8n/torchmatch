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
}: let
  defaultVariant = variants.${defaultVariantName};

  mkVenvApp = {
    name,
    variant ? defaultVariant,
    runtimeInputs ? [],
    text,
  }:
    pkgs.writeShellApplication {
      inherit name;
      runtimeInputs = [variant.venv] ++ runtimeInputs;
      text = ''
        set -euo pipefail
        ${text}
      '';
    };

  mkStdlibPythonApp = {
    name,
    text,
  }:
    pkgs.writeShellApplication {
      inherit name;
      runtimeInputs = [pkgs.python313];
      text = ''
        set -euo pipefail
        ${text}
      '';
    };

  toApp = drv: {
    type = "app";
    program = "${drv}/bin/${drv.meta.mainProgram or drv.pname or drv.name}";
  };

  bench-init = mkVenvApp {
    name = "bench-init";
    text = ''python -m torchmatch.bench init-machine "$@"'';
  };

  bench-collect = mkVenvApp {
    name = "bench-collect";
    text = ''python -m torchmatch.bench collect "$@"'';
  };

  bench-aggregate = mkStdlibPythonApp {
    name = "bench-aggregate";
    text = ''python3 scripts/benchmark_aggregate.py "$@"'';
  };

  bench-validate = mkStdlibPythonApp {
    name = "bench-validate";
    text = ''python3 scripts/benchmark_validate.py benchmarks/results "$@"'';
  };

  perVariantTests = lib.mapAttrs' (vname: variant:
    lib.nameValuePair "test-${vname}" (mkVenvApp {
      name = "test-${vname}";
      variant = variant;
      text = ''pytest tests/ "$@"'';
    }))
  variants;

  perVariantLints = lib.mapAttrs' (vname: variant:
    lib.nameValuePair "lint-${vname}" (mkVenvApp {
      name = "lint-${vname}";
      variant = variant;
      text = ''ruff check . "$@"'';
    }))
  variants;

  perVariantFormats = lib.mapAttrs' (vname: variant:
    lib.nameValuePair "format-${vname}" (mkVenvApp {
      name = "format-${vname}";
      variant = variant;
      text = ''ruff format . "$@"'';
    }))
  variants;

  test = mkVenvApp {
    name = "test";
    text = ''pytest tests/ "$@"'';
  };

  lint = mkVenvApp {
    name = "lint";
    text = ''ruff check . "$@"'';
  };

  format = mkVenvApp {
    name = "format";
    text = ''ruff format . "$@"'';
  };

  # Execute notebooks and export them as markdown pages for the docs site.
  # Prerequisites: uv sync --group notebooks  (adds matplotlib, jupyter, nbconvert, jupytext)
  nb-render = mkVenvApp {
    name = "nb-render";
    text = ''python scripts/nb_render.py "$@"'';
  };

  flatApps =
    {
      inherit bench-init bench-collect bench-aggregate bench-validate;
      inherit test lint format;
      inherit nb-render;
    }
    // perVariantTests
    // perVariantLints
    // perVariantFormats;
in
  lib.mapAttrs (_n: drv: toApp drv) flatApps
