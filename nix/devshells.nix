# Per-variant dev shells for torchmatch, built using tue-p8n/nix.
#
# The shell exposes the uv2nix-built variant venv on PATH (so `python -c
# "import torchmatch"` works without `uv sync`), and additionally keeps
# `uv` available so contributors can `uv sync --extra <variant>` for
# editable iteration on torchmatch itself.
{
  pkgs,
  lib,
  tue-p8n,
  variants,
  docyardCore,
  docyardThemePath,
  defaultVariantName ? "cu128",
}: let
  mkShellFor = {
    name,
    variant,
    accelerator,
  }: let
    # Resolve target accelerator hardware environment
    env = tue-p8n.lib.resolve {
      inherit pkgs;
      inherit accelerator;
    };
  in
    env.uv.mkShell {
      name = "torchmatch-${name}";
      packages = [
        variant.venv
        docyardCore
        pkgs.clang-tools
        pkgs.cmake
        pkgs.ruff
        pkgs.pyright
        pkgs.nodejs_24
        pkgs.pnpm
      ];
      env =
        {
          DOCYARD_THEME_PATH = docyardThemePath;
        }
        // lib.optionalAttrs (accelerator == "cpu") {
          TORCHMATCH_SKIP_CUDA = "1";
        };
      shellHook = ''
        export TORCH_EXTENSIONS_DIR="$VIRTUAL_ENV/torch_extensions"
        echo " >>> torchmatch ${name} shell (uv $(uv --version 2>/dev/null | head -n1))"
      '';
    };

  shells = {
    cpu = mkShellFor {
      name = "cpu";
      variant = variants.cpu;
      accelerator = "cpu";
    };
    cu126 = mkShellFor {
      name = "cu126";
      variant = variants.cu126;
      accelerator = "cuda12_6";
    };
    cu128 = mkShellFor {
      name = "cu128";
      variant = variants.cu128;
      accelerator = "cuda12_8";
    };
    cu130 = mkShellFor {
      name = "cu130";
      variant = variants.cu130;
      accelerator = "cuda13_0";
    };
    cu132 = mkShellFor {
      name = "cu132";
      variant = variants.cu132;
      accelerator = "cuda13_2";
    };
  };
in
  shells // {default = shells.${defaultVariantName};}
