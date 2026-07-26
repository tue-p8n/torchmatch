# Per-variant dev shells for torchmatch.
#
# The shell exposes the uv2nix-built variant venv on PATH (so `python -c
# "import torchmatch"` works without `uv sync`), and additionally keeps
# `uv` available so contributors can `uv sync --extra <variant>` for
# editable iteration on torchmatch itself (which sets
# no-build-isolation-package = ["torchmatch"] in pyproject.toml).
{
  pkgs,
  lib,
  variants,
  docyardCore,
  docyardThemePath,
  defaultVariantName ? "cu128",
}: let
  cRuntimeLibs = with pkgs; [
    stdenv.cc.cc.lib
    zlib
    bzip2
    xz
    zstd
    openssl
    libffi
    ncurses
    libxml2
    expat
  ];
  graphicsLibs = with pkgs; [
    libGL
    libGLU
    libglvnd
    glib
  ];
  mediaLibs = with pkgs; [
    libjpeg
    libpng
    libtiff
    freetype
    fontconfig
  ];
  commonRuntimeLibs = cRuntimeLibs ++ graphicsLibs ++ mediaLibs;

  uvShellHook = ''
    REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || REPO_ROOT=$PWD
    export REPO_ROOT
    export UV_LINK_MODE=copy
    export UV_PYTHON_PREFERENCE=only-managed
    export UV_PYTHON_DOWNLOADS=auto
    export UV_PROJECT_ENVIRONMENT="$REPO_ROOT/.venv"
    export VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT"
    export TORCH_EXTENSIONS_DIR="$VIRTUAL_ENV/torch_extensions"
  '';

  nixLdHook = libPath: ''
    export NIX_LD_LIBRARY_PATH="${libPath}"
    export NIX_LD="${lib.fileContents "${pkgs.stdenv.cc}/nix-support/dynamic-linker"}"
  '';

  hostGpuHook = ''
    if [ -d "/run/opengl-driver/lib" ]; then
      export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
      export TRITON_LIBCUDA_PATH="/run/opengl-driver/lib"
    fi
  '';

  commonShellPackages = with pkgs; [
    uv
    clang-tools
    cmake
    ninja
    ruff
    pyright
    git
    pkg-config
    nodejs_24
    pnpm_9
    docyardCore
  ];

  mkShellFor = {
    name,
    variant,
  }: let
    cudaPkgs = variant.cudaPkgs;
    libPath = lib.makeLibraryPath (
      commonRuntimeLibs
      ++ lib.optional (cudaPkgs != null) cudaPkgs.cudatoolkit
    );
    mkShellOverride = pkgs.mkShell.override {stdenv = variant.hostStdenv;};
    cudaPackagesList = lib.optional (cudaPkgs != null) cudaPkgs.cudatoolkit;
    backendEnvLines =
      if cudaPkgs == null
      then ''
        export TORCHMATCH_SKIP_CUDA=1
      ''
      else let
        mmVersion = lib.versions.majorMinor cudaPkgs.cudatoolkit.version;
        backend = "cu${builtins.replaceStrings ["."] [""] mmVersion}";
      in ''
        export CUDA_HOME="${cudaPkgs.cudatoolkit}"
        export CUDA_PATH="${cudaPkgs.cudatoolkit}"
        export UV_TORCH_BACKEND="${backend}"
        # Probe every visible GPU and join the unique compute caps with `;`
        # so torch builds a fat binary covering all devices on this host.
        # Falls back to a fixed list when no GPU is detected; echoes which
        # path produced the value so silent fallbacks are visible.
        archs=""
        if command -v nvidia-smi > /dev/null 2>&1; then
          archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u | paste -sd ';' -)
        fi
        if [ -n "$archs" ]; then
          export TORCH_CUDA_ARCH_LIST="$archs"
          echo " >>> TORCH_CUDA_ARCH_LIST=$archs (probed via nvidia-smi)"
        else
          : "''${TORCH_CUDA_ARCH_LIST:=8.0;8.6;8.9;9.0}"
          export TORCH_CUDA_ARCH_LIST
          echo " >>> TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST (fallback; no GPU detected)" >&2
        fi
      '';
  in
    mkShellOverride {
      name = "torchmatch-${name}";
      packages = commonShellPackages ++ cudaPackagesList ++ [variant.venv];
      shellHook = ''
        ${uvShellHook}
        ${nixLdHook libPath}
        export LD_LIBRARY_PATH="${libPath}:$LD_LIBRARY_PATH"
        ${hostGpuHook}
        ${backendEnvLines}
        export DOCYARD_THEME_PATH="${docyardThemePath}"
        echo " >>> torchmatch ${name} shell (uv $(uv --version 2>/dev/null | head -n1))"
      '';
    };

  shells = lib.mapAttrs (vname: variant:
    mkShellFor {
      name = vname;
      variant = variant;
    })
  variants;
in
  shells // {default = shells.${defaultVariantName};}
