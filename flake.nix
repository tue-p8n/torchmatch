{
  description = "Linear assignment problem solvers for PyTorch";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";

    # Formatter
    treefmt.url = "github:numtide/treefmt-nix";

    # Pre-commit
    git-hooks.url = "github:cachix/git-hooks.nix";
    git-hooks.inputs.nixpkgs.follows = "nixpkgs";

    # Python projects
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    # Documentation framework
    docyard = {
      url = "github:mapnomad/docyard";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs@{
      self,
      flake-parts,
      nixpkgs,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        inputs.git-hooks.flakeModule
        inputs.treefmt.flakeModule
        inputs.docyard.flakeModules.default
      ];
      systems = [ "x86_64-linux" ];

      perSystem =
        {
          config,
          system,
          pkgs,
          lib,
          ...
        }:
        let
          variants = import ./nix/variants.nix {
            inherit
              pkgs
              lib
              uv2nix
              pyproject-nix
              pyproject-build-systems
              ;
            workspaceRoot = ./.;
          };

          apps = import ./nix/apps.nix {
            inherit pkgs lib variants;
            defaultVariantName = "cu128";
          };

          devShells = import ./nix/devshells.nix {
            inherit pkgs lib variants;
            docyardCore = config.docyard.core;
            docyardThemePath = config.docyard.themePath;
            defaultVariantName = "cu128";
          };

          # Select the named variant matching the active pkgs.cudaPackages version.
          # Users choose the CUDA version by passing a different pkgs scope
          # (e.g. pkgs.cudaPackages_13_0.pkgs) rather than by picking a suffixed name.
          cudaVariant =
            let
              v = pkgs.cudaPackages.cudaMajorMinorVersion;
            in
            if lib.hasPrefix "13.2" v then
              variants.cu132
            else if lib.hasPrefix "13." v then
              variants.cu130
            else if v == "12.8" then
              variants.cu128
            else if v == "12.6" then
              variants.cu126
            else
              throw "torchmatchWithCuda: no wheel for CUDA ${v}. Supported: 12.6, 12.8, 13.x";
        in
        {
          _module.args.pkgs = import inputs.nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
              cudaSupport = true;
              cudaForwardCompat = true;
            };
          };

          inherit apps devShells;

          docyard = {
            enable = true;
            site = {
              directory = "docs/site";
              apis = [
                {
                  language = "python";
                  name = "torchmatch";
                  src = "sources";
                }
              ];
              outDir = "public/docyard";
              preHook = "python3 scripts/benchmark_aggregate.py";
              extraEnv = {
                NODE_OPTIONS = "--max-old-space-size=4096";
              };
            };
          };

          packages = {
            # CPU — no compiled CUDA extension; always available
            torchmatch = variants.cpu.package;
            # CUDA — version follows pkgs.cudaPackages (override pkgs to change it)
            torchmatchWithCuda = cudaVariant.package;
            # Explicit version pins — for CI builds and reproducible pinning
            torchmatchWithCuda_12_6 = variants.cu126.package;
            torchmatchWithCuda_12_8 = variants.cu128.package;
            torchmatchWithCuda_13_0 = variants.cu130.package;
            torchmatchWithCuda_13_2 = variants.cu132.package;
            default = variants.cu128.package;
          };

          # Git Hooks.
          # https://github.com/cachix/git-hooks.nix/blob/master/flake-module.nix
          pre-commit.settings = {
            package = pkgs.prek;
            hooks = {
              # Treefmt (see above)
              treefmt = {
                enable = true;
                package = config.treefmt.build.wrapper;
              };

              # File hygiene.
              check-toml.enable = true;
              check-yaml.enable = true;
              check-json.enable = true;
              check-merge-conflicts.enable = true;
              check-added-large-files.enable = true;
              end-of-file-fixer.enable = true;
              trim-trailing-whitespace = {
                enable = true;

                # Preserve markdown "two trailing spaces = line break" semantics.
                args = [ "--markdown-linebreak-ext=md" ];
              };
            };
          };

          # Treefmt
          # https://github.com/numtide/treefmt-nix
          treefmt = {
            programs = {
              alejandra.enable = true;
              deadnix.enable = true;
              shellcheck.enable = true;
              shfmt.enable = true;
              clang-format.enable = true;
              clang-tidy.enable = true;
              prettier.enable = true;
              ruff.check = true;
              ruff.format = true;
            };
            settings = {
              global.excludes = [
                "docs/site/content/**"
              ];
              formatter = {
                shellcheck.options = [
                  "-s"
                  "bash"
                ];
                ruff-check.priority = 1;
                ruff-check.options = [ "--fix-only" ];
                ruff-format.priority = 2;
                clang-format = {
                  args = [
                    "-i"
                    "--style=${./.clang-format}"
                  ];
                  includes = [
                    "*.c"
                    "*.cc"
                    "*.cpp"
                    "*.h"
                    "*.hh"
                    "*.hpp"
                    "*.glsl"
                    "*.cu"
                    "*.cuh"
                  ];
                };
              };
            };
          };
        };

      # Overlay — lets consumers do:
      #   pkgs.extend inputs.torchmatch.overlays.default
      # or, to pin a specific CUDA version:
      #   pkgs.cudaPackages_13_0.pkgs.extend inputs.torchmatch.overlays.default
      # The CUDA variant is selected by reading final.cudaPackages.cudaMajorMinorVersion,
      # so overriding pkgs.cudaPackages is all that is needed to change the CUDA version.
      flake.overlays.default =
        final: _prev:
        let
          lib = final.lib;
          variantsFromFinal = import ./nix/variants.nix {
            pkgs = final;
            inherit lib;
            inherit (inputs) uv2nix pyproject-nix pyproject-build-systems;
            workspaceRoot = ./.;
          };
          cudaVersion = final.cudaPackages.cudaMajorMinorVersion or null;
          cudaVariant =
            if cudaVersion == null then
              throw "torchmatch overlay: pkgs.cudaPackages has no cudaMajorMinorVersion"
            else if lib.hasPrefix "13.2" cudaVersion then
              variantsFromFinal.cu132
            else if lib.hasPrefix "13." cudaVersion then
              variantsFromFinal.cu130
            else if cudaVersion == "12.8" then
              variantsFromFinal.cu128
            else if cudaVersion == "12.6" then
              variantsFromFinal.cu126
            else
              throw "torchmatch overlay: no wheel for CUDA ${cudaVersion}. Supported: 12.6, 12.8, 13.x";
        in
        {
          torchmatch = variantsFromFinal.cpu.package;
          torchmatchWithCuda = cudaVariant.package;
        };
    };
}
