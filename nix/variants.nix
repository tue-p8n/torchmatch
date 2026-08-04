# Per-variant uv2nix python sets for torchmatch, built using tue-p8n/nix.
#
# Returns an attribute set keyed by variant name; each entry exposes:
#   pythonSet   - the uv2nix python package set;
#   venv        - virtual env containing all locked deps + torchmatch;
#   package     - the torchmatch package derivation;
#   hostStdenv  - the gcc stdenv used to build the package;
#   cudaPkgs    - the cudaPackages_X_Y attrset, or null for the cpu variant.
{
  pkgs,
  lib,
  tue-p8n,
  workspaceRoot,
}: let
  mkVariant = {
    name,
    accelerator,
  }: let
    # Resolve target accelerator hardware environment
    env = tue-p8n.lib.resolve {
      inherit pkgs;
      inherit accelerator;
    };

    stdenv' =
      if lib.hasPrefix "cuda12" accelerator
      then pkgs.gcc13Stdenv
      else env.config.stdenv;

    # Build project via tue-p8n's centralized uv2nix builder
    project = env.uv.mkProject {
      inherit name workspaceRoot;

      # torchmatch owns its C++ extensions; configure the build environment
      overrides = _final: prev: {
        torchmatch = prev.torchmatch.overrideAttrs (old: {
          stdenv = stdenv';
          nativeBuildInputs =
            (old.nativeBuildInputs or [])
            ++ lib.optionals (env.config.tag != "cpu") [
              stdenv'.cc
              env.config.pkgs.cudaPackages.cudatoolkit
            ];
          env =
            (old.env or {})
            // (
              if env.config.tag == "cpu"
              then {
                TORCHMATCH_BUILD_CPU = "1";
                TORCHMATCH_BUILD_TRANSPORT = "1";
              }
              else {
                TORCHMATCH_BUILD_CPU = "1";
                TORCHMATCH_BUILD_CUDA = "1";
                TORCHMATCH_BUILD_TRANSPORT = "1";
                CUDA_HOME = "${env.config.pkgs.cudaPackages.cudatoolkit}";
                CUDA_PATH = "${env.config.pkgs.cudaPackages.cudatoolkit}";
                TORCH_CUDA_ARCH_LIST = "8.0;8.6;8.9;9.0";
                CC = "${stdenv'.cc}/bin/gcc";
                CXX = "${stdenv'.cc}/bin/g++";
              }
            );
        });
      };
    };
  in {
    pythonSet = project.pythonSet;
    venv = project.venv;
    package = project.pythonSet.torchmatch;
    cudaPkgs =
      if accelerator == "cpu"
      then null
      else env.config.pkgs.cudaPackages;
    hostStdenv = stdenv';
  };
in {
  cpu = mkVariant {
    name = "torchmatch";
    accelerator = "cpu";
  };
  cu126 = mkVariant {
    name = "torchmatch";
    accelerator = "cuda12_6";
  };
  cu128 = mkVariant {
    name = "torchmatch";
    accelerator = "cuda12_8";
  };
  cu130 = mkVariant {
    name = "torchmatch";
    accelerator = "cuda13_0";
  };
  cu132 = mkVariant {
    name = "torchmatch";
    accelerator = "cuda13_2";
  };
}
