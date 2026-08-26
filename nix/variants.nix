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
}:
let
  p8n = tue-p8n.lib pkgs;
  pyproject = p8n.uv.readProject workspaceRoot;

  mkVariant =
    {
      accelerator,
      ...
    }:
    let
      accelConfig = p8n.config.build pkgs accelerator;
      stdenv' =
        if lib.hasPrefix "cuda12" accelerator then
          pkgs.gcc13Stdenv
        else
          accelConfig.stdenv;

      extraOverlay = _final: prev: {
        torchmatch = prev.torchmatch.overrideAttrs (old: {
          stdenv = stdenv';
          nativeBuildInputs =
            (old.nativeBuildInputs or [ ])
            ++ lib.optionals (accelConfig.acceleration != "none") [
              stdenv'.cc
              accelConfig.pkgs.cudaPackages.cudatoolkit
            ];
          env =
            (old.env or { })
            // (
              if accelConfig.acceleration == "none" then
                {
                  TORCHMATCH_BUILD_CPU = "1";
                  TORCHMATCH_BUILD_TRANSPORT = "1";
                }
              else
                {
                  TORCHMATCH_BUILD_CPU = "1";
                  TORCHMATCH_BUILD_CUDA = "1";
                  TORCHMATCH_BUILD_TRANSPORT = "1";
                  CUDA_HOME = "${accelConfig.pkgs.cudaPackages.cudatoolkit}";
                  CUDA_PATH = "${accelConfig.pkgs.cudaPackages.cudatoolkit}";
                  TORCH_CUDA_ARCH_LIST = "8.0;8.6;8.9;9.0";
                  CC = "${stdenv'.cc}/bin/gcc";
                  CXX = "${stdenv'.cc}/bin/g++";
                }
            );
        });
      };

      venv = pyproject.mkVenv {
        inherit accelerator;
        overlays = [ extraOverlay ];
      };
    in
    {
      pythonSet = venv.pythonSet;
      inherit venv;
      package = venv.pythonSet.torchmatch;
      cudaPkgs = if accelerator == "cpu" then null else accelConfig.pkgs.cudaPackages;
      hostStdenv = stdenv';
    };
in
{
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
