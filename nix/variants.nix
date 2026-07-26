# Per-variant uv2nix python sets for torchmatch.
#
# Each variant pairs one `pytorch-{cpu,cu126,cu128,cu130}` index from
# pyproject.toml with the matching cudaPackages set (when applicable) and
# the host gcc stdenv that the upstream torch wheel will accept.
#
# Returns an attribute set keyed by variant name; each entry exposes:
#   pythonSet   - the uv2nix python package set (override scope already
#                 composed with the workspace overlay and any per-variant
#                 build-system overrides);
#   venv        - a virtual env derivation containing all locked deps for
#                 the variant + the torchmatch package itself;
#   package     - the torchmatch package derivation (built against the
#                 variant's torch and CUDA toolkit);
#   hostStdenv  - the gcc stdenv used to build the package;
#   cudaPkgs    - the cudaPackages_X_Y attrset, or null for the cpu variant.
{
  pkgs,
  lib,
  uv2nix,
  pyproject-nix,
  pyproject-build-systems,
  workspaceRoot,
}: let
  workspace = uv2nix.lib.workspace.loadWorkspace {inherit workspaceRoot;};

  python = pkgs.python313;

  mkVariant = {
    name,
    hostStdenv,
    cudaPkgs ? null,
  }: let
    # mkPyprojectOverlay must be variant-specific: pyproject.toml declares
    # `cpu` / `cu126` / `cu128` / `cu130` as mutually exclusive extras, and
    # uv2nix's conflict resolver needs `dependencies` to pick one. Wheel
    # preference matters for torch: building torch from source is not
    # feasible, and we want bit-identical match with what runtime users get
    # from `pip install torch`.
    overlay = workspace.mkPyprojectOverlay {
      sourcePreference = "wheel";
      dependencies = {torchmatch = [name];};
    };

    # Opt in to precompiled extensions for nix derivations (setup.py
    # defaults to JIT-only; BUILD_* vars are the explicit opt-in).
    # nvidia-smi probing belongs in the dev shell, not in a
    # host-independent Nix derivation.
    extensionEnv =
      if cudaPkgs == null
      then {
        TORCHMATCH_BUILD_CPU = "1";
        TORCHMATCH_BUILD_TRANSPORT = "1";
      }
      else {
        TORCHMATCH_BUILD_CPU = "1";
        TORCHMATCH_BUILD_CUDA = "1";
        TORCHMATCH_BUILD_TRANSPORT = "1";
        CUDA_HOME = "${cudaPkgs.cudatoolkit}";
        CUDA_PATH = "${cudaPkgs.cudatoolkit}";
        TORCH_CUDA_ARCH_LIST = "8.0;8.6;8.9;9.0";
      };

    # Threading `stdenv = hostStdenv` through callPackage is what actually
    # selects the compiler for the package build; setting it inside an
    # `overrideAttrs` would only add a derivation attribute and silently
    # leave `pkgs.stdenv` driving the build.
    pythonSet =
      (pkgs.callPackage pyproject-nix.build.packages {
        inherit python;
        stdenv = hostStdenv;
      })
      .overrideScope (
        lib.composeManyExtensions [
          pyproject-build-systems.overlays.wheel
          overlay
          (_final: prev: let
            cufileOverride = pkg:
              pkg.overrideAttrs (old: {
                # The cuFile wheel ships an optional RDMA transport
                # (libcufile_rdma.so.1) whose libmlx5 / librdmacm / libibverbs
                # deps live in rdma-core. RDMA is only needed for GPUDirect
                # Storage on InfiniBand fabrics; pulling rdma-core into every
                # CUDA closure for an unused transport is wasteful. Tell
                # auto-patchelf to skip those sonames.
                autoPatchelfIgnoreMissingDeps =
                  (old.autoPatchelfIgnoreMissingDeps or [])
                  ++ ["libmlx5.so.1" "librdmacm.so.1" "libibverbs.so.1"];
              });
            nvshmemOverride = pkg:
              pkg.overrideAttrs (old: {
                # nvshmem ships optional MPI/PMIx/UCX/libfabric/IB transports
                # alongside its core lib. Each transport is a separate plugin
                # `.so` that NVSHMEM dlopen()s only if the user picks that
                # bootstrap mode; the default in-process bootstrap needs none
                # of them. Skip the sonames so the core lib still patches.
                autoPatchelfIgnoreMissingDeps =
                  (old.autoPatchelfIgnoreMissingDeps or [])
                  ++ [
                    "libmpi.so.40"
                    "libpmix.so.2"
                    "libucs.so.0"
                    "libucp.so.0"
                    "libmlx5.so.1"
                    "liboshmem.so.40"
                    "libfabric.so.1"
                  ];
              });
          in
            lib.optionalAttrs (prev ? "nvidia-cufile-cu12") {
              "nvidia-cufile-cu12" = cufileOverride prev."nvidia-cufile-cu12";
            }
            // lib.optionalAttrs (prev ? "nvidia-cufile") {
              "nvidia-cufile" = cufileOverride prev."nvidia-cufile";
            }
            // lib.optionalAttrs (prev ? "nvidia-nvshmem-cu12") {
              "nvidia-nvshmem-cu12" = nvshmemOverride prev."nvidia-nvshmem-cu12";
            }
            // lib.optionalAttrs (prev ? "nvidia-nvshmem-cu13") {
              "nvidia-nvshmem-cu13" = nvshmemOverride prev."nvidia-nvshmem-cu13";
            })
          (final: prev: let
            # Inter-wheel CUDA library references: each nvidia-*-cuXX wheel
            # installs its `.so` under
            #   $out/lib/python3.13/site-packages/nvidia/<dir>/lib/
            # and torch (and the cusolver/cusparse wheels) dlopens them by
            # RPATH-relative path at runtime. That nested layout is deeper
            # than the `$pkg/lib` that auto-patchelf's `gatherLibraries`
            # exposes from buildInputs, so we both register each provider's
            # nested lib dir via `addAutoPatchelfSearchPath` in a preFixup
            # hook AND pass the providers as buildInputs so their store paths
            # become real runtime deps. `final` (not `prev`) makes cusolver
            # see the post-override cusparse.
            #
            # cu12 names everything `nvidia-<x>-cu12`. cu13 dropped the suffix
            # for most components and kept it for a handful (cudnn, cusparselt,
            # nccl, nvshmem). The per-CUDA-major provider map below records the
            # wheel name -> python module name (the `nvidia/<dir>/` directory)
            # for every wheel that ships a `.so` we may need to resolve against.
            #
            # If NVIDIA renames or adds a wheel in a future uv.lock, update these
            # maps. `applyOverrides` filters out any entry whose package is not
            # in `prev`, so dead entries silently no-op.
            # Maps wheel name -> nvidia/<dir>/ subdir name.
            cu12Providers = {
              "nvidia-cublas-cu12" = "cublas";
              "nvidia-cuda-cupti-cu12" = "cuda_cupti";
              "nvidia-cuda-nvrtc-cu12" = "cuda_nvrtc";
              "nvidia-cuda-runtime-cu12" = "cuda_runtime";
              "nvidia-cudnn-cu12" = "cudnn";
              "nvidia-cufft-cu12" = "cufft";
              "nvidia-cufile-cu12" = "cufile";
              "nvidia-curand-cu12" = "curand";
              "nvidia-cusolver-cu12" = "cusolver";
              "nvidia-cusparse-cu12" = "cusparse";
              "nvidia-cusparselt-cu12" = "cusparselt";
              "nvidia-nccl-cu12" = "nccl";
              "nvidia-nvjitlink-cu12" = "nvjitlink";
              "nvidia-nvshmem-cu12" = "nvshmem";
              "nvidia-nvtx-cu12" = "nvtx";
            };

            cu13Providers = {
              "nvidia-cublas" = "cublas";
              "nvidia-cuda-cupti" = "cuda_cupti";
              "nvidia-cuda-nvrtc" = "cuda_nvrtc";
              "nvidia-cuda-runtime" = "cuda_runtime";
              "nvidia-cudnn-cu13" = "cudnn";
              "nvidia-cufft" = "cufft";
              "nvidia-cufile" = "cufile";
              "nvidia-curand" = "curand";
              "nvidia-cusolver" = "cusolver";
              "nvidia-cusparse" = "cusparse";
              "nvidia-cusparselt-cu13" = "cusparselt";
              "nvidia-nccl-cu13" = "nccl";
              "nvidia-nvjitlink" = "nvjitlink";
              "nvidia-nvshmem-cu13" = "nvshmem";
              "nvidia-nvtx" = "nvtx";
            };

            # Detect which CUDA major is present in this variant's env by
            # probing a marker wheel that is uniquely-named per major. cudnn
            # is suffixed in both lines (cu12 and cu13) so it's a stable probe.
            # The CPU variant short-circuits to an empty map; for CUDA
            # variants a missing probe is a real regression (the wheel set
            # has changed and the provider maps are stale), so we throw.
            activeProviders =
              if cudaPkgs == null
              then {}
              else if prev ? "nvidia-cudnn-cu13"
              then cu13Providers
              else if prev ? "nvidia-cudnn-cu12"
              then cu12Providers
              else throw "torchmatch variant ${name}: cudaPkgs is set but neither nvidia-cudnn-cu12 nor nvidia-cudnn-cu13 is present in the python set. The CUDA-major detection probe has gone stale (likely a renamed wheel in uv.lock). Update cu12Providers / cu13Providers in nix/variants.nix.";

            # Map a list of subdir names back to wheel names against the active
            # provider map. Returns a list of pkgs from `final`.
            providersFor = subdirs: let
              byDir = lib.mapAttrs' (wheel: dir: lib.nameValuePair dir wheel) activeProviders;
            in
              map (d: final.${byDir.${d}}) subdirs;

            mkCrossPreFixup = subdirs: let
              byDir = lib.mapAttrs' (wheel: dir: lib.nameValuePair dir wheel) activeProviders;
            in
              lib.concatMapStringsSep "\n" (d: ''
                addAutoPatchelfSearchPath --no-recurse \
                  "${final.${byDir.${d}}}/lib/python3.13/site-packages/nvidia/${d}/lib"
              '')
              subdirs;

            crossHelper = {
              providerSubdirs,
              ignoreSonames ? [],
            }: pkg:
              pkg.overrideAttrs (old: {
                buildInputs = (old.buildInputs or []) ++ providersFor providerSubdirs;
                preFixup = (old.preFixup or "") + "\n" + mkCrossPreFixup providerSubdirs;
                autoPatchelfIgnoreMissingDeps =
                  (old.autoPatchelfIgnoreMissingDeps or [])
                  ++ ignoreSonames;
              });

            # Wheel-name -> override mapping. The override list is the same per
            # CUDA major (only the wheel name differs), so we generate both
            # variants up-front and let `applyOverrides` filter to the active set.

            # Subdirs sibling to cusolver / cusparse that they need at link time.
            cusolverNeeds = ["cublas" "cusparse" "nvjitlink"];
            cusparseNeeds = ["nvjitlink"];
            # torch dlopens libs from every nvidia wheel.
            torchNeeds = lib.attrValues activeProviders;

            # libcuda.so.1 is the userland NVIDIA driver. Never shipped in a
            # wheel; it lives on the host (system driver or nixGL) and is
            # dlopen()d at runtime. Skip at patch time.
            torchIgnore = ["libcuda.so.1"];

            cu12Cross = {
              "nvidia-cusolver-cu12" = crossHelper {providerSubdirs = cusolverNeeds;};
              "nvidia-cusparse-cu12" = crossHelper {providerSubdirs = cusparseNeeds;};
              torch = crossHelper {
                providerSubdirs = torchNeeds;
                ignoreSonames = torchIgnore;
              };
            };
            cu13Cross = {
              "nvidia-cusolver" = crossHelper {providerSubdirs = cusolverNeeds;};
              "nvidia-cusparse" = crossHelper {providerSubdirs = cusparseNeeds;};
              torch = crossHelper {
                providerSubdirs = torchNeeds;
                ignoreSonames = torchIgnore;
              };
            };

            activeCross =
              if cudaPkgs == null
              then {}
              else if prev ? "nvidia-cudnn-cu13"
              then cu13Cross
              else if prev ? "nvidia-cudnn-cu12"
              then cu12Cross
              else throw "torchmatch variant ${name}: cudaPkgs is set but neither nvidia-cudnn-cu12 nor nvidia-cudnn-cu13 is present in the python set. See activeProviders for the same diagnosis.";

            applyOverrides = overrides:
              lib.mapAttrs (n: f: f prev.${n})
              (lib.filterAttrs (n: _: prev ? ${n}) overrides);
          in
            applyOverrides activeCross)
          # torchmatch owns all C++ extensions; inject CUDA build env here.
          (_final: prev: {
            torchmatch = prev.torchmatch.overrideAttrs (old: {
              nativeBuildInputs =
                (old.nativeBuildInputs or [])
                ++ lib.optionals (cudaPkgs != null) [cudaPkgs.cudatoolkit];
              env = (old.env or {}) // extensionEnv;
            });
          })
        ]
      );

    # Merging the chosen conflict extra into torchmatch selects the correct
    # torch index (cpu / cu126 / cu128 / cu130).
    venv = pythonSet.mkVirtualEnv "torchmatch-${name}-venv" (
      workspace.deps.groups
      // {
        torchmatch = (workspace.deps.groups.torchmatch or []) ++ [name];
      }
    );
  in {
    inherit pythonSet venv hostStdenv cudaPkgs;
    package = pythonSet.torchmatch;
  };
in {
  cpu = mkVariant {
    name = "cpu";
    hostStdenv = pkgs.gcc14Stdenv;
  };
  cu126 = mkVariant {
    name = "cu126";
    hostStdenv = pkgs.gcc13Stdenv;
    cudaPkgs = pkgs.cudaPackages_12_6;
  };
  cu128 = mkVariant {
    name = "cu128";
    hostStdenv = pkgs.gcc13Stdenv;
    cudaPkgs = pkgs.cudaPackages_12_8;
  };
  cu130 = mkVariant {
    name = "cu130";
    hostStdenv = pkgs.gcc14Stdenv;
    cudaPkgs = pkgs.cudaPackages_13_0;
  };
  cu132 = mkVariant {
    name = "cu132";
    hostStdenv = pkgs.gcc14Stdenv;
    cudaPkgs = pkgs.cudaPackages_13_2;
  };
}
