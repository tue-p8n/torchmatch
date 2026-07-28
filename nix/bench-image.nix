# OCI image for running the benchmark suite on hosts where Nix itself is
# undesirable (e.g. shared HPC login/compute nodes) via Apptainer/Singularity
# or plain Docker/Podman instead. Built with `nix build .#bench-image`
# (streams a `docker load`-compatible tar to stdout -- see the usage note
# below), pushed to GHCR by .github/workflows/bench-image.yml, and pulled on
# the target host with no Nix involved there at all.
#
# libcuda.so.1 (the userland NVIDIA driver) is deliberately absent from the
# venv's closure -- see variants.nix's `torchIgnore` comment -- and is
# expected to come from the container runtime's GPU passthrough instead:
# Apptainer/Singularity's `--nv` flag, or Docker's `--gpus all` with the
# NVIDIA Container Toolkit. Neither needs any extra wiring on our side.
{
  pkgs,
  variant,
}: let
  testsDir = ../tests;

  # Triton's libcuda_dirs() shells out to `ldconfig` to find libcuda.so.1
  # unless TRITON_LIBCUDA_PATH is already set, and this image has no
  # ldconfig. Apptainer/Singularity's --nv bind-mounts the host driver
  # libraries into the fixed path /.singularity.d/libs regardless of where
  # they actually live on the host, so point Triton there when that
  # directory is present. Docker's --gpus all (NVIDIA Container Toolkit)
  # doesn't create /.singularity.d/libs and needs no such override -- it
  # puts libcuda.so.1 somewhere ldconfig-discoverable already.
  entrypoint = pkgs.writeShellScriptBin "torchmatch-bench-entrypoint" ''
    if [ -d /.singularity.d/libs ] && [ -z "''${TRITON_LIBCUDA_PATH:-}" ]; then
      export TRITON_LIBCUDA_PATH=/.singularity.d/libs
    fi
    exec ${variant.venv}/bin/python3.13 -m torchmatch.bench "$@"
  '';
in
  pkgs.dockerTools.streamLayeredImage {
    name = "ghcr.io/tue-p8n/torchmatch/bench";
    tag = "latest";

    contents = [variant.venv pkgs.bashInteractive pkgs.coreutils entrypoint];

    extraCommands = ''
      mkdir -p app/benchmarks/results
      cp -r ${testsDir} app/tests
    '';

    config = {
      WorkingDir = "/app";
      Entrypoint = ["${entrypoint}/bin/torchmatch-bench-entrypoint"];
      # `--results-root`/`--tests-dir` both default to cwd-relative paths
      # (see sources/torchmatch/bench/__main__.py), so a plain bind-mount of
      # the host's benchmarks/results/ over /app/benchmarks/results is all a
      # caller needs -- no extra flags.
    };
  }
