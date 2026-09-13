# Sourced by run_smoke_test.sh and run_remote_pipeline.sh; not meant to be run directly.
# Activates the venv the Makefile builds, building it first if it is not there. Callers cd
# to the repo root before sourcing this.
#
# The venv itself -- which interpreter, which torch, which dependencies -- is the Makefile's
# job and is described there; this file only decides *where* it goes and activates it. Set
# CUDA=12 or CUDA=13 in the environment to choose the torch build, as for make. Nothing here
# names CUDA, so an existing venv keeps the torch it was built with: a pipeline run does not
# swap a cu129 torch, deliberately installed for the nvdiffrast build, for the driver's
# preferred cu130 behind your back.
#
# Location: REPO_ROOT/.venv is tried first. If creating it fails, as it can under WSL when
# the repo lives on a Windows-mounted drive (/mnt/c/...), whose filesystem may refuse to
# create symlinks or set the executable bit, the venv is created at ~/.venvs/<reponame> on
# WSL's own filesystem instead. That choice is recorded in .venv-external-path, a one-line
# file at the repo root, so later runs go straight there; delete the file to try the local
# .venv again.
#
# VENV_DIR, if set, overrides all of the above and is tried with no fallback.
#
# Safe to source repeatedly: `make venv` is a no-op once the venv is complete.

REPO_ROOT="$(pwd)"
VENV_MARKER="$REPO_ROOT/.venv-external-path"

if [ -n "${VENV_DIR:-}" ]; then
  _venv_target="$VENV_DIR"
elif [ -f "$VENV_MARKER" ]; then
  _venv_target="$(cat "$VENV_MARKER")"
else
  _venv_target="$REPO_ROOT/.venv"
fi

if ! command -v make >/dev/null 2>&1; then
  echo "ERROR: make is not installed, and it is what builds the venv (see the Makefile)." >&2
  echo "  Debian/Ubuntu: sudo apt install make" >&2
  exit 1
fi

# Builds the venv at $1 and installs into it. Returns non-zero, with a half-created
# directory removed, if that does not produce a working bin/activate.
_venv_build() {
  local dir="$1"
  if [ -d "$dir" ] && [ ! -f "$dir/bin/activate" ]; then
    rm -rf "$dir"
  fi
  mkdir -p "$(dirname "$dir")" 2>/dev/null || true
  if make -C "$REPO_ROOT" venv VENV="$dir" 2>&1 | sed 's/^/[venv] /' >&2 \
     && [ -f "$dir/bin/activate" ]; then
    return 0
  fi
  [ -f "$dir/bin/activate" ] || rm -rf "$dir"
  return 1
}

if ! _venv_build "$_venv_target"; then
  if [ -n "${VENV_DIR:-}" ] || [ -f "$VENV_MARKER" ]; then
    # an explicit override or a recorded fallback failed; nothing left to try
    echo "ERROR: could not build a working venv at $_venv_target. See the errors above;" >&2
    echo "       \`make venv\` on its own reproduces them." >&2
    exit 1
  fi
  _fallback="$HOME/.venvs/$(basename "$REPO_ROOT")"
  echo "[venv] could not build a working venv at $_venv_target." >&2
  echo "[venv] If this is WSL and the repo lives on a Windows-mounted drive (/mnt/c/...)," >&2
  echo "[venv] that filesystem can refuse to create symlinks or set the executable bit at" >&2
  echo "[venv] all, which nothing here can work around in place. Trying a venv on WSL's" >&2
  echo "[venv] own filesystem instead:" >&2
  echo "[venv]   $_fallback" >&2
  if _venv_build "$_fallback"; then
    _venv_target="$_fallback"
    echo "$_fallback" > "$VENV_MARKER"
    echo "[venv] using $_fallback from now on (recorded in .venv-external-path;" >&2
    echo "[venv] delete that file to make this try $REPO_ROOT/.venv again)" >&2
  else
    echo "ERROR: could not build a working venv at $_venv_target OR at $_fallback." >&2
    exit 1
  fi
fi

if [ -z "${VIRTUAL_ENV:-}" ] || [ "$VIRTUAL_ENV" != "$_venv_target" ]; then
  # shellcheck disable=SC1091
  source "$_venv_target/bin/activate"
fi

# nvdiffrast, when scripts/setup_toolchain.sh has built it, leaves the library path it needs
# here. Absent on a machine without it, where the callers fall back to the software raster.
if [ -f "$_venv_target/etc/nvdiffrast-env.sh" ]; then
  # shellcheck disable=SC1091
  source "$_venv_target/etc/nvdiffrast-env.sh"
fi

echo "[venv] active: $(command -v python) ($(python --version 2>&1))"
PY=python

# Kept for the one caller that installs an optional package on the fly (thingi10k, in
# run_remote_pipeline.sh's objects stage).
_ensure_pip() {
  if python -m pip --version >/dev/null 2>&1; then
    return 0
  fi
  python -m ensurepip --upgrade >/dev/null 2>&1 && return 0
  echo "ERROR: this venv has no pip and ensurepip could not install it." >&2
  exit 1
}
