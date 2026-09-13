# hac26 environment. One entry point for the install; the pipeline scripts
# (scripts/run_smoke_test.sh, scripts/run_remote_pipeline.sh) reuse the .venv this builds.
#
#   make venv                 # torch for whatever this machine has (see CUDA below)
#   make venv CUDA=12         # torch built against CUDA 12.x  (cu129 wheels)
#   make venv CUDA=13         # torch built against CUDA 13.x  (cu130 wheels)
#   make venv CUDA=cpu        # no CUDA; on macOS this is the MPS build
#   make toolchain            # build nvdiffrast and the CUDA toolkit it compiles against
#   make check                # what the venv actually has: python, torch, CUDA, nvdiffrast
#   make test | test-fast | smoke | pipeline | data
#
# Every variable below can be overridden on the command line:
#   make venv PY_VERSION=3.11 VENV=/scratch/hac26-venv EXTRAS=test
#
# `make venv` is a no-op once the venv is complete, and reinstalls by itself when
# pyproject.toml changes or when the CUDA choice no longer matches the installed torch.
#
# A venv remembers the torch build it was made with, so `make smoke` -- or anything else
# that reaches `make venv` without naming CUDA, as scripts/_venv_setup.sh does -- keeps it
# rather than re-deciding. Name CUDA=12, CUDA=13 or CUDA=auto to change it.

VENV       ?= .venv
PY_VERSION ?= 3.12
CUDA       ?= auto
EXTRAS     ?= test,toolchain
EDITABLE   ?= 1

PYTEST_ARGS ?=

PY   := $(VENV)/bin/python
MARK := $(VENV)/.hac26-deps
TORCH_MARK := $(VENV)/.hac26-torch
UV   := $(shell command -v uv 2>/dev/null)

# ---------------------------------------------------------------------------
# Which torch to install.
#
# CUDA=auto reads the driver rather than the toolkit: a CUDA 13 wheel needs a 580-series
# driver or newer, and the usual failure on a cluster is a new wheel against an older
# driver, which imports fine and then fails at the first kernel launch. No nvidia-smi
# means no GPU, so cpu. Pass CUDA=12 or CUDA=13 to decide it yourself; any other value is
# handed to the wheel index unchanged, so CUDA=cu126 pins an older line.
#
# Auto-detection only ever applies to a venv that has not been built yet. `deps` records the
# tag it installed in $(TORCH_MARK), and that record wins whenever CUDA is not named on this
# invocation: `make venv CUDA=12` followed by `make smoke` has to keep the cu129 torch that
# was asked for, and not quietly put the driver's preference back. Naming CUDA (on the
# command line or in the environment, including CUDA=auto) overrides the record and rewrites
# it; so does deleting the venv.
# ---------------------------------------------------------------------------
# The record carries the machine beside the wheel line, so a venv built on one instruction set
# and reached from another -- which a shared filesystem makes easy -- is rebuilt rather than
# reused. Only the wheel line is read back as the CUDA choice.
ifeq ($(origin CUDA),file)
TORCH_RECORDED := $(shell cat $(TORCH_MARK) 2>/dev/null)
ifeq ($(word 2,$(TORCH_RECORDED)),$(shell uname -m))
CUDA_RECORDED := $(word 1,$(TORCH_RECORDED))
endif
endif

ifneq ($(CUDA_RECORDED),)
TORCH_TAG := $(CUDA_RECORDED)
else

ifeq ($(CUDA),auto)
CUDA_RESOLVED := $(shell \
  if command -v nvidia-smi >/dev/null 2>&1; then \
    d=$$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
         | head -1 | cut -d. -f1); \
    if [ -n "$$d" ] && [ "$$d" -ge 580 ] 2>/dev/null; then echo 13; else echo 12; fi; \
  else echo cpu; fi)
else
CUDA_RESOLVED := $(CUDA)
endif

ifeq ($(CUDA_RESOLVED),12)
TORCH_TAG := cu129
else ifeq ($(CUDA_RESOLVED),13)
TORCH_TAG := cu130
else ifeq ($(CUDA_RESOLVED),cpu)
TORCH_TAG := cpu
else
TORCH_TAG := $(CUDA_RESOLVED)
endif

endif

# macOS has no cpu index of its own; the plain PyPI wheel is the Metal/MPS build.
ifeq ($(TORCH_TAG)/$(shell uname -s),cpu/Darwin)
TORCH_INDEX :=
else
TORCH_INDEX := https://download.pytorch.org/whl/$(TORCH_TAG)
endif
TORCH_INDEX_FLAG := $(if $(TORCH_INDEX),--index-url $(TORCH_INDEX),)

ifeq ($(shell uname -s),Darwin)
ifneq ($(TORCH_TAG),cpu)
$(error CUDA=$(CUDA) but this is macOS, which has no CUDA wheels. \
  Use CUDA=cpu here (the default; it is the MPS build) and CUDA=12 or CUDA=13 on the GPU box.)
endif
endif

# The pytorch index carries torch and nothing else, so torch is installed from it on its
# own and the rest of the dependencies come from PyPI afterwards.
ifeq ($(EDITABLE),1)
PROJECT_SPEC := -e '.[$(EXTRAS)]'
else
# A plain install writes no hac26.egg-info into the repo, which an editable install does
# and which fails on a Windows-mounted WSL path. Every script in scripts/ puts the repo
# root on sys.path itself, so the package need not be editable to work on.
PROJECT_SPEC := '.[$(EXTRAS)]'
endif

ifdef UV
PIP_INSTALL   := uv pip install --python $(PY)
PIP_UNINSTALL := uv pip uninstall --python $(PY)
else
PIP_INSTALL   := $(PY) -m pip install
PIP_UNINSTALL := $(PY) -m pip uninstall -y
endif

.DEFAULT_GOAL := help
.PHONY: help venv deps toolchain check test test-fast smoke pipeline data check-data \
        clean distclean

help:
	@awk '/^#/ {sub(/^# ?/, ""); print; next} {exit}' Makefile
	@echo ""
	@echo "this machine: CUDA=$(CUDA) -> torch $(TORCH_TAG)$(if $(TORCH_INDEX), from $(TORCH_INDEX), from PyPI)$(if $(CUDA_RECORDED), (the build $(VENV) was made with; name CUDA to change it),)"
	@echo "installer:    $(if $(UV),uv ($(UV)),pip)"

# ---------------------------------------------------------------------------
# The venv.
#
# The interpreter is chosen by version, not by whichever python3 is first on PATH: the
# project needs >= 3.10 (pyproject.toml) and cluster images still ship a 3.6 or 3.8 as
# python3, which produces a venv that fails much later and confusingly. uv, when it is
# installed, downloads a matching interpreter itself and is preferred for that reason.
# ---------------------------------------------------------------------------
$(PY):
	@set -e; \
	if [ -n "$(UV)" ]; then \
	  echo "==> uv venv --python $(PY_VERSION) $(VENV)"; \
	  uv venv --python $(PY_VERSION) $(VENV); \
	  exit 0; \
	fi; \
	tried=""; \
	for c in python$(PY_VERSION) python3.13 python3.12 python3.11 python3.10 python3 python; do \
	  p=$$(command -v $$c 2>/dev/null) || continue; \
	  "$$p" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' \
	    >/dev/null 2>&1 || continue; \
	  tried="$$tried $$c"; \
	  echo "==> $$p -m venv $(VENV)  ($$("$$p" --version 2>&1))"; \
	  mkdir -p $(dir $(VENV)); \
	  rm -rf $(VENV); \
	  if "$$p" -m venv --copies $(VENV) >/dev/null 2>&1 \
	     && $(PY) -m pip install -q --upgrade pip >/dev/null 2>&1; then \
	    exit 0; \
	  fi; \
	  echo "    ...that interpreter cannot make a usable venv here; trying the next" >&2; \
	  rm -rf $(VENV); \
	done; \
	echo "ERROR: found no Python able to create a venv for this project." >&2; \
	echo "       It needs >= 3.10 (pyproject.toml) with a working venv and pip." >&2; \
	echo "       python3 here is: $$(python3 --version 2>&1)" >&2; \
	echo "       tried:$${tried:- nothing suitable}" >&2; \
	echo "" >&2; \
	echo "  Install uv, which fetches a matching interpreter itself:" >&2; \
	echo "    curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; \
	echo "  then rerun make venv. Or install python and its venv module:" >&2; \
	echo "    sudo apt install python3.12 python3.12-venv   # Debian/Ubuntu" >&2; \
	exit 1

# Up to date is judged on what the venv actually contains, not on a record kept here: the
# dependencies are reinstalled when pyproject.toml is newer than the last install, and torch
# when the CUDA choice no longer matches the build that is installed. `make deps` reinstalls
# unconditionally.
venv: $(MARK)
	@set -e; \
	if [ "$$(cat $(MARK) 2>/dev/null)" != "$(EXTRAS)" ]; then \
	  echo "==> extras are now [$(EXTRAS)]"; \
	  $(MAKE) --no-print-directory deps; \
	fi
	@set -e; \
	have=$$($(PY) -c "import torch; v = torch.version.cuda; \
	                  print('cu' + v.replace('.', '') if v else 'cpu')" 2>/dev/null || true); \
	if [ "$$have" != "$(TORCH_TAG)" ]; then \
	  [ -z "$$have" ] || echo "==> torch in $(VENV) is $$have, wanted $(TORCH_TAG); replacing it"; \
	  $(MAKE) --no-print-directory deps TORCH_REPLACE=$${have:+1}; \
	  if [ -n "$$have" ] && [ -f $(VENV)/etc/nvdiffrast-env.sh ]; then \
	    echo "==> nvdiffrast here was compiled against the torch just replaced and will not"; \
	    echo "    import under the new one: it links libc10 and libtorch, whose symbols do"; \
	    echo "    not survive a version change."; \
	    if [ "$(TORCH_TAG)" = "cpu" ]; then \
	      echo "    A CPU torch cannot rebuild it; it stays broken until a CUDA torch is back."; \
	    else \
	      echo "==> rebuilding it against $(TORCH_TAG)"; \
	      $(MAKE) --no-print-directory toolchain \
	        || echo "==> that rebuild failed; nvdiffrast stays broken until \`make toolchain\` works"; \
	    fi; \
	  fi; \
	fi; \
	printf '%s\n' "$(TORCH_TAG) $(shell uname -m)" > $(TORCH_MARK)
	@echo "==> $(VENV) ready: $$($(PY) --version), torch $(TORCH_TAG), extras [$(EXTRAS)]"

$(MARK): pyproject.toml | $(PY)
	@$(MAKE) --no-print-directory deps

# TORCH_REPLACE=1 uninstalls torch first. Without it an installed torch of the wrong CUDA
# build satisfies "torch>=2.1" and the install is a silent no-op.
deps: | $(PY)
	@echo "==> torch $(TORCH_TAG)$(if $(TORCH_INDEX), from $(TORCH_INDEX), from PyPI)"
	@[ -z "$(TORCH_REPLACE)" ] || $(PIP_UNINSTALL) torch
	$(PIP_INSTALL) $(TORCH_INDEX_FLAG) "torch>=2.1"
	@printf '%s\n' "$(TORCH_TAG)" > $(TORCH_MARK)
	@echo "==> hac26 and its dependencies, from pyproject.toml"
	$(PIP_INSTALL) $(PROJECT_SPEC)
	@printf '%s\n' "$(EXTRAS)" > $(MARK)

# ---------------------------------------------------------------------------
# nvdiffrast, for the exact forward model. GPU boxes only; the tests and the smoke test
# run the pure-torch rasteriser instead and do not need this. The build wants ninja, which
# rides in on the toolchain extra (see pyproject.toml) that EXTRAS installs by default.
# ---------------------------------------------------------------------------
toolchain: venv
	@if [ "$(TORCH_TAG)" = "cpu" ]; then \
	  echo "ERROR: this venv has a CPU torch; nvdiffrast needs a CUDA build." >&2; \
	  echo "       make venv CUDA=12   (or CUDA=13), then make toolchain" >&2; \
	  exit 1; \
	fi
	VIRTUAL_ENV=$(abspath $(VENV)) PYTHON=$(abspath $(PY)) \
	  PATH="$(abspath $(VENV))/bin:$$PATH" scripts/setup_toolchain.sh

check: $(PY)
	@$(PY) -c "import sys, torch; \
	print('python     ', sys.version.split()[0], sys.executable); \
	print('torch      ', torch.__version__, '(cuda', torch.version.cuda or 'none', ')'); \
	print('cuda avail ', torch.cuda.is_available(), \
	      torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''); \
	print('mps avail  ', getattr(torch.backends,'mps',None) and torch.backends.mps.is_available()); \
	import platform; print('machine    ', platform.machine())"
	@$(PY) -c "import nvdiffrast; print('nvdiffrast  ok', nvdiffrast.__file__)" 2>/dev/null \
	  || echo "nvdiffrast  not importable (make toolchain, or run with HAC26_SOFTWARE_RASTER=1)"

# ---------------------------------------------------------------------------
# Everything below runs inside the venv above.
# ---------------------------------------------------------------------------
test: venv
	$(PY) -m pytest $(PYTEST_ARGS)

# Skips the extraction-scale tests; a few minutes rather than an hour.
test-fast: venv
	$(PY) -m pytest -m "not slow" $(PYTEST_ARGS)

data: venv
	$(PY) scripts/fetch_data.py

check-data: venv
	$(PY) scripts/check_data.py

smoke: venv
	VENV_DIR=$(abspath $(VENV)) scripts/run_smoke_test.sh

pipeline: venv
	VENV_DIR=$(abspath $(VENV)) scripts/run_remote_pipeline.sh

clean:
	rm -rf .pytest_cache hac26.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

distclean: clean
	rm -rf $(VENV)
