#!/usr/bin/env bash
#
# Build everything needed to run Nebula: the simulator (CPU and, if CUDA is
# available, GPU versions), ELSEPA, and a Python environment with cstool for
# generating material files.
#
# Run ./build.sh --help for options.

set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUILD_DIR="build"
BUILD_TYPE="Release"
JOBS="$(nproc 2>/dev/null || echo 4)"
CUDA_ARCH=""
ENABLE_CUDA=1
CUDA_HOME_ARG=""
ELSEPA_ZIP=""
BUILD_PYTHON=1
MATERIALS=""

usage() {
	cat <<EOF
Usage: $0 [options]

Options:
  --build-dir DIR      Build folder (default: build)
  --build-type TYPE    CMake build type (default: Release)
  -j, --jobs N         Parallel build jobs (default: $(nproc 2>/dev/null || echo 4))
  --cuda-arch LIST     CUDA architectures, as for CMAKE_CUDA_ARCHITECTURES,
                       e.g. "80;90;120" or "native". Default: 75, 80, 86, 89,
                       90, 100 and 120 (as far as the CUDA compiler supports
                       them), plus PTX for the newest one.
  --cuda-home DIR      CUDA toolkit to use (default: nvcc on PATH, \$CUDA_HOME,
                       or the newest /usr/local/cuda*)
  --no-cuda            Build the CPU version only
  --elsepa-zip FILE    Use a local copy of elsepa.zip instead of downloading it
  --no-python          Do not create the Python environment with cstool
  --materials "A B"    After building, run cstool on third_party/cstool/data/
                       materials/A.yaml, B.yaml, ... and put the resulting
                       .mat files in BUILD_DIR/materials
  -h, --help           Show this help
EOF
}

die() { echo "error: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
	case "$1" in
		--build-dir)  BUILD_DIR="$2"; shift 2 ;;
		--build-type) BUILD_TYPE="$2"; shift 2 ;;
		-j|--jobs)    JOBS="$2"; shift 2 ;;
		--cuda-arch)  CUDA_ARCH="$2"; shift 2 ;;
		--cuda-home)  CUDA_HOME_ARG="$2"; shift 2 ;;
		--no-cuda)    ENABLE_CUDA=0; shift ;;
		--elsepa-zip) ELSEPA_ZIP="$(realpath "$2")"; shift 2 ;;
		--no-python)  BUILD_PYTHON=0; shift ;;
		--materials)  MATERIALS="$2"; shift 2 ;;
		-h|--help)    usage; exit 0 ;;
		*) usage >&2; die "unknown option $1" ;;
	esac
done

[[ -n "$MATERIALS" && $BUILD_PYTHON -eq 0 ]] && die "--materials needs the Python environment"

command -v cmake >/dev/null || die "cmake not found (version 3.18 or newer is required)"

mkdir -p "$BUILD_DIR"
BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"


# Locate the CUDA compiler
find_nvcc() {
	if [[ -n "$CUDA_HOME_ARG" ]]; then
		[[ -x "$CUDA_HOME_ARG/bin/nvcc" ]] || die "no nvcc in $CUDA_HOME_ARG/bin"
		echo "$CUDA_HOME_ARG/bin/nvcc"; return
	fi
	if command -v nvcc >/dev/null; then
		command -v nvcc; return
	fi
	if [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]]; then
		echo "$CUDA_HOME/bin/nvcc"; return
	fi
	local candidate
	candidate="$(ls -d /usr/local/cuda*/bin/nvcc 2>/dev/null | sort -V | tail -n 1 || true)"
	[[ -n "$candidate" ]] && echo "$candidate"
	return 0
}

CMAKE_ARGS=(-S "$SOURCE_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE")
if [[ $ENABLE_CUDA -eq 1 ]]; then
	NVCC="$(find_nvcc)"
	if [[ -n "$NVCC" ]]; then
		echo "Using CUDA compiler $NVCC"
		CMAKE_ARGS+=(-DNEBULA_ENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER="$NVCC")
	else
		echo "No CUDA compiler found: building the CPU version only"
		CMAKE_ARGS+=(-DNEBULA_ENABLE_CUDA=OFF)
	fi
	[[ -n "$CUDA_ARCH" ]] && CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
else
	CMAKE_ARGS+=(-DNEBULA_ENABLE_CUDA=OFF)
fi
if [[ -n "$ELSEPA_ZIP" ]]; then
	CMAKE_ARGS+=(-DELSEPA_ZIP="$ELSEPA_ZIP")
fi


# Build the C++/CUDA/Fortran parts
echo "==> Configuring"
cmake "${CMAKE_ARGS[@]}"
echo "==> Building"
cmake --build "$BUILD_DIR" -j "$JOBS"


# Python environment with cstool
VENV="$BUILD_DIR/venv"
if [[ $BUILD_PYTHON -eq 1 ]]; then
	echo "==> Setting up Python environment in $VENV"
	PYTHON="${PYTHON:-python3}"
	command -v "$PYTHON" >/dev/null || die "$PYTHON not found"
	if command -v uv >/dev/null; then
		[[ -x "$VENV/bin/python" ]] || uv venv --python "$(command -v "$PYTHON")" "$VENV"
		uv pip install --python "$VENV/bin/python" --reinstall-package cstool \
			"$SOURCE_DIR/third_party/cstool" numpy matplotlib
	else
		[[ -x "$VENV/bin/python" ]] || "$PYTHON" -m venv "$VENV" \
			|| die "could not create a virtual environment (is python3-venv installed?)"
		"$VENV/bin/python" -m pip install --upgrade pip
		"$VENV/bin/python" -m pip install "$SOURCE_DIR/third_party/cstool" numpy matplotlib
	fi
fi


# Environment script
ENV_FILE="$BUILD_DIR/nebula-env.sh"
{
	echo "# Source this file to put Nebula, cstool and ELSEPA on your PATH."
	echo "export PATH=\"$BUILD_DIR/bin\${PATH:+:\$PATH}\""
	[[ -d "$BUILD_DIR/elsepa" ]] && echo "export ELSEPA_DIR=\"$BUILD_DIR/elsepa\""
	[[ -x "$VENV/bin/python" ]] && echo "export PATH=\"$VENV/bin:\$PATH\""
} > "$ENV_FILE"


# Material files
if [[ -n "$MATERIALS" ]]; then
	[[ -x "$BUILD_DIR/elsepa/elscata" ]] || die "ELSEPA was not built; cannot run cstool"
	mkdir -p "$BUILD_DIR/materials"
	for m in $MATERIALS; do
		[[ -f "$BUILD_DIR/materials/$m.mat" ]] && { echo "==> $m.mat exists, skipping"; continue; }
		echo "==> Generating $m.mat (this takes several minutes)"
		# cstool writes <name>.mat to the working directory
		( source "$ENV_FILE" && cd "$BUILD_DIR/materials" \
			&& cstool "$SOURCE_DIR/third_party/cstool/data/materials/$m.yaml" )
	done
fi


echo
echo "==> Done"
echo "Executables:"
for f in "$BUILD_DIR"/bin/* "$BUILD_DIR"/elsepa/elscat?; do [[ -x "$f" ]] && echo "  $f"; done
[[ -x "$BUILD_DIR/bin/nebula_gpu" ]] && \
	echo "CUDA architectures: $(sed -n 's/^CMAKE_CUDA_ARCHITECTURES:[A-Z]*=//p' "$BUILD_DIR/CMakeCache.txt")"
[[ -n "$MATERIALS" ]] && echo "Material files: $BUILD_DIR/materials"
echo
echo "To use: source $ENV_FILE"
