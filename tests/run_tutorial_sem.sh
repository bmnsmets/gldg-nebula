#!/usr/bin/env bash
#
# Run the SEM tutorial (https://nebula-simulator.github.io/tutorial-sem/) with
# the binaries from ./build.sh, and check the results.
#
# Usage: tests/run_tutorial_sem.sh [options] WORKDIR
#
# Options:
#   --build-dir DIR   Build folder of ./build.sh (default: build)
#   --mat-dir DIR     Folder with existing silicon.mat and pmma.mat; if not
#                     given, they are generated with cstool in WORKDIR
#   --no-gpu          Do not run nebula_gpu
#   --no-cpu          Do not run nebula_cpu_mt
#   --seed N          Random seed for Nebula (default: 1)

set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$SOURCE_DIR/build"
MAT_DIR=""
RUN_GPU=1
RUN_CPU=1
SEED=1
WORKDIR=""

while [[ $# -gt 0 ]]; do
	case "$1" in
		--build-dir) BUILD_DIR="$(realpath "$2")"; shift 2 ;;
		--mat-dir)   MAT_DIR="$(realpath "$2")"; shift 2 ;;
		--no-gpu)    RUN_GPU=0; shift ;;
		--no-cpu)    RUN_CPU=0; shift ;;
		--seed)      SEED="$2"; shift 2 ;;
		-h|--help)   sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
		*)           WORKDIR="$1"; shift ;;
	esac
done
[[ -n "$WORKDIR" ]] || { echo "error: no WORKDIR given" >&2; exit 2; }

source "$BUILD_DIR/nebula-env.sh"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

cp "$SOURCE_DIR"/tutorial-sem/{sem.tri,sem-pri.py,sem-analysis.py} .

# Material files
for m in silicon pmma; do
	if [[ -n "$MAT_DIR" ]]; then
		cp "$MAT_DIR/$m.mat" .
	elif [[ ! -f "$m.mat" ]]; then
		echo "==> Generating $m.mat with cstool"
		cstool "$SOURCE_DIR/third_party/cstool/data/materials/$m.yaml"
	fi
done

echo "==> Generating primary electrons"
python sem-pri.py

DET_FILES=()
run_nebula() {
	local exe="$1" out="$2"
	shift 2
	echo "==> Running $exe"
	local start=$SECONDS
	"$exe" --seed="$SEED" "$@" sem.tri sem.pri silicon.mat pmma.mat > "$out"
	echo "    $exe took $((SECONDS - start)) s"
	DET_FILES+=("$out")
}
[[ $RUN_GPU -eq 1 ]] && run_nebula nebula_gpu output-gpu.det
[[ $RUN_CPU -eq 1 ]] && run_nebula nebula_cpu_mt output-cpu.det

echo "==> Running the tutorial analysis script"
MPLBACKEND=Agg python sem-analysis.py "${DET_FILES[0]}"

echo "==> Checking results"
python "$SOURCE_DIR/tests/check_sem_output.py" sem.pri "${DET_FILES[@]}"
