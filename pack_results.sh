#!/bin/bash
# Pack the small result files of ablation suites (or single runs) into one .tar.gz to download.
#
# Usage (on the server, no docker needed):
#   ./pack_results.sh <exp> <suite_id> [<suite_id> ...]      # every run of the suites
#   ./pack_results.sh <exp> --run <run_id> [<run_id> ...]      # single runs
# Examples:
#   ./pack_results.sh medium 20261113-1635
#   ./pack_results.sh full 20261113-1636
#
# Packed (paths relative to <output>/roadpp/<exp>/, so the archive unpacks into the same layout locally):
#   suites/<suite>/  (suite.json, the tables, the logs)
#   runs/<run>/run_info*.json and runs/<run>/step06_output/{paper_metrics.json, paper_events.csv,
#   test_results.json, state_metrics.json, state_compare.png}
# Not packed: models, datasets (*.pt), review videos, code snapshots.
# Output: <output>/roadpp/<exp>/packs/results_<exp>_<name>.tar.gz
#
# Unpack locally:
#   tar -xzf results_<exp>_<name>.tar.gz -C /Users/jing/PycharmProjects/CauVid/src/exp_roadpp/output/roadpp/<exp>/

set -e
OUTPUT_HOST="${CAUVID_ROADPP_OUTPUT_HOST:-/storage-01/ml-jsha/CauVid_output}"

if [[ $# -lt 2 ]]; then
    sed -n '2,20p' "$0"
    exit 1
fi
EXP="$1"; shift
EXP_DIR="$OUTPUT_HOST/roadpp/$EXP"
cd "$EXP_DIR"

RUNS=(); SUITES=(); MODE="suite"
for arg in "$@"; do
    if [[ "$arg" == "--run" ]]; then MODE="run"; continue; fi
    if [[ "$MODE" == "suite" ]]; then
        [[ -d "suites/$arg" ]] || { echo "No suite $EXP_DIR/suites/$arg"; exit 1; }
        SUITES+=("$arg")
        for r in runs/"${arg}"_*; do [[ -d "$r" ]] && RUNS+=("$(basename "$r")"); done
    else
        [[ -d "runs/$arg" ]] || { echo "No run $EXP_DIR/runs/$arg"; exit 1; }
        RUNS+=("$arg")
    fi
done

FILES=()
for s in "${SUITES[@]}"; do FILES+=("suites/$s"); done
for r in "${RUNS[@]}"; do
    for f in runs/"$r"/run_info*.json \
             runs/"$r"/step06_output/paper_metrics.json runs/"$r"/step06_output/paper_events.csv \
             runs/"$r"/step06_output/test_results.json runs/"$r"/step06_output/state_metrics.json \
             runs/"$r"/step06_output/state_compare.png; do
        [[ -f "$f" ]] && FILES+=("$f")
    done
done
if [[ ${#FILES[@]} -eq 0 ]]; then echo "Nothing to pack"; exit 1; fi

NAME="${SUITES[*]:-${RUNS[0]}}"; NAME="${NAME// /_}"
mkdir -p packs
OUT="packs/results_${EXP}_${NAME}.tar.gz"
tar -czf "$OUT" --exclude='*.pt' --exclude='review' --exclude='state_review' "${FILES[@]}"
echo "[pack] ${#RUNS[@]} runs, ${#FILES[@]} entries -> $EXP_DIR/$OUT ($(du -h "$OUT" | cut -f1))"
echo "[pack] unpack locally: tar -xzf $(basename "$OUT") -C <CauVid>/src/exp_roadpp/output/roadpp/$EXP/"
