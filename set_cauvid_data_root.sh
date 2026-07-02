#!/usr/bin/env bash

# Source this file to point Docker and local Python runs at a remote data root.
#
# Usage:
#   source ./set_cauvid_data_root.sh /storage-01/ml-jsha/CauVid_Data
#   source ./set_cauvid_data_root.sh /storage-02/ml-jsha
#   source ./set_cauvid_data_root.sh DATA_ROOT OUTPUT_ROOT
#
# Notes:
#   - This file must be sourced, not executed, so the exports stay in your shell.
#   - If DATA_ROOT ends with CauVid_Data, OUTPUT_ROOT defaults to sibling CauVid_output.
#     Otherwise OUTPUT_ROOT defaults to DATA_ROOT.

_cauvid_data_root_usage() {
    cat <<'EOF'
Usage:
  source ./set_cauvid_data_root.sh DATA_ROOT [OUTPUT_ROOT]

Examples:
  source ./set_cauvid_data_root.sh /storage-01/ml-jsha/CauVid_Data
  source ./set_cauvid_data_root.sh /storage-02/ml-jsha
  source ./set_cauvid_data_root.sh /storage-01/ml-jsha/CauVid_Data /storage-01/ml-jsha/CauVid_output
EOF
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    _cauvid_data_root_usage
    echo
    echo "Error: this script must be sourced:"
    echo "  source $0 DATA_ROOT [OUTPUT_ROOT]"
    exit 2
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
    _cauvid_data_root_usage
    return 2
fi

_cauvid_data_root="${1%/}"
if [[ -z "$_cauvid_data_root" ]]; then
    echo "Error: DATA_ROOT cannot be empty." >&2
    return 2
fi

if [[ $# -eq 2 ]]; then
    _cauvid_output_root="${2%/}"
elif [[ "$(basename "$_cauvid_data_root")" == "CauVid_Data" ]]; then
    _cauvid_output_root="$(dirname "$_cauvid_data_root")/CauVid_output"
else
    _cauvid_output_root="$_cauvid_data_root"
fi

export CAUVID_STORAGE_ROOT="$_cauvid_data_root"
export CAUVID_OUTPUT_ROOT="$_cauvid_output_root"

export CAUVID_RAW_DRIVING_DATASET="$CAUVID_STORAGE_ROOT/driving-video-with-object-tracking"
export CAUVID_DRIVING_MINI_HOST="$CAUVID_STORAGE_ROOT/driving_mini"
export CAUVID_NUSCENES_HOST="$CAUVID_STORAGE_ROOT/nuScenes"

export CAUVID_PIPELINE_OUTPUT_HOST="$CAUVID_OUTPUT_ROOT/pipeline_output"
export CAUVID_OUTPUT_HOST="$CAUVID_OUTPUT_ROOT/output"
export CAUVID_LOGS_HOST="$CAUVID_OUTPUT_ROOT/logs"
export CAUVID_TORCH_CACHE_HOST="$CAUVID_STORAGE_ROOT/.cache/torch"

export CAUVID_CONFIG_HOST="${CAUVID_CONFIG_HOST:-$PWD/configs}"

echo "CauVid data root configured:"
echo "  CAUVID_STORAGE_ROOT=$CAUVID_STORAGE_ROOT"
echo "  CAUVID_OUTPUT_ROOT=$CAUVID_OUTPUT_ROOT"
echo "  CAUVID_RAW_DRIVING_DATASET=$CAUVID_RAW_DRIVING_DATASET"
echo "  CAUVID_DRIVING_MINI_HOST=$CAUVID_DRIVING_MINI_HOST"
echo "  CAUVID_PIPELINE_OUTPUT_HOST=$CAUVID_PIPELINE_OUTPUT_HOST"
echo "  CAUVID_CONFIG_HOST=$CAUVID_CONFIG_HOST"

if [[ ! -d "$CAUVID_RAW_DRIVING_DATASET" ]]; then
    echo "Warning: raw dataset directory does not exist: $CAUVID_RAW_DRIVING_DATASET" >&2
fi

unset _cauvid_data_root
unset _cauvid_output_root
