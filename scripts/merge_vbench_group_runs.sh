#!/usr/bin/env bash
set -euo pipefail

config=""
merged_output=""
group_summary_output=""
force=0

usage() {
  cat <<'EOF'
Usage:
  scripts/merge_vbench_group_runs.sh --config CONFIG [--force]
  scripts/merge_vbench_group_runs.sh --config CONFIG \
    [--merged-output FILE.csv] \
    [--group-summary-output FILE.csv] \
    [--force]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      [[ $# -ge 2 ]] || { echo "Missing value for --config" >&2; exit 2; }
      config="$2"
      shift 2
      ;;
    --merged-output)
      [[ $# -ge 2 ]] || { echo "Missing value for --merged-output" >&2; exit 2; }
      merged_output="$2"
      shift 2
      ;;
    --group-summary-output)
      [[ $# -ge 2 ]] || { echo "Missing value for --group-summary-output" >&2; exit 2; }
      group_summary_output="$2"
      shift 2
      ;;
    --force)
      force=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -n "$config" ]] || { echo "--config is required" >&2; usage >&2; exit 2; }

cmd=(uv run python scripts/merge_vbench_group_runs.py --config "$config")

if [[ -n "$merged_output" ]]; then
  cmd+=(--merged-output "$merged_output")
fi
if [[ -n "$group_summary_output" ]]; then
  cmd+=(--group-summary-output "$group_summary_output")
fi
if [[ $force -eq 1 ]]; then
  cmd+=(--force)
fi

exec "${cmd[@]}"
