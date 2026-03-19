#!/usr/bin/env bash
set -euo pipefail

config=""
force=0
skip_groups=""
vbench_input=""
per_video_output=""
group_summary_output=""
experiment_output=""
profile_output=""

usage() {
  cat <<'EOF'
Usage:
  scripts/summarize_vbench_groups.sh --config CONFIG [--force]
  scripts/summarize_vbench_groups.sh --config CONFIG --skip-groups g2,g3 \
    --per-video-output FILE.csv \
    --group-summary-output FILE.csv \
    [--vbench-input FILE.csv] \
    [--experiment-output FILE.csv] \
    [--profile-output FILE.csv] \
    [--force]

Notes:
  - Group names must match YAML groups[].name exactly.
  - In subset mode (--skip-groups), explicit output file names are required.
  - File-name arguments must be bare CSV file names under paths.output_dir.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      [[ $# -ge 2 ]] || { echo "Missing value for --config" >&2; exit 2; }
      config="$2"
      shift 2
      ;;
    --force)
      force=1
      shift
      ;;
    --skip-groups)
      [[ $# -ge 2 ]] || { echo "Missing value for --skip-groups" >&2; exit 2; }
      skip_groups="$2"
      shift 2
      ;;
    --vbench-input)
      [[ $# -ge 2 ]] || { echo "Missing value for --vbench-input" >&2; exit 2; }
      vbench_input="$2"
      shift 2
      ;;
    --per-video-output)
      [[ $# -ge 2 ]] || { echo "Missing value for --per-video-output" >&2; exit 2; }
      per_video_output="$2"
      shift 2
      ;;
    --group-summary-output)
      [[ $# -ge 2 ]] || { echo "Missing value for --group-summary-output" >&2; exit 2; }
      group_summary_output="$2"
      shift 2
      ;;
    --experiment-output)
      [[ $# -ge 2 ]] || { echo "Missing value for --experiment-output" >&2; exit 2; }
      experiment_output="$2"
      shift 2
      ;;
    --profile-output)
      [[ $# -ge 2 ]] || { echo "Missing value for --profile-output" >&2; exit 2; }
      profile_output="$2"
      shift 2
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

if [[ -n "$skip_groups" ]]; then
  [[ -n "$per_video_output" ]] || {
    echo "--skip-groups requires --per-video-output" >&2
    exit 2
  }
  [[ -n "$group_summary_output" ]] || {
    echo "--skip-groups requires --group-summary-output" >&2
    exit 2
  }
fi

cmd=(uv run python scripts/summarize.py --config "$config")

if [[ $force -eq 1 ]]; then
  cmd+=(--force)
fi
if [[ -n "$skip_groups" ]]; then
  cmd+=(--skip-groups "$skip_groups")
fi
if [[ -n "$vbench_input" ]]; then
  cmd+=(--vbench-input "$vbench_input")
fi
if [[ -n "$per_video_output" ]]; then
  cmd+=(--per-video-output "$per_video_output")
fi
if [[ -n "$group_summary_output" ]]; then
  cmd+=(--group-summary-output "$group_summary_output")
fi
if [[ -n "$experiment_output" ]]; then
  cmd+=(--experiment-output "$experiment_output")
fi
if [[ -n "$profile_output" ]]; then
  cmd+=(--profile-output "$profile_output")
fi

exec "${cmd[@]}"
