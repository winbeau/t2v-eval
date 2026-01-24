// Evaluation metrics data types

export interface VideoMetric {
  video_id: string;
  group: string;
  prompt?: string;
  num_frames?: number;
  duration_sec?: number;
  clip_or_vqa_score?: number;
  vbench_temporal_score?: number;
  flicker_mean?: number;
  niqe_mean?: number;
  fps?: number;
  [key: string]: string | number | undefined;
}

export interface GroupSummary {
  group: string;
  n_videos: number;
  clip_or_vqa_score_mean?: number;
  clip_or_vqa_score_std?: number;
  vbench_temporal_score_mean?: number;
  vbench_temporal_score_std?: number;
  flicker_mean_mean?: number;
  flicker_mean_std?: number;
  niqe_mean_mean?: number;
  niqe_mean_std?: number;
  [key: string]: string | number | undefined;
}

export interface MetricConfig {
  key: string;
  label: string;
  shortLabel: string;
  direction: 'higher' | 'lower';  // higher is better or lower is better
  precision: number;
  unit?: string;
}

export interface TableConfig {
  title: string;
  caption: string;
  label: string;
  metrics: MetricConfig[];
  highlightBest: boolean;
  showStd: boolean;
  transpose: boolean;
}

export interface LatexTableOptions {
  format: 'booktabs' | 'standard';
  alignment: 'c' | 'l' | 'r';
  precision: number;
  highlightBest: boolean;
  showStd: boolean;
  caption: string;
  label: string;
}

export type DataSource = 'per_video' | 'group_summary';
