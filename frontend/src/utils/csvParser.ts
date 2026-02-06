import Papa from 'papaparse';
import type { VideoMetric, GroupSummary } from '../types/metrics';

export async function loadCSV<T>(file: File): Promise<T[]> {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (results) => {
        if (results.errors.length > 0) {
          console.warn('CSV parsing warnings:', results.errors);
        }
        resolve(results.data as T[]);
      },
      error: (error: Error) => {
        reject(error);
      },
    });
  });
}

export async function loadCSVFromURL<T>(url: string): Promise<T[]> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch CSV: ${response.statusText}`);
  }
  const text = await response.text();

  return new Promise((resolve, reject) => {
    Papa.parse(text, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (results) => {
        resolve(results.data as T[]);
      },
      error: (error: Error) => {
        reject(error);
      },
    });
  });
}

export function parseVideoMetrics(data: unknown[]): VideoMetric[] {
  return data.map((row) => {
    const r = row as Record<string, unknown>;
    return {
      video_id: String(r.video_id || ''),
      group: String(r.group || ''),
      prompt: r.prompt ? String(r.prompt) : undefined,
      num_frames: typeof r.num_frames === 'number' ? r.num_frames : undefined,
      duration_sec: typeof r.duration_sec === 'number' ? r.duration_sec : undefined,
      clip_or_vqa_score: typeof r.clip_or_vqa_score === 'number' ? r.clip_or_vqa_score : undefined,
      vbench_temporal_score: typeof r.vbench_temporal_score === 'number' ? r.vbench_temporal_score : undefined,
      flicker_mean: typeof r.flicker_mean === 'number' ? r.flicker_mean : undefined,
      niqe_mean: typeof r.niqe_mean === 'number' ? r.niqe_mean : undefined,
      fps: typeof r.fps === 'number' ? r.fps : undefined,
    };
  });
}

export function parseGroupSummary(data: unknown[]): GroupSummary[] {
  return data.map((row) => {
    const r = row as Record<string, unknown>;
    const result: GroupSummary = {
      group: String(r.group || ''),
      n_videos: typeof r.n_videos === 'number' ? r.n_videos : 0,
    };

    // Dynamically add all _mean and _std columns
    for (const [key, value] of Object.entries(r)) {
      if ((key.endsWith('_mean') || key.endsWith('_std')) && typeof value === 'number') {
        result[key] = value;
      }
    }

    return result;
  });
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function aggregatePerVideoToGroupSummary(data: unknown[]): GroupSummary[] {
  type GroupAccumulator = {
    nVideos: number;
    sums: Record<string, number>;
    sumsSq: Record<string, number>;
    counts: Record<string, number>;
  };

  const byGroup = new Map<string, GroupAccumulator>();
  const ignoredColumns = new Set(['group', 'video_id', 'video_path', 'prompt', 'n_videos']);

  for (const row of data) {
    const r = row as Record<string, unknown>;
    const group = String(r.group || '').trim();
    if (!group) continue;

    if (!byGroup.has(group)) {
      byGroup.set(group, {
        nVideos: 0,
        sums: {},
        sumsSq: {},
        counts: {},
      });
    }

    const acc = byGroup.get(group)!;
    acc.nVideos += 1;

    for (const [key, value] of Object.entries(r)) {
      if (ignoredColumns.has(key)) continue;
      if (!isFiniteNumber(value)) continue;

      acc.sums[key] = (acc.sums[key] || 0) + value;
      acc.sumsSq[key] = (acc.sumsSq[key] || 0) + value * value;
      acc.counts[key] = (acc.counts[key] || 0) + 1;
    }
  }

  const results: GroupSummary[] = [];
  for (const [group, acc] of byGroup.entries()) {
    const summary: GroupSummary = {
      group,
      n_videos: acc.nVideos,
    };

    for (const metric of Object.keys(acc.counts)) {
      const count = acc.counts[metric];
      if (!count) continue;

      const mean = acc.sums[metric] / count;
      const variance = Math.max(0, acc.sumsSq[metric] / count - mean * mean);
      const std = Math.sqrt(variance);

      summary[`${metric}_mean`] = mean;
      summary[`${metric}_std`] = std;
    }

    results.push(summary);
  }

  return results;
}

export function parseCsvAsGroupSummary(data: unknown[]): GroupSummary[] {
  if (data.length === 0) return [];

  const sample = data[0] as Record<string, unknown>;
  const columns = Object.keys(sample);
  const hasMeanColumns = columns.some((c) => c.endsWith('_mean'));
  const hasGroupColumn = columns.includes('group');

  if (hasMeanColumns) {
    return parseGroupSummary(data).filter((row) => row.group.trim().length > 0);
  }

  if (hasGroupColumn) {
    return aggregatePerVideoToGroupSummary(data);
  }

  return parseGroupSummary(data).filter((row) => row.group.trim().length > 0);
}

export function getAvailableMetrics(data: GroupSummary[]): string[] {
  if (data.length === 0) return [];

  const metrics = new Set<string>();
  const sample = data[0];

  if (sample) {
    for (const key of Object.keys(sample)) {
      if (key.endsWith('_mean')) {
        const baseName = key.replace('_mean', '');
        metrics.add(baseName);
      }
    }
  }

  return Array.from(metrics);
}
