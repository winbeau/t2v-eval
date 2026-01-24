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
