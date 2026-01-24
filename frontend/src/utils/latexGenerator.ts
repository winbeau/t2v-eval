import type { GroupSummary, MetricConfig, LatexTableOptions } from '../types/metrics';

// Default metric configurations
export const DEFAULT_METRICS: MetricConfig[] = [
  {
    key: 'clip_or_vqa_score',
    label: 'CLIPScore',  // Default to CLIP (change to VQAScore if using VQA mode)
    shortLabel: 'CLIP',
    direction: 'higher',
    precision: 2,
  },
  {
    key: 'vbench_temporal_score',
    label: 'VBench Temporal',
    shortLabel: 'VBench',
    direction: 'higher',
    precision: 3,
  },
  {
    key: 'flicker_mean',
    label: 'Flicker',
    shortLabel: 'Flicker',
    direction: 'lower',
    precision: 4,
  },
  {
    key: 'niqe_mean',
    label: 'NIQE',
    shortLabel: 'NIQE',
    direction: 'lower',
    precision: 2,
  },
];

export function findBestValue(
  values: (number | undefined)[],
  direction: 'higher' | 'lower'
): number | undefined {
  const validValues = values.filter((v): v is number => v !== undefined && !isNaN(v));
  if (validValues.length === 0) return undefined;

  return direction === 'higher'
    ? Math.max(...validValues)
    : Math.min(...validValues);
}

export function formatValue(
  value: number | undefined,
  precision: number,
  std?: number | undefined,
  showStd: boolean = true
): string {
  if (value === undefined || isNaN(value)) return '-';

  const formatted = value.toFixed(precision);

  if (showStd && std !== undefined && !isNaN(std)) {
    return `${formatted} \\pm ${std.toFixed(precision)}`;
  }

  return formatted;
}

export function generateLatexTable(
  data: GroupSummary[],
  metrics: MetricConfig[],
  options: LatexTableOptions
): string {
  const {
    format,
    alignment,
    highlightBest,
    showStd,
    caption,
    label,
  } = options;

  const lines: string[] = [];
  const useBooktabs = format === 'booktabs';

  // Find best values for each metric
  const bestValues: Map<string, number | undefined> = new Map();
  if (highlightBest) {
    for (const metric of metrics) {
      const values = data.map((row) => row[`${metric.key}_mean`] as number | undefined);
      bestValues.set(metric.key, findBestValue(values, metric.direction));
    }
  }

  // Table header
  lines.push('\\begin{table}[htbp]');
  lines.push('  \\centering');

  if (caption) {
    lines.push(`  \\caption{${caption}}`);
  }
  if (label) {
    lines.push(`  \\label{${label}}`);
  }

  // Column specification
  const colSpec = `l${metrics.map(() => alignment).join('')}`;
  lines.push(`  \\begin{tabular}{${colSpec}}`);

  if (useBooktabs) {
    lines.push('    \\toprule');
  } else {
    lines.push('    \\hline');
  }

  // Header row
  const headerCells = ['Method', ...metrics.map((m) => m.shortLabel)];
  lines.push(`    ${headerCells.join(' & ')} \\\\`);

  if (useBooktabs) {
    lines.push('    \\midrule');
  } else {
    lines.push('    \\hline');
  }

  // Data rows
  for (const row of data) {
    const cells: string[] = [formatGroupName(row.group)];

    for (const metric of metrics) {
      const meanKey = `${metric.key}_mean`;
      const stdKey = `${metric.key}_std`;
      const mean = row[meanKey] as number | undefined;
      const std = row[stdKey] as number | undefined;

      let cellValue = formatValue(mean, metric.precision, std, showStd);

      // Highlight best value
      if (highlightBest && mean !== undefined && mean === bestValues.get(metric.key)) {
        cellValue = `\\textbf{${cellValue}}`;
      }

      // Wrap in math mode if contains \\pm
      if (cellValue.includes('\\pm')) {
        cellValue = `$${cellValue}$`;
      }

      cells.push(cellValue);
    }

    lines.push(`    ${cells.join(' & ')} \\\\`);
  }

  if (useBooktabs) {
    lines.push('    \\bottomrule');
  } else {
    lines.push('    \\hline');
  }

  lines.push('  \\end{tabular}');
  lines.push('\\end{table}');

  return lines.join('\n');
}

export function formatGroupName(name: string): string {
  // Convert snake_case to readable format
  return name
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

export function generateSimpleLatexTable(
  data: GroupSummary[],
  metrics: MetricConfig[],
  showStd: boolean = true,
  highlightBest: boolean = true
): string {
  return generateLatexTable(data, metrics, {
    format: 'booktabs',
    alignment: 'c',
    precision: 3,
    highlightBest,
    showStd,
    caption: 'Evaluation Results',
    label: 'tab:results',
  });
}

// Generate LaTeX for metric direction annotations
export function generateMetricAnnotation(metrics: MetricConfig[]): string {
  const annotations = metrics.map((m) => {
    const arrow = m.direction === 'higher' ? '\\uparrow' : '\\downarrow';
    return `${m.shortLabel} ($${arrow}$)`;
  });

  return `Metrics: ${annotations.join(', ')}. $\\uparrow$ higher is better, $\\downarrow$ lower is better.`;
}
