<script setup lang="ts">
import { computed } from 'vue';
import katex from 'katex';
import type { GroupSummary, MetricConfig } from '../types/metrics';
import { findBestValue, formatGroupName } from '../utils/latexGenerator';

interface Props {
  data: GroupSummary[];
  metrics: MetricConfig[];
  showStd: boolean;
  highlightBest: boolean;
}

const props = defineProps<Props>();

// Find best values for highlighting
const bestValues = computed(() => {
  const result = new Map<string, number | undefined>();
  for (const metric of props.metrics) {
    const values = props.data.map(
      (row) => row[`${metric.key}_mean`] as number | undefined
    );
    result.set(metric.key, findBestValue(values, metric.direction));
  }
  return result;
});

// Render header with KaTeX
function renderHeader(metric: MetricConfig): string {
  const arrow = metric.direction === 'higher' ? '\\uparrow' : '\\downarrow';
  const latex = `\\text{${metric.shortLabel}}\\,${arrow}`;
  try {
    return katex.renderToString(latex, { throwOnError: false });
  } catch {
    return `${metric.shortLabel} ${metric.direction === 'higher' ? '↑' : '↓'}`;
  }
}

// Format cell value with KaTeX
function formatCell(
  mean: number | undefined,
  std: number | undefined,
  precision: number,
  isBest: boolean
): string {
  if (mean === undefined || isNaN(mean)) {
    try {
      return katex.renderToString('\\text{--}', { throwOnError: false });
    } catch {
      return '--';
    }
  }

  let latex = '';
  const meanStr = mean.toFixed(precision);

  if (props.showStd && std !== undefined && !isNaN(std)) {
    const stdStr = std.toFixed(precision);
    if (isBest && props.highlightBest) {
      latex = `\\mathbf{${meanStr}}_{\\pm${stdStr}}`;
    } else {
      latex = `${meanStr}_{\\pm${stdStr}}`;
    }
  } else {
    if (isBest && props.highlightBest) {
      latex = `\\mathbf{${meanStr}}`;
    } else {
      latex = meanStr;
    }
  }

  try {
    return katex.renderToString(latex, { throwOnError: false });
  } catch {
    return isBest ? `<strong>${meanStr}</strong>` : meanStr;
  }
}

function isBestValue(metricKey: string, value: number | undefined): boolean {
  if (value === undefined) return false;
  return value === bestValues.value.get(metricKey);
}
</script>

<template>
  <div class="latex-table-container">
    <!-- Paper-style wrapper -->
    <div class="paper-background">
      <div v-if="data.length > 0 && metrics.length > 0" class="latex-table-wrapper">
        <table class="latex-table">
          <!-- Top rule (thick) -->
          <thead>
            <tr class="toprule">
              <th class="method-header">
                <span class="method-header-text">Method</span>
              </th>
              <th
                v-for="metric in metrics"
                :key="metric.key"
                class="metric-header"
                v-html="renderHeader(metric)"
              />
            </tr>
          </thead>
          <!-- Middle rule after header -->
          <tbody>
            <tr
              v-for="(row, index) in data"
              :key="row.group"
              :class="{ 'first-row': index === 0 }"
            >
              <td class="method-cell">
                <span class="method-text">{{ formatGroupName(row.group) }}</span>
              </td>
              <td
                v-for="metric in metrics"
                :key="`${row.group}-${metric.key}`"
                class="value-cell"
                v-html="formatCell(
                  row[`${metric.key}_mean`] as number | undefined,
                  row[`${metric.key}_std`] as number | undefined,
                  metric.precision,
                  isBestValue(metric.key, row[`${metric.key}_mean`] as number | undefined)
                )"
              />
            </tr>
          </tbody>
          <!-- Bottom rule (thick) - handled by CSS -->
        </table>

        <!-- Caption area (optional) -->
        <div class="table-caption">
          <span class="caption-label">Table 1:</span>
          <span class="caption-text">Quantitative comparison of different methods. Best results are highlighted in <strong>bold</strong>.</span>
        </div>
      </div>

      <div v-else class="empty-state">
        <div class="empty-icon">
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
        </div>
        <p class="empty-title">No data to display</p>
        <p class="empty-subtitle">Upload a CSV file to see the LaTeX table preview</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* Container with slight shadow for paper effect */
.latex-table-container {
  @apply w-full;
}

/* Paper-like background */
.paper-background {
  @apply bg-white rounded-lg shadow-md overflow-x-auto;
  background: linear-gradient(to bottom, #fefefe 0%, #f9f9f9 100%);
  border: 1px solid #e5e5e5;
}

.latex-table-wrapper {
  @apply px-4 pt-4 pb-2;
  display: inline-block;
  width: max-content;
  min-width: 100%;
}

/* Main table - booktabs style */
.latex-table {
  border-collapse: collapse;
  width: max-content;
  min-width: 100%;
  font-family: 'Times New Roman', Times, serif;
  font-size: 0.9rem;
  line-height: 1.15;
}

/* No vertical borders - booktabs style */
.latex-table th,
.latex-table td {
  @apply px-2 py-1;
  border: none;
  text-align: center;
  vertical-align: middle;
}

/* Header row with top rule */
.latex-table thead tr.toprule {
  border-top: 2px solid #000;
  border-bottom: 1px solid #000;
}

.latex-table thead th {
  @apply py-1.5;
  font-weight: normal; /* KaTeX handles bold */
  letter-spacing: 0.01em;
}

.method-header-text {
  font-weight: 700;
}

/* Method column left-aligned */
.latex-table .method-header,
.latex-table .method-cell {
  text-align: left;
  padding-left: 0.35rem;
}

/* First data row has midrule above it (via header border-bottom) */
.latex-table tbody tr.first-row td {
  padding-top: 0.35rem;
}

/* Bottom rule on last row */
.latex-table tbody tr:last-child {
  border-bottom: 2px solid #000;
}

.latex-table tbody tr:last-child td {
  padding-bottom: 0.35rem;
}

/* Value cells */
.latex-table .value-cell {
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}

/* Subtle row hover - very light to maintain paper feel */
.latex-table tbody tr:hover {
  background-color: rgba(0, 0, 0, 0.02);
}

/* KaTeX styling adjustments */
:deep(.katex) {
  font-size: 1em;
}

:deep(.katex .textbf) {
  font-weight: 700;
}

/* Subscript styling for ± std */
:deep(.katex .msupsub) {
  font-size: 0.85em;
}

/* Caption styling - academic paper style */
.table-caption {
  @apply mt-2 text-center;
  font-family: 'Times New Roman', Times, serif;
  font-size: 0.85rem;
  color: #333;
  white-space: nowrap;
}

.caption-label {
  font-weight: 600;
  margin-right: 0.5em;
}

.caption-text {
  color: #555;
}

/* Empty state */
.empty-state {
  @apply flex flex-col items-center justify-center py-16 text-center;
}

.empty-icon {
  @apply w-16 h-16 text-gray-300 mb-4;
}

.empty-icon svg {
  @apply w-full h-full;
}

.empty-title {
  @apply text-gray-500 font-medium mb-1;
  font-family: 'Times New Roman', Times, serif;
}

.empty-subtitle {
  @apply text-gray-400 text-sm;
}

/* Print-friendly styles */
@media print {
  .paper-background {
    box-shadow: none;
    border: none;
  }

  .latex-table-wrapper {
    padding: 0;
  }
}
</style>
