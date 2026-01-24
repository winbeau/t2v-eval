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

function formatCell(
  mean: number | undefined,
  std: number | undefined,
  precision: number,
  isBest: boolean
): string {
  if (mean === undefined || isNaN(mean)) return '-';

  let html = mean.toFixed(precision);

  if (props.showStd && std !== undefined && !isNaN(std)) {
    // Use KaTeX for ± symbol
    const latex = `${mean.toFixed(precision)} \\pm ${std.toFixed(precision)}`;
    try {
      html = katex.renderToString(latex, { throwOnError: false });
    } catch {
      html = `${mean.toFixed(precision)} ± ${std.toFixed(precision)}`;
    }
  }

  if (props.highlightBest && isBest) {
    return `<span class="metric-best">${html}</span>`;
  }

  return html;
}

function isBestValue(metricKey: string, value: number | undefined): boolean {
  if (value === undefined) return false;
  return value === bestValues.value.get(metricKey);
}

function getDirectionIcon(direction: 'higher' | 'lower'): string {
  return direction === 'higher' ? '↑' : '↓';
}
</script>

<template>
  <div class="table-preview">
    <table v-if="data.length > 0 && metrics.length > 0">
      <thead>
        <tr>
          <th>Method</th>
          <th v-for="metric in metrics" :key="metric.key">
            {{ metric.shortLabel }}
            <span
              class="text-xs ml-1"
              :class="metric.direction === 'higher' ? 'text-green-600' : 'text-red-500'"
            >
              {{ getDirectionIcon(metric.direction) }}
            </span>
          </th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in data" :key="row.group">
          <td class="font-medium text-left">{{ formatGroupName(row.group) }}</td>
          <td
            v-for="metric in metrics"
            :key="`${row.group}-${metric.key}`"
            v-html="formatCell(
              row[`${metric.key}_mean`] as number | undefined,
              row[`${metric.key}_std`] as number | undefined,
              metric.precision,
              isBestValue(metric.key, row[`${metric.key}_mean`] as number | undefined)
            )"
          />
        </tr>
      </tbody>
    </table>

    <div v-else class="empty-state">
      <p class="text-gray-400">No data to display</p>
      <p class="text-sm text-gray-300">Upload a CSV file to see the table preview</p>
    </div>
  </div>
</template>

<style scoped>
.table-preview {
  @apply bg-white rounded-lg shadow-sm border border-gray-200 overflow-x-auto;
}

.table-preview table {
  @apply w-full border-collapse min-w-max;
}

.table-preview th,
.table-preview td {
  @apply border border-gray-200 px-4 py-3 text-center;
}

.table-preview th {
  @apply bg-gray-50 font-semibold text-gray-700 text-sm;
}

.table-preview td {
  @apply text-sm;
}

.table-preview tbody tr:hover {
  @apply bg-blue-50;
}

.empty-state {
  @apply flex flex-col items-center justify-center py-12 text-center;
}

:deep(.metric-best) {
  @apply font-bold text-green-600;
}
</style>
