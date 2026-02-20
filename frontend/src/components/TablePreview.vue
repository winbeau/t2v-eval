<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import katex from 'katex';
import { toPng } from 'html-to-image';
import type { GroupSummary, MetricConfig } from '../types/metrics';
import { findBestValue, formatGroupName } from '../utils/latexGenerator';

interface Props {
  data: GroupSummary[];
  metrics: MetricConfig[];
  showStd: boolean;
  highlightBest: boolean;
}

const props = defineProps<Props>();

const containerRef = ref<HTMLElement | null>(null);
const exportRef = ref<HTMLElement | null>(null);
const isDownloading = ref(false);
const containerWidth = ref(0);
const resizeObserver = ref<ResizeObserver | null>(null);

const METHOD_COLUMN_WIDTH = 220;
const METRIC_COLUMN_MIN_WIDTH = 132;
const TABLE_HORIZONTAL_PADDING = 48;
const DEFAULT_METRICS_PER_BLOCK = 5;

function updateContainerWidth() {
  containerWidth.value = containerRef.value?.clientWidth ?? 0;
}

onMounted(() => {
  updateContainerWidth();

  if (containerRef.value && typeof ResizeObserver !== 'undefined') {
    resizeObserver.value = new ResizeObserver(() => {
      updateContainerWidth();
    });
    resizeObserver.value.observe(containerRef.value);
  }

  window.addEventListener('resize', updateContainerWidth);
});

onBeforeUnmount(() => {
  resizeObserver.value?.disconnect();
  window.removeEventListener('resize', updateContainerWidth);
});

const metricsPerBlock = computed(() => {
  if (props.metrics.length === 0) return 0;

  if (containerWidth.value <= 0) {
    return Math.min(DEFAULT_METRICS_PER_BLOCK, props.metrics.length);
  }

  const availableWidth = Math.max(
    containerWidth.value - METHOD_COLUMN_WIDTH - TABLE_HORIZONTAL_PADDING,
    METRIC_COLUMN_MIN_WIDTH
  );
  const fitCount = Math.floor(availableWidth / METRIC_COLUMN_MIN_WIDTH);
  const chunkSize = Math.max(1, fitCount);

  return Math.min(chunkSize, props.metrics.length);
});

const metricBlocks = computed(() => {
  if (props.metrics.length === 0) return [];

  const blocks: MetricConfig[][] = [];
  const chunkSize = metricsPerBlock.value || props.metrics.length;
  for (let i = 0; i < props.metrics.length; i += chunkSize) {
    blocks.push(props.metrics.slice(i, i + chunkSize));
  }

  return blocks;
});

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

function isBestValue(metricKey: string, value: number | undefined): boolean {
  if (value === undefined) return false;
  return value === bestValues.value.get(metricKey);
}

function formatCell(
  mean: number | undefined,
  std: number | undefined,
  precision: number,
  isBest: boolean
): string {
  if (mean === undefined || Number.isNaN(mean)) {
    try {
      return katex.renderToString('\\text{--}', { throwOnError: false });
    } catch {
      return '--';
    }
  }

  const meanStr = mean.toFixed(precision);
  let latex = '';

  if (props.showStd && std !== undefined && !Number.isNaN(std)) {
    const stdStr = std.toFixed(precision);
    latex =
      isBest && props.highlightBest
        ? `\\mathbf{${meanStr}}_{\\pm${stdStr}}`
        : `${meanStr}_{\\pm${stdStr}}`;
  } else {
    latex = isBest && props.highlightBest ? `\\mathbf{${meanStr}}` : meanStr;
  }

  try {
    return katex.renderToString(latex, { throwOnError: false });
  } catch {
    return isBest ? `<strong>${meanStr}</strong>` : meanStr;
  }
}

async function downloadAsPng() {
  if (!exportRef.value || isDownloading.value) return;

  isDownloading.value = true;
  try {
    const dataUrl = await toPng(exportRef.value, {
      backgroundColor: '#ffffff',
      pixelRatio: 2,
      cacheBust: true,
      filter: (node) => !node.classList?.contains('download-btn'),
    });

    const link = document.createElement('a');
    link.download = 'table_preview.png';
    link.href = dataUrl;
    link.click();
  } catch (error) {
    console.error('Failed to download PNG:', error);
  } finally {
    isDownloading.value = false;
  }
}
</script>

<template>
  <div class="latex-table-container" ref="containerRef">
    <button
      v-if="data.length > 0 && metrics.length > 0"
      class="download-btn"
      @click="downloadAsPng"
      :disabled="isDownloading"
      title="Download full preview as PNG"
    >
      <svg
        v-if="!isDownloading"
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        class="download-icon"
      >
        <path
          stroke-linecap="round"
          stroke-linejoin="round"
          stroke-width="2"
          d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
        />
      </svg>
      <svg
        v-else
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
        class="download-icon animate-spin"
      >
        <circle
          class="opacity-25"
          cx="12"
          cy="12"
          r="10"
          stroke="currentColor"
          stroke-width="4"
        ></circle>
        <path
          class="opacity-75"
          fill="currentColor"
          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
        ></path>
      </svg>
    </button>

    <div class="paper-background">
      <div
        v-if="data.length > 0 && metrics.length > 0"
        ref="exportRef"
        class="latex-table-wrapper"
      >
        <section
          v-for="(blockMetrics, blockIndex) in metricBlocks"
          :key="blockMetrics.map((m) => m.key).join('-')"
          class="metric-block"
        >
          <table class="latex-table">
            <thead>
              <tr class="toprule">
                <th class="method-header">
                  <span class="method-header-text">Method</span>
                </th>
                <th
                  v-for="metric in blockMetrics"
                  :key="metric.key"
                  class="metric-header"
                >
                  <span class="metric-title">
                    {{ `${metric.label}\u00A0${metric.direction === 'higher' ? '↑' : '↓'}` }}
                  </span>
                </th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="(row, rowIndex) in data"
                :key="`${blockIndex}-${row.group}`"
                :class="{ 'first-row': rowIndex === 0 }"
              >
                <td class="method-cell">
                  <span class="method-text">{{ formatGroupName(row.group) }}</span>
                </td>
                <td
                  v-for="metric in blockMetrics"
                  :key="`${row.group}-${metric.key}`"
                  class="value-cell"
                  v-html="
                    formatCell(
                      row[`${metric.key}_mean`] as number | undefined,
                      row[`${metric.key}_std`] as number | undefined,
                      metric.precision,
                      isBestValue(metric.key, row[`${metric.key}_mean`] as number | undefined)
                    )
                  "
                />
              </tr>
            </tbody>
          </table>

          <div
            v-if="blockIndex < metricBlocks.length - 1"
            class="block-separator"
            aria-hidden="true"
          ></div>
        </section>

        <div class="paper-caption">
          <span class="paper-caption-label">Table 1:</span>
          <span class="paper-caption-text">
            Quantitative comparison of different methods. Best results are
            highlighted in <strong>bold</strong>.
          </span>
        </div>
      </div>

      <div v-else class="empty-state">
        <div class="empty-icon">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="1.5"
              d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
            />
          </svg>
        </div>
        <p class="empty-title">No data to display</p>
        <p class="empty-subtitle">Upload a CSV file to see the LaTeX table preview</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.latex-table-container {
  @apply w-full relative;
}

.paper-background {
  @apply bg-white rounded-lg shadow-md;
  background: linear-gradient(to bottom, #fefefe 0%, #f9f9f9 100%);
  border: 1px solid #e5e5e5;
}

.download-btn {
  @apply absolute top-3 right-3 p-2 rounded-lg transition-all duration-200;
  @apply bg-gray-100 hover:bg-gray-200 text-gray-500 hover:text-gray-700;
  @apply opacity-0 z-10 shadow-sm;
  @apply disabled:opacity-50 disabled:cursor-not-allowed;
  pointer-events: none;
}

.latex-table-container:hover .download-btn,
.download-btn:focus-visible {
  @apply opacity-100;
  pointer-events: auto;
}

.download-icon {
  @apply w-5 h-5;
}

.animate-spin {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

.latex-table-wrapper {
  @apply px-4 pt-5 pb-3;
}

.metric-block + .metric-block {
  margin-top: 0.9rem;
}

.block-separator {
  margin-top: 0.6rem;
  border-top: 1px solid #cfcfcf;
  opacity: 0.8;
}

.latex-table {
  border-collapse: collapse;
  width: 100%;
  font-family: 'Times New Roman', Times, serif;
  font-size: 0.9rem;
  line-height: 1.2;
  table-layout: auto;
}

.latex-table th,
.latex-table td {
  padding: 0.22rem 0.34rem;
  border: none;
  text-align: center;
  vertical-align: middle;
}

.latex-table thead tr.toprule {
  border-top: 2px solid #000;
  border-bottom: 1px solid #000;
}

.latex-table thead th {
  @apply py-1.5;
  font-weight: 400;
  letter-spacing: 0.01em;
}

.method-header,
.method-cell {
  text-align: left;
  min-width: 12.5rem;
  width: 13.75rem;
  padding-left: 0.28rem;
  padding-right: 0.32rem;
}

.method-header-text {
  font-weight: 700;
}

.method-text {
  display: block;
  white-space: normal;
  word-break: normal;
  overflow-wrap: break-word;
  line-height: 1.16;
}

.metric-header {
  min-width: 6.8rem;
}

.metric-title {
  display: block;
  white-space: normal;
  word-break: normal;
  overflow-wrap: break-word;
  line-height: 1.15;
}

.latex-table tbody tr.first-row td {
  padding-top: 0.35rem;
}

.latex-table tbody tr:last-child {
  border-bottom: 2px solid #000;
}

.latex-table tbody tr:last-child td {
  padding-bottom: 0.35rem;
}

.value-cell {
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}

.latex-table tbody tr:hover {
  background-color: rgba(0, 0, 0, 0.02);
}

:deep(.katex) {
  font-size: 1em;
}

:deep(.katex .textbf) {
  font-weight: 700;
}

:deep(.katex .msupsub) {
  font-size: 0.85em;
}

.paper-caption {
  @apply mt-2 text-center;
  display: block;
  width: 100%;
  font-family: 'Times New Roman', Times, serif;
  font-size: 0.85rem;
  color: #333;
}

.paper-caption-label {
  font-weight: 600;
  margin-right: 0.5em;
}

.paper-caption-text {
  color: #555;
}

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
