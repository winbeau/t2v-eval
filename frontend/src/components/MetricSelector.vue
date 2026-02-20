<script setup lang="ts">
import { computed } from 'vue';
import type { MetricConfig } from '../types/metrics';
import { DEFAULT_METRICS } from '../utils/latexGenerator';

interface Props {
  modelValue: MetricConfig[];
  availableMetrics: string[];
}

const props = defineProps<Props>();

const emit = defineEmits<{
  (e: 'update:modelValue', value: MetricConfig[]): void;
}>();

const localMetrics = computed({
  get: () => props.modelValue,
  set: (value) => emit('update:modelValue', value),
});

// Create a lookup for default metrics
const defaultMetricMap = new Map(DEFAULT_METRICS.map((m) => [m.key, m]));

// All possible metrics with defaults or generated configs
const allMetrics = computed(() => {
  const result: MetricConfig[] = [];

  for (const key of props.availableMetrics) {
    if (defaultMetricMap.has(key)) {
      result.push(defaultMetricMap.get(key)!);
    } else {
      // Generate config for unknown metric
      result.push({
        key,
        label: formatMetricLabel(key),
        shortLabel: formatMetricShortLabel(key),
        direction: guessDirection(key),
        precision: 3,
      });
    }
  }

  return result;
});

function formatMetricLabel(key: string): string {
  return key
    .replace(/_mean$/, '')
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

function formatMetricShortLabel(key: string): string {
  return formatMetricLabel(key);
}

function guessDirection(key: string): 'higher' | 'lower' {
  const lowerBetter = ['flicker', 'niqe', 'brisque', 'error', 'loss', 'mse', 'mae'];
  const keyLower = key.toLowerCase();
  return lowerBetter.some((k) => keyLower.includes(k)) ? 'lower' : 'higher';
}

function isSelected(metric: MetricConfig): boolean {
  return localMetrics.value.some((m) => m.key === metric.key);
}

function toggleMetric(metric: MetricConfig) {
  if (isSelected(metric)) {
    localMetrics.value = localMetrics.value.filter((m) => m.key !== metric.key);
  } else {
    localMetrics.value = [...localMetrics.value, metric];
  }
}

function selectAll() {
  localMetrics.value = [...allMetrics.value];
}

function clearAll() {
  localMetrics.value = [];
}

function toggleDirection(metric: MetricConfig) {
  const index = localMetrics.value.findIndex((m) => m.key === metric.key);
  const current = localMetrics.value[index];
  if (index >= 0 && current) {
    const updated = [...localMetrics.value];
    updated[index] = {
      key: current.key,
      label: current.label,
      shortLabel: current.shortLabel,
      direction: current.direction === 'higher' ? 'lower' : 'higher',
      precision: current.precision,
      unit: current.unit,
    };
    localMetrics.value = updated;
  }
}
</script>

<template>
  <div class="metric-selector">
    <div class="selector-header">
      <h3 class="text-sm font-semibold text-gray-700">Select Metrics</h3>
      <div class="flex gap-2">
        <button class="text-xs text-blue-600 hover:underline" @click="selectAll">
          Select All
        </button>
        <button class="text-xs text-gray-500 hover:underline" @click="clearAll">
          Clear
        </button>
      </div>
    </div>

    <div class="metric-list">
      <div
        v-for="metric in allMetrics"
        :key="metric.key"
        class="metric-item"
        :class="{ selected: isSelected(metric) }"
        @click="toggleMetric(metric)"
      >
        <div class="metric-checkbox">
          <input
            type="checkbox"
            :checked="isSelected(metric)"
            class="w-4 h-4 text-blue-600 rounded"
            @click.stop="toggleMetric(metric)"
          />
        </div>
        <div class="metric-info">
          <span class="metric-label">{{ metric.label }}</span>
          <span
            class="metric-direction"
            :class="metric.direction === 'higher' ? 'text-green-600' : 'text-red-500'"
            @click.stop="toggleDirection(metric)"
            title="Click to toggle direction"
          >
            {{ metric.direction === 'higher' ? '↑ higher' : '↓ lower' }}
          </span>
        </div>
      </div>
    </div>

    <div v-if="availableMetrics.length === 0" class="empty-state">
      <p class="text-gray-400 text-sm">No metrics available</p>
      <p class="text-gray-300 text-xs">Upload a CSV file first</p>
    </div>
  </div>
</template>

<style scoped>
.metric-selector {
  @apply bg-white rounded-lg border border-gray-200 p-4;
}

.selector-header {
  @apply flex items-center justify-between mb-3 pb-2 border-b border-gray-100;
}

.metric-list {
  @apply space-y-2 max-h-64 overflow-y-auto;
}

.metric-item {
  @apply flex items-center gap-1.5 px-1 py-1.5 rounded-lg cursor-pointer transition-colors duration-150;
}

.metric-item:hover {
  @apply bg-gray-50;
}

.metric-item.selected {
  @apply bg-blue-50;
}

.metric-checkbox {
  @apply flex-shrink-0;
}

.metric-info {
  @apply flex items-center gap-1 flex-1 min-w-0;
}

.metric-label {
  @apply text-sm text-gray-700 whitespace-normal leading-5 flex-1 min-w-0;
}

.metric-direction {
  @apply text-xs font-medium cursor-pointer hover:underline whitespace-nowrap flex-shrink-0 ml-1;
}

.empty-state {
  @apply text-center py-4;
}
</style>
