import { ref, computed } from 'vue';
import type { GroupSummary, MetricConfig, LatexTableOptions } from '../types/metrics';
import { loadCSV, parseGroupSummary, getAvailableMetrics } from '../utils/csvParser';
import { generateLatexTable, DEFAULT_METRICS } from '../utils/latexGenerator';

export function useMetricsData() {
  const rawData = ref<GroupSummary[]>([]);
  const isLoading = ref(false);
  const error = ref<string | null>(null);
  const fileName = ref<string | null>(null);

  const availableMetrics = computed(() => getAvailableMetrics(rawData.value));

  const selectedMetrics = ref<MetricConfig[]>([...DEFAULT_METRICS]);

  async function loadFile(file: File) {
    isLoading.value = true;
    error.value = null;
    fileName.value = file.name;

    try {
      const data = await loadCSV<Record<string, unknown>>(file);
      rawData.value = parseGroupSummary(data);
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to load CSV file';
      rawData.value = [];
    } finally {
      isLoading.value = false;
    }
  }

  function clearData() {
    rawData.value = [];
    fileName.value = null;
    error.value = null;
  }

  return {
    rawData,
    isLoading,
    error,
    fileName,
    availableMetrics,
    selectedMetrics,
    loadFile,
    clearData,
  };
}

export function useLatexGenerator(
  data: () => GroupSummary[],
  metrics: () => MetricConfig[]
) {
  const options = ref<LatexTableOptions>({
    format: 'booktabs',
    alignment: 'c',
    precision: 3,
    highlightBest: true,
    showStd: true,
    caption: 'Quantitative Comparison of Different Methods',
    label: 'tab:results',
  });

  const latexCode = computed(() => {
    const d = data();
    const m = metrics();
    if (d.length === 0 || m.length === 0) return '';
    return generateLatexTable(d, m, options.value);
  });

  function updateOptions(newOptions: Partial<LatexTableOptions>) {
    options.value = { ...options.value, ...newOptions };
  }

  return {
    options,
    latexCode,
    updateOptions,
  };
}

export function useClipboard() {
  const copied = ref(false);
  const copyError = ref<string | null>(null);

  async function copyToClipboard(text: string) {
    try {
      await navigator.clipboard.writeText(text);
      copied.value = true;
      copyError.value = null;

      // Reset after 2 seconds
      setTimeout(() => {
        copied.value = false;
      }, 2000);
    } catch (e) {
      copyError.value = 'Failed to copy to clipboard';
      copied.value = false;
    }
  }

  return {
    copied,
    copyError,
    copyToClipboard,
  };
}
