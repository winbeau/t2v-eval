<script setup lang="ts">
import { ref } from 'vue';
import FileUpload from './components/FileUpload.vue';
import TablePreview from './components/TablePreview.vue';
import CodeBlock from './components/CodeBlock.vue';
import MetricSelector from './components/MetricSelector.vue';
import OptionsPanel from './components/OptionsPanel.vue';
import { useMetricsData, useLatexGenerator } from './composables/useMetrics';

const {
  rawData,
  error,
  fileName,
  availableMetrics,
  selectedMetrics,
  loadFile,
  clearData,
} = useMetricsData();

const {
  options,
  latexCode,
} = useLatexGenerator(
  () => rawData.value,
  () => selectedMetrics.value
);

const activeTab = ref<'preview' | 'latex'>('preview');

async function handleFileSelected(files: File[]) {
  const file = files[0];
  if (file) {
    await loadFile(file);
  }
}

function handleClearData() {
  clearData();
}

// Load sample data for demo
async function loadSampleData() {
  const sampleData = [
    {
      group: 'frame_level_baseline',
      n_videos: 50,
      clip_or_vqa_score_mean: 28.45,
      clip_or_vqa_score_std: 3.21,
      vbench_temporal_score_mean: 0.752,
      vbench_temporal_score_std: 0.045,
      flicker_mean_mean: 0.0234,
      flicker_mean_std: 0.0089,
      niqe_mean_mean: 4.56,
      niqe_mean_std: 0.82,
    },
    {
      group: 'head_level_stable_w8',
      n_videos: 50,
      clip_or_vqa_score_mean: 29.12,
      clip_or_vqa_score_std: 2.98,
      vbench_temporal_score_mean: 0.789,
      vbench_temporal_score_std: 0.038,
      flicker_mean_mean: 0.0198,
      flicker_mean_std: 0.0076,
      niqe_mean_mean: 4.32,
      niqe_mean_std: 0.75,
    },
    {
      group: 'head_level_stable_w16',
      n_videos: 50,
      clip_or_vqa_score_mean: 29.87,
      clip_or_vqa_score_std: 2.65,
      vbench_temporal_score_mean: 0.812,
      vbench_temporal_score_std: 0.032,
      flicker_mean_mean: 0.0156,
      flicker_mean_std: 0.0062,
      niqe_mean_mean: 4.18,
      niqe_mean_std: 0.68,
    },
    {
      group: 'head_level_oscillate_w8',
      n_videos: 50,
      clip_or_vqa_score_mean: 28.76,
      clip_or_vqa_score_std: 3.12,
      vbench_temporal_score_mean: 0.768,
      vbench_temporal_score_std: 0.042,
      flicker_mean_mean: 0.0212,
      flicker_mean_std: 0.0081,
      niqe_mean_mean: 4.45,
      niqe_mean_std: 0.79,
    },
    {
      group: 'head_level_oscillate_w16',
      n_videos: 50,
      clip_or_vqa_score_mean: 29.34,
      clip_or_vqa_score_std: 2.87,
      vbench_temporal_score_mean: 0.795,
      vbench_temporal_score_std: 0.036,
      flicker_mean_mean: 0.0178,
      flicker_mean_std: 0.0069,
      niqe_mean_mean: 4.28,
      niqe_mean_std: 0.72,
    },
    {
      group: 'head_level_mixed',
      n_videos: 50,
      clip_or_vqa_score_mean: 30.21,
      clip_or_vqa_score_std: 2.54,
      vbench_temporal_score_mean: 0.825,
      vbench_temporal_score_std: 0.029,
      flicker_mean_mean: 0.0145,
      flicker_mean_std: 0.0058,
      niqe_mean_mean: 4.05,
      niqe_mean_std: 0.65,
    },
  ];

  rawData.value = sampleData;
  fileName.value = 'sample_data.csv';
}
</script>

<template>
  <div class="app-container">
    <!-- Header -->
    <header class="app-header">
      <div class="header-content">
        <div class="header-title">
          <h1 class="text-2xl font-bold text-gray-900">T2V-Eval</h1>
          <span class="text-sm text-gray-500">LaTeX Table Generator</span>
        </div>
        <div class="header-actions">
          <a
            href="https://github.com/YOUR_USERNAME/t2v-eval"
            target="_blank"
            class="github-link"
          >
            <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
              <path
                fill-rule="evenodd"
                d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
                clip-rule="evenodd"
              />
            </svg>
            GitHub
          </a>
        </div>
      </div>
    </header>

    <!-- Main Content -->
    <main class="app-main">
      <div class="main-grid">
        <!-- Left Panel: Upload & Options -->
        <aside class="left-panel">
          <!-- File Upload -->
          <section class="panel-section">
            <h2 class="section-title">Data Source</h2>
            <FileUpload @file-selected="handleFileSelected" />

            <div v-if="fileName" class="file-info">
              <div class="flex items-center justify-between">
                <span class="text-sm text-gray-600">
                  <span class="font-medium">{{ fileName }}</span>
                  <span class="text-gray-400 ml-2">({{ rawData.length }} groups)</span>
                </span>
                <button
                  class="text-xs text-red-500 hover:underline"
                  @click="handleClearData"
                >
                  Clear
                </button>
              </div>
            </div>

            <button
              v-if="!fileName"
              class="sample-btn"
              @click="loadSampleData"
            >
              Load Sample Data
            </button>

            <div v-if="error" class="error-message">
              {{ error }}
            </div>
          </section>

          <!-- Metric Selector -->
          <section class="panel-section">
            <h2 class="section-title">Metrics</h2>
            <MetricSelector
              v-model="selectedMetrics"
              :available-metrics="availableMetrics"
            />
          </section>

          <!-- Options -->
          <section class="panel-section">
            <OptionsPanel v-model="options" />
          </section>
        </aside>

        <!-- Right Panel: Preview & Code -->
        <div class="right-panel">
          <!-- Tab Navigation -->
          <div class="tab-nav">
            <button
              class="tab-btn"
              :class="{ active: activeTab === 'preview' }"
              @click="activeTab = 'preview'"
            >
              <svg
                class="w-4 h-4 mr-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                />
              </svg>
              Table Preview
            </button>
            <button
              class="tab-btn"
              :class="{ active: activeTab === 'latex' }"
              @click="activeTab = 'latex'"
            >
              <svg
                class="w-4 h-4 mr-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"
                />
              </svg>
              LaTeX Code
            </button>
          </div>

          <!-- Tab Content -->
          <div class="tab-content">
            <Transition name="fade" mode="out-in">
              <div v-if="activeTab === 'preview'" key="preview">
                <TablePreview
                  :data="rawData"
                  :metrics="selectedMetrics"
                  :show-std="options.showStd"
                  :highlight-best="options.highlightBest"
                />
              </div>
              <div v-else key="latex">
                <CodeBlock
                  v-if="latexCode"
                  :code="latexCode"
                  language="latex"
                />
                <div v-else class="empty-code">
                  <p class="text-gray-400">No LaTeX code generated</p>
                  <p class="text-sm text-gray-300">Upload data and select metrics first</p>
                </div>
              </div>
            </Transition>
          </div>

          <!-- Usage Tips -->
          <div class="usage-tips">
            <h3 class="text-sm font-semibold text-gray-700 mb-2">Usage Tips</h3>
            <ul class="text-xs text-gray-500 space-y-1">
              <li>• Upload <code class="bg-gray-100 px-1 rounded">group_summary.csv</code> from the evaluation output</li>
              <li>• Select which metrics to include in the table</li>
              <li>• Click the direction indicator (↑/↓) to toggle higher/lower is better</li>
              <li>• Use <code class="bg-gray-100 px-1 rounded">booktabs</code> format for publication-ready tables</li>
              <li>• Copy the LaTeX code and paste into your paper</li>
            </ul>
          </div>
        </div>
      </div>
    </main>

    <!-- Footer -->
    <footer class="app-footer">
      <p class="text-xs text-gray-400">
        T2V-Eval LaTeX Table Generator • Built with Vue 3 + TypeScript + KaTeX
      </p>
    </footer>
  </div>
</template>

<style scoped>
.app-container {
  @apply min-h-screen flex flex-col bg-gray-50;
}

.app-header {
  @apply bg-white border-b border-gray-200 sticky top-0 z-50;
}

.header-content {
  @apply max-w-7xl mx-auto px-4 py-4 flex items-center justify-between;
}

.header-title {
  @apply flex items-baseline gap-3;
}

.github-link {
  @apply flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900 transition-colors;
}

.app-main {
  @apply flex-1 max-w-7xl mx-auto w-full px-4 py-6;
}

.main-grid {
  @apply grid grid-cols-1 lg:grid-cols-3 gap-6;
}

.left-panel {
  @apply lg:col-span-1 space-y-6;
}

.right-panel {
  @apply lg:col-span-2 space-y-4;
}

.panel-section {
  @apply space-y-3;
}

.section-title {
  @apply text-sm font-semibold text-gray-700;
}

.file-info {
  @apply mt-3 p-3 bg-green-50 rounded-lg border border-green-200;
}

.sample-btn {
  @apply mt-3 w-full py-2 text-sm text-blue-600 border border-blue-200 rounded-lg hover:bg-blue-50 transition-colors;
}

.error-message {
  @apply mt-3 p-3 bg-red-50 text-red-600 text-sm rounded-lg border border-red-200;
}

.tab-nav {
  @apply flex gap-2 bg-white p-1 rounded-lg border border-gray-200;
}

.tab-btn {
  @apply flex-1 flex items-center justify-center px-4 py-2 text-sm font-medium rounded-md transition-colors duration-150;
}

.tab-btn:not(.active) {
  @apply text-gray-500 hover:text-gray-700 hover:bg-gray-50;
}

.tab-btn.active {
  @apply bg-blue-600 text-white;
}

.tab-content {
  @apply min-h-[400px];
}

.empty-code {
  @apply flex flex-col items-center justify-center h-64 bg-gray-100 rounded-lg;
}

.usage-tips {
  @apply bg-blue-50 rounded-lg p-4 border border-blue-100;
}

.app-footer {
  @apply bg-white border-t border-gray-200 py-4 text-center;
}

/* Transitions */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.15s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
