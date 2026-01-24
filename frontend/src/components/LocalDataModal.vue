<script setup lang="ts">
import { ref, onMounted } from 'vue';

interface Props {
  visible: boolean;
}

const props = defineProps<Props>();

const emit = defineEmits<{
  (e: 'close'): void;
  (e: 'select', file: string): void;
}>();

const csvFiles = ref<string[]>([]);
const isLoading = ref(true);
const error = ref<string | null>(null);

// Fetch list of CSV files from public/data/
async function fetchCsvFiles() {
  isLoading.value = true;
  error.value = null;

  try {
    // Fetch the file list from a manifest file
    const response = await fetch('/data/manifest.json');
    if (response.ok) {
      const manifest = await response.json();
      csvFiles.value = manifest.files || [];
    } else {
      // If no manifest, show empty state
      csvFiles.value = [];
    }
  } catch (e) {
    // No manifest file exists
    csvFiles.value = [];
  } finally {
    isLoading.value = false;
  }
}

function handleSelect(file: string) {
  emit('select', file);
  emit('close');
}

function handleClose() {
  emit('close');
}

function handleBackdropClick(event: MouseEvent) {
  if (event.target === event.currentTarget) {
    handleClose();
  }
}

onMounted(() => {
  if (props.visible) {
    fetchCsvFiles();
  }
});

// Watch for visibility changes
import { watch } from 'vue';
watch(() => props.visible, (newVal) => {
  if (newVal) {
    fetchCsvFiles();
  }
});
</script>

<template>
  <Teleport to="body">
    <Transition name="modal">
      <div
        v-if="visible"
        class="modal-backdrop"
        @click="handleBackdropClick"
      >
        <div class="modal-content">
          <!-- Header -->
          <div class="modal-header">
            <h2 class="modal-title">Load Local Data</h2>
            <button class="close-btn" @click="handleClose">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          <!-- Body -->
          <div class="modal-body">
            <!-- Loading State -->
            <div v-if="isLoading" class="loading-state">
              <svg class="animate-spin h-8 w-8 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <span class="text-gray-500 mt-2">Loading files...</span>
            </div>

            <!-- Empty State -->
            <div v-else-if="csvFiles.length === 0" class="empty-state">
              <svg class="w-16 h-16 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 13h6m-3-3v6m-9 1V7a2 2 0 012-2h6l2 2h6a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2z" />
              </svg>
              <p class="text-gray-500 mt-4 text-center">No CSV files found</p>
              <p class="text-gray-400 text-sm mt-2 text-center">
                Add CSV files to <code class="bg-gray-100 px-2 py-1 rounded">frontend/public/data/</code>
              </p>
              <p class="text-gray-400 text-xs mt-2 text-center">
                Then create a <code class="bg-gray-100 px-1 rounded">manifest.json</code> with:<br/>
                <code class="bg-gray-100 px-2 py-1 rounded text-xs">{"files": ["file1.csv", "file2.csv"]}</code>
              </p>
            </div>

            <!-- File List -->
            <div v-else class="file-list">
              <button
                v-for="file in csvFiles"
                :key="file"
                class="file-item"
                @click="handleSelect(file)"
              >
                <svg class="w-5 h-5 text-green-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <span class="file-name">{{ file }}</span>
                <svg class="w-4 h-4 text-gray-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
                </svg>
              </button>
            </div>
          </div>

          <!-- Footer -->
          <div class="modal-footer">
            <p class="text-xs text-gray-400">
              Files loaded from <code class="bg-gray-100 px-1 rounded">public/data/</code>
            </p>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.modal-backdrop {
  @apply fixed inset-0 z-50 flex items-center justify-center;
  @apply bg-black/50 backdrop-blur-sm;
}

.modal-content {
  @apply bg-white rounded-xl shadow-2xl w-full max-w-md mx-4;
  @apply transform transition-all;
}

.modal-header {
  @apply flex items-center justify-between px-6 py-4 border-b border-gray-200;
}

.modal-title {
  @apply text-lg font-semibold text-gray-900;
}

.close-btn {
  @apply p-1 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-100 transition-colors;
}

.modal-body {
  @apply px-6 py-4 max-h-80 overflow-y-auto;
}

.loading-state {
  @apply flex flex-col items-center justify-center py-8;
}

.empty-state {
  @apply flex flex-col items-center justify-center py-8;
}

.file-list {
  @apply space-y-2;
}

.file-item {
  @apply w-full flex items-center gap-3 px-4 py-3 rounded-lg;
  @apply bg-gray-50 hover:bg-blue-50 border border-gray-200 hover:border-blue-300;
  @apply transition-all cursor-pointer text-left;
}

.file-item:hover {
  @apply shadow-sm;
}

.file-name {
  @apply flex-1 text-sm font-medium text-gray-700 truncate;
}

.modal-footer {
  @apply px-6 py-3 bg-gray-50 rounded-b-xl border-t border-gray-200;
}

/* Transitions */
.modal-enter-active,
.modal-leave-active {
  transition: all 0.2s ease;
}

.modal-enter-from,
.modal-leave-to {
  opacity: 0;
}

.modal-enter-from .modal-content,
.modal-leave-to .modal-content {
  transform: scale(0.95);
}
</style>
