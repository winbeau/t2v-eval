<script setup lang="ts">
import { ref } from 'vue';

interface Props {
  accept?: string;
  multiple?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  accept: '.csv',
  multiple: false,
});

const emit = defineEmits<{
  (e: 'file-selected', files: File[]): void;
}>();

const isDragging = ref(false);
const fileInput = ref<HTMLInputElement | null>(null);

function handleDragOver(event: DragEvent) {
  event.preventDefault();
  isDragging.value = true;
}

function handleDragLeave() {
  isDragging.value = false;
}

function handleDrop(event: DragEvent) {
  event.preventDefault();
  isDragging.value = false;

  const files = event.dataTransfer?.files;
  if (files && files.length > 0) {
    emitFiles(files);
  }
}

function handleFileInput(event: Event) {
  const input = event.target as HTMLInputElement;
  if (input.files && input.files.length > 0) {
    emitFiles(input.files);
  }
}

function emitFiles(fileList: FileList) {
  const files = Array.from(fileList).filter((file): file is File =>
    file !== undefined && file.name.endsWith('.csv')
  );
  const firstFile = files[0];
  if (firstFile) {
    emit('file-selected', props.multiple ? files : [firstFile]);
  }
}

function triggerFileInput() {
  fileInput.value?.click();
}
</script>

<template>
  <div
    class="file-upload"
    :class="{ 'is-dragging': isDragging }"
    @dragover="handleDragOver"
    @dragleave="handleDragLeave"
    @drop="handleDrop"
    @click="triggerFileInput"
  >
    <input
      ref="fileInput"
      type="file"
      :accept="accept"
      :multiple="multiple"
      class="hidden"
      @change="handleFileInput"
    />

    <div class="upload-content">
      <svg
        class="upload-icon"
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path
          stroke-linecap="round"
          stroke-linejoin="round"
          stroke-width="2"
          d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
        />
      </svg>
      <p class="upload-text">
        <span class="font-semibold">Click to upload</span> or drag and drop
      </p>
      <p class="upload-hint">CSV files only (group_summary.csv)</p>
    </div>
  </div>
</template>

<style scoped>
.file-upload {
  @apply border-2 border-dashed border-gray-300 rounded-xl p-8 text-center cursor-pointer transition-all duration-200;
}

.file-upload:hover,
.file-upload.is-dragging {
  @apply border-blue-500 bg-blue-50;
}

.upload-content {
  @apply flex flex-col items-center gap-2;
}

.upload-icon {
  @apply w-12 h-12 text-gray-400;
}

.file-upload:hover .upload-icon,
.file-upload.is-dragging .upload-icon {
  @apply text-blue-500;
}

.upload-text {
  @apply text-sm text-gray-600;
}

.upload-hint {
  @apply text-xs text-gray-400;
}
</style>
