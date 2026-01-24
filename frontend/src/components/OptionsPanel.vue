<script setup lang="ts">
import type { LatexTableOptions } from '../types/metrics';

interface Props {
  modelValue: LatexTableOptions;
}

const props = defineProps<Props>();

const emit = defineEmits<{
  (e: 'update:modelValue', value: LatexTableOptions): void;
}>();

function updateOption<K extends keyof LatexTableOptions>(
  key: K,
  value: LatexTableOptions[K]
) {
  emit('update:modelValue', { ...props.modelValue, [key]: value });
}
</script>

<template>
  <div class="options-panel">
    <h3 class="text-sm font-semibold text-gray-700 mb-4">LaTeX Options</h3>

    <div class="option-group">
      <label class="option-label">Table Format</label>
      <div class="flex gap-2">
        <button
          class="option-btn"
          :class="{ active: modelValue.format === 'booktabs' }"
          @click="updateOption('format', 'booktabs')"
        >
          Booktabs
        </button>
        <button
          class="option-btn"
          :class="{ active: modelValue.format === 'standard' }"
          @click="updateOption('format', 'standard')"
        >
          Standard
        </button>
      </div>
    </div>

    <div class="option-group">
      <label class="option-label">Column Alignment</label>
      <div class="flex gap-2">
        <button
          class="option-btn"
          :class="{ active: modelValue.alignment === 'l' }"
          @click="updateOption('alignment', 'l')"
        >
          Left
        </button>
        <button
          class="option-btn"
          :class="{ active: modelValue.alignment === 'c' }"
          @click="updateOption('alignment', 'c')"
        >
          Center
        </button>
        <button
          class="option-btn"
          :class="{ active: modelValue.alignment === 'r' }"
          @click="updateOption('alignment', 'r')"
        >
          Right
        </button>
      </div>
    </div>

    <div class="option-group">
      <label class="option-label">Precision (decimal places)</label>
      <input
        type="number"
        :value="modelValue.precision"
        min="0"
        max="6"
        class="option-input"
        @input="updateOption('precision', Number(($event.target as HTMLInputElement).value))"
      />
    </div>

    <div class="option-group">
      <label class="option-label">Caption</label>
      <input
        type="text"
        :value="modelValue.caption"
        class="option-input"
        placeholder="Table caption..."
        @input="updateOption('caption', ($event.target as HTMLInputElement).value)"
      />
    </div>

    <div class="option-group">
      <label class="option-label">Label</label>
      <input
        type="text"
        :value="modelValue.label"
        class="option-input"
        placeholder="tab:results"
        @input="updateOption('label', ($event.target as HTMLInputElement).value)"
      />
    </div>

    <div class="option-group">
      <div class="flex items-center gap-4">
        <label class="toggle-label">
          <input
            type="checkbox"
            :checked="modelValue.highlightBest"
            class="toggle-input"
            @change="updateOption('highlightBest', ($event.target as HTMLInputElement).checked)"
          />
          <span>Highlight Best</span>
        </label>

        <label class="toggle-label">
          <input
            type="checkbox"
            :checked="modelValue.showStd"
            class="toggle-input"
            @change="updateOption('showStd', ($event.target as HTMLInputElement).checked)"
          />
          <span>Show Std (±)</span>
        </label>
      </div>
    </div>
  </div>
</template>

<style scoped>
.options-panel {
  @apply bg-white rounded-lg border border-gray-200 p-4;
}

.option-group {
  @apply mb-4;
}

.option-group:last-child {
  @apply mb-0;
}

.option-label {
  @apply block text-xs font-medium text-gray-500 mb-2;
}

.option-btn {
  @apply px-3 py-1.5 text-xs font-medium rounded-md border border-gray-200 text-gray-600 transition-colors duration-150;
}

.option-btn:hover {
  @apply bg-gray-50;
}

.option-btn.active {
  @apply bg-blue-600 text-white border-blue-600;
}

.option-input {
  @apply w-full px-3 py-2 text-sm border border-gray-200 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent;
}

.toggle-label {
  @apply flex items-center gap-2 text-sm text-gray-600 cursor-pointer;
}

.toggle-input {
  @apply w-4 h-4 text-blue-600 rounded focus:ring-blue-500;
}
</style>
