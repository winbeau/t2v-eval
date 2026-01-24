<script setup lang="ts">
import { computed } from 'vue';
import { useClipboard } from '../composables/useMetrics';

interface Props {
  code: string;
  language?: string;
}

const props = withDefaults(defineProps<Props>(), {
  language: 'latex',
});

const { copied, copyToClipboard } = useClipboard();

const lineNumbers = computed(() => {
  const lines = props.code.split('\n');
  return lines.map((_, i) => i + 1);
});

async function handleCopy() {
  await copyToClipboard(props.code);
}
</script>

<template>
  <div class="code-block">
    <div class="code-header">
      <span class="language-tag">{{ language.toUpperCase() }}</span>
      <button
        class="copy-button"
        :class="{ 'copy-success': copied }"
        @click="handleCopy"
        :title="copied ? 'Copied!' : 'Copy to clipboard'"
      >
        <svg
          v-if="!copied"
          xmlns="http://www.w3.org/2000/svg"
          class="h-4 w-4"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
          />
        </svg>
        <svg
          v-else
          xmlns="http://www.w3.org/2000/svg"
          class="h-4 w-4 text-green-400"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M5 13l4 4L19 7"
          />
        </svg>
        <span class="ml-1 text-xs">{{ copied ? 'Copied!' : 'Copy' }}</span>
      </button>
    </div>

    <div class="code-content">
      <div class="line-numbers">
        <span v-for="num in lineNumbers" :key="num" class="line-number">
          {{ num }}
        </span>
      </div>
      <pre class="code-text"><code>{{ code }}</code></pre>
    </div>
  </div>
</template>

<style scoped>
.code-block {
  @apply bg-gray-900 rounded-lg overflow-hidden shadow-lg;
}

.code-header {
  @apply flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700;
}

.language-tag {
  @apply text-xs font-medium text-gray-400;
}

.copy-button {
  @apply flex items-center px-2 py-1 text-gray-400 hover:text-white rounded transition-colors duration-200;
}

.copy-button.copy-success {
  @apply text-green-400;
}

.code-content {
  @apply flex overflow-x-auto;
}

.line-numbers {
  @apply flex flex-col py-4 px-3 text-right bg-gray-800/50 text-gray-500 select-none border-r border-gray-700;
  min-width: 3rem;
}

.line-number {
  @apply text-xs leading-6 font-mono;
}

.code-text {
  @apply flex-1 py-4 px-4 text-gray-100 font-mono text-sm leading-6 overflow-x-auto;
  tab-size: 2;
}

.code-text code {
  @apply whitespace-pre;
}
</style>
