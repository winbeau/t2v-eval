# T2V-Eval LaTeX Table Generator

A Vue 3 + TypeScript frontend for generating LaTeX tables from T2V evaluation results.

## Features

- **CSV Upload**: Drag & drop or click to upload `group_summary.csv` files
- **LaTeX Generation**: Generate publication-ready LaTeX tables with booktabs format
- **Live Preview**: Real-time table preview with KaTeX rendering
- **Metric Selection**: Choose which metrics to include in the table
- **Customizable Options**:
  - Table format (booktabs/standard)
  - Column alignment (left/center/right)
  - Decimal precision
  - Show/hide standard deviation
  - Highlight best values
  - Custom caption and label

## Quick Start

```bash
# Install dependencies
pnpm install

# Start development server
pnpm dev

# Build for production
pnpm build

# Preview production build
pnpm preview
```

## Usage

1. **Upload Data**: Upload your `group_summary.csv` file from the evaluation output, or click "Load Sample Data" to see a demo

2. **Select Metrics**: Choose which metrics to include in the table. Click the direction indicator (↑/↓) to toggle whether higher or lower values are better

3. **Configure Options**:
   - **Table Format**: `booktabs` for publication-ready tables (requires `\usepackage{booktabs}`)
   - **Alignment**: Column alignment for metric values
   - **Precision**: Number of decimal places
   - **Show Std**: Include standard deviation (±) in cells
   - **Highlight Best**: Bold the best value in each column

4. **Copy LaTeX**: Switch to the "LaTeX Code" tab and copy the generated code

## Expected CSV Format

The CSV should have the following columns:

```csv
group,n_videos,clip_or_vqa_score_mean,clip_or_vqa_score_std,vbench_temporal_score_mean,...
frame_level_baseline,50,28.45,3.21,0.752,...
head_level_stable_w8,50,29.12,2.98,0.789,...
```

Required columns:
- `group`: Group/method name
- `n_videos`: Number of videos evaluated

Metric columns follow the pattern:
- `{metric_name}_mean`: Mean value
- `{metric_name}_std`: Standard deviation

## Generated LaTeX Example

```latex
\begin{table}[htbp]
\centering
\caption{T2V Evaluation Results}
\label{tab:results}
\begin{tabular}{lccc}
\toprule
Method & CLIP/VQA ↑ & VBench ↑ & Flicker ↓ \\
\midrule
Frame Level Baseline & 28.45 $\pm$ 3.21 & 0.752 $\pm$ 0.045 & 0.023 $\pm$ 0.009 \\
Head Level Stable W8 & \textbf{29.12} $\pm$ 2.98 & \textbf{0.789} $\pm$ 0.038 & \textbf{0.020} $\pm$ 0.008 \\
\bottomrule
\end{tabular}
\end{table}
```

## Tech Stack

- **Vue 3** with Composition API
- **TypeScript** for type safety
- **Vite** for fast development and building
- **Tailwind CSS** for styling
- **KaTeX** for LaTeX math rendering
- **PapaParse** for CSV parsing

## Development

```bash
# Type check
pnpm type-check

# Lint
pnpm lint

# Format
pnpm format
```

## License

MIT
