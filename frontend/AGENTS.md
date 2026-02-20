# Repository Guidelines

## Scope
- 本文件作用域为当前目录及其所有子目录。

## Package Manager
- 仅使用 `pnpm` 管理依赖和脚本。
- 禁止使用 `npm` / `yarn`。

## Project Structure
- `src/`: Vue 3 + TypeScript 源码。
- `public/`: 静态资源与示例数据。
- `dist/`: 构建产物（自动生成）。

## Common Commands
- 安装依赖：`pnpm install`
- 本地开发：`pnpm dev`
- 生产构建：`pnpm build`
- 预览构建：`pnpm preview`

## Code Style
- 组件命名使用 `PascalCase`。
- 变量与函数使用 `camelCase`。
- 优先保持 TypeScript 类型完整，避免 `any`。

## Change Rules
- 修改前端功能后，至少运行一次 `pnpm build` 确认可构建。
- 非必要不提交 `dist/` 目录改动。

## Session Memory (Playwright Cache)
- 当前已有可用页面：`http://127.0.0.1:4173/`（用户已明确端口在运行）。
- 默认规则：不要重复启动新端口，不要重复做同一轮 Playwright 回归。
- 最近一次已完成的 Playwright 复现步骤：
  - 打开 `http://127.0.0.1:4173/`
  - 点击 `Load Local Data`
  - 加载 `vbench_per_video.csv`
  - 点击 `Select All` 进入多指标分块预览
- 复现结论（用于后续上下文压缩后直接复用）：
  - “命名列在有明显空白时仍提前换行”的问题与表格列宽分配/断词策略相关。
  - 重点排查对象：`src/components/TablePreview.vue` 中 `method` 列宽、`metric-title` 的 `word-break/overflow-wrap`、以及单元格 padding/gap。
- 最近一次已落地的样式修复（无需重复 Playwright 才能知道改动点）：
  - 文件：`src/components/TablePreview.vue`
  - 调整：移除 `method` 列 `width: 1%` 压缩行为，改为更稳定的列宽；收紧 `th/td` 横向 padding；把 `overflow-wrap: anywhere` 改为 `break-word`，`word-break` 改为 `normal`。
  - 目标：减少“有大量留白但标题/方法名仍过早换行”的现象。
  - 补充：指标头部箭头改为与指标名同一文本节点并使用不换行空格绑定（`label\u00A0↑/↓`），避免箭头单独换行。
- 重新调用 Playwright 的唯一条件：
  - 本次会话中 `src/components/TablePreview.vue` 或相关布局样式已再次修改，且需要最终可视化验收截图。
