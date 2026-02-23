# VBench Alignment Diagnostic Report

## Conclusion

- root_cause: `upstream_dimension_logic_effectively_identical_under_current_setup`

## Logs

```text
[CSV] outputs/Exp-K_StaOscCompression/vbench_Exp-K_StaOscCompression.csv exists=True
[CSV] rows=896 cols=15
[CSV] focus columns=['dynamic_degree', 'motion_smoothness', 'overall_consistency', 'temporal_style', 'imaging_quality', 'aesthetic_quality', 'subject_consistency', 'background_consistency']
  - dynamic_degree: nonnull=896 mean=0.431987 std=0.385258
  - motion_smoothness: nonnull=896 mean=0.984868 std=0.016212
  - overall_consistency: nonnull=896 mean=0.076369 std=0.072885
  - temporal_style: nonnull=896 mean=0.076369 std=0.072885
  - imaging_quality: nonnull=896 mean=68.761003 std=8.935179
  - aesthetic_quality: nonnull=896 mean=0.576219 std=0.077041
  - subject_consistency: nonnull=896 mean=0.970843 std=0.008702
  - background_consistency: nonnull=896 mean=0.960880 std=0.003782
[CSV] overall_consistency == temporal_style equal_ratio=1.0000
[JSON] results_dir=outputs/Exp-K_StaOscCompression/vbench_results exists=True

[JSON:dynamic_degree] outputs/Exp-K_StaOscCompression/vbench_results/dynamic_degree_eval_results.json exists=True
  items=13436 scored=13436 mean=0.431899
  sample:
    {'video_path': '/data/chenjiayu/wenbiao_zhao/t2v-eval/outputs/Exp-K_StaOscCompression/vbench_results/input_videos/split_clip/g1_video_000/g1_video_000_000.mp4', 'video_results': True, 'score': None}

[JSON:motion_smoothness] outputs/Exp-K_StaOscCompression/vbench_results/motion_smoothness_eval_results.json exists=True
  items=13436 scored=13436 mean=0.984866
  sample:
    {'video_path': '/data/chenjiayu/wenbiao_zhao/t2v-eval/outputs/Exp-K_StaOscCompression/vbench_results/input_videos/split_clip/g1_video_000/g1_video_000_000.mp4', 'video_results': 0.9870132541041702, 'score': None}

[JSON:overall_consistency] outputs/Exp-K_StaOscCompression/vbench_results/overall_consistency_eval_results.json exists=True
  items=13436 scored=13436 mean=0.076379
  sample:
    {'video_path': '/data/chenjiayu/wenbiao_zhao/t2v-eval/outputs/Exp-K_StaOscCompression/vbench_results/input_videos/split_clip/g1_video_000/g1_video_000_000.mp4', 'video_results': 0.2995211184024811, 'score': None}

[JSON:temporal_style] outputs/Exp-K_StaOscCompression/vbench_results/temporal_style_eval_results.json exists=True
  items=13436 scored=13436 mean=0.076379
  sample:
    {'video_path': '/data/chenjiayu/wenbiao_zhao/t2v-eval/outputs/Exp-K_StaOscCompression/vbench_results/input_videos/split_clip/g1_video_000/g1_video_000_000.mp4', 'video_results': 0.2995211184024811, 'score': None}

[JSON:imaging_quality] outputs/Exp-K_StaOscCompression/vbench_results/imaging_quality_eval_results.json exists=True
  items=13436 scored=13436 mean=68.760712
  sample:
    {'video_path': '/data/chenjiayu/wenbiao_zhao/t2v-eval/outputs/Exp-K_StaOscCompression/vbench_results/input_videos/split_clip/g1_video_000/g1_video_000_000.mp4', 'video_results': 62.40827453136444, 'score': None}

[JSON:aesthetic_quality] outputs/Exp-K_StaOscCompression/vbench_results/aesthetic_quality_eval_results.json exists=True
  items=13436 scored=13436 mean=0.576210
  sample:
    {'video_path': '/data/chenjiayu/wenbiao_zhao/t2v-eval/outputs/Exp-K_StaOscCompression/vbench_results/input_videos/split_clip/g1_video_000/g1_video_000_000.mp4', 'video_results': 0.5996844172477722, 'score': None}

[JSON:subject_consistency] outputs/Exp-K_StaOscCompression/vbench_results/subject_consistency_eval_results.json exists=True
  items=13436 scored=13436 mean=0.970842
  sample:
    {'video_path': '/data/chenjiayu/wenbiao_zhao/t2v-eval/outputs/Exp-K_StaOscCompression/vbench_results/input_videos/split_clip/g1_video_000/g1_video_000_000.mp4/g1.mp4', 'video_results': 0.978297072072183, 'score': None}

[JSON:background_consistency] outputs/Exp-K_StaOscCompression/vbench_results/background_consistency_eval_results.json exists=True
  items=13436 scored=13436 mean=0.960880
  sample:
    {'video_path': '/data/chenjiayu/wenbiao_zhao/t2v-eval/outputs/Exp-K_StaOscCompression/vbench_results/input_videos/split_clip/g1_video_000/g1_video_000_000.mp4/g1.mp4', 'video_results': 0.9559286794354839, 'score': None}
[PAIR] overall_consistency vs temporal_style:
  {'matched_rows': 13436, 'equal_ratio': 1.000000, 'mean_abs_diff': 0.000000, 'max_abs_diff': 0.000000}
[FULL_INFO] left=outputs/Exp-K_StaOscCompression/vbench_results/overall_consistency_full_info.json exists=True right=outputs/Exp-K_StaOscCompression/vbench_results/temporal_style_full_info.json exists=True
[FULL_INFO] matched=896 prompt_equal_ratio=1.000000 video_list_equal_ratio=1.000000
[FULL_INFO] prompt_like_video_id_ratio left=0.857143 right=0.857143
[SOURCE] similarity ratios: raw=0.970762 normalized=0.977397 dimension_agnostic=1.000000
[SOURCE] normalized_sha overall=d87c222ecb83 temporal=f7c58994f428
[CONFIG] configs/Exp-K_StaOscCompression.yaml exists=True
[CONFIG] backend='vbench_long' mode='long_custom_input' comparison_profile='deep_forcing_8d'
[CONFIG] scale_to_percent includes `overall_consistency`=True (len=6)
[CONCLUSION] root_cause=upstream_dimension_logic_effectively_identical_under_current_setup
```
