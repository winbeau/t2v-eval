# VBench Runner Module Layout

## Structure

- `core.py`: pipeline orchestration, dependency checks, parsing, output sync.
- `dimensions/`: one file per VBench dimension.
  - `base.py`: shared `DimensionSpec`.
  - `registry.py`: canonical 16-dimension registry + profile defaults.
  - `<dimension>.py`: defines `SPEC` only.

## Conventions

- Dimension module file name must equal dimension key (e.g. `subject_consistency.py`).
- Each dimension module exports exactly one symbol: `SPEC`.
- `SPEC.key` must match VBench dimension name used by `vbench.evaluate`.
- Any new dependency requirement should be encoded in `SPEC`
  (`requires_clip`, `requires_pyiqa`) and consumed by `core.py`.

## Profiles

- `long_6`: recommended fast subset for long consistency studies.
- `long_16`: full VBench-Long dimension set.

Set via config:

```yaml
metrics:
  vbench:
    use_long: true
    dimension_profile: "long_16"
```

If `subtasks` is explicitly set, it overrides profile defaults.
