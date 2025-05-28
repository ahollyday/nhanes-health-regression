# Feature Configuration Instructions

This directory contains YAML configuration files that define the features and targets used in the modeling pipeline.

## Available Files

- `features_test_config_?.yaml` files contain alternative input configurations for testing and validating the pipeline.
- To activate a configuration, copy the desired file to `features.yaml`:

```bash
cp features_test_config_1.yaml features.yaml
```

## Configuration Requirements

- The `features.yaml` file must define **at least two numeric targets** using the `role:` field (e.g., `role: target`).
- All other entries should have `role: feature`.

### Log Transformation Guidelines

- Targets with **non-Normal or skewed distributions** should have `log_transform: true`.
- The pipeline will plot distributions to help you decide whether transformation is appropriate.
- **Only targets** may be log-transformed. **Features should not** be log-transformed (i.e., use `log_transform: false` for all features).

### Dropping Extremes

- Set `drop_extrema: true` for features where extreme values should be excluded (e.g., outliers).

### Optional Exclusion

- To exclude datasets or individual features from processing, comment them out using `#`.

## Example `features.yaml` Structure

```yaml
features:
  - name: glycohemoglobin
    source: LBXGH
    file: GHB_I
    type: numerical
    role: feature
    drop_extrema: true
    log_transform: false
    unit: percent

  - name: vitamin_d
    source: LBXVID
    file: VID_I
    type: numerical
    role: target
    drop_extrema: false
    log_transform: true
    unit: ng/mL
```
