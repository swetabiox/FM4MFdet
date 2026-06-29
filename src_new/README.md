# Source code

This directory contains custom MMDetection components used in this project.

## Contents

- `custom_mmdet/backbones/`  
  Custom backbone implementations (e.g. H0, DINOv2, UNI, Virchow).

- `custom_mmdet/necks/`  
  Custom neck implementations (e.g. feature pyramid variants).

These components are registered and imported via `custom_imports`
in the corresponding MMDetection configuration files.

## Configuration

If the directory structure or module names are changed, the import paths
in the configuration files must be updated accordingly, for example:

```python
custom_imports = dict(
    imports=[
        "src.custom_mmdet.backbones.hoptimus0_vit",
        "src.custom_mmdet.necks.simple_feature_pyramid",
    ]
)


Notes
This directory contains only lightweight Python code.
No compiled extensions or external build steps are required.