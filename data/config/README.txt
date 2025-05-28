# Files features_test_config_?.yaml show different input configurations for validating the code. 

# Configure the code by copying the <features_test_config_?.yaml> to <features.yaml>. At least two numeric targets must be assigned in the features.yaml file as "role:". Log transforming the target is recommended when a target distribution is not Normal (distributions are plotted in the pipeline; iteration may be necessary). Datasets can be excluded by commenting them out with "#".  

# Example features.yaml file structure:

# features:
#   - name: glycohemoglobin
#     source: LBXGH
#     file: GHB_I
#     type: numerical
#     role: feature
#     drop_extrema: true
#     log_transform: false
#     unit: percent
  
