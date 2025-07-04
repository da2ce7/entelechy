# meson.build

# This is the Framework Conductor Manifest.
# Its sole responsibility is to discover and delegate build operations to each
# autonomous architecture defined in the 'architectures' sub-directory.
# This hierarchical approach is a core design principle of the framework.

project('ml_engine_framework',
  version: '0.1.0',
  default_options: ['warning_level=2'],
  subproject_dir: 'architectures')

subproject('averaging_ensembled_classifier')

# Example of how you would add another autonomous architecture:
# subdir('architectures/transformer_diffusion_model')
