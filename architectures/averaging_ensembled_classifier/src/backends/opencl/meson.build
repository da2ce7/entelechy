# src/backends/opencl/meson.build — install OpenCL backend Python modules

opencl_sources = files(
  '__init__.py',
  'batch_processor.py',
  'compute_patterns.py',
  'context.py',
  'execution_plan.py',
  'graph_recipes.py',
  'launcher_infra.py',
)

py.install_sources(opencl_sources,
  subdir: 'averaging_ensembled_classifier' / 'src' / 'backends' / 'opencl')

# kernel_bindings sub-package
kernel_binding_sources = files(
  'kernel_bindings/__init__.py',
  'kernel_bindings/phase_1_act.py',
  'kernel_bindings/phase_2_learn_A_production.py',
  'kernel_bindings/phase_2_learn_B_processing.py',
  'kernel_bindings/phase_2_learn_C_reduction.py',
  'kernel_bindings/phase_2_learn_D_backprop.py',
  'kernel_bindings/phase_3_update.py',
)

py.install_sources(kernel_binding_sources,
  subdir: 'averaging_ensembled_classifier' / 'src' / 'backends' / 'opencl' / 'kernel_bindings')
