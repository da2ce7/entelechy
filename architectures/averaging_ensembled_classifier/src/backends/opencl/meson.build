# src/backends/opencl/meson.build — install OpenCL backend Python modules

opencl_sources = files(
  '__init__.py',
  'buffer_allocator.py',
  'context.py',
  'discovery.py',
  'renderer.py',
  'retrieval.py',
  'type_mapping.py',
)

py.install_sources(opencl_sources,
  subdir: 'averaging_ensembled_classifier' / 'src' / 'backends' / 'opencl')

# kernel_bindings sub-package
kernel_binding_sources = files(
  'kernel_bindings/__init__.py',
  'kernel_bindings/base.py',
  'kernel_bindings/binding_phase_1_act.py',
  'kernel_bindings/binding_phase_2_learn_A.py',
  'kernel_bindings/binding_phase_2_learn_B.py',
  'kernel_bindings/binding_phase_2_learn_C.py',
  'kernel_bindings/binding_phase_2_learn_D.py',
  'kernel_bindings/binding_phase_3_update.py',
  'kernel_bindings/dispatch_table.py',
)

py.install_sources(kernel_binding_sources,
  subdir: 'averaging_ensembled_classifier' / 'src' / 'backends' / 'opencl' / 'kernel_bindings')
