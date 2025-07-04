# architectures/averaging_ensembled_classifier/meson.build

# This is the modern Architecture Manifest for 'averaging_ensembled_classifier'.
# It is a well-formed project that can be built standalone OR included as a subproject.

# The first statement MUST be a call to project().
# The language is 'c' because this project contains C-syntax OpenCL kernels.
# 'python' is NOT a language declared here; it is managed by the python module.
project('averaging_ensembled_classifier-sub', 'c', version: '0.1.0')

py = import('python').find_installation()

# --- 1. Automated Source File Discovery ---
py_sources = files(
  'src/batch_processor.py',
  'src/cl_context_manager.py',
  'src/compute_patterns.py',
  'src/execution_plan.py',
  'src/graph_recipes.py',
  'src/kernel_signatures/__init__.py',
  'src/kernel_signatures/phase_1_act.py',
  'src/kernel_signatures/phase_2_learn_A_production.py',
  'src/kernel_signatures/phase_2_learn_B_processing.py',
  'src/kernel_signatures/phase_2_learn_C_reduction.py',
  'src/kernel_signatures/phase_2_learn_D_backprop.py',
  'src/kernel_signatures/phase_3_update.py',
  'src/kernel_signatures/utility_signatures.py',
  'src/launcher_infra.py',
  'src/main_orchestrator.py',
  'src/memory_layout.py',
  'src/model_spec.py',
  'src/parameter_space.py',
  'src/stabilization_policy.py',
  'src/utility_kernels.py',
  'src/workload_primitives.py',
  '__init__.py',
)

cl_kernels = files(
  'kernels/aggregation_kernels.cl.c',
  'kernels/backprop_kernels.cl.c',
  'kernels/chunk_kernels.cl.c',
  'kernels/parameter_optim.cl.c',
)

all_sources = py_sources + cl_kernels

py.install_sources(py_sources, subdir: 'averaging_ensembled_classifier')

install_data(
  cl_kernels,
  install_dir: py.get_install_dir() / 'averaging_ensembled_classifier' / 'kernels',
)

# This tells meson-python about your project's Python dependencies.
py_dep = py.dependency()