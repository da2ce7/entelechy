# architectures/averaging_ensembled_classifier/meson.build

# This is the modern Architecture Manifest for 'averaging_ensembled_classifier'.
# It is a well-formed project that can be built standalone OR included as a subproject.

# The first statement MUST be a call to project().
# The language is 'c' because this project contains C-syntax OpenCL kernels.
# 'python' is NOT a language declared here; it is managed by the python module.
project('averaging_ensembled_classifier-sub', 'c', version: '0.1.0')

py = import('python').find_installation()

# --- Feature-flag probing (ADR-014) ---
backend_vulkan = get_option('aec_backend_vulkan')
backend_cpu    = get_option('aec_backend_cpu')

# OpenCL backend is always enabled (runtime kernel compilation via PyOpenCL).
backend_opencl = true

# --- Build-time configuration (ADR-014) ---
configure_file(
  input:  'src/_build_config.py.in',
  output: '_build_config.py',
  configuration: configuration_data({
    'BACKEND_OPENCL': backend_opencl ? 'True' : 'False',
    'BACKEND_VULKAN': backend_vulkan.allowed() ? 'True' : 'False',
    'BACKEND_CPU':    backend_cpu.allowed() ? 'True' : 'False',
  }),
  install: true,
  install_dir: py.get_install_dir() / 'averaging_ensembled_classifier' / 'src',
)

# --- Top-level src/ Python modules ---
src_root_sources = files(
  'src/__init__.py',
  'src/arch_primitives.py',
  'src/main_orchestrator.py',
  # Shim adapters (backward-compat, removed in Phase 6)
  'src/batch_processor.py',
  'src/cl_context_manager.py',
  'src/compute_patterns.py',
  'src/execution_plan.py',
  'src/graph_recipes.py',
  'src/launcher_infra.py',
)

py.install_sources(src_root_sources,
  subdir: 'averaging_ensembled_classifier' / 'src')

# Shim for kernel_signatures (backward-compat)
py.install_sources(files('src/kernel_signatures/__init__.py'),
  subdir: 'averaging_ensembled_classifier' / 'src' / 'kernel_signatures')

# Backends __init__ markers
py.install_sources(files('src/backends/__init__.py'),
  subdir: 'averaging_ensembled_classifier' / 'src' / 'backends')

# --- Delegate to sub-packages ---
subdir('src/shared')
subdir('src/backends/opencl')

# Conditional backend subdirs (no-op in Phase 0)
if backend_vulkan.allowed()
  # subdir('src/backends/vulkan')  # created in Phase 3
endif
if backend_cpu.allowed()
  subdir('src/backends/cpu')
endif

# --- OpenCL kernel data files ---
cl_kernels = files(
  'kernels/phase_1_act.cl.c',
  'kernels/phase_2_learn_A_production.cl.c',
  'kernels/phase_2_learn_B_processing.cl.c',
  'kernels/phase_2_learn_C_reduction.cl.c',
  'kernels/phase_2_learn_D_backprop.cl.c',
  'kernels/phase_3_update.cl.c',
)

install_data(
  cl_kernels,
  install_dir: py.get_install_dir() / 'averaging_ensembled_classifier' / 'kernels',
)

# This tells meson-python about your project's Python dependencies.
py_dep = py.dependency()