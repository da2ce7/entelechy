# kernels/meson.build — OpenCL kernel sources and host-mode validation (ADR-013)

cl_kernels = files(
  'phase_1_act.cl.c',
  'phase_2_learn_A_production.cl.c',
  'phase_2_learn_B_processing.cl.c',
  'phase_2_learn_C_reduction.cl.c',
  'phase_2_learn_D_backprop.cl.c',
  'phase_3_update.cl.c',
)

# Install kernel sources as data files for OpenCL runtime compilation
install_data(
  cl_kernels,
  install_dir: py.get_install_dir() / 'averaging_ensembled_classifier' / 'kernels',
)

# --- Host-mode kernel validation target (ADR-013) ---
# Compiles kernel files using C compiler with Host/C++ stub definitions
# to catch type mismatches at build time without requiring OpenCL runtime.
# Note: Reduced warning set to avoid noise from host/device math function
# return type differences (C stdlib returns double, OpenCL returns float).
cc = meson.get_compiler('c')
kernel_validate_c_args = cc.get_supported_arguments(
    '-Wall',
    '-Wno-unused-parameter',    # Kernel params often contextual
    '-Wno-unused-variable',     # Tile indices used only in some paths
    '-Wno-sign-conversion',     # size_t vs uint is host-mode artifact
    '-Wno-float-conversion',    # Host math returns double, OpenCL returns float
    '-Wno-double-promotion',    # Same reason
)
static_library(
  'kernel_host_validate',
  cl_kernels,
  c_args: kernel_validate_c_args,
  build_by_default: true,
  install: false,
)
