# src/backends/cpu/meson.build — CPU kernel shared library target (ADR-014)

cpu_kernel_sources = files(
    'kernel_sources/cpu_threads.c',
    'kernel_sources/phase_1_act.c',
    'kernel_sources/phase_2_learn_A_production.c',
    'kernel_sources/phase_2_learn_B_processing.c',
    'kernel_sources/phase_2_learn_C_reduction.c',
    'kernel_sources/phase_2_learn_D_backprop.c',
    'kernel_sources/phase_3_update.c',
)

# ISA flags: use aec_cpu_isa_flags if provided, else -march=native
cpu_isa_flags = get_option('aec_cpu_isa_flags')
if cpu_isa_flags.length() == 0
    cpu_isa_flags = ['-march=native']
endif

cpu_c_args = [
    '-DCPU_KERNELS_BUILDING',
    '-fvisibility=hidden',
] + cpu_isa_flags

# Math library (needed for sqrtf, expf, logf, fabsf on some platforms)
cc = meson.get_compiler('c')
m_dep = cc.find_library('m', required: false)

# Thread library (needed for pthreads fallback)
thread_dep = dependency('threads', required: false)

cpu_lib = shared_library(
    'cpu_kernels',
    cpu_kernel_sources,
    c_args: cpu_c_args,
    include_directories: include_directories('kernel_sources'),
    dependencies: [m_dep, thread_dep],
    install: true,
    install_dir: py.get_install_dir() / 'averaging_ensembled_classifier' / 'backends' / 'cpu',
    gnu_symbol_visibility: 'hidden',
)

# Install Python source files for the CPU backend
cpu_py_sources = files(
    '__init__.py',
    '_ffi_types.py',
    '_loader.py',
    '_dispatch_table.py',
    'discovery.py',
    'type_mapping.py',
    'buffer_allocator.py',
    'retrieval.py',
    'renderer.py',
)

py.install_sources(
    cpu_py_sources,
    subdir: 'averaging_ensembled_classifier' / 'backends' / 'cpu',
)
