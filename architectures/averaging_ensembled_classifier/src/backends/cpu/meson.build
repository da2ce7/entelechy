# src/backends/cpu/meson.build — CPU kernel shared library target (ADR-014)

cpu_kernel_sources = files(
    'kernel_sources/cpu_threads.c',
    'kernel_sources/cpu_kernels.c',
)

# ISA flags: use aec_cpu_isa_flags if provided, else -march=native
cpu_isa_flags = get_option('aec_cpu_isa_flags')
if cpu_isa_flags.length() == 0
    cpu_isa_flags = ['-march=native']
endif

# Math library (needed for sqrtf, expf, logf, fabsf on some platforms)
cc = meson.get_compiler('c')
m_dep = cc.find_library('m', required: false)

# Thread library (needed for pthreads fallback)
thread_dep = dependency('threads', required: false)

# --- Platform-adaptive warning flags ---
# MSVC uses /W4; GCC/Clang use the comprehensive -W set.
if cc.get_id() == 'msvc'
    cpu_warn_args = ['/W4']
else
    cpu_warn_args = cc.get_supported_arguments(
        '-Wall', '-Wextra', '-Wpedantic',
        '-Wconversion', '-Wsign-conversion',
        '-Wshadow', '-Wstrict-prototypes', '-Wmissing-prototypes',
        '-Wformat=2', '-Wdouble-promotion',
        '-Wundef', '-Wpointer-arith', '-Wcast-align',
        '-Wnull-dereference', '-Wwrite-strings',
    )
endif

# --- Base compile flags ---
cpu_c_args = ['-DCPU_KERNELS_BUILDING']

if cc.get_id() == 'msvc'
    # MSVC: no -fvisibility, no -march; use /arch: for ISA targeting.
    # __declspec(dllexport) is handled by cpu_export.h via CPU_KERNELS_BUILDING.
    cpu_c_args += cpu_warn_args
else
    cpu_c_args += cpu_isa_flags + cpu_warn_args
endif

# --- Debug assertions (bounds checks, NaN guards) ---
if get_option('aec_cpu_debug_asserts')
    cpu_c_args += ['-DCPU_DEBUG_ASSERTS=1']
endif

# --- _Float16 detection for FP8+FP16 compute variants (ADR-025 §5.3) ---
# Test for _Float16 arithmetic support using a cast (works with GCC 12+ extension
# in C11/C17 mode, not just C23). This intentionally tests for _Float16 as an
# arithmetic type, NOT Clang's legacy __fp16 extension (which is storage-only
# and would fail on the multiplication). The cast syntax avoids requiring the
# C23-specific `f16` literal suffix while still verifying arithmetic capability.
has_float16 = cc.compiles('''
    _Float16 test_float16(_Float16 x) { return x * (_Float16)2.0; }
''', name: '_Float16 support check')

if has_float16
    message('_Float16 support detected — FP16 compute variants enabled')
    cpu_c_args += ['-DHAS_FLOAT16=1']
else
    warning('_Float16 not supported — FP16 compute variants will be skipped')
    cpu_c_args += ['-DHAS_FLOAT16=0']
endif

# --- Shared library target ---
# Symbol visibility:
#   Unix:    gnu_symbol_visibility='hidden' + CPU_KERNELS_EXPORT attribute
#   Windows: cpu_export.h __declspec(dllexport/dllimport) via CPU_KERNELS_BUILDING
#   (Meson ignores gnu_symbol_visibility on Windows, so it is always safe to set.)
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
    '_compiler.py',
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

# Install C kernel sources for JIT compilation on Linux/macOS
cpu_kernel_headers_and_sources = files(
    'kernel_sources/cpu_export.h',
    'kernel_sources/cpu_fp8.h',
    'kernel_sources/cpu_fp8_lut.gen.h',
    'kernel_sources/cpu_kernels.c',
    'kernel_sources/cpu_kernels.h',
    'kernel_sources/cpu_precision.h',
    'kernel_sources/cpu_simd.h',
    'kernel_sources/cpu_threads.c',
    'kernel_sources/cpu_threads.h',
    'kernel_sources/phase_1_act.inc',
    'kernel_sources/phase_2_learn_A_production.inc',
    'kernel_sources/phase_2_learn_B_processing.inc',
    'kernel_sources/phase_2_learn_C_reduction.inc',
    'kernel_sources/phase_2_learn_D_backprop.inc',
    'kernel_sources/phase_3_update.inc',
)

install_data(
    cpu_kernel_headers_and_sources,
    install_dir: py.get_install_dir() / 'averaging_ensembled_classifier' / 'backends' / 'cpu' / 'kernel_sources',
)

# --- Generate _fp8_variants.py — records which FP8 variants were compiled ---
# (ADR-025 §5.3, Step 9C.4.1)
fp8_all_suffixes = [
    's8e4c32x32', 's8e4c32x64', 's8e4c64x64',
    's8e5c32x32', 's8e5c32x64', 's8e5c64x64',
]
fp8_c16_suffixes = [
    's8e4c16x32', 's8e4c16x64',
    's8e5c16x32', 's8e5c16x64',
    # FP16 state variants (ADR-024, FP8+FP16 state)
    's8e4c16x16', 's8e4c32x16', 's8e4c64x16',
    's8e5c16x16', 's8e5c32x16', 's8e5c64x16',
]

if has_float16
    fp8_all_suffixes += fp8_c16_suffixes
endif

# Build the list string for Python
fp8_list_str = ''
fp8_sep = ''
foreach v : fp8_all_suffixes
    fp8_list_str += fp8_sep + '"' + v + '"'
    fp8_sep = ', '
endforeach

fp8_conf_data = configuration_data()
fp8_conf_data.set('AVAILABLE_FP8_VARIANTS', '[' + fp8_list_str + ']')
if has_float16
    fp8_conf_data.set('HAS_FLOAT16', 'True')
else
    fp8_conf_data.set('HAS_FLOAT16', 'False')
endif

configure_file(
    input: '_fp8_variants.py.in',
    output: '_fp8_variants.py',
    configuration: fp8_conf_data,
    install: true,
    install_dir: py.get_install_dir() / 'averaging_ensembled_classifier' / 'backends' / 'cpu',
)
