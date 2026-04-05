# src/backends/vulkan/kernel_sources/meson.build — SPIR-V compilation via glslc (ADR-014, ADR-023 §4.1)

glslc = find_program('glslc')

# Shaders with storage-role or state-role typed buffer bindings.
# These are compiled once per precision variant.
role_bearing_shaders = [
    'forward_pass.comp',
    'render_logits_chunk.comp',
    'compute_probs_loss_cce_chunk.comp',
    'compute_probs_loss_bce_chunk.comp',
    'calculate_module_param_grads_chunk.comp',
    'backprop_error_to_hidden_chunk.comp',
    'calculate_chunk_temp_gradients.comp',
    'clip_partial_gradients.comp',
    'gather_and_permute_grad_hidden_activations.comp',
    'aggregate_partials.comp',
    'clip_intermediate_grad.comp',
    'stabilize_and_reduce_grad_hidden_activations.comp',
    'backprop_shared_weights_chunk.comp',
    'backprop_shared_biases_chunk.comp',
    'clip_shared_gradients_chunk.comp',
    'adam_update.comp',
    'clamp_temperatures.comp',
]

# Shaders with only compute-role buffers — single build, no precision flags.
compute_only_shaders = [
    'normalize_gradients.comp',
    'aggregate_partials_from_compute.comp',  # ADR-026: Compute-entry variant
]

# Precision variant definitions: [suffix, extra glslc flags]
# Three-axis: STORAGE_TYPE × COMPUTE_TYPE × STATE_TYPE (ADR-024 §5.5)
precision_variants = [
    # storage=16, compute=16
    ['_s16c16x16', ['-DSTORAGE_TYPE=float16_t', '-DCOMPUTE_TYPE=float16_t', '-DSTATE_TYPE=float16_t', '-DENABLE_FP16_EXTENSION=1', '-DNUMERICAL_EPSILON_VALUE=1e-3']],
    ['_s16c16x32', ['-DSTORAGE_TYPE=float16_t', '-DCOMPUTE_TYPE=float16_t', '-DSTATE_TYPE=float', '-DENABLE_FP16_EXTENSION=1', '-DNUMERICAL_EPSILON_VALUE=1e-3']],
    ['_s16c16x64', ['-DSTORAGE_TYPE=float16_t', '-DCOMPUTE_TYPE=float16_t', '-DSTATE_TYPE=double', '-DENABLE_FP16_EXTENSION=1', '-DENABLE_FP64_EXTENSION=1', '-DNUMERICAL_EPSILON_VALUE=1e-3']],
    # storage=16, compute=32
    ['_s16c32x16', ['-DSTORAGE_TYPE=float16_t', '-DCOMPUTE_TYPE=float', '-DSTATE_TYPE=float16_t', '-DENABLE_FP16_EXTENSION=1']],
    ['_s16c32x32', ['-DSTORAGE_TYPE=float16_t', '-DCOMPUTE_TYPE=float', '-DSTATE_TYPE=float', '-DENABLE_FP16_EXTENSION=1']],
    ['_s16c32x64', ['-DSTORAGE_TYPE=float16_t', '-DCOMPUTE_TYPE=float', '-DSTATE_TYPE=double', '-DENABLE_FP16_EXTENSION=1', '-DENABLE_FP64_EXTENSION=1']],
    # storage=16, compute=64
    ['_s16c64x16', ['-DSTORAGE_TYPE=float16_t', '-DCOMPUTE_TYPE=double', '-DSTATE_TYPE=float16_t', '-DENABLE_FP16_EXTENSION=1', '-DENABLE_FP64_EXTENSION=1', '-DNUMERICAL_EPSILON_VALUE=1e-15']],
    ['_s16c64x32', ['-DSTORAGE_TYPE=float16_t', '-DCOMPUTE_TYPE=double', '-DSTATE_TYPE=float', '-DENABLE_FP16_EXTENSION=1', '-DENABLE_FP64_EXTENSION=1', '-DNUMERICAL_EPSILON_VALUE=1e-15']],
    ['_s16c64x64', ['-DSTORAGE_TYPE=float16_t', '-DCOMPUTE_TYPE=double', '-DSTATE_TYPE=double', '-DENABLE_FP16_EXTENSION=1', '-DENABLE_FP64_EXTENSION=1', '-DNUMERICAL_EPSILON_VALUE=1e-15']],
    # storage=32, compute=32
    ['_s32c32x32', []],
    ['_s32c32x64', ['-DSTATE_TYPE=double', '-DENABLE_FP64_EXTENSION=1']],
    # storage=32, compute=64
    ['_s32c64x32', ['-DCOMPUTE_TYPE=double', '-DENABLE_FP64_EXTENSION=1', '-DNUMERICAL_EPSILON_VALUE=1e-15']],
    ['_s32c64x64', ['-DCOMPUTE_TYPE=double', '-DSTATE_TYPE=double', '-DENABLE_FP64_EXTENSION=1', '-DNUMERICAL_EPSILON_VALUE=1e-15']],
    # storage=64
    ['_s64c64x64', ['-DSTORAGE_TYPE=double', '-DCOMPUTE_TYPE=double', '-DSTATE_TYPE=double', '-DENABLE_FP64_EXTENSION=1', '-DNUMERICAL_EPSILON_VALUE=1e-15']],
]

spirv_targets = []

# Multi-variant compilation for role-bearing shaders.
foreach variant : precision_variants
    suffix = variant[0]
    extra_flags = variant[1]
    foreach shader : role_bearing_shaders
        stem = shader.replace('.comp', '')
        spirv_targets += custom_target(
            stem + suffix + '.spv',
            input: shader,
            output: stem + suffix + '.spv',
            command: [glslc,
                      '--target-env=vulkan1.1',
                      '-I', meson.current_source_dir(),
                      extra_flags,
                      '-o', '@OUTPUT@',
                      '@INPUT@'],
            install: true,
            install_dir: py.get_install_dir() / 'averaging_ensembled_classifier' / 'backends' / 'vulkan' / 'kernel_sources',
        )
    endforeach
endforeach

# Compute-only variant definitions: [suffix, extra glslc flags]
compute_only_variants = [
    ['_c16', ['-DCOMPUTE_TYPE=float16_t', '-DENABLE_FP16_EXTENSION=1', '-DNUMERICAL_EPSILON_VALUE=1e-3']],
    ['_c32', []],
    ['_c64', ['-DCOMPUTE_TYPE=double', '-DENABLE_FP64_EXTENSION=1', '-DNUMERICAL_EPSILON_VALUE=1e-15']],
]

# Single-variant compilation for compute-only shaders.
foreach variant : compute_only_variants
    suffix = variant[0]
    extra_flags = variant[1]
    foreach shader : compute_only_shaders
        stem = shader.replace('.comp', '')
        spirv_targets += custom_target(
            stem + suffix + '.spv',
            input: shader,
            output: stem + suffix + '.spv',
            command: [glslc,
                      '--target-env=vulkan1.1',
                      '-I', meson.current_source_dir(),
                      extra_flags,
                      '-o', '@OUTPUT@',
                      '@INPUT@'],
            install: true,
            install_dir: py.get_install_dir() / 'averaging_ensembled_classifier' / 'backends' / 'vulkan' / 'kernel_sources',
        )
    endforeach
endforeach

# Install the package marker and GLSL source for reference.
py.install_sources(
    files('__init__.py'),
    subdir: 'averaging_ensembled_classifier' / 'backends' / 'vulkan' / 'kernel_sources',
)
