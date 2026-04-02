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
]

# Precision variant definitions: [suffix, extra glslc flags]
precision_variants = [
    ['_fp32',    []],
    ['_s16fp32', ['-DSTORAGE_FLOAT=float16_t', '-DSTATE_FLOAT=float', '-DENABLE_FP16_EXTENSION=1']],
    ['_fp16',    ['-DSTORAGE_FLOAT=float16_t', '-DSTATE_FLOAT=float16_t', '-DENABLE_FP16_EXTENSION=1']],
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

# Single-variant compilation for compute-only shaders.
foreach shader : compute_only_shaders
    stem = shader.replace('.comp', '')
    spirv_targets += custom_target(
        stem + '_fp32.spv',
        input: shader,
        output: stem + '_fp32.spv',
        command: [glslc,
                  '--target-env=vulkan1.1',
                  '-I', meson.current_source_dir(),
                  '-o', '@OUTPUT@',
                  '@INPUT@'],
        install: true,
        install_dir: py.get_install_dir() / 'averaging_ensembled_classifier' / 'backends' / 'vulkan' / 'kernel_sources',
    )
endforeach

# Install the package marker and GLSL source for reference.
py.install_sources(
    files('__init__.py'),
    subdir: 'averaging_ensembled_classifier' / 'backends' / 'vulkan' / 'kernel_sources',
)
