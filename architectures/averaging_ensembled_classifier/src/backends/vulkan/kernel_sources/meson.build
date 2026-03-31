# src/backends/vulkan/kernel_sources/meson.build — SPIR-V compilation via glslc (ADR-014)

glslc = find_program('glslc')

shader_sources = [
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
    'normalize_gradients.comp',
    'adam_update.comp',
    'clamp_temperatures.comp',
]

spirv_targets = []
foreach shader : shader_sources
    stem = shader.replace('.comp', '')
    spirv_targets += custom_target(
        stem + '.spv',
        input: shader,
        output: stem + '.spv',
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
