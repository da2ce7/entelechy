# src/backends/vulkan/meson.build — Vulkan backend package (ADR-014)

subdir('kernel_sources')

py.install_sources(
    files(
        '__init__.py',
        'context.py',
        'discovery.py',
        'type_mapping.py',
        'buffer_allocator.py',
        '_pipeline_cache.py',
        '_descriptor_manager.py',
        '_push_constants.py',
        'retrieval.py',
        'renderer.py',
    ),
    subdir: 'averaging_ensembled_classifier' / 'backends' / 'vulkan',
)
