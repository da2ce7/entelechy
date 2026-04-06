# src/shared/meson.build — install shared-layer Python modules

shared_sources = files(
  '__init__.py',
  'buffer_lifecycle.py',
  'engine.py',
  'fp8_scaling.py',
  'hardware_profile.py',
  'memory_layout.py',
  'model_spec.py',
  'parameter_space.py',
  'plan_builder.py',
  'plan_renderer.py',
  'plan_types.py',
  'precision_config.py',
  'precision_suffix.py',
  'problem_type_strategy.py',
  'reduction_tree_plan.py',
  'retrieval_future.py',
  'stabilization_policy.py',
  'streaming_loop_plan.py',
  'ticket.py',
  'workload_primitives.py',
)

py.install_sources(shared_sources,
  subdir: 'averaging_ensembled_classifier' / 'src' / 'shared')

# kernel_contracts sub-package
kernel_contract_sources = files(
  'kernel_contracts/__init__.py',
  'kernel_contracts/phase_1_act.py',
  'kernel_contracts/phase_2_learn_A_production.py',
  'kernel_contracts/phase_2_learn_B_processing.py',
  'kernel_contracts/phase_2_learn_C_reduction.py',
  'kernel_contracts/phase_2_learn_D_backprop.py',
  'kernel_contracts/phase_3_update.py',
)

py.install_sources(kernel_contract_sources,
  subdir: 'averaging_ensembled_classifier' / 'src' / 'shared' / 'kernel_contracts')
