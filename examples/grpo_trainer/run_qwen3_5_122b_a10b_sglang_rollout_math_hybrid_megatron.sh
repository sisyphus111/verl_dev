#!/usr/bin/env bash
# Qwen3.5-122B-A10B ordinary SGLang rollout GRPO with Megatron training.
# Target shape: 8 trainer nodes, 64 GPUs total.

set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1

########################### Quick Config ###########################

NNODES=${NNODES:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

MODEL_PATH=${MODEL_PATH:-/mnt/hdfs/ord/models/Qwen/Qwen3.5-122B-A10B}
TRAIN_FILE=${TRAIN_FILE:-/mnt/hdfs/ord/datasets/DAPO-Math-17k/data/dapo-math-17k.parquet}
TEST_FILE=${TEST_FILE:-/mnt/hdfs/ord/datasets/AIME_2024/data/aime-2024.parquet}

project_name=verl_grpo_qwen3_5_122b_sglang
exp_name=qwen3_5_122b_a10b_64gpu_sglang
TRACE_DIR=${TRACE_DIR:-/mnt/hdfs/ord/qwen35-122b-a10b-sglang-rollout/resp}
CKPTS_DIR=${CKPTS_DIR:-/mnt/hdfs/ord/qwen35-122b-a10b-sglang-rollout/ckpts}
save_contents="['model']"

train_prompt_bsz=32
ppo_mini_bsz=32
grpo_n=8
max_prompt_length=1024
max_response_length=16384
max_token_len_per_gpu=$((max_prompt_length + max_response_length))
total_training_steps=120
total_epochs=1
train_max_samples=$((train_prompt_bsz * total_training_steps * 2))

train_tp=2
train_pp=8
train_cp=1
train_ep=4
train_etp=1

rollout_tp=8
rollout_ep=8
rollout_max_num_seqs=64
rollout_gpu_memory_utilization=0.6

use_dynamic_bsz=False

########################### Parameter Arrays ###########################

DATA=(
    # File paths
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    # Dataset fields
    data.prompt_key=prompt
    data.truncation=left
    # Batch and length configuration
    data.train_batch_size=${train_prompt_bsz}
    data.train_max_samples=${train_max_samples}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
)

MODEL=(
    # Model path
    actor_rollout_ref.model.path="${MODEL_PATH}"
    # Model processing
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.use_fused_kernels=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa
)

ACTOR_ROLLOUT_REF_COMMON=(
    actor_rollout_ref.nccl_timeout=10800
)

ACTOR=(
    # Optimizer
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=10
    actor_rollout_ref.actor.optim.weight_decay=0.1
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    # PPO/GRPO update batch configuration
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_bsz}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_token_len_per_gpu}
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
    # Loss configuration
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.entropy_coeff=0
    # Megatron parallelism and memory
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True
    actor_rollout_ref.actor.megatron.use_remove_padding=False
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${train_etp}
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=True
    actor_rollout_ref.actor.megatron.grad_offload=True
    actor_rollout_ref.actor.megatron.dtype=bfloat16
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=null
    # Transformer architecture overrides
    actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_load_balancing_type=\"none\"
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=False
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.actor.checkpoint.save_contents="${save_contents}"
)

ROLLOUT=(
    # SGLang rollout engine
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.enable_decoupled_spec=False
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.expert_parallel_size=${rollout_ep}
    actor_rollout_ref.rollout.data_parallel_size=1
    actor_rollout_ref.rollout.pipeline_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_memory_utilization}
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs}
    +actor_rollout_ref.rollout.engine_kwargs.sglang.log_level=info
    +actor_rollout_ref.rollout.engine_kwargs.sglang.log_level_http=info
    actor_rollout_ref.rollout.trace.trace_dir="${TRACE_DIR}"
    # Generation parameters
    actor_rollout_ref.rollout.n=${grpo_n}
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    # Old log-prob recomputation
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${max_token_len_per_gpu}
    # Validation generation
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
)

REF=(
    # Ref log-prob recomputation
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${max_token_len_per_gpu}
    # Megatron ref parallelism and memory
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${train_etp}
    actor_rollout_ref.ref.megatron.param_offload=True
)

REWARD=(
    # DAPO reward manager
    reward.reward_manager.name=dapo
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True
    +reward.reward_kwargs.overlong_buffer_cfg.len=2048
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
    +reward.reward_kwargs.overlong_buffer_cfg.log=False
    +reward.reward_kwargs.max_resp_len=${max_response_length}
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.n_gpus_per_node=${GPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=1
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.val_before_train=False
    trainer.test_freq=10
    trainer.total_epochs=${total_epochs}
    trainer.total_training_steps=${total_training_steps}
    +ray_kwargs.ray_init.address=auto
)

########################### Launch ###########################

export HYDRA_FULL_ERROR=1
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name=ppo_megatron_trainer.yaml \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ACTOR_ROLLOUT_REF_COMMON[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "$@"
