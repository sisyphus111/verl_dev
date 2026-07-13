#!/usr/bin/env bash
# Qwen3.5-397B-A17B + Qwen3.5-0.8B decoupled speculative rollout GRPO
# with Megatron training. The trainer topology follows the upstream 397B SFT
# Megatron example (128 trainer GPUs, TP=2 PP=4 CP=1 EP=32). Decoupled-spec
# keeps one extra free draft node, so the default full job shape is 136 GPUs.

set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1

########################### Quick Config ###########################

NNODES=${NNODES:-16}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

MODEL_PATH=${MODEL_PATH:-/mnt/hdfs/ord/models/Qwen/Qwen3.5-397B-A17B}
DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH:-/mnt/hdfs/ord/models/Qwen/Qwen3.5-0.8B}
TRAIN_FILE=${TRAIN_FILE:-/mnt/hdfs/ord/datasets/DAPO-Math-17k/data/dapo-math-17k.parquet}
TEST_FILE=${TEST_FILE:-/mnt/hdfs/ord/datasets/AIME_2024/data/aime-2024.parquet}

project_name=verl_grpo_qwen3_5_397b_decoupled_spec
exp_name=qwen3_5_397b_a17b_136gpu_dspec
DSPEC_OUTPUT_DIR=${DSPEC_OUTPUT_DIR:-/opt/tiger/verl/resp}
save_contents="['model', 'extra', 'optimizer']"

train_prompt_bsz=16
ppo_mini_bsz=16
grpo_n=4
max_response_length=4096
max_token_len_per_gpu=8192

train_tp=2
train_pp=4
train_cp=1
train_ep=32
train_etp=1

rollout_tp=16
rollout_ep=16

rollout_dtype=bfloat16
train_dtype=bfloat16

# keep 397B lower until measured to avoid
# over-reserving Qwen3.5 GDN/mamba-style rollout state at server startup.
rollout_max_num_seqs=32
rollout_gpu_memory_utilization=0.67
update_weights_bucket_megabytes=256
disable_vision_encoder=${DISABLE_VISION_ENCODER:-True}

draft_nnodes=1
draft_ngpus=8
draft_tp=2
speculative_num_steps=5

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
    data.max_response_length=${max_response_length}
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
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
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
    actor_rollout_ref.actor.megatron.dtype=${train_dtype}
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
    # SGLang verifier rollout engine
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.enable_decoupled_spec=True
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.data_parallel_size=1
    actor_rollout_ref.rollout.pipeline_model_parallel_size=1
    actor_rollout_ref.rollout.expert_parallel_size=${rollout_ep}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_memory_utilization}
    actor_rollout_ref.rollout.dtype=${rollout_dtype}
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs}
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${update_weights_bucket_megabytes}
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_vision_encoder=${disable_vision_encoder}
    +actor_rollout_ref.rollout.engine_kwargs.sglang.log_level=info
    +actor_rollout_ref.rollout.engine_kwargs.sglang.log_level_http=info
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

DRAFT=(
    # Decoupled speculative drafter
    draft.model_path="${DRAFT_MODEL_PATH}"
    draft.tokenizer_path="${DRAFT_MODEL_PATH}"
    draft.nnodes=${draft_nnodes}
    draft.ngpus=${draft_ngpus}
    draft.tp_size=${draft_tp}
    draft.speculative_num_steps=${speculative_num_steps}
    draft.output_dir="${DSPEC_OUTPUT_DIR}"
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
    trainer.save_freq=-1
    trainer.val_before_train=False
    trainer.test_freq=10
    trainer.total_epochs=10
    +ray_kwargs.ray_init.address=auto
)

########################### Launch ###########################

export HYDRA_FULL_ERROR=1
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-path=../experimental/decoupled_spec/config \
    --config-name=decoupled_spec_ppo_megatron_trainer.yaml \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ACTOR_ROLLOUT_REF_COMMON[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${DRAFT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "$@"
