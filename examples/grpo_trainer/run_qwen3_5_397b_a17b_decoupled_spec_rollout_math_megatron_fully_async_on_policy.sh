#!/usr/bin/env bash
# Qwen3.5-397B-A17B + Qwen3.5-0.8B decoupled speculative rollout GRPO
# on the fully_async code path, configured in synchronous/on-policy mode.
#
# Default resource shape:
#   - Trainer: 16 nodes * 8 GPUs = 128 GPUs, Megatron TP=2 PP=4 CP=1 EP=32
#   - Verifier rollout: 4 nodes * 8 GPUs = 32 GPUs, SGLang TP=32 DP=4 EP=32
#   - Decoupled drafter: 1 node * 8 GPUs = 8 GPUs total
# Total default allocation: 168 GPUs.

set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

export CUDA_DEVICE_MAX_CONNECTIONS=1

########################### Quick Config ###########################

NNODES_TRAIN=${NNODES_TRAIN:-16}
NNODES_ROLLOUT=${NNODES_ROLLOUT:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

MODEL_PATH=${MODEL_PATH:-/mnt/hdfs/ord/models/Qwen/Qwen3.5-397B-A17B}
DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH:-/mnt/hdfs/ord/models/Qwen/Qwen3.5-0.8B}
TRAIN_FILE=${TRAIN_FILE:-/mnt/hdfs/ord/datasets/DAPO-Math-17k/data/dapo-math-17k.parquet}
TEST_FILE=${TEST_FILE:-/mnt/hdfs/ord/datasets/AIME_2024/data/aime-2024.parquet}

project_name=verl_grpo_qwen3_5_397b_decoupled_spec_fasync
exp_name=qwen3_5_397b_a17b_train128_rollout32_draft8_fasync_onpolicy
DSPEC_OUTPUT_DIR=${DSPEC_OUTPUT_DIR:-/opt/tiger/verl/resp}
save_contents="['model', 'extra', 'optimizer']"

train_prompt_bsz=0
gen_prompt_bsz=1
ppo_mini_bsz=16
grpo_n=4
max_prompt_length=512
max_response_length=4096
max_token_len_per_gpu=8192
total_rollout_steps=null
lr_decay_steps=$((512 * 400))

train_tp=2
train_pp=4
train_cp=1
train_ep=32
train_etp=1

rollout_total_tp=32
rollout_dp=4
if (( rollout_total_tp % rollout_dp != 0 )); then
    echo "rollout_total_tp (${rollout_total_tp}) must be divisible by rollout_dp (${rollout_dp})" >&2
    exit 1
fi
rollout_tp=$((rollout_total_tp / rollout_dp))
rollout_ep=32
rollout_dtype=bfloat16
train_dtype=bfloat16
rollout_max_num_seqs=32
rollout_gpu_memory_utilization=0.5
update_weights_bucket_megabytes=8192

draft_nnodes=1
draft_ngpus=8
draft_tp=2
draft_quantization=bfloat16
speculative_num_steps=5

# On-policy/synchronous mode for the fully_async architecture.
staleness_threshold=0
trigger_parameter_sync_step=1
require_batches=1
partial_rollout=False

use_dynamic_bsz=False

########################### Parameter Arrays ###########################

DATA=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.prompt_key=prompt
    data.truncation=left
    data.trust_remote_code=True
    data.train_batch_size=${train_prompt_bsz}
    data.gen_batch_size=${gen_prompt_bsz}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.return_raw_chat=True
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.use_fused_kernels=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa
)

ACTOR_ROLLOUT_REF_COMMON=(
    actor_rollout_ref.hybrid_engine=False
    actor_rollout_ref.nccl_timeout=10800
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=10
    actor_rollout_ref.actor.optim.lr_decay_style=constant
    actor_rollout_ref.actor.optim.lr_decay_steps=${lr_decay_steps}
    actor_rollout_ref.actor.optim.weight_decay=0.1
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_bsz}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_token_len_per_gpu}
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.actor.use_rollout_log_probs=True
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
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
    rollout.nnodes=${NNODES_ROLLOUT}
    rollout.n_gpus_per_node=${GPUS_PER_NODE}
    rollout.n=${grpo_n}
    rollout.total_rollout_steps=${total_rollout_steps}
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.enable_decoupled_spec=True
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.data_parallel_size=${rollout_dp}
    actor_rollout_ref.rollout.pipeline_model_parallel_size=1
    actor_rollout_ref.rollout.expert_parallel_size=${rollout_ep}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_memory_utilization}
    actor_rollout_ref.rollout.dtype=${rollout_dtype}
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs}
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${update_weights_bucket_megabytes}
    +actor_rollout_ref.rollout.engine_kwargs.sglang.log_level=info
    +actor_rollout_ref.rollout.engine_kwargs.sglang.log_level_http=info
    +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_dp_attention=True
    actor_rollout_ref.rollout.n=${grpo_n}
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${max_token_len_per_gpu}
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
)

DRAFT=(
    +draft.model_path="${DRAFT_MODEL_PATH}"
    +draft.tokenizer_path="${DRAFT_MODEL_PATH}"
    +draft.load_format=auto
    +draft.quantization=${draft_quantization}
    +draft.nnodes=${draft_nnodes}
    +draft.ngpus=${draft_ngpus}
    +draft.tp_size=${draft_tp}
    +draft.speculative_num_steps=${speculative_num_steps}
    +draft.output_dir="${DSPEC_OUTPUT_DIR}"
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${max_token_len_per_gpu}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${train_etp}
    actor_rollout_ref.ref.megatron.param_offload=True
)

REWARD=(
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
    algorithm.rollout_correction.bypass_mode=True
)

ASYNC_TRAINING=(
    async_training.staleness_threshold=${staleness_threshold}
    async_training.trigger_parameter_sync_step=${trigger_parameter_sync_step}
    async_training.require_batches=${require_batches}
    async_training.partial_rollout=${partial_rollout}
    async_training.use_trainer_do_validate=False
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.n_gpus_per_node=${GPUS_PER_NODE}
    trainer.nnodes=${NNODES_TRAIN}
    trainer.save_freq=-1
    trainer.val_before_train=False
    trainer.test_freq=10
    trainer.total_epochs=10
    trainer.resume_mode=auto
    +ray_kwargs.ray_init.address=auto
)

########################### Launch ###########################

export HYDRA_FULL_ERROR=1
PYTHONUNBUFFERED=1 python3 -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path=config \
    --config-name=fully_async_ppo_megatron_trainer.yaml \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ACTOR_ROLLOUT_REF_COMMON[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${DRAFT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${ASYNC_TRAINING[@]}" \
    "${TRAINER[@]}" \
    "$@"
