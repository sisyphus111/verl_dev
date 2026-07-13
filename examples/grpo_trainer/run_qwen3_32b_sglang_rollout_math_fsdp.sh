#!/usr/bin/env bash
# Qwen3-32B ordinary SGLang rollout GRPO with FSDP.
# Target shape: 8 nodes, 64 GPUs total. Rollout uses TP=4.

set -xeuo pipefail

export RAY_DEDUP_LOGS=0

########################### Quick Config ###########################

NNODES=${NNODES:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

MODEL_PATH=${MODEL_PATH:-/mnt/hdfs/ord/models/Qwen/Qwen3-32B}
TRAIN_FILE=${TRAIN_FILE:-/mnt/hdfs/ord/datasets/DAPO-Math-17k/data/dapo-math-17k.parquet}
TEST_FILE=${TEST_FILE:-/mnt/hdfs/ord/datasets/AIME_2024/data/aime-2024.parquet}

project_name=${PROJECT_NAME:-verl_grpo_qwen3_sglang}
exp_name=${EXP_NAME:-qwen3_32b_sglang_tp4_64gpu_fsdp}
TRACE_DIR=${TRACE_DIR:-/mnt/hdfs/ord/${exp_name}/resp}
CKPT_DIR=${CKPT_DIR:-/mnt/hdfs/ord/${exp_name}/ckpt}

train_prompt_bsz=${TRAIN_PROMPT_BSZ:-64}
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-64}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-4}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-4096}
train_max_samples=${TRAIN_MAX_SAMPLES:--1}

rollout_tp=${ROLLOUT_TP:-4}
rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-64}
rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.75}
rollout_trace_max_samples_per_step_per_worker=${ROLLOUT_TRACE_MAX_SAMPLES_PER_STEP_PER_WORKER:-null}

total_training_steps=${TOTAL_TRAINING_STEPS:-200}
total_epochs=${TOTAL_EPOCHS:-1}
use_dynamic_bsz=${USE_DYNAMIC_BSZ:-False}

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
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa
)

ACTOR=(
    # FSDP backend
    actor_rollout_ref.actor.strategy=fsdp
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.use_torch_compile=False
    # Optimizer
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=10
    actor_rollout_ref.actor.optim.weight_decay=0.1
    # PPO/GRPO update batch configuration
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
    # Loss configuration
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
)

ROLLOUT=(
    # SGLang rollout engine
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.enable_decoupled_spec=False
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.data_parallel_size=1
    actor_rollout_ref.rollout.pipeline_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_memory_utilization}
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs}
    actor_rollout_ref.rollout.trace.trace_dir="${TRACE_DIR}"
    actor_rollout_ref.rollout.trace.max_samples_per_step_per_worker=${rollout_trace_max_samples_per_step_per_worker}
    +actor_rollout_ref.rollout.engine_kwargs.sglang.log_level=info
    +actor_rollout_ref.rollout.engine_kwargs.sglang.log_level_http=info
    # Generation parameters
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    # Old log-prob recomputation
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    # Validation generation
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
)

REF=(
    # FSDP ref policy
    actor_rollout_ref.ref.strategy=fsdp
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.use_torch_compile=False
    # Ref log-prob recomputation
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
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
    trainer.logger='["console"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.n_gpus_per_node=${GPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=1
    trainer.default_local_dir="${CKPT_DIR}"
    trainer.max_actor_ckpt_to_keep=200
    trainer.val_before_train=False
    trainer.test_freq=-1
    trainer.total_epochs=${total_epochs}
    trainer.total_training_steps=${total_training_steps}
    +ray_kwargs.ray_init.address=auto
)

########################### Launch ###########################

export HYDRA_FULL_ERROR=1
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name=ppo_trainer.yaml \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "$@"
