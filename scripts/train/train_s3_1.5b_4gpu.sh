#!/usr/bin/env bash
set -euo pipefail

data_name=nq_hotpotqa_train
random_seed="${1:-42}"

export CUDA_VISIBLE_DEVICES=0,1,2
export DATA_DIR="${S3_TRAIN_DATA_DIR:-data/${data_name}}"
export BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct
# Actor/critic init paths. Default to the base model; the supervisor wrapper
# overrides these with the latest checkpoint dir (a HF model folder) to resume.
export ACTOR_MODEL="${S3_ACTOR_MODEL:-$BASE_MODEL}"
export CRITIC_MODEL="${S3_CRITIC_MODEL:-$BASE_MODEL}"
export EXPERIMENT_NAME="s3_1.5b_full_${random_seed}"
export VLLM_ATTENTION_BACKEND=XFORMERS
# Point the reward-model generator (verl/utils/reward_score/rag_2.py ->
# generator_llms/local_inst.py) at the locally served 1.5B generator so the
# vLLM OpenAI endpoint accepts the model name.
export S3_GENERATOR_MODEL="${S3_GENERATOR_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

mkdir -p train_logs verl_checkpoints

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files="$DATA_DIR/train_e5_s3.parquet" \
    data.val_files="$DATA_DIR/test_e5_s3_sampled.parquet" \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=96 \
    data.val_batch_size=24 \
    data.max_prompt_length=8000 \
    data.max_response_length=500 \
    data.max_start_length=2000 \
    data.max_obs_length=1400 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path="$ACTOR_MODEL" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=48 \
    actor_rollout_ref.actor.ppo_micro_batch_size=24 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=24 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=24 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.state_masking=true \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=true \
    critic.optim.lr_warmup_steps_ratio=0.01 \
    critic.model.path="$CRITIC_MODEL" \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size=12 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    trainer.critic_warmup=0 \
    "trainer.logger=['console']" \
    +trainer.val_only=false \
    +trainer.val_before_train="${S3_VAL_BEFORE:-true}" \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=3 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=100 \
    trainer.project_name=SearchAgent \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.total_epochs=4 \
    trainer.total_training_steps=1500 \
    trainer.default_local_dir="verl_checkpoints/$EXPERIMENT_NAME" \
    +data.random_seed="$random_seed" \
    max_turns=3 \
    +generator_llm="$BASE_MODEL" \
    +output_context_dir=data/output_sequences_s3_1.5b \
    retriever.url=http://127.0.0.1:3000/retrieve \
    retriever.topk=8 \
    2>&1 | tee "train_logs/$EXPERIMENT_NAME.log"
