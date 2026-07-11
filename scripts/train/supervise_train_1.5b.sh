#!/usr/bin/env bash
# Supervisor for the 1.5B S3 PPO run.
#
# Runs scripts/train/train_s3_1.5b_4gpu.sh in a restart loop. On any non-zero
# exit (e.g. a rollout worker dies from OOM/SIGSEGV), it cleans up orphaned GPU
# processes and relaunches, resuming from the latest saved checkpoint by pointing
# the actor/critic init paths at the newest global_step_* HF checkpoint dir.
#
# Checkpoints are HF model folders (model.safetensors + config), so "resume"
# just re-inits weights from them. Optimizer state and the global step counter
# are not restored (this fork saves neither), so total step accounting restarts;
# training continues from the saved weights, which is what matters for progress.
#
# Env knobs:
#   S3_LOG_DIR       (default /data/ariy/s3-logs)
#   S3_MAX_ATTEMPTS  (default 100)
#   S3_RESTART_DELAY (default 20 seconds)
set -uo pipefail

SEED="${1:-42}"
EXP="s3_1.5b_full_${SEED}"
CKPT_ROOT="verl_checkpoints/${EXP}"
LOG_DIR="${S3_LOG_DIR:-/data/ariy/s3-logs}"
MAX_ATTEMPTS="${S3_MAX_ATTEMPTS:-100}"
RESTART_DELAY="${S3_RESTART_DELAY:-20}"

mkdir -p "$LOG_DIR"
SUP_LOG="$LOG_DIR/supervisor.log"

log() { echo "===== [$(date '+%Y-%m-%d %H:%M:%S')] $* =====" | tee -a "$SUP_LOG"; }

latest_ckpt() {  # $1 = actor|critic -> prints the checkpoint holding the most-recently-saved weights
  local sub="$CKPT_ROOT/$1" d t newest_dir best=0
  [ -d "$sub" ] || return 0
  # Select by the newest FILE mtime inside each checkpoint dir, not by the step
  # number and not by the directory mtime. On resume this fork restarts
  # global_steps at 0, so newer checkpoints overwrite lower-numbered dirs
  # (e.g. global_step_25) in place. Overwriting files does not bump the parent
  # dir's mtime, so both numeric sort and `ls -dt` would wrongly pick an older,
  # higher-numbered checkpoint. The newest inner-file mtime always identifies
  # the most-trained weights.
  for d in "$sub"/global_step_*/; do
    [ -d "$d" ] || continue
    t=$(find "$d" -maxdepth 1 -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1)
    [ -z "$t" ] && continue
    if awk "BEGIN{exit !($t > $best)}"; then
      best="$t"
      newest_dir="${d%/}"
    fi
  done
  [ -n "$newest_dir" ] && echo "$newest_dir"
}

cleanup() {
  pkill -9 -f "verl.trainer.main_ppo" 2>/dev/null || true
  # Kill anything still holding the training GPUs (0,1,2).
  local g pid
  for g in 0 1 2; do
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" 2>/dev/null); do
      kill -9 "$pid" 2>/dev/null || true
    done
  done
  sleep 8
}

attempt=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
  attempt=$((attempt + 1))
  cleanup

  a=$(latest_ckpt actor)
  c=$(latest_ckpt critic)
  if [ -n "$a" ] && [ -n "$c" ]; then
    export S3_ACTOR_MODEL="$a"
    export S3_CRITIC_MODEL="$c"
    export S3_VAL_BEFORE=false   # skip the pre-train validation on resumes
    log "attempt $attempt/$MAX_ATTEMPTS RESUME actor=$a critic=$c"
  else
    unset S3_ACTOR_MODEL S3_CRITIC_MODEL 2>/dev/null || true
    export S3_VAL_BEFORE=true
    log "attempt $attempt/$MAX_ATTEMPTS FRESH (base model)"
  fi

  bash scripts/train/train_s3_1.5b_4gpu.sh "$SEED" > "$LOG_DIR/training-launch.log" 2>&1
  rc=$?

  if [ "$rc" -eq 0 ]; then
    log "attempt $attempt exited 0 -- training complete"
    break
  fi
  log "attempt $attempt crashed rc=$rc -- restarting in ${RESTART_DELAY}s"
  sleep "$RESTART_DELAY"
done

log "supervisor exiting after $attempt attempt(s)"
