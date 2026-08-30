# ============================================================
# GPU profile: G4 / Ada / L4 / Blackwell class
# All defaults preserve values supplied by the caller.
# ============================================================

# Leave headroom for CUDA runtime and vLLM workspace.
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
# vLLM defaults to 1024; hybrid/mamba models hit "max_num_seqs exceeds available
# Mamba cache blocks" during CUDA graph capture, so cap concurrency here.
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-512}"
# Optional: set a smaller model or context in .env when the assigned GPU is limited.
# export VLLM_MAX_MODEL_LEN=32768

# Speculative decoding. Ornith-1.5-35B-A3B ships 1 MTP layer
# (text_config.mtp_num_hidden_layers=1). method="mtp" with no "model" key makes
# vLLM reuse the target checkpoint's own MTP weights; the older per-arch names
# (qwen3_5_mtp / qwen3_next_mtp) still work but are deprecated.
# num_speculative_tokens=3 mirrors SGLang's --speculative-num-steps 3; vLLM
# then reuses that single MTP layer 3 times, trading acceptance rate for depth.
# Not VLLM_-prefixed: vLLM warns about unknown VLLM_* env vars it doesn't own.
# `${VAR+set}` (not `${VAR:-}`) so an explicitly empty value disables it:
#   export SPECULATIVE_CONFIG=    # -> no --speculative-config at all
if [[ -z "${SPECULATIVE_CONFIG+set}" ]]; then
  SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":3}'
fi
export SPECULATIVE_CONFIG
