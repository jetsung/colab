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
