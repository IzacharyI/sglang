#include <ATen/hip/HIPContext.h>
#include <cstdint>
#include <cstdlib>
#include <torch/all.h>

namespace {

int gdn_sort_threshold() {
  // Sorting improves locality for large TP8/NVH8 batches, but is slower for
  // mid-size batches. Keep the tuned threshold override from the standalone
  // kernel wrapper.
  static int threshold = []() {
    const char* env = std::getenv("HIP_GDN_SORT_IDX_BS");
    return env ? std::atoi(env) : 192;
  }();
  return threshold;
}

struct GdnSortCache {
  const void* last_ptr = nullptr;
  int last_batch_size = 0;
  torch::Tensor sorted_indices;
  torch::Tensor perm_i32;
};

GdnSortCache gdn_sort_cache;

}  // namespace

extern "C" {

void launch_gdn_decode_iasm(
    const void* query,
    const void* key,
    const void* value,
    const void* a_input,
    const void* b_input,
    const void* dt_bias,
    const void* A_log,
    const void* indices,
    void* state,
    void* output,
    int batch_size,
    int seq_length,
    int num_v_blocks,
    bool use_qk_l2norm,
    float scale,
    int num_k_heads,
    int num_v_heads,
    const int* batch_perm,
    hipStream_t stream);

void launch_state_transpose(
    void* state,
    const void* indices,
    int batch_size,
    int num_v_heads,
    hipStream_t stream);

void launch_state_transpose_multi_layer(
    void* state_base,
    const void* indices,
    const void* slot_layout,
    int target_layout,
    int num_layers,
    int batch_size,
    int num_v_heads,
    int64_t layer_stride_floats,
    hipStream_t stream);

}  // extern "C"

void hip_gdn_decode_asm(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor dt_bias,
    torch::Tensor A_log,
    torch::Tensor indices,
    torch::Tensor state,
    torch::Tensor output,
    int64_t batch_size,
    int64_t seq_length,
    int64_t num_v_blocks,
    bool use_qk_l2norm,
    double scale,
    int64_t num_k_heads,
    int64_t num_v_heads) {
  TORCH_CHECK(query.is_cuda(), "query must be a CUDA/HIP tensor");
  TORCH_CHECK(key.is_cuda(), "key must be a CUDA/HIP tensor");
  TORCH_CHECK(value.is_cuda(), "value must be a CUDA/HIP tensor");
  TORCH_CHECK(state.is_cuda(), "state must be a CUDA/HIP tensor");
  TORCH_CHECK(output.is_cuda(), "output must be a CUDA/HIP tensor");
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA/HIP tensor");
  TORCH_CHECK(indices.scalar_type() == torch::kInt32, "indices must be int32");
  TORCH_CHECK(dt_bias.scalar_type() == torch::kBFloat16, "dt_bias must be bfloat16");
  TORCH_CHECK(output.scalar_type() == value.scalar_type(), "output dtype must match value dtype");

  auto stream = at::hip::getCurrentHIPStream().stream();

  const int* batch_perm_ptr = nullptr;
  torch::Tensor sorted_indices;
  torch::Tensor perm_i32;
  const int batch_size_i = static_cast<int>(batch_size);
  const int num_k_heads_i = static_cast<int>(num_k_heads);
  const int num_v_heads_i = static_cast<int>(num_v_heads);
  const int sort_threshold = gdn_sort_threshold();
  if (sort_threshold > 0 && batch_size_i >= sort_threshold && num_k_heads_i == 2 && num_v_heads_i == 8) {
    const void* current_ptr = indices.data_ptr();
    if (gdn_sort_cache.last_ptr == current_ptr && gdn_sort_cache.last_batch_size == batch_size_i) {
      sorted_indices = gdn_sort_cache.sorted_indices;
      perm_i32 = gdn_sort_cache.perm_i32;
    } else {
      auto perm_i64 = torch::argsort(indices, /*dim=*/0, /*descending=*/false);
      sorted_indices = indices.index_select(0, perm_i64);
      perm_i32 = perm_i64.to(torch::kInt32);
      gdn_sort_cache.last_ptr = current_ptr;
      gdn_sort_cache.last_batch_size = batch_size_i;
      gdn_sort_cache.sorted_indices = sorted_indices;
      gdn_sort_cache.perm_i32 = perm_i32;
    }
    batch_perm_ptr = perm_i32.data_ptr<int>();
  }

  launch_gdn_decode_iasm(
      query.data_ptr(),
      key.data_ptr(),
      value.data_ptr(),
      a.data_ptr(),
      b.data_ptr(),
      dt_bias.data_ptr(),
      A_log.data_ptr(),
      batch_perm_ptr ? sorted_indices.data_ptr() : indices.data_ptr(),
      state.data_ptr(),
      output.data_ptr(),
      batch_size_i,
      static_cast<int>(seq_length),
      static_cast<int>(num_v_blocks),
      use_qk_l2norm,
      static_cast<float>(scale),
      num_k_heads_i,
      num_v_heads_i,
      batch_perm_ptr,
      stream);

  const auto err = hipGetLastError();
  TORCH_CHECK(
      err == hipSuccess,
      "hip_gdn_decode_asm launch failed: ",
      hipGetErrorString(err),
      " (BS=",
      batch_size,
      " SQ=",
      seq_length,
      " NKH=",
      num_k_heads,
      " NVH=",
      num_v_heads,
      " NVB=",
      num_v_blocks,
      " l2norm=",
      use_qk_l2norm,
      ")");
}

void hip_gdn_state_transpose(torch::Tensor state, torch::Tensor indices, int64_t batch_size, int64_t num_v_heads) {
  TORCH_CHECK(state.is_cuda(), "state must be a CUDA/HIP tensor");
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA/HIP tensor");
  TORCH_CHECK(indices.scalar_type() == torch::kInt32, "indices must be int32");
  auto stream = at::hip::getCurrentHIPStream().stream();
  launch_state_transpose(
      state.data_ptr(), indices.data_ptr(), static_cast<int>(batch_size), static_cast<int>(num_v_heads), stream);
}

void hip_gdn_state_transpose_multi_layer(
    torch::Tensor state_base,
    torch::Tensor indices,
    torch::Tensor slot_layout,
    int64_t target_layout,
    int64_t num_layers,
    int64_t batch_size,
    int64_t num_v_heads,
    int64_t layer_stride_floats) {
  TORCH_CHECK(state_base.is_cuda(), "state_base must be a CUDA/HIP tensor");
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA/HIP tensor");
  TORCH_CHECK(slot_layout.is_cuda(), "slot_layout must be a CUDA/HIP tensor");
  TORCH_CHECK(indices.scalar_type() == torch::kInt32, "indices must be int32");
  TORCH_CHECK(slot_layout.scalar_type() == torch::kInt8, "slot_layout must be int8");
  TORCH_CHECK(state_base.is_contiguous(), "state_base must be contiguous");

  auto stream = at::hip::getCurrentHIPStream().stream();
  launch_state_transpose_multi_layer(
      state_base.data_ptr(),
      indices.data_ptr(),
      slot_layout.data_ptr(),
      static_cast<int>(target_layout),
      static_cast<int>(num_layers),
      static_cast<int>(batch_size),
      static_cast<int>(num_v_heads),
      layer_stride_floats,
      stream);
}
