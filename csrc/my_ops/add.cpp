#include <torch/extension.h>

namespace my_ops
{
torch::Tensor add(const torch::Tensor & t1_, const torch::Tensor & t2_)
{
  torch::Tensor t1 = t1_.contiguous();
  torch::Tensor t2 = t2_.contiguous();
  torch::Tensor result = torch::empty(t1.sizes(), t1.options());
  const float * t1_ptr = t1.data_ptr<float>();
  const float * t2_ptr = t2.data_ptr<float>();
  float * result_ptr = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); ++i) {
    result_ptr[i] = t1_ptr[i] + t2_ptr[i];
  }
  return result;
}

TORCH_LIBRARY(my_ops, m)
{
  m.def("add", add);
}

}  // namespace my_ops