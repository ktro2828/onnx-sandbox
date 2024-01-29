#include "attention_kernel.hpp"

#include <torch/torch.h>

namespace my_autograd
{
using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::variable_list;

struct AttentionWeightComputation : public Function<AttentionWeightComputation>
{
  static Tensor forward(
    AutogradContext * ctx, Tensor query_batch_cnt, Tensor key_batch_cnt, Tensor index_pair_batch,
    Tensor index_pair, Tensor query_features, Tensor key_features)
  {
    int b = query_batch_cnt.sizes().at(0);

    const auto & query_sizes = index_pair.sizes();
    const auto & key_sizes = key_features.sizes();

    int total_query_num = query_sizes.at(0);
    int local_size = query_sizes.at(1);

    int total_key_num = key_sizes.at(0);
    int nhead = key_sizes.at(1);
    int hdim = key_sizes.at(2);

    // save context
    // non-variable data
    ctx->saved_data["b"] = b;
    ctx->saved_data["total_query_num"] = total_query_num;
    ctx->saved_data["local_size"] = local_size;
    ctx->saved_data["total_key_num"] = total_key_num;
    ctx->saved_data["nhead"] = nhead;
    ctx->saved_data["hdim"] = hdim;
    // variable data
    variable_list to_save = {query_batch_cnt, key_batch_cnt,  index_pair_batch,
                             index_pair,      query_features, key_features};
    ctx->save_for_backward(to_save);

    Tensor output = torch::zeros({total_query_num, local_size, nhead}, torch::device(torch::kCUDA));

    attention_weight_computation_wrapper(
      b, total_query_num, local_size, total_key_num, nhead, hdim, query_batch_cnt, key_batch_cnt,
      index_pair_batch, index_pair, query_features, key_features, output);

    return output;
  }

  static variable_list backward(AutogradContext * ctx, variable_list grad_outputs)
  {
    // load saved context
    auto b = ctx->saved_data["b"].toInt();
    auto total_query_num = ctx->saved_data["total_query_num"].toInt();
    auto local_size = ctx->saved_data["local_size"].toInt();
    auto total_key_num = ctx->saved_data["total_key_num"].toInt();
    auto nhead = ctx->saved_data["nhead"].toInt();
    auto hdim = ctx->saved_data["hdim"].toInt();

    auto saved_variables = ctx->get_saved_variables();

    auto query_batch_cnt = saved_variables.at(0);
    auto key_batch_cnt = saved_variables.at(1);
    auto index_pair_batch = saved_variables.at(2);
    auto index_pair = saved_variables.at(3);
    auto query_features = saved_variables.at(4);
    auto key_features = saved_variables.at(5);

    auto grad_out = grad_outputs.at(0).contiguous();

    auto grad_query_features =
      torch::zeros({total_query_num, nhead, hdim}, torch::device(torch::kCUDA));
    auto grad_key_features =
      torch::zeros({total_key_num, nhead, hdim}, torch::device(torch::kCUDA));

    attention_weight_computation_grad_wrapper(
      b, total_query_num, local_size, total_key_num, nhead, hdim, query_batch_cnt, key_batch_cnt,
      index_pair_batch, index_pair, query_features, key_features, grad_out, grad_query_features,
      grad_key_features);

    Tensor none;
    return {none, none, none, none, grad_query_features, grad_key_features};
  }
};  // struct AttentionWeightComputation

struct AttentionValueComputation : public Function<AttentionValueComputation>
{
  static Tensor forward(
    AutogradContext * ctx, Tensor query_batch_cnt, Tensor key_batch_cnt, Tensor index_pair_batch,
    Tensor index_pair, Tensor attn_weight, Tensor value_features)
  {
    int b = query_batch_cnt.sizes().at(0);

    const auto & query_sizes = index_pair.sizes();
    const auto & value_sizes = value_features.sizes();

    int total_query_num = query_sizes.at(0);
    int local_size = query_sizes.at(1);

    int total_key_num = value_sizes.at(0);
    int nhead = value_sizes.at(1);
    int hdim = value_sizes.at(2);

    // save context
    // non-variable data
    ctx->saved_data["b"] = b;
    ctx->saved_data["total_query_num"] = total_query_num;
    ctx->saved_data["local_size"] = local_size;
    ctx->saved_data["total_key_num"] = total_key_num;
    ctx->saved_data["nhead"] = nhead;
    ctx->saved_data["hdim"] = hdim;
    // variable data
    variable_list to_save = {query_batch_cnt, key_batch_cnt, index_pair_batch,
                             index_pair,      attn_weight,   value_features};
    ctx->save_for_backward(to_save);

    Tensor output = torch::zeros({total_query_num, nhead, hdim}, torch::device(torch::kCUDA));

    attention_value_computation_wrapper(
      b, total_query_num, local_size, total_key_num, nhead, hdim, query_batch_cnt, key_batch_cnt,
      index_pair_batch, index_pair, attn_weight, value_features, output);

    return output;
  }

  static variable_list backward(AutogradContext * ctx, variable_list grad_outputs)
  {
    // load saved context
    auto b = ctx->saved_data["b"].toInt();
    auto total_query_num = ctx->saved_data["total_query_num"].toInt();
    auto local_size = ctx->saved_data["local_size"].toInt();
    auto total_key_num = ctx->saved_data["total_key_num"].toInt();
    auto nhead = ctx->saved_data["nhead"].toInt();
    auto hdim = ctx->saved_data["hdim"].toInt();

    auto saved_variables = ctx->get_saved_variables();

    auto query_batch_cnt = saved_variables.at(0);
    auto key_batch_cnt = saved_variables.at(1);
    auto index_pair_batch = saved_variables.at(2);
    auto index_pair = saved_variables.at(3);
    auto attn_weight = saved_variables.at(4);
    auto value_features = saved_variables.at(5);

    auto grad_out = grad_outputs.at(0).contiguous();

    auto grad_attn_weight =
      torch::zeros({total_query_num, local_size, nhead}, torch::device(torch::kCUDA));
    auto grad_value_features =
      torch::zeros({total_key_num, nhead, hdim}, torch::device(torch::kCUDA));

    attention_value_computation_grad_wrapper(
      b, total_query_num, local_size, total_key_num, nhead, hdim, query_batch_cnt, key_batch_cnt,
      index_pair_batch, index_pair, attn_weight, value_features, grad_out, grad_attn_weight,
      grad_value_features);

    Tensor none;
    return {none, none, none, none, grad_attn_weight, grad_value_features};
  }
};  // struct AttentionValueComputation

Tensor attention_weight_computation(
  Tensor query_batch_cnt, Tensor key_batch_cnt, Tensor index_pair_batch, Tensor index_pair,
  Tensor query_features, Tensor key_features)
{
  return AttentionWeightComputation::apply(
    query_batch_cnt, key_batch_cnt, index_pair_batch, index_pair, query_features, key_features);
}

Tensor attention_value_computation(
  Tensor query_batch_cnt, Tensor key_batch_cnt, Tensor index_pair_batch, Tensor index_pair,
  Tensor attn_weight, Tensor value_features)
{
  return AttentionValueComputation::apply(
    query_batch_cnt, key_batch_cnt, index_pair_batch, index_pair, attn_weight, value_features);
}

TORCH_LIBRARY(my_autograd, m)
{
  m.def(
     "attention_weight_computation(Tensor query_batch_cnt, Tensor key_batch_cnt, Tensor "
     "index_pair_batch, Tensor index_pair, Tensor query_features, Tensor key_features) -> Tensor")
    .def(
      "attention_value_computation(Tensor query_batch_cnt, Tensor key_batch_cnt, Tensor "
      "index_pair_batch, Tensor index_pair, Tensor attn_weight, Tensor value_features) -> Tensor");
}

TORCH_LIBRARY_IMPL(my_autograd, Autograd, m)
{
  m.impl("attention_weight_computation", attention_weight_computation)
    .impl("attention_value_computation", attention_value_computation);
}

}  // namespace my_autograd