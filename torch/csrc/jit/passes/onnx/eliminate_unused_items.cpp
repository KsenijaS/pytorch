#include <torch/csrc/jit/passes/onnx/eliminate_unused_items.h>
#include <torch/csrc/jit/passes/onnx/helper.h>

#include <c10/util/Optional.h>
#include <algorithm>

namespace torch {
namespace jit {

namespace onnx {
using namespace ::c10::onnx;
}

std::vector<at::Tensor> getValues(
    Node* node,
    const ValueToParamPairMap& valsToParamsMap) {
  size_t numInputs = node->inputs().size();
  std::vector<at::Tensor> inputTensorValues;
  inputTensorValues.reserve(numInputs);
  for (auto val : node->inputs()) {
    if (val->node()->kind() == prim::Param) {
      printf("=========================PARAM======================\n");
      auto itr = valsToParamsMap.find(val);
      printf("============FIND===================\n");
      if (itr == valsToParamsMap.end()) {
        continue;
      }
      inputTensorValues.push_back(itr->second.second.toTensor());
    } else if (val->node()->kind() == onnx::Constant) {
      printf("====================CONSTANT=====================\n");
      inputTensorValues.push_back(val->node()->t(attr::value));
    } else {
      printf("=======================CONTINUE======================\n");
      continue;
    }
  }
  //AT_ASSERT(inputTensorValues.size() == numInputs);
  return inputTensorValues;
}

static void fuseConvBachNorm(Block* b, ValueToParamPairMap& valsToParamsMap) {
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    for (auto* child_block : it->blocks()) {
      fuseConvBachNorm(child_block, valsToParamsMap);
    }
    if (it->kind() == onnx::Conv and it->next()->kind() == onnx::BatchNormalization){
      auto bnNode = it->next();
      auto origconvNode = *it;
      auto epsilon = bnNode->f(attr::epsilon);
      auto w_conv_value = getValues(origconvNode, valsToParamsMap);
      auto bn_value = getValues(bnNode, valsToParamsMap);
      auto bn_scale = bn_value[0];
      auto bn_B = bn_value[1];
      auto bn_mean = bn_value[2];
      auto bn_var = bn_value[3];
      auto w_conv = w_conv_value[0];

      bn_var.add(epsilon);
      bn_var.sqrt();
      bn_scale.div(bn_var);

      auto w_conv_flatten = w_conv.flatten(1, -1);
      for (size_t i = 0; i < w_conv.size(0); i++) {
        w_conv_flatten[i].mul(bn_scale[i]);
      }
      auto w_conv_reshape = w_conv_flatten.reshape_as(w_conv);

      bn_mean.mul(bn_scale);
      bn_B.sub(bn_mean);

      printf("=========================== bn_B =========================== %lu \n", bn_B.dim());

      Node* convNode = b->owningGraph()->create(onnx::Conv, bnNode->outputs().size());
      for (size_t i = 0; i < convNode->outputs().size(); ++i) {
        convNode->outputs()[i]->copyMetadata(bnNode->outputs()[i]);
      }

      convNode->copyAttributes(*origconvNode);
      convNode->insertBefore(origconvNode);
      convNode->addInput(origconvNode->inputs().at(0));

      auto conv_W = b->addInput();
      valsToParamsMap.insert({conv_W, std::make_pair(conv_W->debugName(), w_conv_reshape)});
      conv_W->inferTypeFrom(w_conv_reshape);
      convNode->addInput(conv_W);
      
      auto conv_B = b->addInput();
      valsToParamsMap.insert({conv_B, std::make_pair(conv_B->debugName(), bn_B)});
      conv_B->inferTypeFrom(bn_B);
      convNode->addInput(conv_B);

      bnNode->replaceAllUsesWith(convNode);
      bnNode->removeAllInputs();
      //origconvNode->destroy();
      bnNode->destroy();
      //origconvNode->destroy();

      //auto s = w_conv.size(0);
      //w_conv.flatten(0, 1);
      // auto bn_B = it->inputs().at(2)->node()->fs(attr::value);
      // auto bn_mean = it->inputs().at(3)->node()->fs(attr::value);
      // auto bn_var = it->inputs().at(4)->node()->fs(attr::value);
      // auto w_conv = orig_conv_node->inputs().at(1)->node()->fs(attr::value);
      
      // Calculate new value of initializers of conv node
      // bn_var.add(epsilon);
      // bn_var = sqrt(bn_var);
      // bn_scale = bn_scale / bn_var;
      // printf("=======================CONV O================= %lu\n", w_conv.size(0));
      // printf("=======================CONV 1================= %lu\n", w_conv.size(1));
      // printf("=======================CONV 2================= %lu\n", w_conv.size(2));
      // printf("=======================CONV 3================= %lu\n", w_conv.size(3));
      // printf("=======================CONV FLATTEN 0================= %lu\n", w_conv_flatten.size(0));
      // printf("=======================CONV FLATTEN 1================= %lu\n", w_conv_flatten.size(1));
      // printf("=======================CONV FLATTEN DIM================= %lu\n", w_conv_reshape.dim());
      // printf("=======================CONV RESHAPE O================= %lu\n", w_conv_reshape.size(0));
      // printf("=======================CONV RESHAPE 1================= %lu\n", w_conv_reshape.size(1));
      // printf("=======================CONV RESHAPE 2================= %lu\n", w_conv_reshape.size(2));
      // printf("=======================CONV RESHAPE 3================= %lu\n", w_conv_reshape.size(3));
    }
  }
}

void EliminateUnusedItemsONNX(Block* b, ParamMap& paramsDict) {
  auto valsToParamsMap = buildValueToParamsMap(b, paramsDict);
  fuseConvBachNorm(b, valsToParamsMap);
  eraseUnusedValuesFromMap(valsToParamsMap);
  eraseUnusedBlockInputs(b);
  buildParamsMapFromValueToParamsMap(valsToParamsMap, paramsDict);
  return;
}

} // namespace jit
} // namespace torch
