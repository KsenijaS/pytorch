#include <torch/csrc/jit/passes/onnx/fuse_bn_conv.h>
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
      auto itr = valsToParamsMap.find(val);
      if (itr == valsToParamsMap.end()) {
        continue;
      }
      inputTensorValues.push_back(itr->second.second.toTensor());
    } else if (val->node()->kind() == onnx::Constant) {
      inputTensorValues.push_back(val->node()->t(attr::value));
    } else {
      continue;
    }
  }
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
      //auto b_conv =w_conv_value[1];
      at::Tensor b_conv;

      bn_var.add(epsilon);
      bn_var.sqrt();
      bn_scale.div(bn_var);

      // Calculate weight
      auto w_conv_flatten = w_conv.flatten(1, -1);
      for (size_t i = 0; i < w_conv.size(0); i++) {
        w_conv_flatten[i].mul(bn_scale[i]);
      }
      auto w_conv_reshape = w_conv_flatten.reshape_as(w_conv);

      // Calculate bias
      if (origconvNode->inputs().size() == 3) {
        b_conv = w_conv_value[1];
        b_conv.sub(bn_mean);
        b_conv.mul(bn_scale);
        b_conv.add(bn_B);
      } else {
        bn_mean.mul(bn_scale);
        bn_B.sub(bn_mean);
        b_conv = bn_B;
      }

      Node* convNode = b->owningGraph()->create(onnx::Conv, bnNode->outputs().size());
      for (size_t i = 0; i < convNode->outputs().size(); ++i) {
        convNode->outputs()[i]->copyMetadata(bnNode->outputs()[i]);
      }

      convNode->copyAttributes(*origconvNode);
      convNode->insertBefore(bnNode);
      convNode->addInput(origconvNode->inputs().at(0));

      auto conv_W = b->addInput();
      valsToParamsMap.insert({conv_W, std::make_pair(conv_W->debugName(), w_conv_reshape)});
      conv_W->inferTypeFrom(w_conv_reshape);
      convNode->addInput(conv_W);
      
      auto conv_B = b->addInput();
      valsToParamsMap.insert({conv_B, std::make_pair(conv_B->debugName(), b_conv)});
      conv_B->inferTypeFrom(bn_B);
      convNode->addInput(conv_B);

      bnNode->replaceAllUsesWith(convNode);
      bnNode->removeAllInputs();
      it->removeAllInputs();
      bnNode->destroy();
      it.destroyCurrent();
    }
  }
}

void FuseBNConvONNX(Block* b, ParamMap& paramsDict) {
  auto valsToParamsMap = buildValueToParamsMap(b, paramsDict);
  fuseConvBachNorm(b, valsToParamsMap);
  //eraseUnusedValuesFromMap(valsToParamsMap);
  //eraseUnusedBlockInputs(b);
  buildParamsMapFromValueToParamsMap(valsToParamsMap, paramsDict);
  return;
}

} // namespace jit
} // namespace torch
