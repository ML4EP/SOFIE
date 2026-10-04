
#include <SOFIE/RModel.hxx>
#include <SOFIE/RModelParser_ONNX.hxx>

#include <gtest/gtest.h>

#include <regex>
#include <string>
#include <vector>

#ifndef SOFIE_ONNX_MODELS_DIR
#define SOFIE_ONNX_MODELS_DIR "."
#endif

using namespace SOFIE;

namespace {

RModel generate(std::string const &modelName, OptimizationLevel level)
{
   RModelParser_ONNX parser;
   RModel model = parser.Parse(std::string(SOFIE_ONNX_MODELS_DIR) + "/" + modelName + ".onnx");
   model.SetOptimizationLevel(level);
   model.Generate(Options::kNoWeightFile);
   return model;
}

struct AliasCase {
   const char *model;
   const char *tensor;
   const char *owner;
   bool granted;
   const char *why;
};

const std::vector<AliasCase> aliasCases = {
   {"ReshapeAlias", "reshaped", "prod", true, "a reshape only reinterprets the shape of its input"},
   {"SliceIdentityAlias", "sliced", "prod", true, "a slice selecting everything does not change the data"},
   {"IdentityAlias", "ident", "prod", true, "an identity is the same data under another name"},
   {"ReshapeAliasGraphOutput", "out", "prod", false, "a graph output is written into the buffer of the caller"},
   {"AliasAcrossNewTensor", "reshaped", "prod", true, "the alias is read after another tensor was created"},
   {"AliasOwnerReadAfterAlias", "reshaped", "prod", true, "the aliased tensor is read after the last use of the alias"},
   {"AliasDynShape", "ident", "prod", true, "the aliased tensor comes from the dynamic memory pool"},
   {"AliasDynShapeAcrossNewTensor", "ident", "prod", true,
    "the alias is read after another tensor was created in the dynamic memory pool"},
   {"AliasChain", "squeezed", "prod", true, "Squeeze does not change the data"},
   {"AliasChain", "unsqueezed", "squeezed", true, "Unsqueeze does not change the data"},
   {"AliasChain", "flattened", "unsqueezed", true, "Flatten does not change the data"},
   {"AliasChain", "reshaped", "flattened", true, "Reshape does not change the data"},
   {"AliasChainSingle", "reshaped", "prod", true, "the single alias the chain above amounts to"},
};

std::regex aliasRegex(AliasCase const &c)
{
   return std::regex(std::string("auto\\s*\\*\\s*tensor_") + c.tensor + "\\s*=\\s*tensor_" + c.owner + "\\s*;");
}

std::regex copyRegex(AliasCase const &c)
{
   return std::regex(std::string("std::copy\\s*\\(\\s*tensor_") + c.owner + "\\b[^;]*\\btensor_" + c.tensor +
                     "\\s*\\)");
}

bool matches(std::string const &code, std::regex const &pattern)
{
   return std::regex_search(code, pattern);
}

}

TEST(SofieAlias, GrantedAtExtended)
{
   for (auto const &c : aliasCases) {
      SCOPED_TRACE(std::string(c.model) + ": " + c.tensor);
      RModel model = generate(c.model, OptimizationLevel::kExtended);
      EXPECT_EQ(model.IsAliasTensor(c.tensor), c.granted) << c.why;
   }
}

TEST(SofieAlias, RefusedAtBasic)
{
   for (auto const &c : aliasCases) {
      SCOPED_TRACE(std::string(c.model) + ": " + c.tensor);
      RModel model = generate(c.model, OptimizationLevel::kBasic);
      EXPECT_FALSE(model.IsAliasTensor(c.tensor));
   }
}

TEST(SofieAlias, EmittedCodeFollowsDecision)
{
   for (auto const &c : aliasCases) {
      SCOPED_TRACE(std::string(c.model) + ": " + c.tensor);
      const std::regex alias = aliasRegex(c);
      const std::regex copy = copyRegex(c);

      const std::string extended = generate(c.model, OptimizationLevel::kExtended).ReturnGenerated();
      EXPECT_EQ(matches(extended, alias), c.granted);
      EXPECT_EQ(matches(extended, copy), !c.granted);

      const std::string basic = generate(c.model, OptimizationLevel::kBasic).ReturnGenerated();
      EXPECT_FALSE(matches(basic, alias));
      EXPECT_TRUE(matches(basic, copy));
   }
}

TEST(SofieAlias, ChainCostsNoExtraMemory)
{
   const RModel chain = generate("AliasChain", OptimizationLevel::kExtended);
   const RModel single = generate("AliasChainSingle", OptimizationLevel::kExtended);
   EXPECT_EQ(chain.GetIntermediateTensorSize(), single.GetIntermediateTensorSize());
}
