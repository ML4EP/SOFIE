#ifndef SOFIE_ROPERATOR_SubGraph
#define SOFIE_ROPERATOR_SubGraph

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>


namespace SOFIE{

   // operator dealing with subgraphs (such as If , Loop, etc..)

class ROperator_If final : public ROperator
{

private:

   std::string fNX;
   ETensorType fType = ETensorType::UNDEFINED;  // output type (support only one common type)
   std::vector<std::string> fNYs;
   std::shared_ptr<RModel> fModel_then;
   std::shared_ptr<RModel> fModel_else;
   std::string fInputSignature_modelThen;
   std::string fInputSignature_modelElse;

public:
   ROperator_If(){}
   ROperator_If(const std::string & nameX, const std::vector<std::string> & nameYs, std::unique_ptr<RModel> model_then, std::unique_ptr<RModel> model_else):
      fNX(UTILITY::Clean_name(nameX)), fNYs(nameYs), fModel_then(std::move(model_then)), fModel_else(std::move(model_else))
      {
         for (auto & n : fNYs)
            n = UTILITY::Clean_name(n);

         fInputTensorNames = { fNX };
         fOutputTensorNames.assign(fNYs.begin(), fNYs.end());
      }

   // the condition has one element: it is read without strides. The branches read the strided inputs of the model
   // through the strides given to the main Session
   bool SupportsStridedInput() const override { return true; }

   void Initialize(RModel& model) override {
       //input must be a graph input, or already initialized intermediate tensor
      if (model.CheckIfTensorAlreadyExist(fNX) == false){
        throw std::runtime_error("SOFIE If Op Input Tensor is not found in model");
      }
      //add the subgraph model to parent RModel and initialize them
      model.InitializeSubGraph(fModel_then);
      model.InitializeSubGraph(fModel_else);

      // generate input string signature for subgraphs
      fInputSignature_modelThen = fModel_then->GenerateInferSignature(false);
      fInputSignature_modelElse = fModel_else->GenerateInferSignature(false);

      // add the outputs
      for (size_t i = 0; i < fNYs.size(); i++) {
         // assume shape of then tensor is same of else tensor
         // if not need to make a parametric tensor output (tbd)
         auto soutput_name = fModel_then->GetOutputTensorNames()[i];
         // Dim-aware: the "then" branch's output may be dynamic (e.g. it
         // depends on a tensor defined in this If's own enclosing scope);
         // GetTensorShape() would throw for that case, while GetDimTensorShape()
         // (and the AddIntermediateTensor(..., vector<Dim>) overload below)
         // handle both the static and dynamic case uniformly.
         auto shape = fModel_then->GetDimTensorShape(soutput_name);
         auto type = fModel_then->GetTensorType(soutput_name);
         // the branches write directly into the output tensors of the If operator, which need the same static shape
         auto elseName = fModel_else->GetOutputTensorNames()[i];
         if (fModel_then->IsDynamicTensor(soutput_name) || fModel_else->IsDynamicTensor(elseName))
            throw std::runtime_error("SOFIE If Op does not support branch outputs with a dynamic shape");
         if (fModel_else->GetTensorShape(elseName) != fModel_then->GetTensorShape(soutput_name) ||
             fModel_else->GetTensorType(elseName) != type)
            throw std::runtime_error("SOFIE If Op requires the then and else branch outputs to have the same shape and type");
         if (i == 0)
            fType = type;
         else {
            if (type != fType)
               throw std::runtime_error("SOFIE If Op supports only all outputs of the same type");
         }
         model.AddIntermediateTensor(fNYs[i], fType, shape );
      }

   }


   // The branches are generated as sessions on the same queue (members fSession_<branch> of the session, see
   // RModel::GenerateSessionCode_GPU_ALPAKA). The condition is read back on the host to select the branch, which is run
   // on the input tensors of the model and whose outputs are copied into the output tensors of the If operator.
   std::string Generate_GPU_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      if (fType == ETensorType::UNDEFINED) {
         throw std::runtime_error("SOFIE If operator called to Generate without being initialized first");
      }
      std::stringstream out;
      out << "\n//------ If_GPU_ALPAKA\n";
      out << SP << "auto condHost_" << opName << " = alpaka::allocBuf<uint8_t, Idx>(hostAcc, Ext1D::all(Idx{1}));\n";
      out << SP << "alpaka::memcpy(queue, condHost_" << opName << ", deviceBuf_" << fNX << ");\n";
      out << SP << "alpaka::wait(queue);\n";
      auto branch = [&](RModel &model, const std::string &signature) {
         const std::string session = "fSession_" + model.GetName();
         std::string code = SP + SP + session + "._infer_impl(" + signature + ");\n";
         for (size_t i = 0; i < fNYs.size(); i++) {
            const std::string branchOutput = model.ResolveAliasTensor(model.GetOutputTensorNames()[i]);
            code += SP + SP + "alpaka::memcpy(queue, deviceBuf_" + fNYs[i] + ", " + session + ".deviceBuf_" +
                    branchOutput + ");\n";
         }
         return code;
      };
      out << SP << "if (*alpaka::getPtrNative(condHost_" << opName << ")) {\n";
      out << branch(*fModel_then, fModel_then->GenerateInferSignature_GPU_ALPAKA(false));
      out << SP << "} else {\n";
      out << branch(*fModel_else, fModel_else->GenerateInferSignature_GPU_ALPAKA(false));
      out << SP << "}\n";
      return out.str();
   }

   std::string Generate(std::string opName) override {
      opName = "op_" + opName;
      if (fType == ETensorType::UNDEFINED) {
         throw std::runtime_error("SOFIE If operator called to Generate without being initialized first");
      }
      std::stringstream out;
      out << "\n//------ If operator\n";
      // the code is generated inside doInfer, where the session is the (const) argument "session" ("this->" is
      // replaced by it) and the tensors are plain pointers. Each branch writes directly into the output tensors
      // of the If operator, by calling the doInfer function of its sub-session.
      auto call = [&](const RModel &branch, const std::string &signature) {
         std::string args = signature;
         for (const auto &y : fNYs)
            args += (args.empty() ? "" : ", ") + std::string("tensor_") + y;
         return SP + SP + "doInfer(this->fSession_" + branch.GetName() + ", " + args + ");\n";
      };
      out << SP << "if (tensor_" << fNX << "[0]) {\n";
      out << call(*fModel_then, fInputSignature_modelThen);
      out << SP << "} else {\n";
      out << call(*fModel_else, fInputSignature_modelElse);
      out << SP << "}\n";
      return out.str();
   }



};

}//SOFIE

#endif //SOFIE_ROPERATOR_Tanh
