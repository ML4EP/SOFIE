#include <vector>
#include <fstream>
#include <limits>

#include "SOFIE/RModel.hxx"
#include "SOFIE/RModelParser_ONNX.hxx"

#include  "gtest/gtest.h"

#include "test_helpers.h"

bool verbose = true;
int sessionId = 0;

void ExecuteSofieParser(std::string modelName) {
   SOFIE::RModelParser_ONNX parser;
   std::string inputName = modelName + ".onnx";
   std::cout << "parsing file " << inputName << std::endl;
   SOFIE::RModel model = parser.Parse(inputName);
   std::cout << "generating model.....\n";
   model.Generate();
   std::string outputName = modelName + ".hxx";
   std::cout << "writing model as header .....\n";
   model.OutputGenerated(); // outputName);
   std::cout << "output written in  " << outputName << std::endl;
}


std::vector<std::string> declaredModels;

int DeclareCode(std::string modelName)
{
   const std::string source = "#include \"" + modelName + ".hxx\"\nint main() {}\n";
   const std::string name = "sofie_model_check_" + std::to_string(declaredModels.size());
   {
      std::ofstream out(name + ".cxx");
      out << source;
   }
   const std::string command = std::string(SOFIE_TEST_CXX) + " " + SOFIE_TEST_CXX_FLAGS + " -I. -fsyntax-only " + name +
                               ".cxx > " + name + ".log 2>&1";
   if (sofieExec(command) != 0)
      return 0;
   declaredModels.push_back(modelName);
   return static_cast<int>(declaredModels.size());
}

std::vector<float> RunInference(std::vector<float> const &x, int sId)
{
   const std::string modelName = declaredModels.at(sId - 1);
   const std::string name = "sofie_model_run_" + std::to_string(sId);
   std::string source = "#include <cstdio>\n#include <vector>\n#include \"" + modelName + ".hxx\"\nint main()\n{\n";
   source += "   std::vector<float> x = " + floatArrayLiteral(x) + ";\n";
   source += "   SOFIE_" + modelName + "::Session s;\n";
   source += "   std::vector<float> result = s.infer(x.data());\n";
   source += "   for (float v : result) std::printf(\"%a\\n\", v);\n}\n";
   printf("doing inference.....");
   if (!compileGeneratedProgram(name, source))
      throw std::runtime_error("failed to compile the inference program for " + modelName);
   return parseFloatList(runGeneratedProgram(name));
}

void TestLinear(int nbatches, bool useBN = false, int inputSize = 10, int nlayers = 4)
{
   std::string modelName = "LinearModel";
   if (useBN) modelName += "_BN";
   modelName += "_B" + std::to_string(nbatches);

   // network parameters : nbatches, inputDim, nlayers
   std::vector<int> params = {nbatches, inputSize, nlayers};

   std::string command = std::string(SOFIE_TEST_PYTHON) + " LinearModelGenerator.py ";
   for (size_t i = 0; i < params.size(); i++)
      command += "  " + std::to_string(params[i]);
   if (useBN)
      command += "  --bn";

   printf("executing %s\n", command.c_str());
   sofieExec(command.c_str());

   ExecuteSofieParser(modelName);

   int id = DeclareCode(modelName);

   ASSERT_NE(id, 0) << "Declareing model code to interpreter failed!";

   // input data
   std::vector<float> xinput(nbatches * inputSize);
   for (int ib = 0; ib < nbatches; ib++) {
      std::vector<float> x1(inputSize, float(ib + 1));
      std::copy(x1.begin(), x1.end(), xinput.begin() + ib * inputSize);
   }

   auto result = RunInference(xinput, id);

   // read reference value from test file
   std::vector<float> refValue(result.size());

   std::ifstream f(std::string(modelName + ".out").c_str());
   for (size_t i = 0; i < refValue.size(); ++i) {
      f >> refValue[i];
      if (verbose)
         std::cout << " result " << result.at(i) << " reference " << refValue[i] << std::endl;
      EXPECT_NEAR(result.at(i), refValue[i], 10 * std::numeric_limits<float>::epsilon());
   }
}

void TestConv( std::string type, int nbatches, bool useBN = false, int ngroups = 2, int nchannels = 2, int nd = 4, int nlayers = 4, int usePool = 0)
{
   std::string modelName = "Conv" + type + "Model";
   if (useBN) modelName += "_BN";
   if (usePool == 1) modelName += "_MAXP";
   if (usePool == 2) modelName += "_AVGP";
   modelName += "_B" + std::to_string(nbatches);

   // input size is fixed to (nb, nc, nd, nd)
   int inputDim = nd;
   if (type == "2d") inputDim *= nd;
   if (type == "3d") inputDim *= nd*nd;

   const int inputSize = nchannels * inputDim;

   //const char *argv[5] = {}
   std::string argv[5];

   argv[0] = std::to_string(nbatches);
   //.c_str();
   argv[1] = std::to_string(nchannels);
   argv[2] = std::to_string(nd);
   argv[3] = std::to_string(ngroups); // for 3d this is depth size
   argv[4] = std::to_string(nlayers);
   std::string command = std::string(SOFIE_TEST_PYTHON) + " Conv" + type + "ModelGenerator.py ";
   for (int i = 0; i < 5; i++) {
      command += " ";
      command += argv[i];
   }
   if (useBN) command += "  --bn";
   if (usePool == 1) command += " --maxpool";
   if (usePool == 2) command += " --avgpool";
   printf("executing %s\n", command.c_str());
   sofieExec(command.c_str());

   // some model needs some simplifications
#ifdef USE_ONNXSIM
   if (usePool == 2) {
      printf("simplify onnx model using onnxsim tool \n");
      std::string cmd = std::string(SOFIE_TEST_PYTHON) + " -m onnxsim " + modelName + ".onnx " + modelName + ".onnx";
      int ret = sofieExec(cmd.c_str());
      if (ret != 0) {
         std::cout << "Error when simplifing ONNX model with AveragePool layer using onnx-simplifier (onnxsim) - skip the test" << std::endl;
         GTEST_SKIP();
         return;
      }
   }
#endif


   ExecuteSofieParser(modelName);

   int id = DeclareCode(modelName);

   ASSERT_NE(id, 0) << "Declareing model code to interpreter failed!";

   // input data
   std::vector<float> xinput(nbatches*inputSize);
   for (int ib = 0; ib < nbatches; ib++) {
      std::vector<float> x1(inputDim, float(ib + 1));
      std::vector<float> x2(inputDim, -float(ib + 1));
      // x1 and x2 are the two channels, if more channels will be with zero
      std::copy(x1.begin(), x1.end(), xinput.begin() + ib * inputSize);
      if (nchannels > 1)
         std::copy(x2.begin(), x2.end(), xinput.begin() + ib * inputSize + x1.size());
   }

   auto result = RunInference(xinput, id);


   // read reference value from test file
   std::vector<float> refValue(result.size());

   std::ifstream f(std::string(modelName + ".out").c_str());
   for (size_t i = 0; i < refValue.size(); ++i) {
      f >> refValue[i];
      if (verbose) std::cout << " result " << result.at(i) << " reference " << refValue[i] << std::endl;
      EXPECT_NEAR(result.at(i), refValue[i], 10 * std::numeric_limits<float>::epsilon());
   }
}

void TestRecurrent(std::string type, int nbatches, int inputSize = 5, int seqSize = 10, int hiddenSize = 3, int nlayers = 1)
{

   if (type.empty()) type = "RNN";
   std::string modelName = type + "Model";
   modelName += "_B" + std::to_string(nbatches);

   // network parameters : nbatches, inputDim, nlayers
   std::vector<int> params = {nbatches, inputSize, seqSize, hiddenSize, nlayers};

   std::string command = std::string(SOFIE_TEST_PYTHON) + " RecurrentModelGenerator.py ";
   for (size_t i = 0; i < params.size(); i++)
      command += "  " + std::to_string(params[i]);
   if (type == "LSTM")
      command += "  --lstm";
   else if (type == "GRU")
      command += "  --gru";

   printf("executing %s\n", command.c_str());
   sofieExec(command.c_str());
   // need to simplify obtained recurrent ONNX model
#ifdef USE_ONNXSIM
   printf("simplify onnx model using onnxsim tool \n");
   std::string cmd = std::string(SOFIE_TEST_PYTHON) + " -m onnxsim " + modelName + ".onnx " + modelName + ".onnx";
   int ret = sofieExec(cmd.c_str());
   if (ret != 0) {
      std::cout << "Error when simplifing ONNX Recurrent model using onnx-simplifier (onnxsim) - skip the test" << std::endl;
      GTEST_SKIP();
      return;
   }
#endif

   ExecuteSofieParser(modelName);

   int id = DeclareCode(modelName);

   std::cout << "id " << id << std::endl;

   ASSERT_NE(id, 0) << "Declareing model code to interpreter failed!";

   // input data
   std::vector<float> xinput(nbatches * seqSize * inputSize);
   for (int ib = 0; ib < nbatches; ib++) {
      for (int it = 0; it < seqSize; it++) {
         std::vector<float> x1(inputSize, std::pow(-1, ib + 2) * float(it + 1));
         std::copy(x1.begin(), x1.end(), xinput.begin() + ib * inputSize*seqSize + it * inputSize);
      }
   }
   if (verbose) {
      std::cout << " input data \n";
      int k = 0;
      for (int i = 0; i < nbatches; ++i) {
         for (int j = 0; j < seqSize; j++) {
            for (int l = 0; l < inputSize; l++) {
               std::cout << xinput[k++] << ", ";
            }
             std::cout << "\n";
         }
         std::cout << "\n";
      }
   }

   auto result = RunInference(xinput, id);

   // read reference value from test file
   std::vector<float> refValue(result.size());

   std::ifstream f(std::string(modelName + ".out").c_str());
   for (size_t i = 0; i < refValue.size(); ++i) {
      f >> refValue[i];
      if (verbose)
         std::cout << " result " << result.at(i) << " reference " << refValue[i] << std::endl;
      EXPECT_NEAR(result.at(i), refValue[i], 10 * std::numeric_limits<float>::epsilon());
   }
}

void TestConvTranspose( std::string type, int nbatches, bool useBN = false, int ngroups = 1, int nchannels = 2, int nd = 4, int nlayers = 4, int usePool = 0)
{
   std::string modelName = "ConvTrans" + type + "Model";
   if (useBN) modelName += "_BN";
   if (usePool == 1) modelName += "_MAXP";
   if (usePool == 2) modelName += "_AVGP";
   modelName += "_B" + std::to_string(nbatches);

   // input size is fixed to (nb, nc, nd, nd)
   int inputDim = nd;
   if (type == "2d") inputDim *= nd;
   if (type == "3d") inputDim *= nd*nd;

   const int inputSize = nchannels * inputDim;

   //const char *argv[5] = {}
   std::string argv[5];

   argv[0] = std::to_string(nbatches);
   //.c_str();
   argv[1] = std::to_string(nchannels);
   argv[2] = std::to_string(nd);
   argv[3] = std::to_string(ngroups); // for 3d this is depth size
   argv[4] = std::to_string(nlayers);
   std::string command = std::string(SOFIE_TEST_PYTHON) + " ConvTrans" + type + "ModelGenerator.py ";
   for (int i = 0; i < 5; i++) {
      command += " ";
      command += argv[i];
   }
   if (useBN) command += "  --bn";
   if (usePool == 1) command += " --maxpool";
   if (usePool == 2) command += " --avgpool";
   printf("executing %s\n", command.c_str());
   sofieExec(command.c_str());

   // some model needs some semplifications
   if (usePool == 2) {
      printf("simplify onnx model using onnxsim tool \n");
      std::string cmd = std::string(SOFIE_TEST_PYTHON) + " -m onnxsim " + modelName + ".onnx " + modelName + ".onnx";
      int ret = sofieExec(cmd.c_str());
      if (ret != 0) {
         std::cout << "Error when simplifing ONNX model with AveragePool layer using onnx-simplifier (onnxsim) - skip the test" << std::endl;
         GTEST_SKIP();
         return;
      }
   }


   ExecuteSofieParser(modelName);

   int id = DeclareCode(modelName);

   ASSERT_NE(id, 0) << "Declareing model code to interpreter failed!";

   // input data
   std::vector<float> xinput(nbatches*inputSize);
   for (int ib = 0; ib < nbatches; ib++) {
      std::vector<float> x1(inputDim, float(ib + 1));
      std::vector<float> x2(inputDim, -float(ib + 1));
      for (int i = 0; i < inputDim; i++) x1[i] = float(i)*float(ib+1);
      for (int i = 0; i < inputDim; i++) x2[i] = -float(i)*float(ib+1);
      // x1 and x2 are the two channels, if more channels will be with zero
      std::copy(x1.begin(), x1.end(), xinput.begin() + ib * inputSize);
      if (nchannels > 1)
         std::copy(x2.begin(), x2.end(), xinput.begin() + ib * inputSize + x1.size());
   }

   auto result = RunInference(xinput, id);


   // read reference value from test file
   std::vector<float> refValue(result.size());

   std::ifstream f(std::string(modelName + ".out").c_str());
   for (size_t i = 0; i < refValue.size(); ++i) {
      f >> refValue[i];
      if (verbose) std::cout << " result " << result.at(i) << " reference " << refValue[i] << std::endl;
      EXPECT_NEAR(result.at(i), refValue[i], 10 * std::numeric_limits<float>::epsilon());
   }
}


TEST(SOFIE, Linear_B1) {
   TestLinear(1);
}
TEST(SOFIE, Linear_B4)
{
   // test batch =4 (equal output size)
   TestLinear(4);
}
TEST(SOFIE,Conv2d_B1) {
   TestConv("2d", 1);
}
TEST(SOFIE, Conv2d_B4)
{
   TestConv("2d",4);
}
// test with batch normalization
TEST(SOFIE, Linear_BNORM_B8)
{
   TestLinear(8,true,5,4);
}
TEST(SOFIE, Conv2d_BNORM_B5)
{
   TestConv("2d",5,true);
}
// test with max pooling
TEST(SOFIE, Conv2d_MAXPOOL_B2)
{
   TestConv("2d",2,false,1,2,3,1,1);
}
// test with avg pooling
TEST(SOFIE, Conv2d_AVGPOOL_B2)
{
   TestConv("2d", 2, false, 1, 2, 4, 1, 2);
}

// test conv1d
TEST(SOFIE, Conv1d_B1)
{
   TestConv("1d", 1, false, 1, 2, 10, 1, 0);
}

// test conv3d
TEST(SOFIE, Conv3d_B1)
{
   TestConv("3d", 1, false, 3, 2, 3, 1, 0);
}

// Tets recurrent network
// test with avg pooling
TEST(SOFIE, RNN_B1)
{
   TestRecurrent("RNN", 1, 3, 5, 4, 1);
}

TEST(SOFIE, LSTM_B1)
{
   TestRecurrent("LSTM", 1, 3, 5, 4, 1);
}

TEST(SOFIE, GRU_B1)
{
   TestRecurrent("GRU", 1, 3, 5, 4, 1);
}

TEST(SOFIE, CONVTRANS2D_B1)
{
   TestConvTranspose("2d", 1);
}
