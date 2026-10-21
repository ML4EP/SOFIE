// Formatting of the generated code: the indentation of the code and the layout of the includes of the header.

#include "SOFIE/model/RModel.hxx"
#include "SOFIE/operators/ROperator_Relu.hxx"
#include "SOFIE/operators/ROperator_Gemm.hxx"

#include "SOFIE/utils/SOFIE_formatter.hxx"

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace SOFIE;

namespace {

std::string WithoutWhitespace(const std::string &s)
{
   std::string out;
   for (char c : s)
      if (!std::isspace(static_cast<unsigned char>(c)))
         out += c;
   return out;
}

std::vector<std::string> Lines(const std::string &s)
{
   std::vector<std::string> lines;
   std::istringstream in(s);
   for (std::string l; std::getline(in, l);)
      lines.push_back(l);
   return lines;
}

} // namespace

TEST(FormatGeneratedCode, IndentsBlocks)
{
   const std::string code = "struct S {\nint a;\nvoid f() {\nif (a) {\nreturn;\n}\n}\n};\n";
   const std::string expected = "struct S {\n   int a;\n   void f() {\n      if (a) {\n         return;\n      }\n   }\n};\n";
   EXPECT_EQ(FormatGeneratedCode(code), expected);
}

TEST(FormatGeneratedCode, ContentsOfNamespacesAreNotIndented)
{
   const std::string code = "namespace A{\nnamespace B {\nstruct S {\nint a;\n};\n}\n}\n";
   const std::string expected = "namespace A{\nnamespace B {\nstruct S {\n   int a;\n};\n}\n}\n";
   EXPECT_EQ(FormatGeneratedCode(code), expected);
}

TEST(FormatGeneratedCode, BracesInStringsAndCommentsAreIgnored)
{
   const std::string code = "void f() {\nout << \"{ \";\nchar c = '}';\n// a { in a comment\n/* and } here */\nint x;\n}\n";
   const std::string expected =
      "void f() {\n   out << \"{ \";\n   char c = '}';\n   // a { in a comment\n   /* and } here */\n   int x;\n}\n";
   EXPECT_EQ(FormatGeneratedCode(code), expected);
}

TEST(FormatGeneratedCode, BodiesWithoutBraces)
{
   const std::string code = "void f() {\nif (a)\nx = 1;\nelse\nx = 2;\nfor (int i = 0; i < n; i++)\ny += i;\nz = 0;\n}\n";
   const std::string expected =
      "void f() {\n   if (a)\n      x = 1;\n   else\n      x = 2;\n   for (int i = 0; i < n; i++)\n      y += i;\n   z = 0;\n}\n";
   EXPECT_EQ(FormatGeneratedCode(code), expected);
}

TEST(FormatGeneratedCode, ContinuationLines)
{
   const std::string code = "void f() {\nfoo(a,\nb,\nc);\nint x = a\n+ b;\n}\n"
                            "S::S(int q)\n: m(q)\n{\n}\n";
   const std::string expected = "void f() {\n   foo(a,\n      b,\n      c);\n   int x = a\n      + b;\n}\n"
                                "S::S(int q)\n   : m(q)\n{\n}\n";
   EXPECT_EQ(FormatGeneratedCode(code), expected);
}

TEST(FormatGeneratedCode, AccessLabelsPreprocessorAndBlankLines)
{
   const std::string code = "struct S {\n\nint a;\npublic:\n\n\n#if defined(X)\nint b;\n#endif\n\n};\n";
   const std::string expected = "struct S {\n   int a;\npublic:\n\n   #if defined(X)\n   int b;\n   #endif\n};\n";
   EXPECT_EQ(FormatGeneratedCode(code), expected);
}

TEST(FormatGeneratedCode, SwitchCases)
{
   const std::string code = "void f(int v) {\nswitch (v) {\ncase 0:\nreturn 1;\ncase 1: {\nint t = 2;\nreturn t;\n}\ndefault:\nreturn 0;\n}\n}\n";
   const std::string expected =
      "void f(int v) {\n   switch (v) {\n      case 0:\n         return 1;\n      case 1: {\n         int t = 2;\n         return t;\n      }\n      default:\n         return 0;\n   }\n}\n";
   EXPECT_EQ(FormatGeneratedCode(code), expected);
}

TEST(FormatGeneratedCode, RawStringsAreKeptVerbatim)
{
   const std::string code = "const char *t() {\nreturn R\"X(\n  keep {\n\n    this   \n)X\";\n}\nint after;\n";
   const std::string expected = "const char *t() {\n   return R\"X(\n  keep {\n\n    this   \n)X\";\n}\nint after;\n";
   EXPECT_EQ(FormatGeneratedCode(code), expected);
}

TEST(FormatGeneratedCode, InitializerListElementsAreAligned)
{
   const std::string code = "constexpr std::array<T, 2> dims{\nmakeDims(a),\nmakeDims(b)\n};\nenum class E {\nA,\nB\n};\n";
   const std::string expected = "constexpr std::array<T, 2> dims{\n   makeDims(a),\n   makeDims(b)\n};\nenum class E {\n   A,\n   B\n};\n";
   EXPECT_EQ(FormatGeneratedCode(code), expected);
}

TEST(FormatGeneratedCode, WrappedTemplateAndLambdaBodies)
{
   const std::string code = "template <typename T,\ntypename U>\nvoid f() {\nauto c = [&](int x) {\nreturn x;\n};\nfoo(a, [&] {\nbody();\n});\n}\n";
   const std::string expected =
      "template <typename T,\n   typename U>\nvoid f() {\n   auto c = [&](int x) {\n      return x;\n   };\n   foo(a, [&] {\n      body();\n   });\n}\n";
   EXPECT_EQ(FormatGeneratedCode(code), expected);
}

TEST(FormatGeneratedCode, MultiLineMacroAndStringContinuation)
{
   const std::string code = "#define M(x) \\\n do { \\\n x; \\\n } while (0)\nconst char *s = \"a \\\n   b\";\n";
   const std::string expected = "#define M(x) \\\n   do { \\\n   x; \\\n   } while (0)\nconst char *s = \"a \\\n   b\";\n";
   EXPECT_EQ(FormatGeneratedCode(code), expected);
}

TEST(FormatGeneratedCode, MalformedInputIsHandled)
{
   const std::vector<std::string> inputs = {"", "\n", "{", "}", "}}}\n)))\nint a;\n", "{{{{\n((((\nint a;\n",
                                            "const char *s = R\"x(\n never closed {\nint a;\n", "int a;\n/* never closed {\n  int b;\n",
                                            "const char *s = \"abc\nint a;\n", "\\\nint a;\n", "void f() {\r\nint a;\r\n}\r\n"};
   for (const auto &input : inputs) {
      const std::string once = FormatGeneratedCode(input);
      EXPECT_EQ(WithoutWhitespace(once), WithoutWhitespace(input)) << input;
      EXPECT_EQ(FormatGeneratedCode(once), once) << input;
   }
}

TEST(FormatGeneratedCode, IndentationIsBoundedForPathologicalNesting)
{
   std::string code;
   for (int i = 0; i < 2000; i++)
      code += "{\n";
   code += "x;\n";
   for (int i = 0; i < 2000; i++)
      code += "}\n";
   const std::string once = FormatGeneratedCode(code);
   EXPECT_LT(once.size(), 2000u * 2 * 200u);
   EXPECT_EQ(FormatGeneratedCode(once), once);
}

TEST(FormatGeneratedCode, IndependentOfTheInputIndentation)
{
   const std::string code = "namespace N{\nstruct S {\npublic:\nS(int x)\n: a(x)\n{\n}\nint a;\nvoid f(int x,\nint y) {\nif (x)\nreturn;\nelse if (y)\ny++;\nswitch (x) {\ncase 1:\nbreak;\n}\n}\n};\n}\n";
   std::string indented;
   for (const auto &l : Lines(code))
      indented += std::string(7, ' ') + l + "\n";
   EXPECT_EQ(FormatGeneratedCode(indented), FormatGeneratedCode(code));
}

TEST(FormatGeneratedCode, OnlyWhitespaceChangesAndIsIdempotent)
{
   const std::string code = "namespace N{\nstruct S {\n\tint a;\n   void f(int x,\n\t         int y) {\n if (x) { return; }\n}\n};\n}//N\n";
   const std::string once = FormatGeneratedCode(code);
   EXPECT_EQ(WithoutWhitespace(once), WithoutWhitespace(code));
   EXPECT_EQ(FormatGeneratedCode(once), once);
}

namespace {

// the include lines of the generated code
std::vector<std::string> Includes(const std::string &code)
{
   std::vector<std::string> inc;
   for (const auto &l : Lines(code))
      if (l.rfind("#include", 0) == 0)
         inc.push_back(l);
   return inc;
}

bool IsStandardHeader(const std::string &include) { return include.rfind("#include <", 0) == 0 && include.find('/') == std::string::npos; }

} // namespace

TEST(FormatGeneratedCode, CpuIncludesAreMergedSortedWithoutRepetitions)
{
   // Gemm needs helper functions, which need standard headers the model already includes (e.g. <algorithm>)
   RModel model("IncludesCpu", "now");
   model.AddInputTensorInfo("A", ETensorType::FLOAT, std::vector<size_t>{2, 3});
   model.AddInputTensorName("A");
   std::vector<float> weights(12, 1.f);
   model.AddInitializedTensor("B", ETensorType::FLOAT, std::vector<std::size_t>{3, 4}, weights.data());
   model.AddOperator(std::make_unique<ROperator_Gemm<float>>(1.f, 0.f, 0, 0, "A", "B", "Y"));
   model.AddOutputTensorNameList({"Y"});
   model.Generate(Options::kNoWeightFile);
   const auto includes = Includes(model.ReturnGenerated());
   ASSERT_FALSE(includes.empty());

   EXPECT_EQ(std::set<std::string>(includes.begin(), includes.end()).size(), includes.size()) << "repeated include";
   std::vector<std::string> standard;
   for (const auto &i : includes)
      if (IsStandardHeader(i))
         standard.push_back(i);
   EXPECT_TRUE(std::is_sorted(standard.begin(), standard.end()));
   // one block of standard headers
   EXPECT_EQ(std::find(includes.begin(), includes.end(), standard.front()) - includes.begin(), 0);
   EXPECT_EQ(std::find(includes.begin(), includes.end(), standard.back()) - includes.begin(),
             static_cast<long>(standard.size()) - 1);
}

TEST(FormatGeneratedCode, GpuIncludesAreStandardThenAlpakaAndSofieBlas)
{
   RModel model("IncludesGpu", "now");
   model.AddInputTensorInfo("X", ETensorType::FLOAT, std::vector<Dim>{Dim(3), Dim(5)});
   model.AddInputTensorName("X");
   model.AddOperator(std::make_unique<ROperator_Relu<float>>("X", "Y"));
   model.AddOutputTensorNameList({"Y"});
   model.GenerateGPU_ALPAKA(Options::kNoWeightFile);

   const auto lines = Lines(model.ReturnGenerated());
   std::vector<std::string> includes;
   size_t firstAlpaka = lines.size();
   for (size_t i = 0; i < lines.size(); i++) {
      if (lines[i].rfind("#include", 0) != 0)
         continue;
      includes.push_back(lines[i]);
      if (lines[i] == "#include <alpaka/alpaka.hpp>")
         firstAlpaka = i;
   }
   ASSERT_LT(firstAlpaka, lines.size());
   ASSERT_GE(firstAlpaka, 1u);
   EXPECT_EQ(lines[firstAlpaka - 1], "") << "blank line between the standard headers and alpaka";
   EXPECT_EQ(lines[firstAlpaka + 1], "#include <sofieBLAS/sofieBLAS.hpp>");
   EXPECT_EQ(std::set<std::string>(includes.begin(), includes.end()).size(), includes.size()) << "repeated include";
   // standard headers sorted and all before alpaka
   std::vector<std::string> standard;
   for (const auto &i : includes)
      if (IsStandardHeader(i))
         standard.push_back(i);
   EXPECT_TRUE(std::is_sorted(standard.begin(), standard.end()));
   EXPECT_EQ(includes[standard.size() - 1], standard.back());
}

TEST(FormatGeneratedCode, GeneratedSessionIsIndented)
{
   RModel model("IndentGpu", "now");
   model.AddInputTensorInfo("X", ETensorType::FLOAT, std::vector<Dim>{Dim(3), Dim(5)});
   model.AddInputTensorName("X");
   model.AddOperator(std::make_unique<ROperator_Relu<float>>("X", "Y"));
   model.AddOutputTensorNameList({"Y"});
   model.GenerateGPU_ALPAKA(Options::kNoWeightFile);
   const std::string code = model.ReturnGenerated();
   // idempotent: the code leaves the generation already formatted
   EXPECT_EQ(FormatGeneratedCode(code), code);
   const auto lines = Lines(code);
   auto session = std::find(lines.begin(), lines.end(), "struct Session {");
   ASSERT_NE(session, lines.end());
   EXPECT_EQ(session[1].rfind("   ", 0), 0u) << "the members of Session are indented: " << session[1];
   EXPECT_EQ(std::count(code.begin(), code.end(), '\t'), 0) << "no tabs";
}
