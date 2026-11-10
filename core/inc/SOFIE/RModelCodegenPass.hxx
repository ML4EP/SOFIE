#ifndef SOFIE_RMODELCODEGENPASS
#define SOFIE_RMODELCODEGENPASS

#include "SOFIE/SOFIE_common.hxx"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace SOFIE {

class RModel;

// Generator being run, named in the model's own terms so a pass keys its target-specific work
// off this enum instead of a vocabulary of its own.
enum class ECodegenTarget { CPU, ALPAKA };

// Hooks in code generation where a pass library rewrites the model or contributes to the
// generated text. Code generation runs unchanged when no pass is installed.
class RModelCodegenPass {
public:
   virtual ~RModelCodegenPass() = default;

   // Runs at the start of Initialize, before any operator initializes. This is the only hook
   // that rewrites the parsed operator list itself (e.g. into a canonical form).
   virtual void Normalize(RModel & /*model*/) {}

   // Runs during Initialize, before the emitted operator set is fixed.
   virtual void Analyze(RModel &model) = 0;

   // Returns true when the model carries data the pass writes to a weight file instead of
   // embedding it in the generated source.
   virtual bool RequiresWeightFile(const RModel &model) const = 0;

   // Replaces operators with lowered equivalents for this target, leaving the parsed
   // graph intact. Runs once per generation.
   virtual void BuildLoweredView(RModel &model, ECodegenTarget target) = 0;

   // Declares the headers the pass's own runtime support needs, before lowering runs.
   virtual void ContributeSupportHeaders(RModel &model, ECodegenTarget target) = 0;

   // Returns true when the initializer is backed by storage the pass owns, so the generator
   // leaves it out of the embedded constants.
   virtual bool HasExternalStorage(const RModel &model, const std::string &tensorName) const = 0;

   // Returns comment lines appended to the generated header, each prefixed with "//".
   virtual std::vector<std::string> DiagnosticComments(const RModel &model) const = 0;
};

// Installs the pass every model consults while generating, replacing any previous one; nullptr
// removes it. InstalledCodegenPass returns the current pass or null.
void InstallCodegenPass(std::unique_ptr<RModelCodegenPass> pass);
RModelCodegenPass *InstalledCodegenPass();

} // namespace SOFIE

#endif
