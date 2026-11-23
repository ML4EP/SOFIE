#ifndef SOFIE_RMODELPLUGIN
#define SOFIE_RMODELPLUGIN

#include "SOFIE/SOFIE_common.hxx"

#include <memory>
#include <string>
#include <string_view>

namespace SOFIE {

class RModel;

// Generator being run, so a plugin keys its target-specific work off the model's own terms.
enum class EGenerationTarget { CPU, ALPAKA };

// Points in the model's life where an optional library (e.g. quantization) rewrites the model or
// contributes to the generated code. Base SOFIE cannot link such a library, so the library
// installs itself here; generation runs unchanged when no plugin is installed.
class RModelPlugin {
public:
   virtual ~RModelPlugin() = default;

   // Runs at the start of Initialize, before any operator initializes. The only hook that may
   // replace the parsed operator list (RModel::UpdateOperators) or change operator attributes
   // that shape inference reads.
   virtual void OnParsed(RModel & /*model*/) {}

   // Runs at the end of the operator Initialize loop, once every tensor has a shape and type.
   virtual void OnInitialized(RModel & /*model*/) {}

   // Runs once per generation, after the generation options are applied and before any code is
   // written: rejects an unsupported target and installs lowered operators.
   virtual void PrepareGeneration(RModel & /*model*/, EGenerationTarget /*target*/) {}

   // Returns true when an initializer the generator would embed as a constant is instead kept as
   // a model weight (weight file and device buffer).
   virtual bool OwnsStorage(const RModel & /*model*/, const std::string & /*tensorName*/) const { return false; }

   // Returns code placed after the generated header's preamble and before the model code, given
   // the model code already generated: support functions that code calls, and comment lines.
   virtual std::string SupportCode(const RModel & /*model*/, std::string_view /*generatedCode*/) const { return {}; }
};

// Installs the plugin every model consults, replacing any previous one; nullptr removes it.
// ActiveModelPlugin returns the current plugin or null.
void InstallModelPlugin(std::unique_ptr<RModelPlugin> plugin);
RModelPlugin *ActiveModelPlugin();

} // namespace SOFIE

#endif
