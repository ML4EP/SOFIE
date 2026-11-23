#include "SOFIE/RModelPlugin.hxx"

namespace SOFIE {

namespace {
std::unique_ptr<RModelPlugin> &ModelPluginSlot()
{
   static std::unique_ptr<RModelPlugin> slot;
   return slot;
}
} // namespace

void InstallModelPlugin(std::unique_ptr<RModelPlugin> plugin)
{
   ModelPluginSlot() = std::move(plugin);
}

RModelPlugin *ActiveModelPlugin()
{
   return ModelPluginSlot().get();
}

} // namespace SOFIE
