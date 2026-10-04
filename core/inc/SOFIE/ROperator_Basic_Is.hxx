// Deprecated forwarding header: renamed to follow the ROperator_<Name>.hxx convention.
// Include "SOFIE/ROperator_BasicIs.hxx" instead.
#ifndef SOFIE_ROperator_Basic_Is_FWD
#define SOFIE_ROperator_Basic_Is_FWD

#include "SOFIE/ROperator_BasicIs.hxx"

namespace SOFIE {

template <EBasicIsOperator Op>
using ROperator_Basic_Is = ROperator_BasicIs<Op>;

} // namespace SOFIE

#endif
