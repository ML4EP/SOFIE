// Deprecated forwarding header: the header was renamed to fix a typo.
// Include "SOFIE/ROperator_Comparison.hxx" instead.
#ifndef SOFIE_ROperator_Comparision_FWD
#define SOFIE_ROperator_Comparision_FWD

#include "SOFIE/ROperator_Comparison.hxx"

namespace SOFIE {

using EComparisionOperator = EComparisonOperator;

template <typename T, EComparisonOperator Op>
using ROperator_Comparision = ROperator_Comparison<T, Op>;

} // namespace SOFIE

#endif
