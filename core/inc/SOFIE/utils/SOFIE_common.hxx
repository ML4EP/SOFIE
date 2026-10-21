#ifndef SOFIE_SOFIE_COMMON
#define SOFIE_SOFIE_COMMON

#include "SOFIE/utils/RTensor.hxx"

#include <span>

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace SOFIE {

enum class ETensorType{
   UNDEFINED = 0, FLOAT = 1, UINT8 = 2, INT8 = 3, UINT16 = 4, INT16 = 5, INT32 = 6, INT64 = 7, STRING = 8, BOOL = 9, //order sensitive
    FLOAT16 = 10, DOUBLE = 11, UINT32 = 12, UINT64 = 13, COMPLEX64 = 14, COMPLEX28 = 15, BFLOAT16 = 16
};

enum class EActivationType{
   UNDEFINED = 0, RELU = 1, SOFTMAX = 2, SIGMOID = 3, LEAKYRELU = 4, TANH = 5, ELU = 6
};

constexpr size_t GetTypeSize(ETensorType type) {
    switch (type) {
        case ETensorType::FLOAT:     return sizeof(float);
        case ETensorType::DOUBLE:    return sizeof(double);
        case ETensorType::UINT8:     return sizeof(uint8_t);
        case ETensorType::INT8:      return sizeof(int8_t);
        case ETensorType::UINT16:    return sizeof(uint16_t);
        case ETensorType::INT16:     return sizeof(int16_t);
        case ETensorType::INT32:     return sizeof(int32_t);
        case ETensorType::INT64:     return sizeof(int64_t);
        case ETensorType::UINT32:    return sizeof(uint32_t);
        case ETensorType::UINT64:    return sizeof(uint64_t);
        case ETensorType::BOOL:      return sizeof(bool);
        case ETensorType::STRING:    return sizeof(std::string);
        default: return 0;
    }
}

typedef std::int64_t int_t;

std::string ConvertTypeToString(ETensorType type);
ETensorType ConvertStringToType(std::string type);

// find if a string represents a number
bool IsInteger(const std::string & s);

struct Dim{
   bool isParam = false;
   size_t dim = 0;
   std::string param;

    // default constructor (for I/O)
   Dim() {}

   // constructor for a parametric dimension with the option to pass a default dim value
   // We use -1 for dim to indicate that the param dimension is an expression (e.g. "d1+d2")
   // in case the string represents a number make Dim not parametric
   Dim(const std::string & p, size_t d = 0) : isParam(true), dim(d), param(p)
   {
      if (IsInteger(p)) {
            isParam = false;
            dim = std::stoi(p);
      }
   }

   // constructor for a non-parametric dimension
   Dim(size_t d) : dim(d) {}

   std::string GetVal() const {
      // cast to int64_t for negative shape values
      return (isParam) ? param : std::to_string(static_cast<int64_t>(dim));
   }

   std::ostream& operator<< (std::ostream& os) const {
      os << GetVal();
      return os;
   }

   bool operator==(const Dim& rhs) const {
       return (isParam && rhs.isParam) ? param == rhs.param : dim == rhs.dim;
   }
   bool operator!=(const Dim& rhs) const {
       return !(*this == rhs);
   }
};

//bool operator==(const Dim& lhs, const Dim& rhs);
inline std::ostream & operator<< (std::ostream &os, const Dim &d) {
   os << d.GetVal();
   return os;
}

struct InputTensorInfo{
   ETensorType type;
   std::vector<Dim> shape;
};

struct TensorInfo{
   ETensorType type;
   std::vector<size_t> shape;
};

struct DynamicTensorInfo{
   ETensorType type;
   std::vector<Dim> shape;
};

// template traits for Tensor Shape
template <typename T>
struct TensorShape {};
template<>
struct TensorShape<Dim> {
   static bool IsDim() { return true; }
};
template<>
struct TensorShape<size_t> {
   static bool IsDim() { return false; }
};

// template traits for Tensor type
template <typename T>
struct TensorType {};
template<>
struct TensorType<float> {
   static const std::string Name() { return "float"; }
};
template<>
struct TensorType<double> {
   static const std::string Name() { return "double"; }
};
template<>
struct TensorType<int64_t> {
   static const std::string Name() { return "int64_t"; }
};
template<>
struct TensorType<int32_t> {
   static const std::string Name() { return "int32_t"; }
};
template<>
struct TensorType<uint32_t> {
   static const std::string Name() { return "uint32_t"; }
};
template<>
struct TensorType<uint64_t> {
   static const std::string Name() { return "uint64_t"; }
};
template<>
struct TensorType<bool> {
   static const std::string Name() { return "bool"; }
};
template<>
struct TensorType<int8_t> {
   static const std::string Name() { return "int8_t"; }
};
template<>
struct TensorType<uint8_t> {
   static const std::string Name() { return "uint8_t"; }
};

struct TensorMemoryInfo {
   std::string_view tensor_name;
   size_t tensor_size;

   TensorMemoryInfo split(const std::string_view new_name, size_t new_size) {
        if (new_size > tensor_size) {
            throw std::invalid_argument("New size exceeds available tensor size.");
        }
        tensor_size -= new_size;
        return TensorMemoryInfo{new_name, new_size};
   }

    // Method to merge another struct into this one
   void merge(const TensorMemoryInfo& other) {
        tensor_size += other.tensor_size;
   }
};

struct MemoryPoolInfo {

   // ordered map with chunk_idx as key and TensorMemoryInfo as value
   std::map<size_t, TensorMemoryInfo> total_stack;

   // ordered map with chunk_idx as key and chunk_size as value
   std::map<size_t, size_t> available_stack;
};

std::vector<Dim> ConvertShapeToDim(const std::vector<size_t> & shape);

std::vector<size_t> ConvertShapeToInt(const std::vector<Dim> & shape);

#include "SOFIE/utils/Shared/ConvertShapeToLength.inc"
std::size_t ConvertShapeToLength(const std::vector<Dim> & shape);

#include "SOFIE/utils/Shared/ConvertShapeToString.inc"
std::string ConvertDimShapeToString(const std::vector<Dim> & shape);

std::string ConvertDimShapeToLength(const std::vector<Dim> & shape);


template<class T>
std::string ConvertValToString(T value) {
   std::stringstream ret;
   ret << std::to_string(value);
   return ret.str();
}
// float specialization
template<>
inline std::string ConvertValToString<float>(float value) {
   std::stringstream ret;
   // special case for infinity and Nan
   if (std::isinf(value))
         ret << (value > 0 ? "std::numeric_limits<float>::infinity()" :
                                  "-std::numeric_limits<float>::infinity()");
   else if (std::isnan(value))
         ret << "std::numeric_limits<float>::quiet_NaN()";
   else {
      ret << std::setprecision(std::numeric_limits<float>::max_digits10);
      ret << value;
   }
   return ret.str();
}
// double specialization
template<>
inline std::string ConvertValToString<double>(double value) {
   std::stringstream ret;
   // special case for infinity and Nan
   if (std::isinf(value))
         ret << (value > 0 ? "std::numeric_limits<double>::infinity()" :
                                  "-std::numeric_limits<double>::infinity()");
   else if (std::isnan(value))
         ret << "std::numeric_limits<double>::quiet_NaN()";
   else {
      ret << std::setprecision(std::numeric_limits<double>::max_digits10);
      ret << value;
   }
   return ret.str();
}
// int64_t specialization for INT64_MIN
template<>
inline std::string ConvertValToString<int64_t>(int64_t value) {
   std::stringstream ret;
   if (value == INT64_MIN)
      ret << "INT64_MIN";
   else
      ret << std::to_string(value);
   return ret.str();
}


// convert list of values in a string taking into account the precision
template<class T>
std::string ConvertValuesToString(size_t n, const T * data, size_t maxprint = -1) {
   std::stringstream ret;
   ret << "{ ";
   for (size_t i = 0; i < std::min(n,maxprint); i++) {
      ret << ConvertValToString(data[i]);
      if (i < n-1) ret << ", ";
      if (i < n-1 && i == maxprint-1) ret << "..... ";
   }
   ret << "}";
   return ret.str();
}
template<class T>
std::string ConvertValuesToString(const std::vector<T> & data, size_t maxprint = 5) {
  return ConvertValuesToString(data.size(), data.data(), maxprint);
}

class InitializedTensor {
public:
   InitializedTensor() = default;
   InitializedTensor(ETensorType type, std::span<std::size_t> shape, std::shared_ptr<void> data, bool typeConstant = false)
      : fConstant(typeConstant), fType{type}, fShape{shape.begin(), shape.end()}, fData{data}
   {
   }

   ETensorType const &type() const { return fType; }
   std::vector<std::size_t> const &shape() const { return fShape; }
   std::shared_ptr<void> const &sharedptr() const { return fData; }
   // query if tensor comes from a Constant operator
   bool IsConstantTensor() const { return fConstant;}
   // query if tensor needs to be written in a weight file. Constant tensors are not written in a separate file
   bool IsWeightTensor() const { return !fConstant && !fIsNotWritable;}
   // check if a Tensor is Writable (need to be written in the file or in the generated code (e.g. as a constant tensor)
   // if an initialized tensors is used in a constant operator at compile time does not need to be written and can be omitted in
   // the generated code
   bool IsNotWritable() const { return fIsNotWritable; }
   // set not writable initialized tensors - i.e. tensor that must not be written in a file
   void SetNotWritable() { fIsNotWritable = true;}
   // set writable initialized tensors - i.e. tensor that must be written in a file
   void SetWritable() { fIsNotWritable = false;}
   // set as constant (needed for non-float initialized tensors)
   void SetConstant() { fConstant = true;}

   template <class T = void>
   T const *data() const
   {
      return static_cast<T const *>(fData.get());
   }

private:
   bool  fConstant = false;      ///< Flag specifying if tensor is a Constant one (coming from a Constant operator)
   bool  fIsNotWritable = false; ///< Flag to indicate that tensor values do not need to be written as weight or generated code
   ETensorType fType;               ///< Encodes the type of the data
   std::vector<std::size_t> fShape; ///< The shape of the data in terms of elements in each dimension
   std::shared_ptr<void> fData;     ///<! Transient shared data
};

template <typename T>
ETensorType GetTemplatedType(T /*obj*/ ){
   if (std::is_same<T, float>::value) return ETensorType::FLOAT;
   if (std::is_same<T, uint8_t>::value) return ETensorType::UINT8;
   if (std::is_same<T, int8_t>::value) return ETensorType::INT8;
   if (std::is_same<T, uint16_t>::value) return ETensorType::UINT16;
   if (std::is_same<T, int16_t>::value) return ETensorType::INT16;
   if (std::is_same<T, int32_t>::value) return ETensorType::INT32;
   if (std::is_same<T, int64_t>::value) return ETensorType::INT64;
   if (std::is_same<T, std::string>::value) return ETensorType::STRING;
   if (std::is_same<T, bool>::value) return ETensorType::BOOL;
   //float16 unimplemented
   if (std::is_same<T, double>::value) return ETensorType::DOUBLE;
   if (std::is_same<T, uint32_t>::value) return ETensorType::UINT32;
   if (std::is_same<T, uint64_t>::value) return ETensorType::UINT64;
   //complex 64, 28, bfloat 16 unimplemented
}

namespace UTILITY{



// clean operator and tensor names
std::string Clean_name(std::string input_tensor_name);

// Check if two shapes are equal
bool AreSameShape(const std::vector<size_t>&, const std::vector<size_t>&);
bool AreSameShape(const std::vector<size_t>&, const std::vector<Dim>&);
bool AreSameShape(const std::vector<Dim>&, const std::vector<Dim>&);


// Multidirectional broadcast a list of tensors to the same shape
std::vector<size_t> MultidirectionalBroadcastShape(std::vector<std::vector<size_t>>);

// Multidirectional broadcast two shapes to the same shape

std::pair<int, std::vector<size_t>> MultidirectionalBroadcastShape(std::vector<size_t> &, std::vector<size_t> &);
std::vector<size_t> UnidirectionalBroadcastShape(std::vector<size_t> &, std::vector<size_t> &);

std::pair<int, std::vector<Dim>> MultidirectionalBroadcastShape(std::vector<Dim> &, std::vector<Dim> &);



#include "SOFIE/utils/Shared/BroadcastConvBias.inc"
#include "SOFIE/utils/Shared/Broadcast.inc"

/// compute stride of a tensor given its shape (assume layout is row-major)
std::vector<size_t> ComputeStrideFromShape(const std::vector<size_t> & shape);
std::vector<Dim> ComputeStrideFromShape(const std::vector<Dim> & shape);

/// layout of the independent 1-D slices of a row-major tensor along one axis, as code expressions
struct SliceInfo {
   std::string nBefore;        // slice groups before the axis ("1" for axis 0)
   std::string nAfter;         // slices per group after the axis
   std::string nSlices;        // nBefore * nAfter
   std::string nElements;      // length of a slice (the axis dimension)
   std::string strideAxis;     // step along the axis
   std::string strideBefore;   // step between slice groups ("0" for axis 0)
};
SliceInfo ComputeSliceInfo(const std::vector<Dim> & shape, size_t axis);


}  // end namespace UTILITY

//Utility functions to generate code
void EmitNestedLoops(std::stringstream &out, size_t loopRank, const std::vector<Dim> shape);
void CloseNestedLoops(std::stringstream &out, size_t loopRank);

//emit the out_0..out_{D-1} coordinate declarations of the flat kernel index elem_idx
//from the output strides and shape
void EmitOutputCoordsFromThreadIdx(std::string &op, const std::vector<Dim> &strides, const std::vector<Dim> &shape);


#include "SOFIE/utils/Shared/MemoryPlanner.inc"

inline std::string ConvertOutputTypeToString(ETensorType t) {
   // The std::vector<bool> is a special type that is not wrapping continuous memory.
   // We don't want to use it as a return type.
   if (t == ETensorType::BOOL) t = ETensorType::UINT8;
   return ConvertTypeToString(t);
}

struct HelperFunctionsCode {
   std::set<std::string> stdHeaders; // standard headers needed by the definitions
   std::string definitions;
   std::string cladDefinitions;
};

///
///
HelperFunctionsCode GenerateHelperFunctionsCode(const std::set<std::string> & neededHelpers,
                                                const std::string & modelNamespace,
                                                bool sgemmAlreadyDeclared = false);


} // namespace SOFIE

#endif
