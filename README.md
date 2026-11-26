# SOFIE

SOFIE (System for Optimized Fast Inference code Emit) converts trained machine
learning models in the [ONNX](https://onnx.ai) format into C++ code that
evaluates them. The generated code only depends on the C++ standard library,
so it can be compiled and used in any C++ program without depending on SOFIE
or on any machine learning framework.

SOFIE itself has no dependencies besides a C++17 compiler: the ONNX files are
read with a small self-contained decoder of their format.

## Building

```bash
git clone https://github.com/ML4EP/SOFIE.git
cmake -S SOFIE -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX=install
cmake --build build --target install -j10
```

This builds and installs the shared library `libSOFIE`. The library is meant
to be loaded at runtime, so its headers are not installed: the functions of its
C interface are looked up when the library is loaded, as in the examples below.

## Generating the code of a model

The C interface of SOFIE, declared in `inc/SOFIE/SOFIE.h`, only uses C types, so that
`libSOFIE` can be loaded at runtime (for example with `dlopen`) and its
functions called by looking up their symbols, without including any SOFIE
header or linking against the library:

```c
size_t SOFIE_GenerateCodeFromONNX(const char *onnx, size_t size, const char *modelName,
                                  unsigned int flags, char **code, char **error);
void SOFIE_Free(char *str);
```

`SOFIE_GenerateCodeFromONNX` takes the content of an ONNX file and returns the
code to evaluate it, with the weights of the model embedded in the code. It
returns the length of the code, or 0 if it can't be generated, in which case
`error` holds the error message. The returned strings are released with
`SOFIE_Free`.

The flags select optional features of the generated code:
- `SOFIE_USE_BLAS`: call the routines of a BLAS library, which must then be
  loaded when the code is used, instead of the plain C++ implementation that
  is part of the generated code
- `SOFIE_USE_VDT`: use the fast math functions of
  [vdt](https://github.com/dpiparo/vdt) if its headers are found when the
  generated code is compiled

### From Python

With `ctypes`:

```python
import ctypes

# The library has to be in the library path, e.g. in LD_LIBRARY_PATH
sofie = ctypes.CDLL("libSOFIE.so")  # libSOFIE.dylib on macOS
sofie.SOFIE_GenerateCodeFromONNX.argtypes = [
    ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_uint,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
]
sofie.SOFIE_GenerateCodeFromONNX.restype = ctypes.c_size_t
sofie.SOFIE_Free.argtypes = [ctypes.c_void_p]

with open("MyModel.onnx", "rb") as f:
    onnx = f.read()

code, error = ctypes.c_void_p(), ctypes.c_void_p()
length = sofie.SOFIE_GenerateCodeFromONNX(onnx, len(onnx), b"MyModel", 0, ctypes.byref(code), ctypes.byref(error))
try:
    if length == 0:
        raise RuntimeError(ctypes.string_at(error).decode())
    with open("MyModel.hxx", "w") as f:
        f.write(ctypes.string_at(code, length).decode())
finally:
    sofie.SOFIE_Free(code)
    sofie.SOFIE_Free(error)
```

### From C++

With `dlopen`, without including any SOFIE header (on Windows, use
`LoadLibrary` and `GetProcAddress` instead):

```cpp
#include <dlfcn.h>

#include <cstddef>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

// The functions of the C interface of SOFIE, which are looked up at runtime
using GenerateFunc_t = std::size_t (*)(const char *onnx, std::size_t size, const char *modelName,
                                       unsigned int flags, char **code, char **error);
using FreeFunc_t = void (*)(char *str);

int main()
{
   // The library has to be in the library path, e.g. in LD_LIBRARY_PATH
   void *sofie = dlopen("libSOFIE.so", RTLD_NOW); // libSOFIE.dylib on macOS
   if (!sofie) {
      std::cerr << dlerror() << std::endl;
      return 1;
   }
   auto generate = reinterpret_cast<GenerateFunc_t>(dlsym(sofie, "SOFIE_GenerateCodeFromONNX"));
   auto sofieFree = reinterpret_cast<FreeFunc_t>(dlsym(sofie, "SOFIE_Free"));

   std::ifstream input{"MyModel.onnx", std::ios::binary};
   const std::string onnx{std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};

   char *code = nullptr;
   char *error = nullptr;
   const std::size_t length = generate(onnx.data(), onnx.size(), "MyModel", 0, &code, &error);
   if (length == 0)
      std::cerr << error << std::endl;
   else
      std::ofstream{"MyModel.hxx"}.write(code, length);
   sofieFree(code);
   sofieFree(error);
   return length == 0;
}
```

## Using the generated code

The code of a model is placed in the namespace `SOFIE_<modelName>`. A `Session`
holds the weights and the intermediate buffers of the model, and evaluates it:

```cpp
#include "MyModel.hxx"

int main()
{
   SOFIE_MyModel::Session session;
   std::vector<float> input(10, 0.5);
   std::vector<float> output = session.infer(input.data());
}
```

A `Session` is not thread-safe: use one `Session` per thread.

## Tests

The tests are built with `-DSOFIE_BUILD_TESTS=ON` and run with `ctest`:

```bash
cmake -S SOFIE -B build -DSOFIE_BUILD_TESTS=ON
cmake --build build -j10
ctest --test-dir build -j10
```

They need [GoogleTest](https://github.com/google/googletest). Most tests also
need Python with the `onnx` module, to generate the ONNX models of the tests,
the tests with models created with PyTorch need the `torch` module, and the
tests of the weight files in the safetensors format need
[nlohmann/json](https://github.com/nlohmann/json). The tests whose optional
dependencies are not found are disabled.

## Inspiration
The standalone version of SOFIE is developed with inspiration from the standalone version of RooFit developed by Jonas Rembser that can be found [here](https://github.com/guitargeek/roofit).
