#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "convolution.h"

namespace py = pybind11;

py::array_t<float> apply_convolution(
    py::array_t<uint8_t> input,
    py::array_t<float> kernel
){
    py::buffer_info input_info = input.request();
    py::buffer_info kernel_info = kernel.request();

    uint8_t* input_ptr = static_cast<uint8_t*>(input_info.ptr);
    float* kernel_ptr = static_cast<float*>(kernel_info.ptr);

    int img_width = input_info.shape[1];
    int img_height = input_info.shape[0];
    int channels = input_info.shape[2];
    int kernel_width = kernel_info.shape[1];
    int kernel_height = kernel_info.shape[0];

    auto output = py::array_t<float>({img_height, img_width, channels});
    py::buffer_info output_info = output.request();
    float* output_ptr = static_cast<float*>(output_info.ptr);

    convolution(
        input_ptr, kernel_ptr, output_ptr,
        img_width, img_height, channels,
        kernel_width, kernel_height
    );

    return output;
}

PYBIND11_MODULE(image_processing_native, m) {
    m.def("apply_convolution", &apply_convolution,
          "Apply convolution to an image using a given kernel.");
}
