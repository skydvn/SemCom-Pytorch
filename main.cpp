// ----------------------------------------------------------------------------
// Title:    : 
// Project   : 
// ----------------------------------------------------------------------------
// Author    : Nguyen Thi Hoai Linh
// Email     : 
// Date      : 2025-08-17 22:32:27
// Last Modified : 2025-08-17 22:32:27
// Modified By   : Nguyen Thi Hoai Linh
// ----------------------------------------------------------------------------
// Description: 
// 
// ----------------------------------------------------------------------------
// HISTORY:
// Date      	By	Comments
// ----------	---	---------------------------------------------------------
// ----------------------------------------------------------------------------
#include <pybind11/embed.h>
#include <iostream>
#include <tuple>

namespace py = pybind11;

int main(int argc, char **argv)
{
    py::scoped_interpreter guard{}; // Khởi động Python

    try
    {
        // Import loader.py
        py::module loader = py::module::import("load_lib");

        // Đường dẫn checkpoint
        std::string checkpoint_path = "out/checkpoints/CIFAR10_0.16666666666666666_AWGN13_swinjscc_15h45m22s_on_Aug_04_2025/epoch_199.pkl";

        // Gọi load_model (đã sửa để có channel)
        py::object load_result = loader.attr("load_model")(checkpoint_path);

        // Giải nén tuple (model, images)
        py::object model = load_result.attr("__getitem__")(0);
        py::object images = load_result.attr("__getitem__")(1);

        std::cout << "[C++] Model và images đã load thành công\n";

        // Gọi encode_and_save để sinh feature + mask
        double snr = 8.8;
        py::object encode_result = model.attr("encode_and_save")(images, snr);

        // Trả về tuple (feature, mask)
        py::tuple feature_mask = encode_result.cast<py::tuple>();
        py::object feature = feature_mask[0];
        py::object mask = feature_mask[1];

        // Gọi channel_and_decode
        py::object recon_image = model.attr("channel_and_decode")(feature, mask, images, snr);

        std::cout << "[C++] Recon image shape: "
                  << std::string(py::str(recon_image.attr("shape"))) << std::endl;
    }
    catch (py::error_already_set &e)
    {
        std::cerr << "[C++] Python error:\n"
                  << e.what() << std::endl;
        return 1;
    }

    return 0;
}
