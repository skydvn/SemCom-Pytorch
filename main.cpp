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
#include <pybind11/stl.h>
#include <iostream>
#include <string>

namespace py = pybind11;

int main(int argc, char **argv)
{
    // ======== Cấu hình đường dẫn checkpoint & flag dữ liệu =========
    const std::string ckpt_path =
        "out/checkpoints/CIFAR10_0.16666666666666666_AWGN13_swinjscc_15h45m22s_on_Aug_04_2025/epoch_199.pkl";
    const int flag_cifar = 0; // 1: dùng CIFAR-10, 0: ảnh ngoài CIFAR
    const double snr = 8.8;

    // ======== Khởi động Python ========
    py::scoped_interpreter guard{};

    try
    {
        // Đảm bảo Python tìm thấy các file .py trong thư mục hiện tại
        py::module_ sys = py::module_::import("sys");
        sys.attr("path").attr("insert")(0, ".");

        // Import load_lib.py
        py::module_ loader = py::module_::import("load_lib");

        // Gọi load_model(checkpoint_path) -> (model, args)
        py::object tuple_model = loader.attr("load_model")(ckpt_path);
        py::tuple tup_m = tuple_model.cast<py::tuple>();
        if (tup_m.size() < 2)
        {
            throw std::runtime_error("load_model() must return (model, args).");
        }
        py::object model = tup_m[0];
        py::object args = tup_m[1];

        std::cout << "[C++] Loaded model & args from Python.\n";

        // Gọi load_data(flag_cifar, args) -> (images, labels)
        py::object tuple_data = loader.attr("load_data")(flag_cifar, args);
        py::tuple tup_d = tuple_data.cast<py::tuple>();
        if (tup_d.size() < 2)
        {
            throw std::runtime_error("load_data() must return (images, labels).");
        }
        py::object images = tup_d[0];
        py::object labels = tup_d[1];

        std::cout << "[C++] Loaded images & labels.\n";

        // Gọi model.encode_and_save(images, snr) -> (feature, mask)
        py::tuple enc = model.attr("encode_and_save")(images, snr).cast<py::tuple>();
        if (enc.size() < 2)
        {
            throw std::runtime_error("encode_and_save() must return (feature, mask).");
        }
        py::object feature = enc[0];
        py::object mask = enc[1];

        std::cout << "[C++] encode_and_save() OK.\n";

        // Gọi model.channel_and_decode(feature, mask, images, snr) -> recon_image (torch.Tensor)
        py::object recon_image = model.attr("channel_and_decode")(feature, mask, images, snr);
        std::cout << "[C++] channel_and_decode() OK.\n";

        // In shape
        py::object shape = recon_image.attr("shape");
        std::cout << "[C++] Recon shape: " << std::string(py::str(shape)) << "\n";

        // (Tuỳ chọn) Convert về NumPy nếu cần:
        // py::object recon_np = recon_image.attr("detach")().attr("cpu")().attr("numpy")();
        // std::cout << "[C++] Converted to NumPy. dtype=" << std::string(py::str(recon_np.attr("dtype"))) << "\n";
    }
    catch (const py::error_already_set &e)
    {
        std::cerr << "[C++] Python error:\n"
                  << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[C++] C++ error:\n"
                  << e.what() << std::endl;
        return 1;
    }

    return 0;
}
