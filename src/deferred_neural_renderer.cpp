#include "deferred_neural_renderer.h"

#include "stb/stb_image_write.h"

#include "timer.h"

#include <memory>

//#include <glad/glad.h> // holds all OpenGL type declarations

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace F = torch::nn::functional;

DNRenderer::DNRenderer(int height, int width, const std::string& model_filename)
    : DNRenderer(height, width)
{
    load(model_filename);
}

DNRenderer::DNRenderer(int height, int width)
    : m_autograd_mode(false)
    , m_render_height(height)
    , m_render_width(width)
    , index(0)
{
#ifdef __APPLE__
    std::cout << "Warning: CUDA not suppored on Apple device. Using CPU.\n";
#endif

    std::cout << "Benchmarked: " << torch::globalContext().benchmarkCuDNN() << "\n";
    std::cout << "User enabled cudnn: " << torch::globalContext().userEnabledCuDNN() << "\n";
    torch::globalContext().setUserEnabledCuDNN(true);
    torch::globalContext().setBenchmarkCuDNN(true);
    torch::globalContext().setFlushDenormal(true);
}

int DNRenderer::load(const std::string& model_filename) {
    try {
        std::cout << "Loading module\n";
        m_model = torch::jit::load(model_filename);
        m_model.eval();

#ifndef __APPLE__
        m_model.to(at::kCUDA);
#endif

        std::cout << "Loaded module\n";
    }
    catch (const c10::Error& e) {
        std::cout << "Failed to module\n";
        std::cerr << "error loading the model\n";
        return -1;
    }

    return 0;
}

// WARNING: This function probably modifies the content of data
void DNRenderer::render(float *data, int rows, int cols, bool writeout) {
    /*
    Timer& timer = Timer::get();
    timer.checkpoint("checkpoint test 1");
    timer.checkpoint("checkpoint test 2");
    timer.checkpoint("checkpoint test 3");
    timer.checkpoint("checkpoint test 4");
    timer.checkpoint("checkpoint test 5");
    timer.checkpoint("checkpoint test 6");
*/
//    timer.checkpoint("torch from blob");
    auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(at::kCUDA);
    auto input = torch::from_blob(data, {rows, cols, 4}, options);

//    std::cout << "0 GPU Pointer: " << input.data_ptr() << "\n";

//    timer.checkpoint("index");
    // TODO: More efficient way to remove/ignore extra data?
    // Remode Blue and Alpha data added by opengl
    input = input.index({torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, 3)});

//    timer.checkpoint("pass input to cuda");

//#ifndef __APPLE__
//    input = input.to(at::kCUDA);
//#endif

//    std::cout << "1 GPU Pointer: " << input.data_ptr() << "\n";

    //timer.checkpoint("flip");
    input = input.flip({0});

//    std::cout << "2 GPU Pointer: " << input.data_ptr() << "\n";
//    timer.checkpoint("unsqueeze");
    //input = input.permute({2, 0, 1}).unsqueeze(0);
    //input = input.permute({2, 0, 1}).unsqueeze(0);
     input = input.unsqueeze(0);
//    std::cout << "3 GPU Pointer: " << input.data_ptr() << "\n";
    //input = input.unsqueeze(0);

    //std::cout << "input shape: " << input.sizes() << "\n";

    //timer.checkpoint("pass input to cuda");
    //input = input.to(at::kCUDA);

//    timer.checkpoint("sample from grid");

    //std::cout << "input shape: " << input.sizes() << "\n";
    /*
    input = F::grid_sample(input, m_grid, F::GridSampleFuncOptions()
                                  .mode(torch::kNearest)
                                  .padding_mode(torch::kBorder)
                                  .align_corners(false));
                                  */
    //std::cout << "input shape: " << input.sizes() << "\n";
//    std::cout << "4 GPU Pointer: " << input.data_ptr() << "\n";
//    timer.checkpoint("permute sample");
    //input = input.permute({0, 3, 2, 1});
    //input = input.permute({0, 2, 3, 1});
//    std::cout << "5 GPU Pointer: " << input.data_ptr() << "\n";

//    std::cout << "input shape: " << input.sizes() << "\n";

//    timer.checkpoint("build jit vector");
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);
//    timer.checkpoint("forward pass");
//    std::cout << "forward pass\n";
    auto output = m_model.forward(inputs).toTensor();
//    std::cout << "6 GPU Pointer: " << output.data_ptr() << "\n";
    //timer.checkpoint("to cpu");
    //torch::Tensor output = output_f.to(at::kCPU);

    //std::cout << "output shape: " << output.sizes() << "\n";

    write(output, writeout);
}

void DNRenderer::write(torch::Tensor& tens, bool write) {
//    std::cout << "7 GPU Pointer: " << tens.data_ptr() << "\n";
//    Timer& timer = Timer::get();
//    timer.checkpoint("permute output");
    tens = tens.squeeze().permute({1, 2, 0});
//    std::cout << "8 GPU Pointer: " << tens.data_ptr() << "\n";

    //if (!write) {
        //timer.checkpoint("flip output");
        tens = tens.flip({0});
    //}

//    std::cout << "9 GPU Pointer: " << tens.data_ptr() << "\n";
//    timer.checkpoint("rescale output");
    tens = ((tens + 1.0) / 2.0) * 255;
//    std::cout << "10 GPU Pointer: " << tens.data_ptr() << "\n";
//    timer.checkpoint("round output");
    tens = torch::round(tens);
//    std::cout << "11 GPU Pointer: " << tens.data_ptr() << "\n";
//    timer.checkpoint("convert output to uint8_t");
    tens = tens.to(torch::kUInt8);
//    std::cout << "12 GPU Pointer: " << tens.data_ptr() << "\n";

    // TODO: More efficient way to increase size of pixel color dimension?
    auto tens_shape = tens.sizes();
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(at::kCUDA);
//    timer.checkpoint("create full tensor");
    auto alpha = torch::full({tens_shape[0], tens_shape[1], 1}, 255, options);
//    timer.checkpoint("create RGBA tensor");
    auto output = torch::cat({tens, alpha}, 2);
//    std::cout << "13 GPU Pointer: " << output.data_ptr() << "\n";

//    timer.checkpoint("make output contiguous");
    output = output.contiguous();
//    std::cout << "14 GPU Pointer: " << output.data_ptr() << "\n";


//#ifndef __APPLE__
    //timer.checkpoint("to cpu");
    //output = output.to(at::kCPU);
//#endif

    m_output = output;
//    std::cout << "15 GPU Pointer: " << m_output.data_ptr() << "\n";
/*
    if (write) {
        timer.checkpoint("get data pointer");
        std::cout << "write shape: " << output.sizes() << "\n";
        uint8_t* data_out = m_output.data_ptr<uint8_t>();
        timer.checkpoint("write to file");
        std::string filename = std::string("output/") + std::to_string(index);
        stbi_write_jpg((filename + ".jpg").c_str(), m_render_width, m_render_height, 3, data_out, 100);
        index++;
    }
    */
//    timer.end();


    //glTexSubImage2D(GL_TEXTURE_2D, 0 ,0, 0, SCR_WIDTH, SCR_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, (GLvoid*)data_out);

    //glLoadIdentity();
    //glRasterPos2i(0, 0);
    //glDrawPixels(m_render_width, m_render_height, GL_RGB, GL_UNSIGNED_INT, data_out);

   // glReadBuffer(GL_BACK);
    //glWindowPos2i(0,0);
    //glDrawPixels(m_render_width, m_render_height, GL_RGB, GL_UNSIGNED_BYTE, data_out);

    //std::string filename = std::string("output/test") + std::to_string(index);
    //stbi_write_jpg((filename + ".jpg").c_str(), m_render_width, m_render_height, 3, data_out, 100);
    //index++;

}
