#include "deferred_neural_renderer.h"

#include "stb_image_write.h"

#include "timer.h"

#include <memory>

#include <glad/glad.h> // holds all OpenGL type declarations

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace F = torch::nn::functional;

DNRenderer::DNRenderer(int height, int width)
    : m_render_height(height)
    , m_render_width(width)
    , index(0)
{
#ifdef __APPLE__
    std::cout << "Warning: CUDA not suppored on Apple device. Using CPU.\n";
#endif
    m_grid = torch::ones({1, m_render_height, m_render_width, 2}, torch::kFloat32);

    for (int row = 0; row < m_render_height; row++) {
        for (int col = 0; col < m_render_width; col++) {
            m_grid[0][row][col][0] = (2.0 * (row / (float)(m_render_height - 1)) - 1.0);
            m_grid[0][row][col][1] = 2.0 * (col / (float)(m_render_width - 1)) - 1.0;
        }
    }
#ifndef __APPLE__
    m_grid = m_grid.to(at::kCUDA);
#endif
}

int DNRenderer::load(const std::string& model_filename) {
    try {
        std::cout << "Loading module\n";
        m_model = torch::jit::load(model_filename);

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
void DNRenderer::render(float* data, int rows, int cols, bool writeout) {
    Timer& timer = Timer::get();
    timer.checkpoint("checkpoint test 1");
    timer.checkpoint("checkpoint test 2");
    timer.checkpoint("checkpoint test 3");
    timer.checkpoint("checkpoint test 4");
    timer.checkpoint("checkpoint test 5");
    timer.checkpoint("checkpoint test 6");

    timer.checkpoint("torch from blob");
    auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided);
    auto input = torch::from_blob(data, {rows, cols, 2}, options);

    timer.checkpoint("pass input to cuda");

#ifndef __APPLE__
    input = input.to(at::kCUDA);
#endif

    timer.checkpoint("flip");
    input = input.flip({0});
    timer.checkpoint("unsqueeze");
    //input = input.permute({2, 0, 1}).unsqueeze(0);
    input = input.permute({2, 1, 0}).unsqueeze(0);
    //input = input.unsqueeze(0);

    //std::cout << "input shape: " << input.sizes() << "\n";

    //timer.checkpoint("pass input to cuda");
    //input = input.to(at::kCUDA);

    timer.checkpoint("sample from grid");
    input = F::grid_sample(input, m_grid, F::GridSampleFuncOptions()
                                  .mode(torch::kNearest)
                                  .padding_mode(torch::kBorder)
                                  .align_corners(false));
    timer.checkpoint("permute sample");
    //input = input.permute({0, 3, 2, 1});
    input = input.permute({0, 2, 3, 1});

    std::cout << "input shape: " << input.sizes() << "\n";

    timer.checkpoint("build jit vector");
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);
    timer.checkpoint("forward pass");
    std::cout << "forward pass\n";
    auto output = m_model.forward(inputs).toTensor();
    //timer.checkpoint("to cpu");
    //torch::Tensor output = output_f.to(at::kCPU);

    //std::cout << "output shape: " << output.sizes() << "\n";

    write(output, writeout);

}

void DNRenderer::write(torch::Tensor& output, bool write) {
    Timer& timer = Timer::get();
    timer.checkpoint("permute output");
    output = output.squeeze().permute({1, 2, 0});

    if (!write) {
        timer.checkpoint("flip output");
        output = output.flip({0});
    }

    timer.checkpoint("rescale output");
    output = ((output + 1.0) / 2.0) * 255;
    timer.checkpoint("round output");
    output = torch::round(output);
    timer.checkpoint("convert output to uint8_t");
    output = output.to(torch::kUInt8);
    timer.checkpoint("make output contiguous");
    output = output.contiguous();

#ifndef __APPLE__
    timer.checkpoint("to cpu");
    output = output.to(at::kCPU);
#endif

    m_output = output;

    if (write) {
        timer.checkpoint("get data pointer");
        std::cout << "write shape: " << output.sizes() << "\n";
        uint8_t* data_out = m_output.data_ptr<uint8_t>();
        timer.checkpoint("write to file");
        std::string filename = std::string("output/test") + std::to_string(index);
        stbi_write_jpg((filename + ".jpg").c_str(), m_render_width, m_render_height, 3, data_out, 100);
        index++;
    }
    timer.end();


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
