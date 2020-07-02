#ifndef DEFERRED_NEURAL_RENDERER_H
#define DEFERRED_NEURAL_RENDERER_H

#include <torch/torch.h>
#include <torch/script.h>
#include <memory>

#include <string>

class DNRenderer {
public:
    DNRenderer(int height, int width);

    int load(const std::string& model_filename);

    void render(float* data, int rows, int cols, bool write);

    torch::Tensor m_output;

private:
    void write(torch::Tensor& out, bool write);

    torch::jit::script::Module m_model;
    torch::Tensor m_grid;

    const int m_render_height;
    const int m_render_width;

    int index;
};

#endif // DEFERRED_NEURAL_RENDERER_H
