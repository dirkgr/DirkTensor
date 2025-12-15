#include "OlmoMlp.h"

#include <format>
#include <xtensor/core/xmath.hpp>
#include <xtensor/core/xnoalias.hpp>
#include <xtensor/io/xnpy.hpp>
#include <xtensor-blas/xlinalg.hpp>

OlmoMlp::OlmoMlp(const std::string& folder, const unsigned int index) {
    m_upProjection.w =
        xt::load_npy<float>(
            std::format("{}/model.layers.{}.mlp.up_proj.weight.npy", folder, index));
    m_gateProjection.w =
        xt::load_npy<float>(
            std::format("{}/model.layers.{}.mlp.gate_proj.weight.npy", folder, index));
    m_downProjection.w =
        xt::load_npy<float>(
            std::format("{}/model.layers.{}.mlp.down_proj.weight.npy", folder, index));
}

xt::xtensor<float, 3> OlmoMlp::forward(const xt::xtensor<float, 3>& input) {
    const size_t batch_size = input.shape(0);
    const size_t seq_len = input.shape(1);
    const size_t d_model = input.shape(2);
    const size_t hidden_size = m_upProjection.shape(0);  // 8192

    // Reshape input from [batch, seq, d_model] to [batch*seq, d_model]
    m_act_input_2d = xt::reshape_view(input, {batch_size * seq_len, d_model});

    // Perform matrix multiplications using direct BLAS GEMM via dot
    // Weights are stored as [hidden_size, d_model], so we need to transpose them
    // input_2d is [batch*seq, d_model], weights transposed become [d_model, hidden_size]
    m_act_up = xt::linalg::dot(m_act_input_2d, xt::transpose(m_upProjection.w));
    m_act_gate = xt::linalg::dot(m_act_input_2d, xt::transpose(m_gateProjection.w));

    // Apply SiLU activation: gate * sigmoid(gate) * projected
    // Use scalar operations to avoid xtensor expression template overhead
    m_act_activated_2d = xt::zeros<float>({batch_size * seq_len, hidden_size});
    const float* const gate_ptr = m_act_gate.data();
    const float* const up_ptr = m_act_up.data();
    float* const act_ptr = m_act_activated_2d.data();
    const size_t total_elements = batch_size * seq_len * hidden_size;
    for (size_t i = 0; i < total_elements; ++i) {
        const float g = gate_ptr[i];
        const float silu = g / (1.0f + std::exp(-g));
        act_ptr[i] = silu * up_ptr[i];
    }

    // Final projection back to d_model
    // m_downProjection is [d_model, hidden_size], so we need to transpose it
    // activated_2d is [batch*seq, hidden_size], transposed weight is [hidden_size, d_model]
    auto output_2d = xt::linalg::dot(m_act_activated_2d, xt::transpose(m_downProjection.w));

    // Reshape back to 3D [batch, seq, d_model]
    auto output = xt::reshape_view(output_2d, {batch_size, seq_len, d_model});

    return output;
}

xt::xtensor<float, 3> OlmoMlp::backward(const xt::xtensor<float, 3>& d_output) {
    const size_t batch_size = d_output.shape(0);
    const size_t seq_len = d_output.shape(1);
    const size_t d_model = d_output.shape(2);
    const size_t hidden_size = m_upProjection.shape(0);  // 8192

    auto d_output_2d = xt::reshape_view(d_output, {batch_size * seq_len, d_model});

    if (m_downProjection.grad.size() == 0)
        m_downProjection.grad = xt::zeros_like(m_downProjection.w);
    m_downProjection.grad += xt::linalg::dot(
        xt::transpose(d_output_2d), // (d_model, tokens)
        m_act_activated_2d              // (tokens, hidden_size)
    );

    const auto d_activated_2d = xt::linalg::dot( // (tokens, hidden_size)
        d_output_2d,                           // (tokens, d_model)
        m_downProjection.w                     // (d_model, hidden_size)
    );

    // Compute d_up and d_gate element-wise in a single pass
    // This avoids intermediate allocations and handles up=0 correctly
    xt::xtensor<float, 2> d_up = xt::empty<float>({batch_size * seq_len, hidden_size});
    xt::xtensor<float, 2> d_gate = xt::empty<float>({batch_size * seq_len, hidden_size});

    const float* const d_act_ptr = d_activated_2d.data();
    const float* const up_ptr = m_act_up.data();
    const float* const gate_ptr = m_act_gate.data();
    float* const d_up_ptr = d_up.data();
    float* const d_gate_ptr = d_gate.data();

    const size_t total = batch_size * seq_len * hidden_size;
    for (size_t i = 0; i < total; ++i) {
        const float d_act = d_act_ptr[i];
        const float up = up_ptr[i];
        const float gate = gate_ptr[i];

        // sig = sigmoid(gate)
        const float sig = 1.0f / (1.0f + std::exp(-gate));
        // silu(gate) = gate * sigmoid(gate)
        const float silu = gate * sig;

        // d_up = d_activated * silu(gate)
        // This is correct even when up=0 (avoids division by zero)
        d_up_ptr[i] = d_act * silu;

        // d_silu = d_activated * up
        const float d_silu = d_act * up;

        // d_gate = d_silu * sig * (1 + gate * (1 - sig))
        // This is the derivative of silu: d(silu)/d(gate) = sig * (1 + gate * (1 - sig))
        d_gate_ptr[i] = d_silu * sig * (1.0f + gate * (1.0f - sig));
    }

    if (m_gateProjection.grad.size() == 0)
        m_gateProjection.grad = xt::zeros_like(m_gateProjection.w);
    m_gateProjection.grad += xt::linalg::dot(  // (hidden_size, d_model)
        xt::transpose(d_gate),                 // (hidden_size, tokens)
        m_act_input_2d                         // (tokens, d_model)
    );

    if (m_upProjection.grad.size() == 0)
        m_upProjection.grad = xt::zeros_like(m_upProjection.w);
    m_upProjection.grad += xt::linalg::dot( // (hidden_size, d_model)
        xt::transpose(d_up),                // (hidden_size, tokens)
        m_act_input_2d                      // (tokens, d_model)
    );

    /*
    // Fast but wrong
    // d_input = d_gate @ m_gateProjection
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        batch_size * seq_len, d_model, hidden_size,
        1.0f,            // alpha
        d_gate.data(),
        hidden_size,           // lda: stride of d_gate
        m_gateProjection.w.data(),
        d_model,               // ldb: stride of m_gateProjection
        0.0f,                  // beta = 0 means overwrite what is already there
        d_input.data(),        // write directly into d_input's buffer
        d_model                // ldc: stride of output
    );

    // d_input += d_up @ m_upProjection
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        batch_size * seq_len, d_model, hidden_size,
        1.0f,            // alpha
        d_up.data(),
        hidden_size,           // lda: stride of d_gate
        m_upProjection.w.data(),
        d_model,               // ldb: stride of m_upProjection
        1.0f,                  // beta = 1 means C += result
        d_input.data(),        // write directly into d_input's buffer
        d_model                // ldc: stride of output
    );
    */

    xt::xtensor<float, 2> d_input_2d =
        xt::linalg::dot(d_gate, m_gateProjection.w) +
        xt::linalg::dot(d_up, m_upProjection.w);

    return xt::eval(xt::reshape_view(d_input_2d, {batch_size, seq_len, d_model}));
}