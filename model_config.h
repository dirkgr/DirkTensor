#pragma once

static constexpr unsigned int head_dim = 128;
static constexpr unsigned int seq_len = 4096;
static constexpr unsigned int d_model = 2048;
static constexpr unsigned int n_heads = d_model / head_dim;
static constexpr unsigned int n_layers = 16;
