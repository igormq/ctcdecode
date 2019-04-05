#include <torch/torch.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "scorer.h"
#include "utf8.h"
#include "ctc_beam_search_decoder.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> beam_decoder(torch::Tensor probs,
                 torch::Tensor seq_lengths,
                 const char *labels,
                 int vocab_size,
                 int beam_size,
                 int num_processes,
                 double cutoff_prob,
                 int cutoff_top_n,
                 int blank_id,
                 bool log_input,
                 void *scorer=nullptr);

void* get_scorer(double alpha,
                 double beta,
                 const char* lm_path,
                 const char* labels,
                 int vocab_size);

int is_character_based(void *scorer);
size_t get_max_order(void *scorer);
size_t get_dict_size(void *scorer);
void reset_params(void *scorer, double alpha, double beta);
double get_alpha(void *scorer);
void set_alpha(void *scorer, double alpha);
void set_beta(void *scorer, double beta);
double get_beta(void *scorer);