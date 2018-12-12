#include <torch/torch.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "scorer.h"
#include "utf8.h"
#include "ctc_beam_search_decoder.h"

int beam_decoder(torch::Tensor probs,
                 torch::Tensor seq_lengths,
                 const char *labels,
                 int vocab_size,
                 size_t beam_size,
                 size_t num_processes,
                 double cutoff_prob,
                 size_t cutoff_top_n,
                 size_t blank_id,
                 void *scorer,
                 torch::Tensor output,
                 torch::Tensor timesteps,
                 torch::Tensor scores,
                 torch::Tensor output_length);

void *get_scorer(double alpha,
                 double beta,
                 const char *lm_path,
                 const char *labels,
                 int vocab_size);

int is_character_based(void *scorer);
size_t get_max_order(void *scorer);
size_t get_dict_size(void *scorer);
void reset_params(void *scorer, double alpha, double beta);
