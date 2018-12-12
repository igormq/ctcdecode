
#include <torch/torch.h>

#include "binding.h"

int utf8_to_utf8_char_vec(const char *labels, std::vector<std::string> &new_vocab)
{
    const char *str_i = labels;
    const char *end = str_i + strlen(labels) + 1;
    do
    {
        char u[5] = {0, 0, 0, 0, 0};
        uint32_t code = utf8::next(str_i, end);
        if (code == 0)
        {
            continue;
        }
        utf8::append(code, u);
        new_vocab.push_back(std::string(u));
    } while (str_i < end);
}

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
                 torch::Tensor output_length)
{

    AT_CHECK(!probs.type().is_cuda(), 'probs must be on CPU');

    std::vector<std::string> new_vocab;
    utf8_to_utf8_char_vec(labels, new_vocab);
    std::cout << '1' << std::endl;
    Scorer *ext_scorer = NULL;
    if (scorer != NULL)
    {
        std::cout << 'null' << std::endl;
        ext_scorer = static_cast<Scorer *>(scorer);
    }
    std::cout << '2' << std::endl;

    auto probs_a = probs.accessor<float, 3>();
    auto seq_lengths_a = seq_lengths.accessor<int32_t, 1>();
    std::cout << '3' << std::endl;

    const int64_t batch_size = probs.size(0);
    const int64_t max_time = probs.size(1);
    const int64_t num_classes = probs.size(2);

    std::vector<std::vector<std::vector<double>>> inputs;
    for (int b = 0; b < batch_size; ++b)
    {
        auto seq_len = std::min(seq_lengths_a[b], (int)max_time);
        std::vector<std::vector<double>> temp(seq_len, std::vector<double>(num_classes));
        for (int t = 0; t < seq_len; ++t)
        {
            for (int n = 0; n < num_classes; ++n)
            {
                temp[t][n] = probs_a[b][t][n];
            }
        }
        inputs.push_back(temp);
    }
    std::cout << '4' << std::endl;

    std::vector<std::vector<std::pair<double, Output>>> batch_results =
        ctc_beam_search_decoder_batch(inputs, new_vocab, beam_size, num_processes, cutoff_prob, cutoff_top_n, blank_id, ext_scorer);

    std::cout << '5' << std::endl;

    auto output_a = output.accessor<float, 3>();
    auto timesteps_a = timesteps.accessor<int32_t, 3>();

    auto output_length_a = output_length.accessor<int32_t, 2>();
    auto scores_a = scores.accessor<float, 2>();
    std::cout << '6' << std::endl;

    for (int b = 0; b < batch_results.size(); ++b)
    {
        std::vector<std::pair<double, Output>> results = batch_results[b];
        for (int p = 0; p < results.size(); ++p)
        {
            std::pair<double, Output> n_path_result = results[p];
            Output out = n_path_result.second;
            std::vector<int> output_tokens = out.tokens;
            std::vector<int> output_timesteps = out.timesteps;
            for (int t = 0; t < output_tokens.size(); ++t)
            {
                output_a[b][p][t] = output_tokens[t];
                timesteps_a[b][p][t] = output_timesteps[t];
            }
            scores_a[b][p] = n_path_result.first;
            output_length_a[b][p] = output_tokens.size();
        }
    }
    return 1;
}

void *get_scorer(double alpha,
                 double beta,
                 const char *lm_path,
                 const char *labels,
                 int vocab_size)
{
    std::vector<std::string> new_vocab;
    utf8_to_utf8_char_vec(labels, new_vocab);
    Scorer *scorer = new Scorer(alpha, beta, lm_path, new_vocab);
    return static_cast<void *>(scorer);
}

int is_character_based(void *scorer)
{
    Scorer *ext_scorer = static_cast<Scorer *>(scorer);
    return ext_scorer->is_character_based();
}
size_t get_max_order(void *scorer)
{
    Scorer *ext_scorer = static_cast<Scorer *>(scorer);
    return ext_scorer->get_max_order();
}
size_t get_dict_size(void *scorer)
{
    Scorer *ext_scorer = static_cast<Scorer *>(scorer);
    return ext_scorer->get_dict_size();
}

void reset_params(void *scorer, double alpha, double beta)
{
    Scorer *ext_scorer = static_cast<Scorer *>(scorer);
    ext_scorer->reset_params(alpha, beta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("beam_decoder", &beam_decoder, "CTC Beam Search Decoder");
    m.def("get_scorer", &get_scorer, "Language Model scorer");
    m.def("is_character_based", &is_character_based, "is character based");
    m.def("get_max_order", &get_max_order, "get max order");
    m.def("get_dict_size", &get_dict_size, "get dict size");
    m.def("reset_params", &reset_params, "reset params");
}
