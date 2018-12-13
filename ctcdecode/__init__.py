import torch

from . import _C


class CTCBeamDecoder(object):
    def __init__(self,
                 labels,
                 model_path=None,
                 alpha=0,
                 beta=0,
                 cutoff_top_n=40,
                 cutoff_prob=1.0,
                 beam_width=100,
                 num_processes=4,
                 blank_id=0):
        self.cutoff_top_n = cutoff_top_n
        self._beam_width = beam_width
        self._scorer = None
        self._num_processes = num_processes
        self._labels = ''.join(labels).encode()
        self._num_labels = len(labels)
        self._blank_id = blank_id
        if model_path:
            self._scorer = _C.get_scorer(alpha, beta, model_path.encode(), self._labels,
                                         self._num_labels)
        self._cutoff_prob = cutoff_prob

    def decode(self, probs, seq_lens=None):
        # We expect batch x seq x label_size
        probs = probs.cpu().float()
        batch_size, max_seq_len = probs.size(0), probs.size(1)

        if seq_lens is None:
            seq_lens = torch.full((batch_size, ), max_seq_len, dtype=torch.int, device='cpu')
        else:
            seq_lens = seq_lens.to('cpu').int()

        output, scores, timesteps, out_seq_len = _C.beam_decoder(
            probs, seq_lens, self._labels, self._num_labels, self._beam_width, self._num_processes,
            self._cutoff_prob, self.cutoff_top_n, self._blank_id, self._scorer)

        return output, scores, timesteps, out_seq_len

    def character_based(self):
        return _C.is_character_based(self._scorer) if self._scorer else None

    def max_order(self):
        return _C.get_max_order(self._scorer) if self._scorer else None

    def dict_size(self):
        return _C.get_dict_size(self._scorer) if self._scorer else None

    def reset_params(self, alpha, beta):
        if self._scorer is not None:
            _C.reset_params(self._scorer, alpha, beta)
