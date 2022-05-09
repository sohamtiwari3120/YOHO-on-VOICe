import torch
import numpy as np
from config import add_leaf_to_path, hparams
add_leaf_to_path()
hp = hparams()
from leaf_pytorch.frontend import Leaf  # NOQA


if __name__ == '__main__':
    fe = Leaf(n_filters = hp.n_mels,
            sample_rate = hp.sample_rate,
            window_len = hp.mel_win_len//hp.sample_rate,
            window_stride = hp.mel_hop_len//hp.sample_rate,
            init_min_freq = hp.fmin,
            init_max_freq = hp.fmax)
    x = torch.randn(1, 1, int(hp.window_len_secs * hp.sample_rate))
    print(x.shape)
    o = fe(x)
    print(o.shape)
    print(o[0][1])
