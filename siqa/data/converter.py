import os

import numpy as np
from tqdm import tqdm

from siqa.data.provider import Norm, KSODataProvider
from siqa.tools.common import Enqueuer


def convertProvider(provider, path, validation_split=0.):
    os.makedirs(path, exist_ok=True)
    if validation_split != 0:
        os.makedirs(path + '_validation', exist_ok=True)
    data_generator = Enqueuer(provider, batch_size=1, n_workers=32)
    try:
        for idx in tqdm(range(int(len(data_generator) * (1 - validation_split)))):
            data = next(data_generator)
            np.save(os.path.join(path, '%d.npy' % idx), data)
        for idx in tqdm(range(int(len(data_generator) * validation_split))):
            data = next(data_generator)
            np.save(os.path.join(path + '_validation', '%d.npy' % idx), data)
    finally:
        data_generator.stop()

if __name__ == '__main__':
    provider = KSODataProvider("/gss/r.jarolim/data/kso_general/quality2", ext="*.fts.gz",
                               patch_shape=(128, 128), norm=Norm.NONE)
    convertProvider(provider, '/gss/r.jarolim/data/converted/test_q2_128_v2', validation_split=0)

    provider = KSODataProvider("/gss/r.jarolim/data/kso_general/quality1", ext="*.fts.gz",
                               patch_shape=(128, 128), norm=Norm.NONE)
    convertProvider(provider, '/gss/r.jarolim/data/converted/test_q1_128_v2', validation_split=0)