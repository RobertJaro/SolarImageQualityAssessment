import glob
import logging
import os
import pickle
import random
import uuid
from enum import Enum
from typing import List

import numpy as np
from astropy import units as u
from keras.utils import Sequence
from tqdm import tqdm

from siqa.data.editor import Editor, PatchEditor, \
    ShapePrepEditor, ContrastNormalizeEditor, LoadMapEditor, SubMapEditor, MapToDataEditor, KSOPrepEditor, \
    RescaleEditor, LoadNumpyEditor, \
    RandomFlipEditor, ImageNormalizeEditor, PyramidRescaleEditor, \
    PeakNormalizeEditor, PassEditor
from siqa.tools.common import Enqueuer


class Norm(Enum):
    CONTRAST = 'contrast'
    IMAGE = 'image'
    PEAK = 'adjusted'
    NONE = 'none'


class DataProvider(Sequence):

    def __init__(self, data, editors: List[Editor]):
        self.data = data
        self.editors = editors

        logging.info("Using {} samples".format(len(self.data)))
        super().__init__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.convertData(self.data[idx])

    def convertData(self, data):
        kwargs = {}
        for editor in self.editors:
            data, kwargs = editor.convert(data, **kwargs)
        return data

    def on_epoch_end(self):
        pass

    def addEditor(self, editor):
        self.editors.append(editor)


class KSODataProvider(DataProvider):

    def __init__(self, paths, coords=None, ext="*.fits", patch_shape=(64, 64),
                 limit=None, norm=Norm.IMAGE, threshold=1.5):
        coords = coords if coords is not None else ([-1000, 1000] * u.arcsec, [-1000, 1000] * u.arcsec)
        map_paths = []
        if not isinstance(paths, list):
            paths = [paths]
        for p in paths:
            map_paths += glob.glob(os.path.join(p, "**", ext), recursive=True)
        if limit:
            map_paths = random.sample(map_paths, limit)

        norm_editor = None
        if norm == Norm.IMAGE:
            norm_editor = ImageNormalizeEditor(peak_adjust=True)
        if norm == Norm.PEAK:
            norm_editor = PeakNormalizeEditor()
        if norm == Norm.CONTRAST:
            norm_editor = ContrastNormalizeEditor(use_median=True, threshold=threshold)
        if norm == Norm.NONE:
            norm_editor = PassEditor()
        assert norm_editor is not None, 'Unknown value for norm: %s' % str(norm)
        editors = [LoadMapEditor(),
                   KSOPrepEditor(),
                   SubMapEditor(coords),
                   MapToDataEditor(),
                   PyramidRescaleEditor(patch_shape),
                   norm_editor,
                   PatchEditor(patch_shape),
                   ShapePrepEditor()]
        super().__init__(map_paths, editors=editors)


class NumpyProvider(DataProvider):

    def __init__(self, path, ext="*.npy", limit=None, patch_shape=None, ext_editors=[], flip_prob=0):
        paths = glob.glob(os.path.join(path, "**", ext), recursive=True)

        if limit:
            paths = random.sample(paths, limit)

        editors = [LoadNumpyEditor(),
                   *ext_editors,
                   PatchEditor(patch_shape),
                   RandomFlipEditor(flip_prob)]
        super().__init__(paths, editors=editors)


class StoreWrapper(Sequence):

    def __init__(self, data_provider, store_path=None, refresh_rate=0):
        self.data_provider = data_provider
        self.store_path = os.path.join(store_path, str(uuid.uuid4())) if store_path else None
        self.refresh_rate = refresh_rate

        self.memory = {}

        if self.store_path is not None:
            os.makedirs(self.store_path)

    def __len__(self):
        return len(self.data_provider)

    def __getitem__(self, idx):
        if self.store_path is None:
            return self.loadMemory(idx)
        else:
            return self.loadFileSystem(idx)

    def loadFileSystem(self, idx):
        file_path = os.path.join(self.store_path, "%d.pickle" % idx)
        if os.path.exists(file_path):
            with open(file_path, "rb", buffering=0) as file:
                d = pickle.load(file)
            if random.random() < self.refresh_rate:
                os.remove(file_path)
        else:
            d = self.data_provider[idx]
            with open(file_path, "wb", buffering=0) as file:
                pickle.dump(d, file)
        return d

    def loadMemory(self, idx):
        if idx in self.memory:
            d = self.memory[idx]
        else:
            d = self.data_provider[idx]
            self.memory[idx] = d
        if random.random() < self.refresh_rate:
            del self.memory[idx]
        return d


class LoadWrapper(Sequence):

    def __init__(self, provider, progress=False):
        data_generator = Enqueuer(provider, shuffle=False)
        iter = tqdm(range(len(data_generator))) if progress else range(len(data_generator))
        self.data_set = np.concatenate([next(data_generator) for _ in iter], axis=0)  # flatten batches
        data_generator.stop()

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        return self.data_set[idx]


class ZipWrapper(Sequence):

    def __init__(self, provider):
        self.provider = provider

    def __len__(self):
        return len(self.provider)

    def __getitem__(self, idx):
        return (self.provider[idx], self.provider.data[idx])
