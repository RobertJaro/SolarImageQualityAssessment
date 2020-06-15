import glob
import logging
import os
import random
import shutil
import uuid
import warnings
from abc import ABC, abstractmethod

import numpy as np
import sunpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.visualization import ImageNormalize, AsinhStretch
from dateutil.parser import parse
from imageio import imwrite
from skimage.transform import pyramid_reduce
from sunpy.coordinates import frames
from sunpy.coordinates.sun import angular_radius
from sunpy.instr.aia import aiaprep
from sunpy.map import Map, header_helper


class Editor(ABC):

    def convert(self, data, **kwargs):
        result = self.call(data, **kwargs)
        if isinstance(result, tuple):
            data, add_kwargs = result
            kwargs.update(add_kwargs)
        else:
            data = result
        return data, kwargs

    @abstractmethod
    def call(self, data, **kwargs):
        raise NotImplementedError()


class LoadFITSEditor(Editor):

    def call(self, map_path, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore warnings
            dst = shutil.copy(map_path, os.path.join(os.environ.get("TMPDIR"), os.path.basename(map_path)))
            hdul = fits.open(dst)
            os.remove(dst)
            hdul.verify("fix")
            data, header = hdul[0].data, hdul[0].header
            hdul.close()
        return data, {"header": header}


class PrepKSOEditor(Editor):

    def call(self, map_path, **kwargs):
        hdul = fits.open(map_path)
        hdu = hdul[0]
        hdu.verify('fix')
        d, h = hdu.data, hdu.header

        tmp_file = "demo%d.jpg" % uuid.uuid4()
        imwrite(tmp_file, d)
        myCmd = os.popen('/home/rja/PythonProjects/SpringProject/spring/limbcenter/sunlimb demo.jpg').read()
        center_x, center_y, radius, d_radius = map(float, myCmd.splitlines())
        os.remove(tmp_file)

        obs_time = parse(h["DATE_OBS"])
        rsun = angular_radius(obs_time)
        coord = SkyCoord(0 * u.arcsec, 0 * u.arcsec, obstime=obs_time, observer='earth',
                         frame=frames.Helioprojective)

        # create WCS header info
        scale = rsun / (radius * u.pix)
        header = header_helper.make_fitswcs_header(
            d, coord,
            rotation_angle=0 * u.deg,
            reference_pixel=u.Quantity([center_x, center_y] * u.pixel),
            scale=u.Quantity([scale, scale]))

        s_map = Map(d.astype(np.float32), header)

        return s_map, {"header": header}


class LoadMapEditor(Editor):

    def call(self, data, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore warnings
            # dst = shutil.copy(data, os.path.join(os.environ.get("TMPDIR"), os.path.basename(data)))
            s_map = Map(data)
            # os.remove(dst)
            return s_map, {'path': data}


class ToMapEditor(Editor):

    def call(self, data, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore warnings
            # dst = shutil.copy(data, os.path.join(os.environ.get("TMPDIR"), os.path.basename(data)))
            print(kwargs['header'])
            s_map = Map(data, kwargs['header'])
            # os.remove(dst)
            return s_map


class SubMapEditor(Editor):

    def __init__(self, coords):
        self.coords = coords

    def call(self, map, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore warnings
            return map.submap(SkyCoord(*self.coords, frame=map.coordinate_frame))


class MapToDataEditor(Editor):

    def call(self, map, **kwargs):
        return map.data, {"header": map.meta}


class ExposureNormalizationEditor(Editor):

    def call(self, data, **kwargs):
        header = kwargs["header"]

        keyword = None
        if "exp_time" in header:
            keyword = "exp_time"
        if "exptime" in header:
            keyword = "exptime"
        if keyword is None:
            return data

        data = np.true_divide(data, header[keyword], out=data)
        header[keyword] = 1
        return data, {"header": header}


class ExposureNormalizationMapEditor(Editor):

    def call(self, map, **kwargs):
        keyword = "exp_time" if "exp_time" in map.meta else "exptime"
        adjusted_data = map.data / map.meta[keyword]
        map.meta[keyword] = 1
        return Map(adjusted_data, map.meta)


class ResizeArcsPPEditor(Editor):

    def __init__(self, target_pp, scaling=1):
        self.target_pp = target_pp
        self.scaling = scaling

    def call(self, data, **kwargs):
        header = kwargs["header"]
        pp = header["arcs_pp"]
        rescale_factor = self.target_pp / pp / self.scaling
        if rescale_factor == 1:
            return data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore warnings
            data = pyramid_reduce(data, downscale=rescale_factor, order=2)
        return data


class PyramidRescaleEditor(Editor):

    def __init__(self, target_shape):
        self.target_shape = target_shape

    def call(self, data, **kwargs):
        scale = data.shape[0] / self.target_shape[0]
        data = pyramid_reduce(data, downscale=scale, order=3)
        return data


class PatchEditor(Editor):
    def __init__(self, patch_shape):
        self.patch_shape = patch_shape

    def call(self, data, **kwargs):
        assert data.shape[0] >= self.patch_shape[0] and data.shape[1] >= self.patch_shape[
            1], "Patch size larger than image size"
        x = random.randint(0, data.shape[0] - self.patch_shape[0])
        y = random.randint(0, data.shape[1] - self.patch_shape[1])
        return data[x:x + self.patch_shape[0], y:y + self.patch_shape[1]]


class SliceEditor(Editor):
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

    def call(self, data, **kwargs):
        return data[..., self.start:self.stop]


class ListPatchEditor(Editor):
    def __init__(self, patch_shape):
        self.patch_shape = patch_shape

    def call(self, data_list, **kwargs):
        x_range = data_list[0].shape[0] - self.patch_shape[0]
        y_range = data_list[0].shape[1] - self.patch_shape[1]

        # pad to patch size
        if x_range < 0 or y_range < 0:
            data_list = [np.pad(d,
                                (-x_range if x_range < 0 else 0, -y_range if y_range < 0 else 0),
                                'reflect') for d in data_list]
            x_range = 0 if x_range < 0 else x_range
            y_range = 0 if y_range < 0 else y_range

        return self._extractPatch(data_list, x_range, y_range)

    def _extractPatch(self, data_list, x_range, y_range):
        x = random.randint(0, x_range)
        y = random.randint(0, y_range)
        return [data[x:x + self.patch_shape[0], y:y + self.patch_shape[1]] for data in data_list]


class ShapePrepEditor(Editor):

    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def call(self, data, **kwargs):
        data = data.reshape((*data.shape, 1))
        return data.astype(self.dtype, copy=False)


class ContrastNormalizeEditor(Editor):

    def __init__(self, use_median=False, threshold=False):
        self.use_median = use_median
        self.threshold = threshold

    def call(self, data, **kwargs):
        shift = np.median(data, (0, 1), keepdims=True) if self.use_median else np.mean(data, (0, 1), keepdims=True)
        data = (data - shift) / (np.std(data, (0, 1), keepdims=True) + 10e-8)
        if self.threshold:
            data[data > self.threshold] = self.threshold
            data[data < -self.threshold] = -self.threshold
            data /= self.threshold
        return data


class PeakNormalizeEditor(Editor):

    def __init__(self, min=-5, max=3, **kwargs):
        super().__init__(**kwargs)

        self.min = min
        self.max = max

    def call(self, data, **kwargs):
        data_crop = data[data > np.mean(data)].ravel()

        hist, bin_edges = np.histogram(data_crop, 50)
        bins = (bin_edges[1:] + bin_edges[:-1]) / 2
        peak = bins[np.argmax(hist)]

        data = (data - peak) / np.std(data_crop)

        data[data < self.min] = self.min
        data[data > self.max] = self.max

        data = (data - (self.max + self.min) / 2) / ((self.max - self.min) / 2)
        return data


class ImageNormalizeEditor(Editor):

    def __init__(self, peak_adjust=False):
        self.peak_adjust = peak_adjust

    def call(self, data, **kwargs):
        if self.peak_adjust:
            peak = np.median(data) + 3 * np.std(data)
            data[data > peak] = peak
        data = (data - np.min(data, (0, 1), keepdims=True)) / (
                np.max(data, (0, 1), keepdims=True) - np.min(data, (0, 1), keepdims=True)) * 2 - 1
        return data


class DepthwiseContrastNormalizeEditor(Editor):

    def __init__(self, use_median=True, running_norm=False, fixed_mean=None, fixed_std=None):
        self.median = use_median
        self.running_norm = running_norm
        self.channels_mean = {}
        self.channels_std = {}

        self.fixed_mean = {} if fixed_mean is None else fixed_mean
        self.fixed_std = {} if fixed_std is None else fixed_std

    def call(self, data, **kwargs):
        for i in range(data.shape[-1]):
            d = data[..., i]
            if i in self.fixed_mean and i in self.fixed_std:
                d = (d - self.fixed_mean[i]) / (self.fixed_std[i] + 10e-6)
                data[..., i] = d
                continue
            if not self.running_norm:
                np.subtract(d, np.mean(d), out=data[..., i])
                np.true_divide(data[..., i], (np.std(d) + 10e-6), out=data[..., i])
                continue
            if i not in self.channels_mean:
                self.channels_mean[i] = []
                self.channels_std[i] = []

            shift = np.median(d) if self.median else np.mean(d)
            std = np.std(d)

            self.channels_mean[i].append(shift)
            self.channels_std[i].append(std)

            if len(self.channels_mean[i]) >= 50000:
                self.fixed_mean[i] = np.mean(self.channels_mean[i])
                self.fixed_std[i] = np.mean(self.channels_std[i])
                logging.info(
                    "Fixed data normalization: CHANNEL %d MEAN %f STD %f" % (i, self.fixed_mean[i], self.fixed_std[i]))

            d = (d - np.mean(self.channels_mean[i])) / (np.mean(self.channels_std[i]) + 10e-6)
            data[..., i] = d
        return data


class StretchEditor(Editor):
    def __init__(self, vmin, vmax, stretch, clip=False):
        self.norm = ImageNormalize(stretch=stretch, vmin=vmin, vmax=vmax, clip=clip)

    def call(self, data, **kwargs):
        return self.norm(data) * 2 - 1  # stretch between -1 and 1


class AsinhStretchEditor(Editor):
    def call(self, data, **kwargs):
        return AsinhStretch(0.1)(data, out=data, clip=False)


class NormalizeEditor(Editor):

    def __init__(self, fixed_min=None, fixed_max=None):
        self.channels_min = {}
        self.channels_max = {}

        self.fixed_max = {} if fixed_max is None else fixed_max
        self.fixed_min = {} if fixed_min is None else fixed_min

    def call(self, data, **kwargs):
        for i in range(data.shape[-1]):
            d = data[..., i]
            if i in self.fixed_min:
                d = ImageNormalize(vmin=self.fixed_min[i], vmax=self.fixed_max[i])(d) * 2 - 1
                data[..., i] = d
                continue
            if i not in self.channels_min:
                self.channels_min[i] = []
                self.channels_max[i] = []

            min = np.min(data)
            max = np.max(data)

            self.channels_min[i].append(min)
            self.channels_max[i].append(max)

            if len(self.channels_min[i]) >= 1000:
                self.fixed_min[i] = np.min(self.channels_min[i])
                self.fixed_max[i] = np.max(self.channels_max[i])
                logging.info(
                    "Fixed data normalization: CHANNEL %d MIN %f MAX %f" % (i, self.fixed_min[i], self.fixed_max[i]))

            d = ImageNormalize(vmin=np.min(self.channels_min[i]), vmax=np.max(self.channels_max[i]))(d) * 2 - 1
            data[..., i] = d
        return data


class NanEditor(Editor):
    def call(self, data, **kwargs):
        data = np.nan_to_num(data)
        return data


class DOTDataEditor(Editor):
    def call(self, data, **kwargs):
        header = kwargs["header"]
        if "D_LAMBDA" not in header:
            raise Exception("INVALID WAVELENGTH")
        d_lambda = eval(header["D_LAMBDA"])
        lambdas = [abs(l + header["LAMBDA"] - 6563) for l in d_lambda]
        if min(lambdas) > 0.2:
            raise Exception("INVALID WAVELENGTH")
        data = data[lambdas.index(min(lambdas)) + 1]

        return data


class AIAPrepEditor(Editor):
    def call(self, aia_map, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore warnings
            if "lvl_num" not in aia_map.meta or aia_map.meta["lvl_num"] != 1.5:
                aia_map = aiaprep(aia_map)

            aia_map.meta["arcs_pp"] = aia_map.scale[0].value
            return aia_map


class EITPrepEditor(Editor):
    def call(self, data, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore warnings
            eit_map = data

            if 'N_MISSING_BLOCKS' in eit_map.meta:
                assert eit_map.meta['N_MISSING_BLOCKS'] == 0, 'Invalid quality of SOHO image'
            if 'obs_type' in eit_map.meta:
                assert eit_map.meta['obs_type'] == 'MAGNI', 'Invalid MDI map'

            target_scale = 2.4

            # angle = -eit_map.meta["sc_roll"]
            # c = np.cos(np.deg2rad(angle))
            # s = np.sin(np.deg2rad(angle))
            #
            # eit_map.meta["PC1_1"] = c
            # eit_map.meta["PC1_2"] = -s
            # eit_map.meta["PC2_1"] = s
            # eit_map.meta["PC2_2"] = c

            scale = target_scale * u.arcsec
            scale_factor = eit_map.scale[0] / scale
            tempmap = eit_map.rotate(recenter=True, scale=scale_factor.value, missing=eit_map.min())

            center = np.floor(tempmap.meta['crpix1'])
            range_side = (center + np.array([-1, 1]) * eit_map.data.shape[0] / 2) * u.pix
            eit_map = tempmap.submap(u.Quantity([range_side[0], range_side[0]]),
                                     u.Quantity([range_side[1], range_side[1]]))
            eit_map.meta['lvl_num'] = 1.5

            return eit_map


class STEREOPrepEditor(Editor):
    def call(self, data, **kwargs):
        stereo_map = data
        if 'NMISSING' in stereo_map.meta:
            assert stereo_map.meta["NMISSING"] == 0.0, 'Invalid quality of STEREO image'
        assert stereo_map.meta['obsrvtry'] == 'STEREO_A', 'Invalid STEREO data'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore warnings

            target_scale = 2.4

            scale = target_scale * u.arcsec
            scale_factor = stereo_map.scale[0] / scale
            tempmap = stereo_map.rotate(recenter=True, scale=scale_factor.value, missing=stereo_map.min())

            center = np.floor(tempmap.meta['crpix1'])
            range_side = (center + np.array([-1, 1]) * stereo_map.data.shape[0] / 2) * u.pix
            stereo_map = tempmap.submap(u.Quantity([range_side[0], range_side[0]]),
                                        u.Quantity([range_side[1], range_side[1]]))
            stereo_map.meta['lvl_num'] = 1.5

            return stereo_map


class KSOPrepEditor(Editor):
    def __init__(self, add_rotation=False):
        self.add_rotation = add_rotation

        super().__init__()

    def call(self, kso_map, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore warnings
            kso_map.meta["waveunit"] = "ag"
            kso_map.meta["arcs_pp"] = kso_map.scale[0].value

            if self.add_rotation:
                angle = -kso_map.meta["angle"]
            else:
                angle = 0
            c = np.cos(np.deg2rad(angle))
            s = np.sin(np.deg2rad(angle))

            kso_map.meta["PC1_1"] = c
            kso_map.meta["PC1_2"] = -s
            kso_map.meta["PC2_1"] = s
            kso_map.meta["PC2_2"] = c

            return kso_map


class KSOPlatePrepEditor(Editor):
    def call(self, kso_data, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore warnings
            header = kwargs["header"]

            coord = SkyCoord(0 * u.arcsec, 0 * u.arcsec, obstime=header["DATE"], observer='earth',
                             frame=frames.Helioprojective)

            kso_data = kso_data[0]
            coord_header = sunpy.map.header_helper.make_fitswcs_header(
                kso_data, coord,
                reference_pixel=u.Quantity([header["CENTER_X"], header["CENTER_Y"]] * u.pixel),
                scale=u.Quantity([header["CDELT1"], header["CDELT2"]] * u.arcsec / u.pixel), )
            # rotation_angle=header["SOLAR_P0"] * u.deg)

            plate_map = sunpy.map.Map(kso_data, coord_header)

            return plate_map


class RescaleEditor(Editor):
    def __init__(self, target_arcspp):
        self.arcspp = target_arcspp

    def call(self, s_map, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore warnings
            scale_factor = s_map.scale[0] / self.arcspp
            tempmap = s_map.rotate(recenter=True, scale=scale_factor.value, missing=s_map.min())

            center = np.floor(tempmap.meta['crpix1'])
            range_side = (center + np.array([-1, 1]) * s_map.data.shape[0] / 2) * u.pix
            tempmap = tempmap.submap(u.Quantity([range_side[0], range_side[0]]),
                                     u.Quantity([range_side[1], range_side[1]]))
            return tempmap


class StackWrapper(Editor):
    def __init__(self, editors):
        self.editors = editors

    def call(self, data, **kwargs):
        stack = []
        kwargs_stack = []
        kwargs = kwargs["kwargs_stack"] if "kwargs_stack" in kwargs else [kwargs] * len(data)
        for d, d_kwargs in zip(data, kwargs):
            for editor in self.editors:
                d, d_kwargs = editor.convert(d, **d_kwargs)
            stack.append(d)
            kwargs_stack.append(d_kwargs)
        return stack, {"kwargs_stack": kwargs_stack}


class LoadFilesFromDirectoryEditor(Editor):

    def __init__(self, extension="*.fts.gz"):
        self.extension = extension

    def call(self, dir, **kwargs):
        file_paths = sorted(glob.glob(os.path.join(dir, self.extension)))
        return file_paths


class LoadFilesFromDirectoriesEditor(Editor):

    def __init__(self, directories):
        self.directories = directories

    def call(self, file_id, **kwargs):
        file_paths = [os.path.join(d, file_id) for d in self.directories]
        return file_paths


class StackImagesEditor(Editor):

    def call(self, data, **kwargs):
        # prevent errors due to single pixel shifts
        # min_x = min([d[0].shape[0] for d in data])
        # min_y = min([d[0].shape[1] for d in data])
        # images = [d[0][0:min_x, 0:min_y] for d in data]
        dstack = np.dstack(data)
        return dstack


class LoadNumpyEditor(Editor):

    def call(self, data, **kwargs):
        # dst = shutil.copy(data, os.path.join(os.environ.get("TMPDIR"), os.path.basename(data)))
        d = np.load(data)
        if d.shape[0] == 1:
            d = d[0]  # batch flatten
        # os.remove(dst)
        return d


class RandomFlipEditor(Editor):

    def __init__(self, probability, **kwargs):
        self.probability = probability
        super().__init__(**kwargs)

    def call(self, data, **kwargs):
        if random.random() < self.probability:
            data = np.flip(data, 0)
        if random.random() < self.probability:
            data = np.flip(data, 1)
        return data


class LambdaEditor(Editor):

    def __init__(self, func):
        self.func = func

    def call(self, data, **kwargs):
        return self.func(data, **kwargs)


class KeywordsEditor(Editor):

    def __init__(self, keywords, include_header=True):
        self.keywords = keywords
        self.include_header = include_header

    def call(self, data, **kwargs):
        if self.include_header:
            kwargs.update(kwargs['header'])
        return (data, *[kwargs[k] for k in self.keywords]), kwargs


class PassEditor(Editor):

    def call(self, data, **kwargs):
        return data
