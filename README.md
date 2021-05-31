# Image Quality Assessment for Full-Disk Solar Observations with Generative Adversarial Networks
[![Watch the video](results/full_overview_1.jpg)](https://youtu.be/rJEe27osgTI)
[![Watch the video](results/full_overview_2.jpg)](https://youtu.be/U-8uEmErupE)
[![Watch the video](results/full_overview_3.jpg)](https://youtu.be/YygyymxqVFk)
[![Watch the video](results/full_overview_4.jpg)](https://youtu.be/OlmC4BnD020)
[![Watch the video](results/full_overview_5.jpg)](https://youtu.be/sCKDFREpJEw)
[![Watch the video](results/full_overview_6.jpg)](https://youtu.be/9MvdLDtxKBo)
## \>>> Watch the videos: [Quality-Scale](https://youtu.be/9MvdLDtxKBo), [2018-09-27](https://youtu.be/rJEe27osgTI), [2018-09-28](https://youtu.be/U-8uEmErupE), [2018-09-29](https://youtu.be/YygyymxqVFk), [2018-09-30](https://youtu.be/OlmC4BnD020), [2019-01-26](https://youtu.be/sCKDFREpJEw)

# [Paper](#paper) --- [Guide](#guide) --- [Citation](#citation) --- [Contact](#contact)

## Abstract
_Context._ Within the last decades, solar physics has entered the era of big data and the amount of data being constantly produced from ground- and space-based observatories can no longer be purely analyzed by human observers.

_Aims._ In order to assure a stable series of recorded images of sufficient quality for further scientific analysis, an objective image quality measure is required. Especially when dealing with ground-based observations, which are subject to varying seeing conditions and clouds, the quality assessment has to take multiple effects into account and provide information about the affected regions. The robust detection of degrading effects in real-time is a critical task, in order to maximise the scientific return from the observation series and to allow for robust event detections in real-time. In this study, we develop a deep learning method that is suited to identify anomalies and provide an image quality assessment of solar full-disk Hα filtergrams. The approach is based on the structural appearance and the true image distribution of high-quality observations.

_Methods._ We employ a neural network with an encoder-decoder architecture to perform an identity transformation of selected high-quality observations. Hereby we use the encoder network to achieve a compressed representation of the input data, which is reconstructed to the original by the decoder. We use adversarial training to recover truncated information based on the high-quality image distribution. When images with reduced quality are transformed, the reconstruction of unknown features (e.g., clouds, contrails, partial occultation) shows deviations from the original. This difference is used to quantify the quality of the observations and to identify the corresponding regions. In addition, we present an extension of this architecture by using also low-quality samples in the training step, which takes the characteristics of both quality domains into account and improves the sensitivity for minor distortions.

_Results._ We apply our method to full-disk Hα filtergrams from Kanzelhöhe Observatory recorded during 2012-2019 and demonstrate its capability to perform reliable detections of various distortion effects. Our quality metric achieves an accuracy of 98.5% in distinguishing observations with degrading effects from distortion-free observations and provides a continuous quality measure which is in good agreement with the human perception.

_Conclusion._ The developed method is capable of providing a reliable image quality assessment in real-time, without the requirement of reference observations. Our approach has the potential for further application to similar astrophysical observations and requires only little effort of manual labeling.


## Guide
If you want to train a model with your own data, you can write your own training script. This 
section guides you through the available tools and how to setup training. See also ``siqa.kso_train`` as a reference.

The application is based on keras and sunpy. Make sure to have the all python packages installed:
```shell script
conda install keras
or
conda install keras-gpu -c anaconda

conda install sunpy -c conda-forge
conda install scikit-learn
conda install tqdm
```
Clone/Download this repository and create your own training script within the project: ``siqa.my_train.py``

First we need to import the required modules:
```python
import os
import matplotlib
# In case you are working with server resources you might need:
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# matplotlib.use("agg")

from siqa.data.converter import convertProvider
from siqa.data.editor import ContrastNormalizeEditor, ImageNormalizeEditor
from siqa.model import IQAModel, DiscriminatorMode
from siqa.train import IQATrainer
import logging
from siqa.data.provider import NumpyProvider, Norm, KSODataProvider
from siqa.evaluation.callback import IQAPlot, IQAProgressPlot
```
Next we define the paths and parameters:
```python
n_compress_channels = 8
classifier_mode = True
resolution = 128

base_path = "path for logs and monitoring"
q2_ds_path = "low quality data set"
q1_ds_path = "high quality data set"
q2_ds_path_validation = "low quality data set for validation"
q1_ds_path_validation = "high quality data set for validation"
```
Create the path and setup logging:
```python
prediction_path = os.path.join(base_path, "prediction")
os.makedirs(prediction_path, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("{0}/{1}.log".format(base_path, "info_log")),
        logging.StreamHandler()
    ])
```
Define the normalization:
```python
norm = Norm.IMAGE
or
norm = Norm.CONTRAST
```
Create data providers for both quality classes, which automatically fetch FITS files from the specified directory and
converts them to numpy arrays:
```python
provider_q2 = KSODataProvider(q2_ds_path, ext="*.fts.gz", patch_shape=(resolution, resolution), norm=norm)
provider_q1 = KSODataProvider(q1_ds_path, ext="*.fts.gz", patch_shape=(resolution, resolution), norm=norm)
```
If a different data preprocessing is requires see section [Data Provider](#data-provider).

Initialize the model:
```python
model = IQAModel((resolution, resolution, 1), depth=3, discriminator_depth=4, res_blocks=9, base_n_filters=64,
                     activation='tanh', learning_rate=2e-4, lambda_discriminator=1, lambda_mse=0, lambda_content=1,
                     discriminator_mode=DiscriminatorMode.LSGAN, n_feature_layers=None,
                     n_compress_channels=n_compress_channels, classifier_mode=classifier_mode, region_mode=False)
```
Here the following parameters can be specified:
* depth (number of strided convolution blocks in the encoder decoder architecture)
* discriminator_depth (number of strided convolutions in the discriminator)
* n_discriminators (number of discriminators used)
* res_blocks (number of residual blocks before upsampling in the decoder)
* base_n_filters (number of filters in the first layer. Afterwards the filters are increased by a factor of 2 for each decrease in resolution)
* base_n_filters_discriminator (same for discriminator)
* n_compress_channels (amount of compression channels in the encoder)
* n_centers (amount of centers used for quantizatin)
* activation (activation function for the image. For [0, 1] use 'sigmoid', for [-1, 1] use 'tanh', else None)
* learning_rate
* batch_size
* lambda_discriminator (weight for the discriminator loss)
* lambda_mse (weight for the MSE loss)
* lambda_content (weight for the Content loss)
* discriminator_mode (either DiscriminatorMode.LSGAN or DiscriminatorMode.WGAN)
* n_feature_layers (the amount of activation layers included in the content loss. Counted from the deepest layer. Use None to use all)
* classifier_mode (whether to use the classifier architecture)

Define callbacks for plots during training (with 10 random samples):
```python
valid_provider_q2 = KSODataProvider(q2_ds_path_validation, ext="*.fts.gz", patch_shape=(resolution, resolution), norm=norm, limit=10)
valid_provider_q1 = KSODataProvider(q1_ds_path_validation, ext="*.fts.gz", patch_shape=(resolution, resolution), norm=norm, limit=10)

q1_callback = IQAPlot(valid_provider_q1, model, prediction_path, log_iteration=1000, plot_id='Q1')
q2_callback = IQAPlot(valid_provider_q2, model, prediction_path, log_iteration=1000, plot_id='Q2')
```
Define callback for plotting the progress:
```python
validation_progress = IQAProgressPlot(prediction_path, log_iteration=1000)
```
Initialize the data providers for the validation set:
```python
valid_provider_q2 = KSODataProvider(q2_ds_path_validation, ext="*.fts.gz", patch_shape=(resolution, resolution), norm=norm, limit=None)
valid_provider_q1 = KSODataProvider(q1_ds_path_validation, ext="*.fts.gz", patch_shape=(resolution, resolution), norm=norm, limit=None)
```
Start the model training:
```python
trainer = IQATrainer(model,
                     train_provider=(provider_q2, provider_q1), base_path=base_path,
                     callbacks=[q1_callback, q2_callback, validation_progress],
                     validation_provider=(valid_provider_q2, valid_provider_q1),
                     log_interval=1000, batch_size=1)
trainer.fit(300000)
```
The python script can be executed with:
```shell script
cd <<path_to_project>>
python -m siqa.kso_train
```

### Data Provider
In case a different data processing pipeline is required, a custom data provider can be implemented.
The provider has to be a subclass of siqa.data.provider.DataProvider and initializes a list of editors, which 
are processed sequentially in the data pipeline.

The sample below is a simple FITS loader with image normalization:
```python
import glob
import os 
from siqa.data.provider import DataProvider
from astropy import units as u
from siqa.data.editor import ImageNormalizeEditor, ShapePrepEditor, LoadMapEditor, \
    SubMapEditor, MapToDataEditor, PyramidRescaleEditor

class FITSProvider(DataProvider):

    def __init__(self, path, ext="*.fits", patch_shape=(64, 64)):
        coords =  ([-1000, 1000] * u.arcsec, [-1000, 1000] * u.arcsec)
        map_paths = glob.glob(os.path.join(path, "**", ext), recursive=True)

        editors = [LoadMapEditor(),
                   SubMapEditor(coords),
                   MapToDataEditor(),
                   PyramidRescaleEditor(patch_shape),
                   ImageNormalizeEditor(),
                   ShapePrepEditor()]
        super().__init__(map_paths, editors=editors)
```
The available editors are listed in ``siqa.data.editor``. In order to create a new editor
create a subclass of ``siqa.data.editor.Editor`` and implement the ``call`` method. Each editor receives the 
output of the previous editor. The sample below gives an editor which loads a map from a given path:
```python
from sunpy.map import Map
from siqa.data.editor import Editor

class LoadMapEditor(Editor):

    def call(self, data, **kwargs):
        s_map = Map(data)
        return s_map
```

### Store Converted Images
In case the data pipeline requires too much computational resources, the data set can be converted first into 
numpy data files and loaded during training, which avoids converting the data multiple times.
This can be achieved with the available converter function:
```python
from siqa.data.converter import convertProvider
convertProvider(provider, 'path_to_converted_set', validation_split=0.1)
``` 
Note that also a validation set can be randomly split from the training set during this process.
The converted data set can be used with the NumpyProvider afterwards:
```python
from siqa.data.provider import NumpyProvider
provider = NumpyProvider('path_to_converted_set', ext_editors=[], patch_shape=(128, 128))
provider_validation = NumpyProvider('path_to_converted_set' + '_validation', ext_editors=[], patch_shape=(128, 128))
```
This also allows to specify additional editors which are applied after loading the numpy file (ext_editors).
The NumpyProvider can then be used same as the usual provider.

## Paper

Open-access available online: https://doi.org/10.1051/0004-6361/202038691 

## Citation

@article{jarolim2020siqa,
  title={Image-quality assessment for full-disk solar observations with generative adversarial networks},
  author={Jarolim, Robert and Veronig, AM and P{\"o}tzi, Werner and Podladchikova, Tatiana},
  journal={Astronomy \& Astrophysics},
  volume={643},
  pages={A72},
  year={2020},
  publisher={EDP Sciences}
}

## Contact

Robert Jarolim<br/>
[robert.jarolim@uni-graz.at](mailto:robert.jarolim@uni-graz.at)

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](https://github.com/RobertJaro/SolarAnomalyDetection/blob/master/LICENSE) file for details.