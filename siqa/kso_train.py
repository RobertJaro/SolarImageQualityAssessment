import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import matplotlib
matplotlib.use("agg")

from siqa.data.converter import convertProvider
from siqa.data.editor import ContrastNormalizeEditor, ImageNormalizeEditor, \
    PeakNormalizeEditor
from siqa.model import IQAModel, DiscriminatorMode
from siqa.train import IQATrainer
import logging
from siqa.data.provider import NumpyProvider, Norm, KSODataProvider
from siqa.evaluation.callback import IQAPlot, IQAProgressPlot

###################### PARAMETERS ##############################
n_compress_channels = 8
norm = Norm.CONTRAST
classifier_mode = True
resolution = 128
base_path = "/gss/r.jarolim/prediction_anomaly_v14/kso_conf2"
q2_ds_path = "/gss/r.jarolim/data/anomaly_data_set/quality2"
q1_ds_path = "/gss/r.jarolim/data/anomaly_data_set/quality1"
q2_converted_path = '/gss/r.jarolim/data/converted/q2_128_v3'
q1_converted_path = '/gss/r.jarolim/data/converted/q1_128_v3'
################################################################

prediction_path = os.path.join(base_path, "prediction")
os.makedirs(prediction_path, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("{0}/{1}.log".format(base_path, "info_log")),
        logging.StreamHandler()
    ])

######################### INIT PROVIDERS ################################
norm_editor = None
if norm == Norm.IMAGE:
    norm_editor = ImageNormalizeEditor(peak_adjust=True)
if norm == Norm.PEAK:
    norm_editor = PeakNormalizeEditor()
if norm == Norm.CONTRAST:
    norm_editor = ContrastNormalizeEditor(use_median=True, threshold=1.5)
assert norm_editor is not None, 'Invalid Normalization'

q2_converted_path_validation = q2_converted_path + '_validation'
q1_converted_path_validation = q1_converted_path + '_validation'
if not os.path.exists(q2_converted_path):
    provider = KSODataProvider(q2_ds_path, ext="*.fts.gz",
                               patch_shape=(resolution, resolution), norm=Norm.NONE)
    convertProvider(provider, q2_converted_path, validation_split=0.1)
if not os.path.exists(q1_converted_path):
    provider = KSODataProvider(q1_ds_path, ext="*.fts.gz",
                               patch_shape=(resolution, resolution), norm=Norm.NONE)
    convertProvider(provider, q1_converted_path, validation_split=0.1)

provider_A = NumpyProvider(q2_converted_path, ext_editors=[norm_editor],
                           patch_shape=(128, 128))
provider_B = NumpyProvider(q1_converted_path, ext_editors=[norm_editor],
                           patch_shape=(128, 128))

######################### INIT MODEL ################################
model = IQAModel((resolution, resolution, 1), depth=3, discriminator_depth=4, res_blocks=9, base_n_filters=64,
                 activation='tanh', learning_rate=2e-4, lambda_discriminator=1, lambda_mse=0, lambda_content=1,
                 discriminator_mode=DiscriminatorMode.LSGAN, n_feature_layers=None,
                 n_compress_channels=n_compress_channels, classifier_mode=classifier_mode)

######################### INIT CALLBACKS ################################
valid_provider_A = NumpyProvider(q2_converted_path_validation,
                                 ext_editors=[norm_editor], patch_shape=(128, 128), limit=10)
valid_provider_B = NumpyProvider(q1_converted_path_validation,
                                 ext_editors=[norm_editor], patch_shape=(128, 128), limit=10)

q1_callback = IQAPlot(valid_provider_B, model, prediction_path, log_iteration=1000, plot_id='Q1')
q2_callback = IQAPlot(valid_provider_A, model, prediction_path, log_iteration=1000, plot_id='Q2')
validation_progress = IQAProgressPlot(prediction_path, log_iteration=1000)

valid_provider_A = NumpyProvider(q2_converted_path_validation, ext_editors=[norm_editor], patch_shape=(128, 128), limit=218)
valid_provider_B = NumpyProvider(q1_converted_path_validation, ext_editors=[norm_editor], patch_shape=(128, 128), limit=218)

trainer = IQATrainer(model,
                     train_provider=(provider_A, provider_B), base_path=base_path,
                     callbacks=[q1_callback, q2_callback, validation_progress],
                     validation_provider=(valid_provider_A, valid_provider_B),
                     log_interval=1000, batch_size=1)
trainer.fit(int(10e6))
