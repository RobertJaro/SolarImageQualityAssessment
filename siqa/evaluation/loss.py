import os

import pandas
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm

from siqa.data.editor import ImageNormalizeEditor, PeakNormalizeEditor, ContrastNormalizeEditor
from siqa.data.provider import Norm, NumpyProvider
from siqa.model import IQAModel, DiscriminatorMode
from siqa.tools.common import Enqueuer
from siqa.train import validateProvider, inferenceTime

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import matplotlib

matplotlib.use("agg")

import os

from matplotlib import pyplot as plt, pylab
import numpy as np

base_path = "/gss/r.jarolim/prediction_anomaly_v14/kso_conf2"
normalization = Norm.CONTRAST
classifier_mode = True
n_compress_channels = 8

prediction_path = os.path.join(base_path, "evaluation")
os.makedirs(prediction_path, exist_ok=True)

resolution = 128

model = IQAModel((resolution, resolution, 1), depth=3, discriminator_depth=4, res_blocks=9, base_n_filters=64,
                     activation='tanh', learning_rate=2e-4, lambda_discriminator=1, lambda_mse=0, lambda_content=1,
                     discriminator_mode=DiscriminatorMode.LSGAN, n_feature_layers=None,
                     n_compress_channels=n_compress_channels,
                     classifier_mode=classifier_mode)
model.combined_generator.load_weights(os.path.join(base_path, "model_%06d.h5") % 300000)
if model.classifier_mode:
    model.classifier.load_weights(os.path.join(base_path, "classifier_model_%06d.h5") % 300000)

norm_editor = None
if normalization == Norm.IMAGE:
    norm_editor = ImageNormalizeEditor(peak_adjust=True)
if normalization == Norm.PEAK:
    norm_editor = PeakNormalizeEditor()
if normalization == Norm.CONTRAST:
    norm_editor = ContrastNormalizeEditor(use_median=True, threshold=1.5)
assert norm_editor is not None, 'Invalid Normalization'
provider_q2 = NumpyProvider('/gss/r.jarolim/data/converted/test_q2_128_v2',
                            ext_editors=[norm_editor], patch_shape=(128, 128))
provider_q1 = NumpyProvider('/gss/r.jarolim/data/converted/test_q1_128_v2',
                            ext_editors=[norm_editor], patch_shape=(128, 128))
provider_validation = NumpyProvider('/gss/r.jarolim/data/converted/q1_128_v3_validation',
                                    ext_editors=[norm_editor], patch_shape=(128, 128))

queue_q2 = Enqueuer(provider_q2, batch_size=128, shuffle=False)
queue_q1 = Enqueuer(provider_q1, batch_size=128, shuffle=False)
queue_validation = Enqueuer(provider_validation, batch_size=128)
try:
    disc_fake_q2, disc_real_q2, ssim_q2, mse_q2, content_q2, classify_q2 = validateProvider(model, queue_q2, True)
    disc_fake_q1, disc_real_q1, ssim_q1, mse_q1, content_q1, classify_q1 = validateProvider(model, queue_q1, True)
    _, _, ssim_valid, mse_valid, content_valid, classify_valid = validateProvider(model, queue_validation, True)
finally:
    queue_q2.stop()
    queue_q1.stop()
    queue_validation.stop()

# provider_q2 = KSOFullDiscNormDataProvider("/gss/r.jarolim/data/kso_general/quality2", ext="*.fts.gz",
#                                        patch_shape=(resolution, resolution), arcs_pp=scale, norm=normalization)
# provider_q1 = KSOFullDiscNormDataProvider("/gss/r.jarolim/data/kso_general/quality1", ext="*.fts.gz",
#                                        patch_shape=(resolution, resolution), arcs_pp=scale, norm=normalization)
# provider_q1.data = provider_q1.data[:512]
# print('Quality 2 inference time:', inferenceTime(model, provider_q2), 'Samples %d' % len(provider_q2))
# print('Quality 1 inference time:', inferenceTime(model, provider_q1), 'Samples %d' % len(provider_q1))
# print('Quality Valid inference time:', inferenceTime(model, provider_validation),
#       'Samples %d' % len(provider_validation))


def plotHist(reference, data_q1, data_q2):
    # data_q1, data_q2 = np.copy(data_q1), np.copy(data_q2)
    mean, std = norm.fit(reference)
    split_line = mean + 3 * std
    bins = np.linspace(min(data_q1), min(data_q1) + 3 * (split_line - min(data_q1)), 100)
    # data_q1[data_q1 > max(bins)] = max(bins)
    # data_q2[data_q2 > max(bins)] = max(bins)
    plt.hist(data_q1, bins, color='#32447A', alpha=0.75)
    plt.hist(data_q2, bins, color='#EFC726', alpha=0.75)
    plt.axvline(split_line, color='red')
    x = np.linspace(min(bins), max(bins), 1000)
    p = norm.pdf(x, mean, std)
    binwidth = (max(bins) - min(bins)) / len(bins)
    p = p * (len(data_q1) * binwidth)
    plt.plot(x, p, '--', linewidth=2.5, color='black')
    plt.xlim((min(bins), max(bins)))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))


def plotHistInverted(reference, data_q1, data_q2):
    # data_q1, data_q2 = np.copy(data_q1), np.copy(data_q2)
    mean, std = norm.fit(reference)
    split_line = mean - 3 * std
    bins = np.linspace(max(data_q1) + 3 * (split_line - max(data_q1)), max(data_q1), 100)
    # data_q1[data_q1 > max(bins)] = max(bins)
    # data_q2[data_q2 > max(bins)] = max(bins)
    plt.hist(data_q1, bins, color='#32447A', alpha=0.75)
    plt.hist(data_q2, bins, color='#EFC726', alpha=0.75)
    plt.axvline(split_line, color='red')
    x = np.linspace(min(bins), max(bins), 1000)
    p = norm.pdf(x, mean, std)
    binwidth = (max(bins) - min(bins)) / len(bins)
    p = p * (len(data_q1) * binwidth)
    plt.plot(x, p, '--', linewidth=2.5, color='black')
    plt.xlim((min(bins), max(bins)))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    plt.gca().invert_xaxis()


params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)

plt.figure(figsize=(20, 10))
plt.subplot(411)
plt.title('Binary Classification')
bins = np.linspace(0, 3, 100)

classify_q1 = np.array(classify_q1)
classify_q2 = np.array(classify_q2)
classify_q1[classify_q1 > 3] = 3
classify_q1[classify_q1 < 0] = 0
classify_q2[classify_q2 > 3] = 3
classify_q2[classify_q2 < 0] = 0
plt.hist(3 - classify_q1, bins, color='#32447A', alpha=0.75)
plt.hist(3 - classify_q2, bins, color='#EFC726', alpha=0.75)
plt.axvline(1, color='red')
plt.xlim((min(bins), max(bins)))
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))

plt.subplot(412)
plt.title('Mean-Squared-Error')
plotHist(mse_valid, mse_q1, mse_q2)

plt.subplot(413)
plt.title('Structural-Similarity-Index')
plotHistInverted(ssim_valid, np.array(ssim_q1), np.array(ssim_q2))

plt.subplot(414)
plt.title('Content-Loss')
plotHist(content_valid, content_q1, content_q2)

plt.tight_layout()
plt.savefig(os.path.join(prediction_path, "histogram.jpg"), dpi=300)
plt.close()

high_mse = np.mean(mse_q1)
margin_content = np.abs(np.mean(content_q1) - np.mean(content_q2))
margin_mse = np.abs(np.mean(mse_q1) - np.mean(mse_q2))
margin_ssim = np.abs(np.mean(ssim_q1) - np.mean(ssim_q2))

content_q1 = np.array(content_q1)
content_q2 = np.array(content_q2)
content_valid = np.array(content_valid)
if model.classifier_mode:
    threshold = np.mean(content_valid) + 3 * np.std(content_valid)
    predict_q1_dsq1 = (3 - classify_q1) < 1
    predict_q2_dsq1 = (3 - classify_q1) >= 1
    predict_q1_dsq2 = (3 - classify_q2) < 1
    predict_q2_dsq2 = (3 - classify_q2) >= 1
    # positive = anomaly
    # q1 = classified as q1 + content < threshold
    # q2 = classified as q2 or content > threshold
    tn = np.sum(content_q1[predict_q1_dsq1] < threshold)
    tp = np.sum(predict_q2_dsq2) + np.sum(content_q2[predict_q1_dsq2] >= threshold)
    fn = np.sum(np.logical_and(predict_q1_dsq2, content_q2 < threshold))
    fp = np.sum(predict_q2_dsq1) + np.sum(content_q1[predict_q1_dsq1] >= threshold)
    # tn = np.sum(predict_q1_dsq1)
    # tp = np.sum(predict_q2_dsq2)
    # fn = np.sum(predict_q2_dsq1)
    # fp = np.sum(predict_q1_dsq2)
else:
    threshold = np.mean(content_valid) + 2 * np.std(content_valid)
    tn = np.sum(content_q1 < threshold)
    tp = np.sum(content_q2 >= threshold)
    fn = np.sum(content_q2 < threshold)
    fp = np.sum(content_q1 >= threshold)

acc = (tp + tn) / (tp + tn + fp + fn)
tss = tp / (tp + fn) - fp / (fp + tn)

threshold = np.mean(content_valid) + 3 * np.std(content_valid)
scale = (min(content_valid), threshold)
result = {'high_mse': high_mse, 'margin_content': margin_content, 'scale': scale, 'margin_mse': margin_mse,
          'margin_ssim': margin_ssim, 'tp': tp, 'tn': tn,
          'fn': fn, 'fp': fp, 'acc': acc, 'tss': tss}
pandas.DataFrame([result]).to_csv(os.path.join(prediction_path, 'result.csv'))
