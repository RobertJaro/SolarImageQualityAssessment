import os
from abc import ABC, abstractmethod

import numpy as np
import pandas
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm
from skimage.restoration import denoise_tv_chambolle

from siqa.data.provider import LoadWrapper
from siqa.model import IQAModel


class Callback(ABC):

    def __init__(self, log_iteration=1000):
        self.log_iteration = log_iteration

    @abstractmethod
    def call(self, iteration, **kwargs):
        raise NotImplementedError()


class BasicPlot(Callback):

    def __init__(self, data_provider, model: IQAModel, path, plot_id, plot_settings, dpi=100, **kwargs):
        self.data_provider = LoadWrapper(data_provider)
        self.path = path
        self.model = model
        self.plot_settings = plot_settings
        self.dpi = dpi
        self.plot_id = plot_id

        super().__init__(**kwargs)

    def call(self, iteration, **kwargs):
        data = self.loadData()

        rows = len(data)
        columns = len(data[0])

        f, axarr = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows))
        for i in range(rows):
            for j in range(columns):
                plot_settings = self.plot_settings[j].copy()
                ax = axarr[i, j]
                ax.axis("off")
                ax.set_title(plot_settings.pop("title", None))
                ax.imshow(data[i][j], **plot_settings)
        plt.tight_layout()
        path = os.path.join(self.path, "%s_iteration%06d.jpg" % (self.plot_id, iteration))
        plt.savefig(path, dpi=self.dpi)
        plt.close()
        del f, axarr, data

    def loadData(self):
        input_data = [np.array([self.data_provider[i]]) for i in range(len(self.data_provider))]
        prediction = [self.predict(d) for d in input_data]
        prediction = list(map(list, zip(*prediction)))  # transpose
        batch_data = [np.concatenate(input_data), ] + [np.concatenate(d) for d in prediction]
        return [[d[j, ..., i] for d in batch_data for i in range(d.shape[-1])] for j in range(len(input_data))]

    def predict(self, input_data):
        raise NotImplementedError()


class IQAPlot(BasicPlot):
    def __init__(self, data_provider, model: IQAModel, path, plot_id="", **kwargs):
        plot_settings_A = [{"cmap": "gray", 'vmin': -1, 'vmax': 1}] * model.shape[-1] * 2

        plot_settings = [*plot_settings_A, {}, {'vmin': 0, 'vmax': 1}]

        super().__init__(data_provider, model, path, plot_id, plot_settings, **kwargs)

    def predict(self, input_data):
        translated = self.model.translator.predict(input_data)[0]
        difference = np.sqrt(np.abs(translated - input_data))

        diff = np.abs(translated - input_data)
        smooth = denoise_tv_chambolle(diff, multichannel=True)
        smooth[smooth <= 0.05] = 0
        smooth[smooth > 0.4] = 0.4
        smooth = smooth / 0.4
        mask = np.mean(np.sqrt(smooth), -1, keepdims=True)

        return translated, difference, mask


class IQAProgressPlot(Callback):

    def __init__(self, path, dpi=100, **kwargs):
        self.path = path
        self.dpi = dpi

        super().__init__(**kwargs)

    def plotHist(self, data_q1, data_q2):
        # data_q1, data_q2 = np.copy(data_q1), np.copy(data_q2)
        mean, std = norm.fit(data_q1)
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
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    def plotHistInverted(self, data_q1, data_q2):
        # data_q1, data_q2 = np.copy(data_q1), np.copy(data_q2)
        mean, std = norm.fit(data_q1)
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
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().invert_xaxis()

    def call(self, iteration, **kwargs):
        history = kwargs["validation_history"]
        x = range(self.log_iteration, (len(history["disc_A_real"]) + 1) * self.log_iteration, self.log_iteration)

        plt.plot(x, history["disc_A_real"], label="Discriminator A Real")
        plt.plot(x, history["disc_B_real"], label="Discriminator B Real")
        plt.plot(x, history["disc_A_fake"], label="Discriminator A Fake")
        plt.plot(x, history["disc_B_fake"], label="Discriminator B Fake")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.path, "validation_disc.jpg"), dpi=self.dpi)
        plt.close()

        plt.plot(x, history["ssim_A"], label="SSIM A")
        plt.plot(x, history["ssim_B"], label="SSIM B")
        plt.plot(x, np.abs(np.subtract(history["ssim_A"], history["ssim_B"])), label="Margin")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.path, "validation_ssim.jpg"), dpi=self.dpi)
        plt.close()

        plt.plot(x, history["mse_A"], label="MSE A")
        plt.plot(x, history["mse_B"], label="MSE B")
        plt.plot(x, np.abs(np.subtract(history["mse_A"], history["mse_B"])), label="Margin")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.path, "validation_mse.jpg"), dpi=self.dpi)
        plt.close()

        plt.plot(x, history["content_A"], label="Content A")
        plt.plot(x, history["content_B"], label="Content B")
        plt.plot(x, np.abs(np.subtract(history["content_A"], history["content_B"])), label="Margin")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.path, "validation_content.jpg"), dpi=self.dpi)
        plt.close()

        distance_A = np.abs(np.subtract(history["disc_A_real"], history["disc_A_fake"]))
        distance_B = np.abs(np.subtract(history["disc_B_real"], history["disc_B_fake"]))

        ssim_A = 1 - np.array(history["ssim_A"])
        ssim_B = 1 - np.array(history["ssim_B"])

        idx = np.round(np.linspace(0, len(distance_A) - 1, 10)).astype(np.int) if len(distance_A) > 10 else range(
            len(distance_A))

        distance_A, distance_B, ssim_A, ssim_B = distance_A[idx], distance_B[idx], ssim_A[idx], ssim_B[idx]

        plt.scatter(ssim_A, distance_A, label="A")
        plt.scatter(ssim_B, distance_B, label="B")
        for i, label in enumerate(idx):
            plt.annotate(str(label + 1), (ssim_A[i], distance_A[i]))
            plt.annotate(str(label + 1), (ssim_B[i], distance_B[i]))
        plt.xlabel("Distortion (1 - SSIM)")
        plt.ylabel("Perception (W-Distance)")
        plt.gca().set_ylim(bottom=0)
        plt.xlim([0, 1])
        plt.legend()
        plt.savefig(os.path.join(self.path, "perception_distortion.jpg"), dpi=self.dpi)
        plt.close()

        content_q1 = history["full_content_B"]
        mse_q1 = history["full_mse_B"]
        disc_q1 = history["full_disc_B"]
        classify_q1 = history["full_classify_B"]
        ssim_q1 = history["full_ssim_B"]
        content_q2 = history["full_content_A"]
        mse_q2 = history["full_mse_A"]
        disc_q2 = history["full_disc_A"]
        classify_q2 = history["full_classify_A"]
        ssim_q2 = history["full_ssim_A"]

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
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

        plt.subplot(412)
        plt.title('Mean-Squared-Error')
        self.plotHist(mse_q1, mse_q2)

        plt.subplot(413)
        plt.title('Structural-Similarity-Index')
        self.plotHistInverted(np.array(ssim_q1), np.array(ssim_q2))

        plt.subplot(414)
        plt.title('Content-Loss')
        self.plotHist(content_q1, content_q2)

        plt.tight_layout()
        plt.savefig(os.path.join(self.path, "loss%06d.jpg" % iteration), dpi=self.dpi)
        plt.close()

        high_mse = np.mean(mse_q1)
        margin_content = np.abs(np.mean(content_q1) - np.mean(content_q2))
        margin_mse = np.abs(np.mean(mse_q1) - np.mean(mse_q2))
        margin_ssim = np.abs(np.mean(ssim_q1) - np.mean(ssim_q2))

        content_q1 = np.array(content_q1)
        content_q2 = np.array(content_q2)
        if len(classify_q1) != 0:
            threshold = np.mean(content_q1) + 3 * np.std(content_q1)
            predict_q1_dsq1 = (3 - classify_q1) < 1
            predict_q2_dsq1 = (3 - classify_q1) >= 1
            predict_q1_dsq2 = (3 - classify_q2) < 1
            predict_q2_dsq2 = (3 - classify_q2) >= 1
            # positive = anomaly
            # q1 = classified as q1 + content < threshold
            # q2 = classified as q2 or content > threshold
            tn = np.sum(content_q1[predict_q1_dsq1] < threshold)
            tp = np.sum(predict_q2_dsq2) + np.sum(content_q2[predict_q1_dsq2] >= threshold)
            fn = np.sum(predict_q2_dsq1) + np.sum(content_q1[predict_q1_dsq1] >= threshold)
            fp = np.sum(np.logical_and(predict_q1_dsq2, content_q2 < threshold))
            # tn = np.sum(predict_q1_dsq1)
            # tp = np.sum(predict_q2_dsq2)
            # fn = np.sum(predict_q2_dsq1)
            # fp = np.sum(predict_q1_dsq2)
        else:
            threshold = np.mean(content_q1) + 2 * np.std(content_q1)
            tn = np.sum(content_q1 < threshold)
            tp = np.sum(content_q2 >= threshold)
            fn = np.sum(content_q1 >= threshold)
            fp = np.sum(content_q2 < threshold)

        acc = (tp + tn) / (tp + tn + fp + fn)
        tss = tp / (tp + fn + 1e-8) - fp / (fp + tn + 1e-8)

        result = {'high_mse': high_mse, 'margin_content': margin_content, 'threshold': threshold,
                  'margin_mse': margin_mse,
                  'margin_ssim': margin_ssim, 'tp': tp, 'tn': tn, 'fn': fn, 'fp': fp, 'acc': acc, 'tss': tss,
                  'time': history['inference_time']}
        df = pandas.DataFrame([result])
        df_path = os.path.join(self.path, 'loss.csv')
        df = pandas.read_csv(df_path, index_col=0).append(df) if os.path.exists(df_path) else df
        df.to_csv(df_path)
