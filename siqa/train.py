import gc
import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
from keras import backend as K
from keras.utils import plot_model
from skimage import measure
from tqdm import tqdm

from siqa.model import IQAModel, DiscriminatorMode
from siqa.data.provider import DataProvider
from siqa.evaluation.callback import Callback
from siqa.tools.common import Enqueuer


class IQATrainer:
    def __init__(self, model: IQAModel, train_provider: Tuple[DataProvider], base_path,
                 callbacks: List[Callback], validation_provider: Tuple[DataProvider] = None,
                 log_interval=1000, batch_size=1):
        self.model = model
        self.provider_A, self.provider_B = train_provider if isinstance(train_provider, tuple) else (
            None, train_provider)
        validation_provider_A, validation_provider_B = validation_provider if validation_provider else (None, None)
        self.validation_provider_A = Enqueuer(validation_provider_A, 32, 4) \
            if validation_provider_A else None
        self.validation_provider_B = Enqueuer(validation_provider_B, 32, 4) \
            if validation_provider_B else None

        self.callbacks = callbacks
        self.batch_size = batch_size

        self.discriminator_objective = []
        self.generator_objective = []
        self.classifier_objective = []
        self.region_identifier_objective = []
        for i in range(self.model.n_discriminators):
            disc_patch = self.model.discriminator_output_shape
            disc_patch = (disc_patch[0] // 2 ** i, disc_patch[1] // 2 ** i, disc_patch[2])
            dummy_features = np.zeros((batch_size,))

            if self.model.discriminator_mode == DiscriminatorMode.WGAN:
                valid = np.ones((batch_size,) + disc_patch)
                fake = -np.ones((batch_size,) + disc_patch)
                dummy = np.zeros((batch_size,) + disc_patch)

                self.discriminator_objective.extend([valid, fake, dummy])
                self.classifier_objective.extend([valid, fake])
                self.region_identifier_objective.extend([valid])
                self.generator_objective.extend([valid, dummy_features])
            else:
                valid = np.ones((batch_size,) + disc_patch)
                fake = np.zeros((batch_size,) + disc_patch)

                self.discriminator_objective.extend([valid, fake])
                self.classifier_objective.extend([valid, fake])
                self.region_identifier_objective.extend([valid])
                self.generator_objective.extend([valid, dummy_features])

        ############################## INIT PATHS ###############################
        os.makedirs(base_path, exist_ok=True)
        self.model_path = os.path.join(base_path, "model_%06d.h5")
        self.classifier_model_path = os.path.join(base_path, "classifier_model_%06d.h5")
        self.optimizer_path_generator = os.path.join(base_path, "optimizer_generator.npy")
        self.optimizer_path_discriminator = os.path.join(base_path, "optimizer_discriminator.npy")
        self.history_path = os.path.join(base_path, "history.pickle")
        self.validation_history_path = os.path.join(base_path, "validation_history.pickle")

        ############################## Restore Model ############################
        model_path = self.model_path % 0
        classifier_model_path = self.classifier_model_path % 0
        if os.path.exists(model_path):
            self.model.combined_generator.load_weights(model_path)
            logging.info("Loaded model from: %s" % model_path)
        if self.model.classifier_mode and os.path.exists(classifier_model_path):
            self.model.classifier.load_weights(classifier_model_path)
            logging.info("Loaded model from: %s" % classifier_model_path)
        # if os.path.exists(self.optimizer_path_generator):
        #     self.model.optimizer_generator.set_weights(np.load(self.optimizer_path_generator, allow_pickle=True))
        # if os.path.exists(self.optimizer_path_discriminator):
        #     self.model.optimizer_discriminator.set_weights(np.load(self.optimizer_path_discriminator, allow_pickle=True))

        if os.path.exists(self.history_path):
            file = open(self.history_path, "rb")
            self.history = pickle.load(file)
            file.close()
            logging.info("Continue training from iteration %d" % len(self.history.get("loss", [])))
        else:
            self.history = {}
        if os.path.exists(self.validation_history_path):
            file = open(self.validation_history_path, "rb")
            self.validation_history = pickle.load(file)
            file.close()
        else:
            self.validation_history = {}

        self.log_interval = log_interval
        self.initial_iteration = len(self.history.get("loss", []))

        # Plot models
        if self.initial_iteration == 0:
            plot_model(self.model.generator, os.path.join(base_path, 'generator.png'), show_shapes=True)
            plot_model(self.model.discriminator, os.path.join(base_path, 'discriminator.png'), show_shapes=True)

    def fit(self, iterations):
        data_generator_B = Enqueuer(self.provider_B, self.batch_size)
        data_generator_A = Enqueuer(self.provider_A,
                                    self.batch_size) if self.model.classifier_mode or self.model.region_mode else None

        out_labels = self.model.combined_discriminator.metrics_names + \
                     self.model.combined_generator.metrics_names
        if self.model.classifier_mode:
            out_labels += self.model.combined_classifier.metrics_names
        if self.model.region_mode:
            out_labels += self.model.combined_region_identifier.metrics_names

        try:
            for iteration in range(self.initial_iteration + 1, iterations + 1):
                iteration_start_time = datetime.now()
                train_time = timedelta(0)

                imgs = next(data_generator_B)

                # Train Discriminators
                start = datetime.now()
                do = [d[:imgs.shape[0]] for d in self.discriminator_objective]  # adjust batch size
                d_loss = self.model.combined_discriminator.train_on_batch(imgs, do)
                train_time += datetime.now() - start

                c_loss = []
                if self.model.classifier_mode:
                    imgs_A = next(data_generator_A)
                    start = datetime.now()
                    batch_size = min([imgs_A.shape[0], imgs.shape[0]])
                    co = [d[:batch_size] for d in self.classifier_objective]  # adjust batch size
                    imgs_A_adj = imgs_A[:batch_size]  # adjust batch size
                    imgs_B_adj = imgs[:batch_size]  # adjust batch size
                    c_loss = self.model.combined_classifier.train_on_batch([imgs_B_adj, imgs_A_adj], co)
                    train_time += datetime.now() - start
                r_loss = []
                if self.model.region_mode:
                    imgs_A = next(data_generator_A)
                    start = datetime.now()
                    ro = [d[:imgs_A.shape[0]] for d in self.region_identifier_objective]  # adjust batch size
                    r_loss = self.model.combined_region_identifier.train_on_batch([imgs_A], ro)
                    train_time += datetime.now() - start

                # Train Generator

                imgs = next(data_generator_B)

                start = datetime.now()
                go = [d[:imgs.shape[0]] for d in self.generator_objective]  # adjust batch size
                g_loss = self.model.combined_generator.train_on_batch(imgs, [imgs] + go)
                train_time += datetime.now() - start

                batch_loss = {label: loss for label, loss in zip(out_labels, [*d_loss, *g_loss, *c_loss, *r_loss])}
                for label, loss in batch_loss.items():
                    if label not in self.history:
                        self.history[label] = []
                    self.history[label].append(loss)

                iteration_time = datetime.now() - iteration_start_time
                if iteration_time - train_time > timedelta(seconds=0.1) and iteration > 10:
                    logging.warning("Inefficient data pipeline. Optimize pipeline or increase computational resources.")

                c_loss_str = ''
                if self.model.classifier_mode:
                    c_loss_str = '[C loss: %.03f/%.03f/%.03f]' % (
                        batch_loss['combined_classifier_valid0_loss'] + batch_loss['combined_classifier_invalid0_loss'],
                        batch_loss['combined_classifier_valid1_loss'] + batch_loss['combined_classifier_invalid1_loss'],
                        batch_loss['combined_classifier_valid2_loss'] + batch_loss[
                            'combined_classifier_invalid2_loss'],)

                r_loss_str = ''
                if self.model.region_mode:
                    r_loss_str = '[R loss: %.03f/%.03f/%.03f]' % (
                        batch_loss['combined_region_identifier_discriminator0_loss'],
                        batch_loss['combined_region_identifier_discriminator1_loss'],
                        batch_loss['combined_region_identifier_discriminator2_loss'],)

                logging.info(
                    "[Iteration %d/%d] [D loss: %.03f/%.03f/%.03f] [G loss: %.03f, adv: %.03f/%.03f/%.03f, content: %.05f, mse: %.05f] %s %s time: %s s" \
                    % (iteration, iterations,
                       batch_loss["combined_discriminator_real0_loss"],
                       batch_loss["combined_discriminator_real1_loss"],
                       batch_loss["combined_discriminator_real2_loss"],
                       batch_loss["loss"],
                       batch_loss["generator_discriminator0_loss"],
                       batch_loss["generator_discriminator1_loss"],
                       batch_loss["generator_discriminator2_loss"],
                       batch_loss["content0_loss"] + batch_loss["content1_loss"] + batch_loss["content2_loss"],
                       batch_loss['generator_mse_loss'],
                       c_loss_str,
                       r_loss_str,
                       str(iteration_time).split(":")[-1]))

                if iteration % self.log_interval == 0:
                    if iteration % (self.log_interval * 10) == 0:
                        self.model.combined_generator.save_weights(self.model_path % iteration)
                        if self.model.classifier_mode:
                            self.model.classifier.save_weights(self.classifier_model_path % iteration)
                    self.model.combined_generator.save_weights(self.model_path % 0)
                    if self.model.classifier_mode:
                        self.model.classifier.save_weights(self.classifier_model_path % 0)
                    np.save(self.optimizer_path_generator, self.model.optimizer_generator.get_weights())
                    np.save(self.optimizer_path_discriminator, self.model.optimizer_discriminator.get_weights())
                    if self.validation_provider_A:
                        self.validate()
                    self.save_history()
                    gc.collect()
                for callback in self.callbacks:
                    if iteration % callback.log_iteration == 0:
                        callback.call(iteration, history=self.history, validation_history=self.validation_history,
                                      model=self.model)
        except Exception as e:
            logging.error(e, exc_info=True)
            raise
        finally:
            data_generator_B.stop()
            if data_generator_A is not None:
                data_generator_A.stop()
            if self.validation_provider_A is not None:
                self.validation_provider_A.stop()
            if self.validation_provider_B is not None:
                self.validation_provider_B.stop()
            K.clear_session()

    def validate(self):
        if "disc_A_real" not in self.validation_history:
            self.validation_history["disc_A_real"] = []
            self.validation_history["disc_A_fake"] = []
            self.validation_history["ssim_A"] = []
            self.validation_history["mse_A"] = []
            self.validation_history["content_A"] = []
            self.validation_history["disc_B_real"] = []
            self.validation_history["disc_B_fake"] = []
            self.validation_history["ssim_B"] = []
            self.validation_history["mse_B"] = []
            self.validation_history["content_B"] = []

        start_time = datetime.now()
        disc_fake, disc_real, ssim, mse, content, classify = validateProvider(self.model, self.validation_provider_A)
        self.validation_history["disc_A_real"].append(np.mean(disc_real))
        self.validation_history["disc_A_fake"].append(np.mean(disc_fake))
        self.validation_history["ssim_A"].append(np.mean(ssim))
        self.validation_history["mse_A"].append(np.mean(mse))
        self.validation_history["content_A"].append(np.mean(content))

        self.validation_history["full_content_A"] = content
        self.validation_history["full_mse_A"] = mse
        self.validation_history["full_disc_A"] = disc_real
        self.validation_history["full_ssim_A"] = ssim
        self.validation_history["full_classify_A"] = classify

        disc_fake, disc_real, ssim, mse, content, classify = validateProvider(self.model, self.validation_provider_B)
        self.validation_history["disc_B_real"].append(np.mean(disc_real))
        self.validation_history["disc_B_fake"].append(np.mean(disc_fake))
        self.validation_history["ssim_B"].append(np.mean(ssim))
        self.validation_history["mse_B"].append(np.mean(mse))
        self.validation_history["content_B"].append(np.mean(content))

        self.validation_history['inference_time'] = (datetime.now() - start_time) / (
                len(self.validation_provider_A) * self.validation_provider_A.batch_size +
                len(self.validation_provider_B) * self.validation_provider_B.batch_size)
        self.validation_history["full_content_B"] = content
        self.validation_history["full_mse_B"] = mse
        self.validation_history["full_disc_B"] = disc_real
        self.validation_history["full_ssim_B"] = ssim
        self.validation_history["full_classify_B"] = classify

        gc.collect()

    def save_history(self):
        file = open(self.history_path, "wb")
        pickle.dump(self.history, file)
        file.close()
        if self.validation_provider_A:
            file = open(self.validation_history_path, "wb")
            pickle.dump(self.validation_history, file)
            file.close()


def validateProvider(model, provider, progress=False):
    disc_real = []
    disc_fake = []
    ssim = []
    mse = []
    content = []
    classify = []
    iter = tqdm(range(len(provider))) if progress else range(len(provider))
    for i in iter:
        img = next(provider)
        img_fake = model.generator.predict(img)

        discriminator_output = model.combined_discriminator.predict(img)
        discriminator_output = [np.mean(d, (1, 2, 3)) for d in discriminator_output]
        if model.discriminator_mode == DiscriminatorMode.WGAN:
            del discriminator_output[2::3]
        disc_fake.extend(np.mean(discriminator_output[1::2], 0))
        disc_real.extend(np.mean(discriminator_output[0::2], 0))

        for i in range(img.shape[0]):
            for j in range(img.shape[-1]):
                ssim.append(calculateSSIM(img[i, ..., j], img_fake[i, ..., j]))
        mse.extend(np.square(img - img_fake).mean(axis=(1, 2, 3)))
        generator_out = model.combined_generator.predict(img)
        content.extend(np.sum(generator_out[2::2], (0, -1)))

        if model.classifier_mode:
            c = model.classifier.predict(img)
            c = np.sum([np.mean(d, (1, 2, 3)) for d in c], 0)
            classify.extend(c)
    return disc_fake, disc_real, ssim, mse, content, classify

def inferenceTime(model, provider):
    queue = Enqueuer(provider, batch_size=1, shuffle=False, workers=15)
    try:
        imgs = np.concatenate([next(queue) for _ in tqdm(range(len(queue)))])
    finally:
        queue.stop()
    start = datetime.now()
    out = model.translator.predict(imgs, verbose=1, batch_size=128)
    return (datetime.now() - start) / len(provider)

def validateData(model, imgs, batch_size=32):
    disc_real = []
    disc_fake = []
    ssim = []
    mse = []
    content = []
    classify = []

    img_fake = model.generator.predict(imgs)
    discriminator_output = model.combined_discriminator.predict(imgs, batch_size=batch_size)
    discriminator_output = [np.mean(d, (1, 2, 3)) for d in discriminator_output]
    if model.discriminator_mode == DiscriminatorMode.WGAN:
        del discriminator_output[2::3]
    disc_fake.extend(np.mean(discriminator_output[1::2], 0))
    disc_real.extend(np.mean(discriminator_output[0::2], 0))

    for i in range(imgs.shape[0]):
        for j in range(imgs.shape[-1]):
            ssim.append(calculateSSIM(imgs[i, ..., j], img_fake[i, ..., j]))
    mse.extend(np.square(imgs - img_fake).mean(axis=(1, 2, 3)))
    generator_out = model.combined_generator.predict(imgs)
    content.extend(np.sum(generator_out[2::2], (0, -1)))

    if model.classifier_mode:
        c = model.classifier.predict(imgs)
        c = np.sum([np.mean(d, (1, 2, 3)) for d in c], 0)
        classify.extend(c)
    return disc_fake, disc_real, ssim, mse, content, classify


def calculateSSIM(img_A, img_B):
    # img_A = (img_A - np.min(img_A)) / (np.max(img_A) - np.min(img_A))
    # img_B = (img_B - np.min(img_B)) / (np.max(img_B) - np.min(img_B))
    return measure.compare_ssim(img_A, img_B, data_range=2)
