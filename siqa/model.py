import logging
from datetime import datetime
from enum import Enum
from functools import partial

from keras import Input
from keras import backend as K
from keras.initializers import RandomNormal
from keras.layers import Conv2D, LeakyReLU, Activation, Add, SeparableConv2D, Conv2DTranspose, \
    Subtract, Lambda, Reshape, AveragePooling2D
from keras.models import Model
from keras.optimizers import Adam

from siqa.tools.common import NameLayer, wasserstein_loss, RandomWeightedAverage, gradient_penalty_loss
from siqa.tools.common import QuantizationLayer
from siqa.tools.common import ReflectionPadding2D, InstanceNormalization


class DiscriminatorMode(Enum):
    WGAN = 'wgan'
    LSGAN = 'lsgan'


class IQAModel:
    def __init__(self, shape, depth=3, discriminator_depth=4, n_discriminators=3, res_blocks=9, base_n_filters=64,
                 base_n_filters_discriminator=None, n_compress_channels=8, n_centers=5, activation=None,
                 learning_rate=2e-4, batch_size=None, lambda_discriminator=1, lambda_mse=0, lambda_content=10,
                 discriminator_mode: DiscriminatorMode = DiscriminatorMode.LSGAN, n_feature_layers=None,
                 classifier_mode=False):
        logging.info("######################### Model Configuration ##########################")
        logging.info("shape: %s" % str(shape))
        logging.info("Depth:   %d" % depth)
        logging.info("Discriminator Depth:   %d" % discriminator_depth)
        logging.info("Number of Discriminators:   %d" % n_discriminators)
        logging.info("Discriminator Mode:   %s" % discriminator_mode)
        logging.info("Classifier Mode:   %s" % classifier_mode)
        logging.info("Number of Feature Layers:   %s" % str(n_feature_layers))
        logging.info("Residual Blocks:   %d" % res_blocks)
        logging.info("Base Filters:   %d" % base_n_filters)
        logging.info("Compress Channels:   %s" % str(n_compress_channels))
        logging.info("Bites:   %s" % str(n_centers))
        logging.info("Activation:   %s" % str(activation))
        logging.info("Learning Rate:   %f" % learning_rate)
        logging.info("Batch Size:   %s" % str(batch_size))
        logging.info("Lambda Discriminator Loss:   %f" % lambda_discriminator)
        logging.info("Lambda MSE Loss:   %f" % lambda_mse)
        logging.info("Lambda Content Loss:   %f" % lambda_content)
        logging.info("START TIME:   %s" % datetime.now())
        logging.info("########################################################################")
        self.shape = shape
        self.discriminator_output_shape = (shape[0] // 2 ** discriminator_depth,
                                           shape[1] // 2 ** discriminator_depth, 1)

        self.depth = depth
        self.discriminator_depth = discriminator_depth
        self.res_blocks = res_blocks
        self.n_discriminators = n_discriminators
        self.discriminator_mode = discriminator_mode
        self.classifier_mode = classifier_mode
        self.n_feature_layers = n_feature_layers if n_feature_layers is not None else discriminator_depth

        self.base_n_filters = base_n_filters
        self.base_n_filters_discriminator = base_n_filters_discriminator if base_n_filters_discriminator is not None else base_n_filters
        self.n_compress_channels = n_compress_channels
        self.n_bites = n_centers
        self.activation = activation
        self.batch_size = batch_size

        self.initializer = RandomNormal(stddev=0.02)

        ############################## LOSS WEIGHTS ###############################
        self.lambda_discriminator = lambda_discriminator
        self.lambda_content = lambda_content
        self.lambda_mse = lambda_mse

        self.optimizer_discriminator = Adam(learning_rate, beta_1=0.5, beta_2=0.9)
        self.optimizer_generator = Adam(learning_rate, beta_1=0.5, beta_2=0.9)

        ############################## Build Models ###############################
        self.build()

        self.combined_generator, self.combined_discriminator, self.combined_classifier, self.mask_generator, self.content_loss_model = self.buildCombined()

        ############################## Set Primary Discriminator ###############################
        self.discriminator = self.discriminators[0][0]
        self.classifier = self.buildClassifier('classifier') if self.classifier_mode else None
        self.translator = self.buildTranslator()

    def build(self):
        self.discriminators = [
            self.buildDiscriminator((self.shape[0] // 2 ** i, self.shape[1] // 2 ** i, self.shape[2]),
                                    "discriminator%i" % i) for i in range(self.n_discriminators)]

        if self.classifier_mode:
            self.classifiers = [
                self.buildDiscriminator((self.shape[0] // 2 ** i, self.shape[1] // 2 ** i, self.shape[2]),
                                        "classifier%i" % i) for i in range(self.n_discriminators)]
        else:
            self.classifiers = []

        logging.info("######################### Discriminator ##############################")
        self.discriminators[0][0].summary(print_fn=logging.info)

        logging.info("######################### Feature Extractor ##############################")
        self.discriminators[0][1].summary(print_fn=logging.info)

        generator = self.buildGenerator("generator")

        logging.info("######################### Generator #################################")
        generator.summary(print_fn=logging.info)

        self.generator = generator

    def buildCombined(self):
        if self.classifier_mode:
            # select discriminator + content loss from classifier
            discriminators, feature_extractors = [d for d, f in self.discriminators], [f for d, f in self.classifiers]
        else:
            discriminators, feature_extractors = zip(*self.discriminators)

        self.setTrainable(generator_trainable=True, discriminator_trainable=False)
        combined_generator = self.combineGenerator(self.generator, discriminators, feature_extractors,
                                                   "combined_generator")

        mask_generator = self.buildMaskGenerator(self.generator, feature_extractors, 'mask')
        content_loss_model = self.buildCombinedContentLoss(feature_extractors, 'content_loss')

        logging.info("######################### Combined Generator #####################################")
        combined_generator.summary(print_fn=logging.info)

        self.setTrainable(generator_trainable=False, discriminator_trainable=True)
        combined_discriminator = self.combineDiscriminator(self.generator, discriminators, feature_extractors,
                                                           "combined_discriminator")

        logging.info("######################### Combined Discriminator #####################################")
        combined_discriminator.summary(print_fn=logging.info)

        combined_classifier = self.buildCombinedClassifier('combined_classifier') if self.classifier_mode else None

        return combined_generator, combined_discriminator, combined_classifier, mask_generator, content_loss_model

    def combineDiscriminator(self, generator, discriminators, feature_extractors, name):
        img = Input(shape=self.shape, name=name + '_input')
        fake = generator(img)

        discriminator_output = []
        discriminator_loss = []
        for i, (discriminator, feature_extractor) in enumerate(zip(discriminators, feature_extractors)):
            fake_resized = AveragePooling2D(2 ** i)(fake)
            img_resized = AveragePooling2D(2 ** i)(img)

            disc_out_fake = discriminator(fake_resized)
            disc_out_real = discriminator(img_resized)

            # Rename Outputs
            disc_out_real = NameLayer(name=name + "_real%d" % i)(disc_out_real)
            disc_out_fake = NameLayer(name=name + "_fake%d" % i)(disc_out_fake)

            averaged_samples = RandomWeightedAverage()([img_resized, fake_resized])
            disc_out_avg = discriminator(averaged_samples)

            partial_gp_loss = partial(gradient_penalty_loss,
                                      averaged_samples=averaged_samples,
                                      gradient_penalty_weight=10)
            partial_gp_loss.__name__ = name + 'gradient_penalty%d' % i

            d_out = [disc_out_real, disc_out_fake] if self.discriminator_mode == DiscriminatorMode.LSGAN else \
                [disc_out_real, disc_out_fake, disc_out_avg]
            d_loss = ['mse', 'mse'] if self.discriminator_mode == DiscriminatorMode.LSGAN else \
                [wasserstein_loss, wasserstein_loss, partial_gp_loss]

            discriminator_output.extend(d_out)
            discriminator_loss.extend(d_loss)

        discriminator_model = Model(inputs=[img],
                                    outputs=discriminator_output,
                                    name=name)

        discriminator_model.compile(optimizer=self.optimizer_discriminator,
                                    loss=discriminator_loss)
        return discriminator_model

    def buildCombinedClassifier(self, name):
        img_valid = Input(shape=self.shape, name=name + '_input_valid')
        img_invalid = Input(shape=self.shape, name=name + '_input_invalid')

        discriminator_output = []
        discriminator_loss = []
        for i, (discriminator, feature_extractor) in enumerate(self.classifiers):
            resized_invalid = AveragePooling2D(2 ** i)(img_invalid)
            resized_valid = AveragePooling2D(2 ** i)(img_valid)

            disc_out_invalid = discriminator(resized_invalid)
            disc_out_valid = discriminator(resized_valid)

            # Rename Outputs
            disc_out_valid = NameLayer(name=name + "_valid%d" % i)(disc_out_valid)
            disc_out_invalid = NameLayer(name=name + "_invalid%d" % i)(disc_out_invalid)

            averaged_samples = RandomWeightedAverage()([resized_valid, resized_invalid])
            disc_out_avg = discriminator(averaged_samples)

            partial_gp_loss = partial(gradient_penalty_loss,
                                      averaged_samples=averaged_samples,
                                      gradient_penalty_weight=10)
            partial_gp_loss.__name__ = name + 'gradient_penalty%d' % i

            d_out = [disc_out_valid, disc_out_invalid] if self.discriminator_mode == DiscriminatorMode.LSGAN else \
                [disc_out_valid, disc_out_invalid, disc_out_avg]
            d_loss = ['mse', 'mse'] if self.discriminator_mode == DiscriminatorMode.LSGAN else \
                [wasserstein_loss, wasserstein_loss, partial_gp_loss]

            discriminator_output.extend(d_out)
            discriminator_loss.extend(d_loss)

        discriminator_model = Model(inputs=[img_valid, img_invalid],
                                    outputs=discriminator_output,
                                    name=name)

        discriminator_model.compile(optimizer=self.optimizer_discriminator,
                                    loss=discriminator_loss)
        return discriminator_model

    def buildClassifier(self, name):
        img = Input(shape=self.shape, name=name + '_input')

        discriminator_output = []
        discriminator_loss = []
        for i, (discriminator, feature_extractor) in enumerate(self.classifiers):
            img_resized = AveragePooling2D(2 ** i)(img)

            disc_out = discriminator(img_resized)

            # Rename Outputs
            disc_out = NameLayer(name=name + "_real%d" % i)(disc_out)

            discriminator_output.extend([disc_out])
            discriminator_loss.extend(['mse'])

        discriminator_model = Model(inputs=[img],
                                    outputs=discriminator_output,
                                    name=name)

        discriminator_model.compile(optimizer=self.optimizer_discriminator,
                                    loss=discriminator_loss)
        return discriminator_model

    def combineGenerator(self, generator, discriminators, feature_extractors, name):

        ############################## INPUTS #########################################
        img = Input(shape=self.shape, name=name + '_imageA')

        ############################## TRANSLATE IMAGES ###############################
        fake = generator(img)

        discriminator_output = []
        for i, (discriminator, feature_extractor) in enumerate(zip(discriminators, feature_extractors)):
            fake_resized = AveragePooling2D(2 ** i)(fake)
            img_resized = AveragePooling2D(2 ** i)(img)

            valid = discriminator(fake_resized)
            features_original = feature_extractor(img_resized)
            features_generated = feature_extractor(fake_resized)

            content = self._getContentDiff(features_original, features_generated, "content%d" % i)

            valid = NameLayer(name="generator_discriminator%d" % i)(valid)
            discriminator_output.extend([valid, content])

        fake = NameLayer(name="generator_mse")(fake)

        combined = Model(inputs=[img], outputs=[fake] + discriminator_output, name=name)
        g_loss = ['mse', 'mae'] if self.discriminator_mode == DiscriminatorMode.LSGAN else [wasserstein_loss, 'mae']
        combined.compile(loss=['mse'] + g_loss * self.n_discriminators,
                         loss_weights=[self.lambda_mse] +
                                      [self.lambda_discriminator, self.lambda_content] * self.n_discriminators,
                         optimizer=self.optimizer_generator)

        return combined

    def buildCombinedContentLoss(self, feature_extractors, name):

        ############################## INPUTS #########################################
        img = Input(shape=self.shape, name=name + '_image')

        ############################## TRANSLATE IMAGES ###############################
        fake = Input(shape=self.shape, name=name + '_fake')

        discriminator_output = []
        for i, feature_extractor in enumerate(feature_extractors):
            fake_resized = AveragePooling2D(2 ** i)(fake)
            img_resized = AveragePooling2D(2 ** i)(img)

            features_original = feature_extractor(img_resized)
            features_generated = feature_extractor(fake_resized)

            content = self._getContentDiff(features_original, features_generated, "content%d" % i)

            discriminator_output.extend([content])

        combined = Model(inputs=[img, fake], outputs=discriminator_output, name=name)
        return combined

    def buildMaskGenerator(self, generator, feature_extractors, name):
        ############################## INPUTS #########################################
        img = Input(shape=self.shape, name=name + '_imageA')

        ############################## TRANSLATE IMAGES ###############################
        fake = generator(img)

        masks = []
        for i, feature_extractor in enumerate(feature_extractors):
            fake_resized = AveragePooling2D(2 ** i)(fake)
            img_resized = AveragePooling2D(2 ** i)(img)

            features_original = feature_extractor(img_resized)
            features_generated = feature_extractor(fake_resized)

            for o, g in zip(features_original, features_generated):
                mask = Lambda(lambda x: K.mean(K.abs(x[0] - x[1]), -1))([o, g])
                masks.append(mask)

        combined = Model(inputs=[img], outputs=masks, name=name)
        return combined

    def _getContentDiff(self, features_original, features_generated, name):
        diffs = []
        for i, (original, generated) in enumerate(zip(features_original, features_generated)):
            s = Subtract(name=name + '_subtract%d' % i)([original, generated])
            s = Lambda(lambda x: K.mean(K.abs(x), [1, 2, 3]),
                       output_shape=lambda input_shape: input_shape[0:1], name=name + '_MSE%d' % i)(s)
            s = Reshape((1,), name=name + '_reshape%d' % i)(s)
            diffs.append(s)
        content = Add(name=name)(diffs)
        return content

    def buildGenerator(self, name):
        n_filters = self.base_n_filters

        img_input = Input(shape=self.shape, name=name + '_input')
        x = img_input

        x = ReflectionPadding2D((3, 3), name=name + '_input_pad')(x)
        x = Conv2D(n_filters, 7, name=name + '_input_conv', kernel_initializer=self.initializer,
                   bias_initializer=self.initializer)(x)
        x = InstanceNormalization(name=name + '_input_in')(x)
        x = Activation("relu", name=name + '_input_relu')(x)

        for i in range(self.depth):
            n_filters *= 2
            x = self._downBlock(x, n_filters, name + '_down%d' % i)

        encoded = ReflectionPadding2D((1, 1), name=name + '_encoded_pad')(x)
        encoded = Conv2D(self.n_compress_channels, kernel_size=3, strides=1, activation="sigmoid",
                         name=name + '_encoded_conv', kernel_initializer=self.initializer,
                         bias_initializer=self.initializer)(encoded)
        encoded = QuantizationLayer(self.n_bites, name=name + '_encoded_quant')(encoded)

        x = ReflectionPadding2D((1, 1), name=name + '_reshape_pad')(encoded)
        x = Conv2D(n_filters, 3, name=name + '_reshape_conv', kernel_initializer=self.initializer,
                   bias_initializer=self.initializer)(x)
        x = InstanceNormalization(name=name + '_reshape_in')(x)
        x = Activation("relu", name=name + '_reshape_relu')(x)

        for i in range(self.res_blocks):
            x = self._identityBlock(x, n_filters, name + '_identity%d' % (i + self.res_blocks))

        for i in range(self.depth):
            n_filters = n_filters // 2
            x = self._upBlock(x, n_filters, name=name + '_up%d' % i)

        x = ReflectionPadding2D((3, 3), name=name + '_output_pad')(x)
        output_img = Conv2D(self.shape[-1], kernel_size=7, strides=1, activation=self.activation,
                            name=name + '_output', kernel_initializer=self.initializer,
                            bias_initializer=self.initializer)(x)
        return Model(img_input, output_img, name=name)

    def buildDiscriminator(self, shape, name):
        n_filters = self.base_n_filters_discriminator

        img = Input(shape=shape, name=name + '_input')
        x = img
        layers = []

        for i in range(self.discriminator_depth):
            normalize = False if self.discriminator_mode == DiscriminatorMode.WGAN else (i != 0)
            x = self._discBlock(x, n_filters, name=name + '_block%d' % i, normalize=normalize)
            layers.append(x)
            n_filters *= 2

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same', name=name + '_validity',
                          kernel_initializer=self.initializer, bias_initializer=self.initializer)(x)
        discriminator = Model(img, validity, name=name)

        feature_extractor = Model(img, layers[-self.n_feature_layers:], name=name + "_feature_extractor")

        return discriminator, feature_extractor

    def _downBlock(self, input_tensor, n_filters, name):

        x = ReflectionPadding2D((1, 1), name=name + '_conv_pad')(input_tensor)
        x = Conv2D(n_filters, 3, strides=2, name=name + '_conv', kernel_initializer=self.initializer,
                   bias_initializer=self.initializer)(x)
        x = InstanceNormalization(name=name + '_conv_in')(x)
        x = Activation("relu", name=name + '_conv_relu')(x)

        return x

    def _upBlock(self, input_tensor, n_filters, name):
        x = Conv2DTranspose(n_filters, 3, strides=2, padding='same', name=name + '_conv2T',
                            kernel_initializer=self.initializer, bias_initializer=self.initializer)(input_tensor)
        x = InstanceNormalization(name=name + '_conv2_in')(x)
        x = Activation('relu', name=name + '_relu')(x)
        return x

    def _identityBlock(self, x, n_filters, name):
        y = x

        x = ReflectionPadding2D((1, 1), name=name + '_conv1_pad')(x)
        x = Conv2D(n_filters, 3, name=name + '_conv1', kernel_initializer=self.initializer,
                   bias_initializer=self.initializer)(x)
        x = InstanceNormalization(name=name + '_conv1_in')(x)
        x = Activation("relu", name=name + '_conv1_relu')(x)

        x = ReflectionPadding2D((1, 1), name=name + '_conv2_pad')(x)
        x = Conv2D(n_filters, 3, name=name + '_conv2', kernel_initializer=self.initializer,
                   bias_initializer=self.initializer)(x)
        x = InstanceNormalization(name=name + '_conv2_in')(x)

        x = Add(name=name + '_add')([x, y])
        # x = Activation("relu", name=name + '_relu')(x)

        return x

    def _separableIdentityBlock(self, x, n_filters, name):
        y = x

        x = ReflectionPadding2D((1, 1), name=name + '_conv1_pad')(x)
        x = SeparableConv2D(n_filters, 3, name=name + '_sepconv1', kernel_initializer=self.initializer,
                            bias_initializer=self.initializer)(x)
        x = InstanceNormalization(name=name + '_conv1_in')(x)
        x = Activation("relu", name=name + '_conv1_relu')(x)

        x = ReflectionPadding2D((1, 1), name=name + '_conv2_pad')(x)
        x = SeparableConv2D(n_filters, 3, name=name + '_sepconv2', kernel_initializer=self.initializer,
                            bias_initializer=self.initializer)(x)
        x = InstanceNormalization(name=name + '_conv2_in')(x)

        x = Add(name=name + '_add')([x, y])
        x = Activation("relu", name=name + '_relu')(x)

        return x

    def _discBlock(self, input_tensor, n_filters, name, normalize=True):
        x = Conv2D(n_filters, 4, padding="same", strides=2, name=name + '_conv', kernel_initializer=self.initializer,
                   bias_initializer=self.initializer)(input_tensor)
        if normalize:
            x = InstanceNormalization(name=name + '_in')(x)
        x = LeakyReLU(0.2, name=name + '_lrelu')(x)
        return x

    def setTrainable(self, generator_trainable, discriminator_trainable):
        for layer in self.generator.layers:
            layer.trainable = generator_trainable
        self.generator.trainable = generator_trainable

        for discriminator, feature_extractor in self.discriminators:
            for layer in discriminator.layers:
                layer.trainable = discriminator_trainable
            discriminator.trainable = discriminator_trainable
            for layer in feature_extractor.layers:
                layer.trainable = discriminator_trainable
            feature_extractor.trainable = discriminator_trainable

        for discriminator, feature_extractor in self.classifiers:
            for layer in discriminator.layers:
                layer.trainable = discriminator_trainable
            discriminator.trainable = discriminator_trainable
            for layer in feature_extractor.layers:
                layer.trainable = discriminator_trainable
            feature_extractor.trainable = discriminator_trainable

    def buildTranslator(self):
        img = Input(shape=(None, None, self.shape[-1]))
        reconstr_img = self.generator(img)

        classifier_out = []
        for i, (classifier, feature_extractor) in enumerate(self.classifiers):
            fake_resized = AveragePooling2D(2 ** i)(reconstr_img)
            img_resized = AveragePooling2D(2 ** i)(img)

            c = classifier(fake_resized)
            features_original = feature_extractor(img_resized)
            features_generated = feature_extractor(fake_resized)

            content = self._getContentDiff(features_original, features_generated, "content%d" % i)

            classifier_out.extend([c, content])

        translator = Model(img, [reconstr_img, *classifier_out])
        return translator
