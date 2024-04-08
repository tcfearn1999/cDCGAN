import numpy as np
from numpy.random import randn, randint
from keras.datasets.fashion_mnist import load_data
from keras.optimizers.legacy import Adam
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, Embedding, Concatenate

class CGAN:
    def __init__(self, latent_dim=100, n_classes=10, img_shape=(28, 28, 1),learning_rate=0.0002, beta=0.5, batch=128, epoch=100):
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.img_shape = img_shape
        self.lr = learning_rate
        self.beta = beta
        self.batch = batch
        self.epoch = epoch
        self.optimizer = Adam(learning_rate=self.lr, beta_1=self.beta)
        
        # Build and compile discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        
        # Build and compile generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        
        # Combined model
        self.gan = self.build_gan()
        self.gan.compile(loss='binary_crossentropy', optimizer=self.optimizer)

    def build_discriminator(self):
        in_label = Input(shape=(1,))
        li = Embedding(self.n_classes, 50)(in_label)
        n_nodes = self.img_shape[0] * self.img_shape[1]
        li = Dense(n_nodes)(li)
        li = Reshape((self.img_shape[0], self.img_shape[1], 1))(li)
        in_image = Input(shape=self.img_shape)
        merge = Concatenate()([in_image, li])
        fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(merge)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = Flatten()(fe)
        fe = Dropout(0.4)(fe)
        out_layer = Dense(1, activation='sigmoid')(fe)
        model = Model([in_image, in_label], out_layer)
        return model

    def build_generator(self):
        in_label = Input(shape=(1,))
        li = Embedding(self.n_classes, 50)(in_label)
        n_nodes = 7 * 7
        li = Dense(n_nodes)(li)
        li = Reshape((7, 7, 1))(li)
        in_lat = Input(shape=(self.latent_dim,))
        n_nodes = 128 * 7 * 7
        gen = Dense(n_nodes)(in_lat)
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = Reshape((7, 7, 128))(gen)
        merge = Concatenate()([gen, li])
        gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
        gen = LeakyReLU(alpha=0.2)(gen)
        out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
        model = Model([in_lat, in_label], out_layer)
        return model

    def build_gan(self):
        self.discriminator.trainable = False
        gen_noise, gen_label = self.generator.input
        gen_output = self.generator.output
        gan_output = self.discriminator([gen_output, gen_label])
        model = Model([gen_noise, gen_label], gan_output)
        return model

    def load_real_samples(self):
        (trainX, trainy), (_, _) = load_data()
        X = np.expand_dims(trainX, axis=-1)
        X = X.astype('float32')
        X = (X - 127.5) / 127.5
        return [X, trainy]

    def generate_real_samples(self, dataset, n_samples):
        images, labels = dataset
        ix = randint(0, images.shape[0], n_samples)
        X, labels = images[ix], labels[ix]
        y = np.ones((n_samples, 1))
        return [X, labels], y

    def generate_latent_points(self, n_samples):
        x_input = randn(self.latent_dim * n_samples)
        z_input = x_input.reshape(n_samples, self.latent_dim)
        labels = randint(0, self.n_classes, n_samples)
        return [z_input, labels]

    def generate_fake_samples(self, n_samples):
        z_input, labels_input = self.generate_latent_points(n_samples)
        images = self.generator.predict([z_input, labels_input])
        y = np.zeros((n_samples, 1))
        return [images, labels_input], y

    def train(self, dataset):
        bat_per_epo = int(dataset[0].shape[0] / self.batch)
        half_batch = int(self.batch / 2)
        for i in range(self.epoch):
            for j in range(bat_per_epo):
                [X_real, labels_real], y_real = self.generate_real_samples(dataset, half_batch)
                d_loss1, _ = self.discriminator.train_on_batch([X_real, labels_real], y_real)
                [X_fake, labels], y_fake = self.generate_fake_samples(half_batch)
                d_loss2, _ = self.discriminator.train_on_batch([X_fake, labels], y_fake)
                [z_input, labels_input] = self.generate_latent_points(self.batch)
                y_gan = np.ones((self.batch, 1))
                g_loss = self.gan.train_on_batch([z_input, labels_input], y_gan)
                print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                      (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))

        self.generator.save('cgan_generator.keras')

# Example usage:
cgan = CGAN()
dataset = cgan.load_real_samples()
cgan.train(dataset)
