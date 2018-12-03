import numpy as np


# noise_factor is how much noise you want from 0 to 1.
def noise_images(images, noise_factor):
    images_noisy = images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    images_noisy = np.clip(images_noisy, 0., 1.)
    return images_noisy


def normalize_reshape(images):
    images = images.astype('float32') / 255.
    images = images.reshape((len(images), np.prod(images.shape[1:])))
    return images

