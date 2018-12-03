from keras.datasets import mnist
from keras.engine.saving import load_model
import imageProcessing as ip
import plotImages

model = load_model('models/result.model')

(_, _), (x_test, _) = mnist.load_data()

x_test = ip.normalize_reshape(x_test)
x_test = ip.noise_images(x_test, 0.5)

denoised_images = model.predict(x_test)

plotImages.plot(x_test, denoised_images, 10)
