from keras.datasets import mnist
import imageProcessing as ip
import networkModel

model = networkModel.def_model_simple_ff()
model.compile(optimizer='adadelta', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()
(y_train, _), (y_test, _) = mnist.load_data()

x_train = ip.normalize_reshape(x_train)
x_test = ip.normalize_reshape(x_test)
y_train = ip.normalize_reshape(y_train)
y_test = ip.normalize_reshape(y_test)

x_train = ip.noise_images(x_train, 0.5)
x_test = ip.noise_images(x_test, 0.5)

model.fit(x_train, y_train,
          epochs=100,
          batch_size=256,
          shuffle=True,
          validation_data=(x_test, y_test))

model.save('models/result.model')
