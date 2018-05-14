import keras
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.layers.convolutional import ZeroPadding2D
from keras.utils import np_utils
import keras.backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
from readcifar import read_cifar, to_image  # it is my personal module


def my_model():
    input = Input(shape=[32, 32, 3])
    x = ZeroPadding2D((85, 85))(input)
    basic_model = InceptionV3(weights='imagenet', include_top=False)
    x = basic_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(218, activation='relu')(x)
    logits = Dense(10, activation='softmax')(x)
    model = Model(inputs=input, outputs=logits)
    for layer in basic_model.layers:
        layer.trainable = False
    return model


version = 'v1.0'
epochs = 1
batch_size = 128

dir_model = 'model'
if not os.path.exists(dir_model):
    os.mkdir(dir_model)
if not os.path.exists('tensorboard'):
    os.mkdir('tensorboard')

dir = r'../../dataset/cifar-10-batches-py'
if not os.path.exists(r'model'):
    os.mkdir(r'model')

(x_trains, y_trains), (x_test, y_test), label_names = read_cifar(dir)
y_trains = np_utils.to_categorical(y_trains, 10)
y_test = np_utils.to_categorical(y_test, 10)
if not os.path.exists(r'dir_data'):
    os.mkdir(r'dir_data')
    x_test = to_image(x_test, 10000)
    x_trains = to_image(x_trains, 50000)
    np.savez(r'dir_data/out.npz', x_trains, x_test)
r = np.load(r'dir_data/out.npz')
x_trains = r['arr_0']
x_test = r['arr_1']

if os.path.exists('model/model_{}.h5'.format(version)):
    model = load_model('model/model_{}.h5'.format(version))
else:
    model = my_model()

model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001, rho=0.9),
              loss='categorical_crossentropy')
mdl = keras.callbacks.ModelCheckpoint(dir_model, monitor='val_loss', save_best_only=True, verbose=2)
save_model = keras.callbacks.ModelCheckpoint(r'model/model_{}.h5'.format(version),
                                             monitor='val_loss',
                                             verbose=1)
tb = keras.callbacks.TensorBoard('tensorboard', histogram_freq=1, batch_size=batch_size)
history = model.fit(x_trains[:500], y_trains[:500], batch_size=batch_size,
          epochs=1, verbose=1,
          callbacks=[mdl, save_model, tb], validation_split=0.2)
loss, accuracy = model.evaluate(x_test[:128], y_test[:128], batch_size=batch_size)
print('test loss:{}  test accuracy:{}\n'.format(loss, accuracy))

print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model_accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'test'])
plt.show()


# img_path = '../../dataset/pictures/African_elephant2.jpg'
# img = image.load_img(img_path, target_size=(224, 224))
# # plt.imshow(img)
# # plt.show()
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

# preds = model.predict(x)
# # decode the results into a list of tuples (class, description, probability)
# # (one such list for each sample in the batch)
# print('Predicted:', decode_predictions(preds, top=3)[0])
# # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), \
# # (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
