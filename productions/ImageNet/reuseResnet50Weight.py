import keras
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Input

from keras.utils import np_utils
import keras.backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
from readcifar import read_cifar, to_image  # it is my personal module


def _to_202x202_images(data, num):
    block = 20
    quotient = num // block
    remainder = num % block
    for i in range(quotient):
        temp_x = np.pad(data[block * i:(block * i + block)],
                        pad_width=[[0, 0], [85, 85], [85, 85], [0, 0]],
                        mode='constant',
                        constant_values=[0, ])
        if i == 0:
            images_x = temp_x
        else:
            images_x = np.concatenate([images_x, temp_x], axis=0)  # this operation occupy too much memory
        del temp_x
        gc.collect()
    if remainder != 0:
        temp_x = np.pad(data[quotient * block:(quotient * block + remainder)],
                        pad_width=[[0, 0], [85, 85], [85, 85], [0, 0]],
                        mode='constant',
                        constant_values=[0, ])
        if quotient != 0:
            images_x = np.concatenate([images_x, temp_x], axis=0)
        else:
            images_x = temp_x
        del temp_x
        gc.collect()
    return images_x


def split_dataset(x, y, ratio):
    num = int((1 - ratio) * len(x))
    x_tra = x[:num]
    x_val = x[num:]
    y_tra = y[:num]
    y_val = y[num:]
    return x_tra, y_tra, x_val, y_val


epochs = 1
batch_size = 128

dir_model = 'model'
if not os.path.exists(dir_model):
    os.mkdir(dir_model)

# dir_data = 'data'
# if not os.path.exists(dir_data):
#     os.mkdir(dir_data)
#     r = np.load(r'../cifar/dir_data/out.npz')
#     x_train = r['arr_0']
#     y_train = r['arr_1']
#     x_test = r['arr_2']
#     y_test = r['arr_3']
#     label_names = r['arr_4']
#     x_test = _to_202x202_images(x_test, len(x_test))
#     x_train = _to_202x202_images(x_train, len(x_train))
#     np.savez('data/out.npz', x_train, y_train, x_test, y_test, label_names)

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

basic_model = InceptionV3(weights='imagenet', include_top=False)

x = basic_model.output

x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)

logits = Dense(10, activation='softmax')(x)
model = Model(inputs=basic_model.input, outputs=logits)

for i, layer in enumerate(basic_model.layers):
   print(i, layer.name)
for layer in basic_model.layers:
    layer.trainable = False

if os.path.exists('model/cifar_with_v3_pretrain.h5'):
    del model
    model = load_model('model/cifar_with_v3_pretrain.h5')
model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001, rho=0.9),
              loss='categorical_crossentropy')
# mdl = keras.callbacks.ModelCheckpoint(dir_model, monitor='val_loss', save_best_only=True, verbose=2)
# model.fit(x_train, y_train, batch_size=128,
#           epochs=5, verbose=1,
#           callbacks=[mdl], validation_split=0.2)
x_tra, y_tra, x_val, y_val = split_dataset(x_trains, y_trains, 0.2)
loss_every_steps = []
accuracy_every_steps = []
loss_every_epoch_tra = []
accuracy_every_epoch_tra = []
loss_every_epoch_val = []
accuracy_every_epoch_val = []
for i in range(epochs):
    indices = np.random.permutation(len(x_tra))
    quotient = len(x_tra) // batch_size
    for j in range(quotient):
        x_batch = x_tra[indices[128 * j: (j + 1) * 128]]
        y_batch = y_tra[indices[128 * j: (j + 1) * 128]]
        x_batch = _to_202x202_images(x_batch, len(x_batch))

        loss_temp, accuracy_temp = model.train_on_batch(x_batch, y_batch)
        if j % 20 == 0:
            loss_every_steps.append(loss_temp)
            accuracy_every_steps.append(accuracy_temp)
            print('loss_temp:{}  accuracy_temp:{}'.format(loss_temp, accuracy_temp))
    loss_every_epoch_tra.append(np.mean(loss_every_steps))
    accuracy_every_epoch_tra.append(np.mean(accuracy_every_steps))

    loss_temp, accuracy_temp = model.evaluate(x_val, y_val, batch_size=batch_size)
    loss_every_epoch_val.append(loss_temp)
    accuracy_every_epoch_val.append(accuracy_temp)
    print('loss_val:{}  accuracy_val:{}'.format(loss_temp, accuracy_temp))
model.save('model/cifar_with_v3_pretrain.h5')

loss, accuracy = model.evaluate(x_val, y_val, batch_size=batch_size)
print('test loss:{}  test accuracy:{}'.format(loss, accuracy))
plt.figure(1)
plt.plot(list(range(epochs)), loss_every_epoch_tra)
plt.plot(list(range(epochs)), loss_every_epoch_val)
plt.title('loss')
plt.show()
plt.figure(2)
plt.plot(list(range(epochs)), accuracy_every_epoch_tra)
plt.plot(list(range(epochs)), accuracy_every_epoch_val)
plt.title('accuracy')
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
