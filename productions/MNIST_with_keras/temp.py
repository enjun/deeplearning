from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt


model = ResNet50(weights='D:\workspace\dataset\\resnet50_weights\\resnet50_weights_tf_dim_ordering_tf_kernels.h5')

img_path = 'D:\workspace\dataset\pictures\African_elephant2.jpg'
img = image.load_img(img_path, target_size=(224, 224))
# plt.imshow(img)
# plt.show()
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), \
# (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]