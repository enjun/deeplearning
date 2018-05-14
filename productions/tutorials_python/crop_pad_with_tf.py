# %% 读图片代码
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
import tensorflow as tf

test_image = 'D:\pywork\gitWork\deeplearning\dataset\pictures\一寸照.jpg'
image_raw_data_jpg = tf.gfile.FastGFile(test_image, 'rb').read()
with tf.Session() as sess:
    # output the original JPG image
    lena = mpimg.imread(test_image)
    plt.figure(0)
    plt.imshow(lena)  # 显示图片
    plt.show()

    # Decode a JPG image
    img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg)
    img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.float32)

    # resize it to 299 by 299 using default method.
    resized_image = tf.image.resize_images(img_data_jpg, [299, 299])
    plt.figure(1)
    plt.imshow(resized_image.eval())  # 显示图片
    plt.show()

    # crop it to 299 by 299 using default method.
    cropped_image = tf.image.resize_image_with_crop_or_pad(img_data_jpg, 299, 299)
    plt.figure(2)
    plt.imshow(cropped_image.eval())  # 显示图片
    plt.show()  