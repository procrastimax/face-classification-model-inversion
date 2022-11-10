import tensorflow as tf
import numpy as np
import os

# image sizes are 92x112
image_height = 112
image_width = 92
data_dir = "test-images"

class_labels = ['s1', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's2', 's20', 's21', 's22', 's23', 's24', 's25', 's26', 's27', 's28', 's29', 's3', 's30', 's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38', 's39', 's4', 's40', 's5', 's6', 's7', 's8', 's9']

print("Loading test images")

# a dict containing the class name and the keras image
img_dict: dict = {}

#for dir in os.listdir(data_dir):
#    dir_path = os.path.join(data_dir, dir)
#    if os.path.isdir(dir_path):
#        for img in os.listdir(dir_path):
#            img_path = os.path.join(dir_path, img)
#            keras_img = tf.keras.utils.load_img(img_path, target_size=(image_height, image_width), color_mode="grayscale")
#            img_array = tf.keras.utils.img_to_array(keras_img)
#            img_array = img_array / 255.0
#            img_array = tf.expand_dims(img_array, 0)
#            img_dict[dir] = img_array

keras_img = tf.keras.utils.load_img("./test-images/s1/1.png", target_size=(image_height, image_width), color_mode="grayscale")
img_array = tf.keras.utils.img_to_array(keras_img)
img_array = img_array / 255.0
img_array = tf.expand_dims(img_array, 0)
img_dict["s1"] = img_array

keras_img = tf.keras.utils.load_img("./img_random_start_img.png", target_size=(image_height, image_width), color_mode="grayscale")
img_array = tf.keras.utils.img_to_array(keras_img)
img_array = img_array / 255.0
img_array = tf.expand_dims(img_array, 0)
img_dict["random"] = img_array

keras_img = tf.keras.utils.load_img("./img_adversary_img.png", target_size=(image_height, image_width), color_mode="grayscale")
img_array = tf.keras.utils.img_to_array(keras_img)
img_array = img_array / 255.0
img_array = tf.expand_dims(img_array, 0)
img_dict["adversary"] = img_array

print("Loading model")
loaded_model = tf.keras.models.load_model("model")

for img_path in img_dict:
    predictions = loaded_model(img_dict[img_path])
    print(f"Image: {img_path}, predicted: {class_labels[np.argmax(predictions)]}, probability: {np.max(predictions) * 100:.2f}%")
