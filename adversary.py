import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# image sizes are 92x112
image_height = 112
image_width = 92
data_dir = "test-images"

class_labels = ['s1', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's2', 's20', 's21', 's22', 's23', 's24', 's25', 's26', 's27', 's28', 's29', 's3', 's30', 's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38', 's39', 's4', 's40', 's5', 's6', 's7', 's8', 's9']

loaded_model: tf.keras.Model = tf.keras.models.load_model("model")
print("Model loaded\n")


def show_image(img_array, title: str):
    img = tf.math.multiply(img_array, 255)
    img = img.numpy().astype("uint8").squeeze()

   # plt.imshow(img, cmap='gray')
   # plt.axis("off")
   # plt.title = title
   # plt.show()

    print(f"Saving image {title}")
    pil_image = Image.fromarray(img)
    pil_image.save(f"img_{title}.png")


#adv_image = np.random.rand(image_height, image_width)
adv_image = np.zeros((image_height, image_width))
adv_image = tf.expand_dims(adv_image, axis=0)
adv_image = tf.Variable(adv_image)

#optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)


# create array for classes, but only set first class to 1 - since we only want to optimize for the first class
class_labels = np.zeros(shape=(1, 40))
class_labels[0][0] = 1

curr_pred = 0


def custom_loss():
    for var in optimizer.variables():
        var.assign(tf.zeros_like(var))
    global curr_pred
    prediction = loaded_model(adv_image)
    curr_pred = tf.math.reduce_max(tf.math.multiply(prediction, class_labels))
    loss = 1 - curr_pred
    return loss


epochs = 5000
for i in range(epochs):
    optimizer.minimize(custom_loss, [adv_image])
    print(f"Optimizing in epoch {i+1}/{epochs} - Prediction: {curr_pred:.2f}")
    if i % 500:
        show_image(adv_image, "adversary_img")

show_image(adv_image, "adversary_img")
