import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# image sizes are 92x112
image_height = 112
image_width = 92
data_dir = "test-images"

class_labels = ['s1', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's2', 's20', 's21', 's22', 's23', 's24', 's25', 's26', 's27', 's28', 's29', 's3', 's30', 's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38', 's39', 's4', 's40', 's5', 's6', 's7', 's8', 's9']

loaded_model: tf.keras.Model = tf.keras.models.load_model("model")
print("Model loaded\n")


def custom_cost_function(y_true, y_pred, loaded_model):
    predictions = loaded_model.predict_on_batch(y_pred)
    score = tf.nn.softmax(predictions[0])
    probability = 1 - score.numpy()[0]  # we want to maximize the score for the first class
    return probability


adv_image = np.random.rand(image_height, image_width, 1)
adv_image = tf.cast(adv_image, tf.float32)
adv_image = adv_image[None, :, :, :]
adv_image = tf.Variable(adv_image)

adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

class_labels = np.zeros(shape=(1, 40))
class_labels[0][0] = 1


def custom_loss():
    for var in adam_optimizer.variables():
        var.assign(tf.zeros_like(var))
    prediction = loaded_model(adv_image)
    softmax = tf.nn.softmax(prediction)
    loss = - tf.math.multiply(tf.math.log(softmax), class_labels)
    return loss


epochs = 2000
for i in range(epochs):
    print(f"Optimizing in epoch {i+1}/{epochs}")
    adam_optimizer.minimize(custom_loss, [adv_image])


adv_image = tf.math.multiply(adv_image, 255)
np_image = adv_image.numpy().astype("uint8").squeeze()

plt.imshow(np_image, cmap='gray')
plt.title("Inverted image")
plt.axis("off")

plt.show()
