import tensorflow as tf
import matplotlib.pyplot as plt

# image sizes are 92x112
image_height = 112
image_width = 92

# test out the batch size here a bit
batch_size = 32
data_dir = "face-images"

train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=data_dir,
    validation_split=0.34,
    subset="training",
    seed=1337,
    image_size=(image_height, image_width),
    batch_size=batch_size,
    color_mode="grayscale"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    directory=data_dir,
    validation_split=0.34,
    subset="validation",
    seed=1337,
    image_size=(image_height, image_width),
    batch_size=batch_size,
    color_mode="grayscale"
)

#data_augmentation = tf.keras.Sequential(
#   [
#       tf.keras.layers.RandomFlip("horizontal",
#                                  input_shape=(image_height,
#                                               image_width,
#                                               1)),
#       tf.keras.layers.RandomRotation(0.1),
#       tf.keras.layers.RandomZoom(0.1),
#   ]
#)


def make_model(image_size, num_classes: int):
    model = tf.keras.Sequential([
        #data_augmentation,
        tf.keras.layers.Rescaling(1. / 255, input_shape=(image_height, image_width, 1)),

        tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding="same", activation="relu", input_shape=image_size),
        tf.keras.layers.MaxPool2D(),

        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(),

        tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation="relu"),
        tf.keras.layers.Dense(num_classes)
    ])
    return model


model = make_model((image_height, image_width, 1), 40)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs = 50
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

print("Saving model")
model.save("model")
