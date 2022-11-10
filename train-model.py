import tensorflow as tf
import matplotlib.pyplot as plt

# image sizes are 92x112
image_height = 112
image_width = 92

# test out the batch size here a bit
batch_size = 32
#data_dir = "face-images/"
data_dir = "face-images-all/"

train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=data_dir,
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=(image_height, image_width),
    batch_size=batch_size,
    color_mode="grayscale"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    directory=data_dir,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=(image_height, image_width),
    batch_size=batch_size,
    color_mode="grayscale"
)

normalization_layer = tf.keras.layers.Rescaling(1. / 255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))


def make_model(num_classes: int):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(image_height, image_width)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    return model


model = make_model(num_classes=40)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

model.summary()

epochs = 50
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']

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
