import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers
import random
import matplotlib.pyplot as plt



# brain
x_val = np.load('/home/miranda/Documents/code/3DCNN/data/x_val_b.npy')
x_test = np.load('/home/miranda/Documents/code/3DCNN/data/x_test_b.npy')

y_val = np.load('/home/miranda/Documents/code/3DCNN/data/y_val_b.npy')
y_test = np.load('/home/miranda/Documents/code/3DCNN/data/y_test_b.npy')

# vessel 
x_val2 = np.load('/home/miranda/Documents/code/3DCNN/data/x_val2_b.npy')
x_test2 = np.load('/home/miranda/Documents/code/3DCNN/data/x_test2_b.npy')

y_val2 = np.load('/home/miranda/Documents/code/3DCNN/data/y_val2_b.npy')
y_test2 = np.load('/home/miranda/Documents/code/3DCNN/data/y_test2_b.npy')


y_val = keras.utils.to_categorical(y_val, 3)
y_test = keras.utils.to_categorical(y_test, 3)

y_val2 = keras.utils.to_categorical(y_val2, 3)
y_test2 = keras.utils.to_categorical(y_test2, 3)

@tf.function
def preprocessing(volume, label):
	"""Process training data by rotating and adding a channel."""
	volume = tf.expand_dims(volume, axis=3)
	return volume, label


batch_size = 2
# validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
validation_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))
validation_dataset = (
	validation_loader.shuffle(len(x_val))
	.map(preprocessing)
	.batch(batch_size)
	.prefetch(2)
)


# validation_loader_d = tf.data.Dataset.from_tensor_slices((x_val2, y_val2))
validation_loader_d = tf.data.Dataset.from_tensor_slices((x_test2, y_test2))
validation_dataset_d = (
	validation_loader.shuffle(len(x_val2))
	.map(preprocessing)
	.batch(batch_size)
	.prefetch(2)
)

g_model = tf.keras.models.load_model('../saved_model/g_model')
# g_model.summary()

history = g_model.fit(validation_dataset)
print(history.history.keys())


d_model = tf.keras.models.load_model('../saved_model/d_model')

histy = d_model.fit(validation_dataset_d)
print(histy.history.keys())



print(history.history['accuracy'])
# plt.figure()
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy (Global)')
# plt.ylim([0, 1])
# plt.legend(loc='lower right')
# plt.savefig("../pic/Accuracy_global.png")


print(histy.history['accuracy'])
# plt.figure()
# plt.plot(histy.history['accuracy'], label='accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy (Detail)')
# plt.ylim([0, 1])
# plt.legend(loc='lower right')
# plt.savefig("../pic/Accuracy_detail.png")


# # Load best weights.
# g_model.load_weights("../G_3d_image_classification.h5")
# # remove the output layer
model = keras.Model(inputs=g_model.inputs, outputs=g_model.layers[-2].output)

# # get extracted features
features = model.predict(np.expand_dims(x_val[0], axis=0))[0]
# print(features)
print(features.shape)

# d_model.load_weights("../D_3d_image_classification.h5")
# # remove the output layer
model2 = keras.Model(inputs=d_model.inputs, outputs=d_model.layers[-2].output)

# # get extracted features
features2 = model2.predict(np.expand_dims(x_val[0], axis=0))[0]
# print(features2)
print(features2.shape)

inputs = (features + features2)/2.0
print(inputs.shape)

def make_model(input_shape = 512):
	# # add new classifier layers
	inputs = keras.Input((input_shape, ))

	# flat1 = layers.Flatten()(fea)
	x = layers.Dense(1024, activation='relu')(inputs)
	x = layers.Dense(128, activation='relu')(x)
	x = keras.layers.Dropout(0.2)(x)

	outputs = layers.Dense(3, activation='softmax')(x)

	# Define the model.
	model = keras.Model(inputs, outputs, name="3dcnn_d")
	# score = tf.nn.softmax(output[0])
	return model


epochs = 50

callbacks = [
    keras.callbacks.ModelCheckpoint("hybrid_classification.h5",save_best_only=True),
]

early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15)

model = make_model()

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_dataset_d,
    validation_data=validation_dataset_d,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[callbacks,early_stopping_cb],
)


# class_names = ["CE", "LAA", "SV"]

# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )

# # plt.show()