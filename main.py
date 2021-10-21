import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers
import random

# brain
x_train = np.load('/srv/scratch/z3533133/data/x_train_spm.npy')
x_val = np.load('/srv/scratch/z3533133/data/x_val_spm.npy')
x_test = np.load('/srv/scratch/z3533133/data/y_test_spm.npy')

y_train = np.load('/srv/scratch/z3533133/data/y_train_spm.npy')
y_val = np.load('/srv/scratch/z3533133/data/y_val_spm.npy')
y_test = np.load('/srv/scratch/z3533133/data/y_test_spm.npy')

# WM 
x_train2 = np.load('/srv/scratch/z3533133/data/x_train_wm_spm.npy')
x_val2 = np.load('/srv/scratch/z3533133/data/x_val_wm_spm.npy')
x_test2 = np.load('/srv/scratch/z3533133/data/x_test_wm_spm.npy')

y_train2 = np.load('/srv/scratch/z3533133/data/y_train_wm_spm.npy')
y_val2 = np.load('/srv/scratch/z3533133/data/y_val_wm_spm.npy')
y_test2 = np.load('/srv/scratch/z3533133/data/y_test_wm_spm.npy')


y_train = keras.utils.to_categorical(y_train, 3)
y_val = keras.utils.to_categorical(y_val, 3)
y_test = keras.utils.to_categorical(y_test, 3)

y_train2 = keras.utils.to_categorical(y_train2, 3)
y_val2 = keras.utils.to_categorical(y_val2, 3)
y_test2 = keras.utils.to_categorical(y_test2, 3)


print(
    "Number of samples in train and validation are %d and %d."
    % (x_train.shape[0], x_val.shape[0])
)



@tf.function
def preprocessing(volume, label):
	"""Process training data by rotating and adding a channel."""
	# volume = normalize(volume)
	volume = tf.expand_dims(volume, axis=3)
	return volume, label



# Define data loaders.
batch_size = 2

# model G
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

train_dataset = (
	train_loader.shuffle(len(x_train))
	.map(preprocessing)
	.batch(batch_size)
	.prefetch(2)
)

validation_dataset = (
	validation_loader.shuffle(len(x_val))
	.map(preprocessing)
	.batch(batch_size)
	.prefetch(2)
)


# model D
train_loader_d = tf.data.Dataset.from_tensor_slices((x_train2, y_train2))
validation_loader_d = tf.data.Dataset.from_tensor_slices((x_val2, y_val2))

train_dataset_d = (
	train_loader_d.shuffle(len(x_train2))
	.map(preprocessing)
	.batch(batch_size)
	.prefetch(2)
)

validation_dataset_d = (
	validation_loader_d.shuffle(len(x_val2))
	.map(preprocessing)
	.batch(batch_size)
	.prefetch(2)
)


def residual_block(input):
	x = layers.Conv3D(64, kernel_size=3, strides=1, padding='same')(input)
	x = layers.BatchNormalization(momentum=0.8)(x)
	x = layers.PReLU(shared_axes=[1,2])(x)            
	x = layers.Conv3D(64, kernel_size=3, strides=1, padding='same')(x)
	x = layers.BatchNormalization(momentum=0.8)(x)
	x = layers.Add()([x, input])
	return x

def G_model(width=128, height=128, depth=128):

	inputs = keras.Input((width, height, depth, 1), name="input")

	x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
	x = layers.MaxPool3D(pool_size=2)(x)
	x = layers.BatchNormalization()(x)
	

	r = residual_block(x)
	residual_blocks = 2
	for _ in range(residual_blocks - 1):
		r = residual_block(r)

	# Post-residual block
	x = layers.Conv3D(filters=64, kernel_size=3, strides=1, padding='same')(r)
	x = layers.MaxPool3D(pool_size=2)(x)
	x = layers.BatchNormalization()(x)

	# # x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
	# # x = layers.MaxPool3D(pool_size=2)(x)
	# # x = layers.BatchNormalization()(x)


	# x = layers.GlobalAveragePooling3D()(r)
	# x = layers.Dense(units=512, activation="relu")(x)
	# x = layers.Dropout(0.3)(x)

	# x = layers.Conv3D(filters=32, kernel_size=3, activation="relu", name="conv_1")(inputs)
	# x = layers.MaxPool3D(pool_size=2)(x)
	# x = layers.BatchNormalization()(x)

	# x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", name="conv_2")(x)
	# x = layers.MaxPool3D(pool_size=2)(x)
	# x = layers.BatchNormalization()(x)

	# x = layers.Conv3D(filters=128, kernel_size=3, activation="relu", name="conv_3")(x)
	# x = layers.MaxPool3D(pool_size=2)(x)
	# x = layers.BatchNormalization()(x)

	# x = layers.Conv3D(filters=256, kernel_size=3, activation="relu", name="conv_4")(x)
	# x = layers.MaxPool3D(pool_size=2)(x)
	# x = layers.BatchNormalization()(x)

	# x = layers.GlobalAveragePooling3D()(x)
	# x = layers.Dense(units=512, activation="relu")(x)
	# # x = layers.Dropout(0.3)(x)

	outputs = layers.Dense(units=3, activation="sigmoid", name="output")(x)

	# Define the model.
	model = keras.Model(inputs, outputs, name="3dcnn_g")

	return model

g_model = G_model()


def D_model(width=128, height=128, depth=128):

	inputs = keras.Input((width, height, depth, 1), name="input")

	# x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
	# x = layers.MaxPool3D(pool_size=2)(x)
	# x = layers.BatchNormalization()(x)
	

	# r = residual_block(inputs)
	# residual_blocks = 2
	# for _ in range(residual_blocks - 1):
	# 	r = residual_block(r)

	# # # Post-residual block
	# # x = layers.Conv3D(filters=64, kernel_size=3, strides=1, padding='same')(x)
	# # x = layers.MaxPool3D(pool_size=2)(x)
	# # x = layers.BatchNormalization()(x)

	# # x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
	# # x = layers.MaxPool3D(pool_size=2)(x)
	# # x = layers.BatchNormalization()(x)


	# x = layers.GlobalAveragePooling3D()(r)
	# x = layers.Dense(units=512, activation="relu")(x)
	# x = layers.Dropout(0.3)(x)

	x = layers.Conv3D(filters=32, kernel_size=3, activation="relu", name="conv_1")(inputs)
	x = layers.MaxPool3D(pool_size=2)(x)
	x = layers.BatchNormalization()(x)

	x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", name="conv_2")(x)
	x = layers.MaxPool3D(pool_size=2)(x)
	x = layers.BatchNormalization()(x)

	x = layers.Conv3D(filters=128, kernel_size=3, activation="relu", name="conv_3")(x)
	x = layers.MaxPool3D(pool_size=2)(x)
	x = layers.BatchNormalization()(x)

	x = layers.Conv3D(filters=256, kernel_size=3, activation="relu", name="conv_4")(x)
	x = layers.MaxPool3D(pool_size=2)(x)
	x = layers.BatchNormalization()(x)

	x = layers.GlobalAveragePooling3D()(x)
	x = layers.Dense(units=512, activation="relu")(x)
	# x = layers.Dropout(0.3)(x)

	outputs = layers.Dense(units=3, activation="sigmoid", name="output")(x)

	# Define the model.
	model = keras.Model(inputs, outputs, name="3dcnn_d")

	return model

d_model = D_model()

dot_img_file1 = './g_model.png'
tf.keras.utils.plot_model(g_model, to_file=dot_img_file1, show_shapes=True)
dot_img_file2 = './d_model.png'
tf.keras.utils.plot_model(d_model, to_file=dot_img_file2, show_shapes=True)

# def train_step_2nd_g(x,y_true):
# 	# second order optimization for global image model

# 	Acc=tf.keras.metrics.CategoricalAccuracy()

# 	with tf.GradientTape() as gen_tape2:
# 		with tf.GradientTape() as gen_tape:
# 			g_output = g_model(x, training=True)
# 			# loss
# 			g_loss = tf.keras.losses.categorical_crossentropy(y_true, g_output)
# 		# first order gradients
# 		gradients_of_g = gen_tape.gradient(g_loss, g_model.trainable_variables)
# 	# second order gradients
# 	gradients_of_g2 = gen_tape2.gradient(gradients_of_g, g_model.trainable_variables)

# 	# normalize gradients
# 	gradients_of_g = [tf.math.l2_normalize(w) for w in gradients_of_g]
# 	gradients_of_g2 = [tf.math.l2_normalize(w) for w in gradients_of_g2]

# 	# Combine first-order and second-order gradients
# 	grads_g = [0.2 * w1 + 0.8 * w2 for (w1, w2) in zip(gradients_of_g2, gradients_of_g)]

# 	keras.optimizers.Adam(learning_rate=lr_schedule).apply_gradients(zip(grads_g, g_model.trainable_variables))
	
# 	Acc.update_state(y_true, g_output)

# 	g_acc = Acc.result()

# 	return g_loss, g_acc

# def train_step_2nd_d(x,y_true):
# 	# second order optimization for vessel model

# 	Acc=tf.keras.metrics.CategoricalAccuracy()

# 	with tf.GradientTape() as gen_tape2:
# 		with tf.GradientTape() as gen_tape:
# 			d_output = d_model(x, training=True)
# 			# loss
# 			d_loss = tf.keras.losses.categorical_crossentropy(y_true, d_output)
# 		# first order gradients
# 		gradients_of_d = gen_tape.gradient(d_loss, d_model.trainable_variables)
# 	# second order gradients
# 	gradients_of_d2 = gen_tape2.gradient(gradients_of_d, d_model.trainable_variables)

# 	# normalize gradients
# 	gradients_of_d = [tf.math.l2_normalize(w) for w in gradients_of_d]
# 	gradients_of_d2 = [tf.math.l2_normalize(w) for w in gradients_of_d2]

# 	# Combine first-order and second-order gradients
# 	grads_d = [0.2 * w1 + 0.8 * w2 for (w1, w2) in zip(gradients_of_d2, gradients_of_d)]

# 	keras.optimizers.Adam(learning_rate=lr_schedule).apply_gradients(zip(grads_d, d_model.trainable_variables))
	
# 	Acc.update_state(y_true, d_output)

# 	d_acc = Acc.result()

# 	return d_loss, d_acc



def train(dataset1, dataset2, epochs):
	for epoch in range(epochs):
		for image_batch, label_batch in dataset1:
			g_loss, g_acc = train_step_2nd_g(image_batch, label_batch)
		for image_batch, label_batch in dataset2:
			d_loss, d_acc = train_step_2nd_d(image_batch, label_batch)

		# Save the model every 15 epochs
		if (epoch + 1) % 15 == 0:
			checkpoint.save(file_prefix = checkpoint_prefix)
			print("g_loss", g_loss.numpy())
			print("g_acc", g_acc.numpy())
			print("d_loss", d_loss.numpy())
			print("d_acc", d_acc.numpy())
			# print("g_loss {:1.2f}".format(g_loss.numpy()))
			# print("d_loss {:1.2f}".format(d_loss.numpy()))
			# print("g_acc {:1.2f}".format(g_acc.numpy()))
			# print("d_acc {:1.2f}".format(d_acc.numpy()))



initial_learning_rate = 0.00001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)

lr_schedule_d = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)

epochs = 200
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(g_optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                                 d_optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                                 g_model=g_model,
                                 d_model=d_model)

# train(train_dataset, train_dataset_d, epochs)


	
g_model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=['accuracy'],
    run_eagerly=True
)


d_model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule_d),
    metrics=['accuracy'],
    run_eagerly=True
)



# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "G_3d_image_classification.h5", save_best_only=True
)
checkpoint_db = keras.callbacks.ModelCheckpoint(
    "D_3d_image_classification.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=30)


# tb_callback = tf.keras.callbacks.TensorBoard(log_dir='./log',
#                                              profile_batch=(10, 15))

history = g_model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb, early_stopping_cb],
)

history2 = d_model.fit(
    train_dataset_d,
    validation_data=validation_dataset_d,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_db, early_stopping_cb],
)

np.save('g_model_history.npy',history.history)
np.save('d_model_history.npy',history2.history)

g_model.save('saved_model/g_model')
d_model.save('saved_model/d_model')


print("Evaluate on test data")
results = g_model.evaluate(x_test, y_test, batch_size=2)
dict(zip(g_model.metrics_names, results))
results2 = d_model.evaluate(x_test2, y_test2, batch_size=2)
dict(zip(d_model.metrics_names, results2))

y_pred = g_model.predict(x_test, verbose=1)
y_pred_2 = d_model.predict(x_test2, verbose=1)
np.save('g_model_pred.npy',y_pred)
np.save('d_model_pred.npy',y_pred_2)

print("test loss, test acc:", results)
print("test loss d, test acc d:", results2)

print("average acc:", (results[1] + results2[1])/2.0)

# =======================================
# # Load best weights.
# # g_model.load_weights("G_3d_image_classification.h5")

# # remove the output layer
# model = keras.Model(inputs=g_model.inputs, outputs=g_model.layers[-2].output)

# # get extracted features
# features = model.predict(np.expand_dims(x_val[0], axis=0))[0]

# d_model.load_weights("D_3d_image_classification.h5")
# # remove the output layer
# model2 = keras.Model(inputs=d_model.inputs, outputs=d_model.layers[-2].output)

# # get extracted features
# features2 = model2.predict(np.expand_dims(x_val[0], axis=0))[0]

# # add new classifier layers
# fea = layers.Add()([features, features2])
# flat1 = layers.Flatten()(fea)
# class1 = layers.Dense(520, activation='relu')(flat1)
# output = layers.Dense(3, activation='softmax')(class1)
# score = tf.nn.softmax(output[0])
# print('score', score)

# class_names = ["CE", "LAA", "SV"]

# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )
