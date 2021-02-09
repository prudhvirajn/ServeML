import numpy as np
import os
import tensorflow as tf
from functools import partial
import keras
from keras import backend as K

import argparse

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
IMAGE_SIZE = [256, 256]
NUM_CLASSES = 5

feature_extractor_dict = {'mobilenetv2': tf.keras.applications.MobileNetV2, 'nasnet_mobile': tf.keras.applications.NASNetMobile, 'resnet50_v2': tf.keras.applications.ResNet50V2}

def to_float32(image):
	return tf.cast(image, tf.float32)

def decode_image(image_data, image_size):
	image = tf.image.decode_jpeg(image_data, channels=3)
	image = tf.reshape(image, [*image_size, 3]) # explicit size needed for TPU
	image = tf.reverse(image, axis=[-1])
	image = tf.image.resize(image, IMAGE_SIZE)  
	return image

def read_labeled_tfrecord(example):
	LABELED_TFREC_FORMAT = {
		"image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
		"height": tf.io.FixedLenFeature([], tf.int64),
		"width": tf.io.FixedLenFeature([], tf.int64),
		"depth": tf.io.FixedLenFeature([], tf.int64),
		"image_path": tf.io.FixedLenFeature([], tf.string),
		"label": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
	}
	example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
	image = decode_image(example['image'], [example['height'], example['width']])
	label = tf.one_hot(tf.cast(example["label"], tf.int32), NUM_CLASSES)
	return image, label # returns a dataset of (image, label) pairs


def load_dataset(filenames, labeled=True):
	ignore_order = tf.data.Options()
	ignore_order.experimental_deterministic = False  # disable order, increase speed
	dataset = tf.data.TFRecordDataset(
		filenames
	)  # automatically interleaves reads from multiple files
	dataset = dataset.with_options(
		ignore_order
	)  # uses data as soon as it streams in, rather than in its original order
	dataset = dataset.map(
		read_labeled_tfrecord, num_parallel_calls=AUTOTUNE
	)
	# returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False
	return dataset

def get_dataset(filenames, labeled=True):
	dataset = load_dataset(filenames, labeled=labeled)
	dataset = dataset.shuffle(5000)
	dataset = dataset.prefetch(buffer_size=AUTOTUNE)
	dataset = dataset.batch(BATCH_SIZE)
	return dataset


def main(train_tfrecord_filepath, validation_tfrecord_file_path, checkpoint_file_dir, feature_extractor_name='mobilenetv2'):
	
	if not os.path.isdir(checkpoint_file_dir):
		os.makedirs(checkpoint_file_dir)

	train_dataset = get_dataset(train_tfrecord_filepath)
	vlad_dataset = get_dataset(validation_tfrecord_file_path)

	feature_extractor = feature_extractor_dict[feature_extractor_name]

	epochs=120

	with tf.device("/GPU:0"):
		inputs = tf.keras.layers.Input([None, None, 3])
		x = tf.keras.layers.experimental.preprocessing.Rescaling(1/255.)(inputs)
		x = tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")(x)
		x = tf.keras.layers.experimental.preprocessing.RandomRotation(0.5)(x)
		x = tf.keras.layers.experimental.preprocessing.RandomContrast(0.5)(x)
		x = tf.keras.layers.experimental.preprocessing.RandomZoom((0, 0.5), width_factor=(0, 0.5))(x)
		x = tf.keras.layers.experimental.preprocessing.RandomCrop(IMAGE_SIZE[0], IMAGE_SIZE[1])(x)
		outputs = feature_extractor(input_shape=[*IMAGE_SIZE, 3], weights=None, classes=NUM_CLASSES)(x)
		model = tf.keras.Model(inputs=inputs, outputs=outputs)

		opt = tf.keras.optimizers.Adam()

		filepath = os.path.join(checkpoint_file_dir, feature_extractor_name+"-{epoch:02d}-{val_accuracy:.2f}.h5")
		save_model = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=5)
		model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

		hist = model.fit(
			train_dataset,
			validation_data=vlad_dataset,
			epochs=epochs,
			callbacks=[save_model])
		np.save(f"{feature_extractor_name}_hist.npy", hist.history)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Parameters to train neural network')
	parser.add_argument('-train_tfrecord_filepath', '--train_fp', type=str, required=True, help='Filepath to Train Dataset TFRecord filepath')
	parser.add_argument('-valid_tfrecord_filepath', '--valid_fp', type=str, required=True, help='Filepath to Validation Dataset TFRecord filepath')
	parser.add_argument('-checkpoint_file_dir', '--cpkt_dir', type=str, required=True, help='Path to directory to store checkpoint files')
	parser.add_argument('-feature_extractor_name', type=str, required=True, help='Name of feature_extractor to use')
	parser.add_argument('-batch_size', type=int, required=False, default=32, help='Batch size to train')
	parser.add_argument('-num_classes', type=int, required=False, default=5, help='Number of classes')

	args = parser.parse_args()

	BATCH_SIZE = args.batch_size
	NUM_CLASSES = args.num_classes

	main(args.train_fp, args.valid_fp, args.cpkt_dir, args.feature_extractor_name)