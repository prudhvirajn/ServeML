import numpy as np
import os
import tensorflow as tf
from collections import deque
import os
import cv2


def to_float32(image, label):
	return tf.cast(image, tf.float32), label

def decode_image(image_data, image_size):
	image = tf.image.decode_jpeg(image_data, channels=3)
	#image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
	image = tf.reshape(image, [*image_size, 3]) # explicit size needed for TPU
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
	label = tf.cast(example['label'], tf.int32)
	return image, label # returns a dataset of (image, label) pairs

def _bytes_feature(value):
	"""Returns a bytes_list from a string / byte."""
	if isinstance(value, type(tf.constant(0))):
		value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
	"""Returns a float_list from a float / double."""
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
	"""Returns an int64_list from a bool / enum / int / uint."""
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(image, image_path, label, height, width, depth):
	feature = {
		'image': _bytes_feature(image),
		'height': _int64_feature(height),
		'width': _int64_feature(width),
		'depth': _int64_feature(depth),
		'image_path': _bytes_feature(image_path),
		'label': _int64_feature(label)
	}
	example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
	return example_proto

def main(test_root_dir, tfrecord_path='test_imgnet_300_shuffled.tfrecord'):

	root_dir = test_root_dir
	class_dirs = os.listdir(root_dir)

	path_labels = deque()

	for label in range(len(class_dirs)):
			cl_dir = class_dirs[label]
			src_dir = os.path.join(root_dir, cl_dir)
			paths = os.listdir(src_dir)

			for path in paths:
				path_labels.append((os.path.join(src_dir, path), label))

	np.random.shuffle(path_labels)


	with tf.io.TFRecordWriter(tfrecord_path) as writer:
		for path, label in path_labels:
			img_path = path

			img = cv2.imread(img_path)

			if img is None:
				print(f"{img_path} could not be loaded successfully")
				continue 

			img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

			height, width, depth = img.shape
			img_string = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 100))[1].tostring()
			tf_example = serialize_example(img_string, str.encode(path), label, height, width, depth)
			writer.write(tf_example.SerializeToString())

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Convert dataset to TFRecords')
	parser.add_argument('-image_directory', '--img_dir', type=str, required=True, help='Filepath to Test Dataset root directory')
	parser.add_argument('-output_filepath', type=str, required=True, help='Output filepath for TFRecord')
	args = parser.parse_args()

	main(args.image_directory, args.output_filepath)