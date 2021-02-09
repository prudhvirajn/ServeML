import numpy as np
import os
import tensorflow as tf
from collections import deque
import os
import cv2

import argparse

IMAGE_SIZE = [256, 256]

def main(test_dir, model_path):

	root_dir = test_dir
	class_dirs = os.listdir(root_dir)

	path_labels = deque()

	correct_score = 0
	total_score = 0

	model = tf.keras.models.load_model(model_path)

	for label in range(len(class_dirs)):
			cl_dir = class_dirs[label]
			src_dir = os.path.join(root_dir, cl_dir)
			paths = os.listdir(src_dir)

			for path in paths:
				path_labels.append((os.path.join(src_dir, path), label))

	for path, label in path_labels:
		img_path = path

		img = cv2.imread(img_path)

		if img is None:
			continue

		img = tf.image.resize(img, IMAGE_SIZE).numpy()
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

		y_prob = model.layers[-1].predict(np.expand_dims(img / 255, axis=0))
		y_classes = y_prob.argmax(axis=-1)

		if y_classes[0] == label:
			correct_score += 1
		total_score += 1

	print(correct_score)
	print(total_score)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Parameters to train neural network')
	parser.add_argument('-test_dir', '--train_fp', type=str, required=True, help='Filepath to Test Dataset root directory')
	parser.add_argument('-model_path', type=str, required=True, help='Filepath to trained model')
	parser.add_argument('-image_size', default=256, type=int, required=False, 'Input Image Size')
	args = parser.parse_args()

	IMAGE_SIZE = args.image_size

	main(args.test_dir, args.model_path)
