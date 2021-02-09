# Getting Started
This repo contains code for converting dataset to tfrecords, training models and evaluating models.

## Preqrequistes

To following libraries are required:
* Python 3
* Tensorflow
* Numpy
* Keras
* OpenCV

## Getting Dataset

Image Dataset is not provided as images were downloaded using imagenet API contained trojan. 

A list of IMAGENET class ids have been given, it is advisable to download the entire IMAGENET dataset and use these specific ids.

Other datasets can be used as long as the follow the same directory structure as the data directory shown below:
```
.
├── bird
│   └── bird.txt
├── boat
│   └── boat.txt
├── bottle
│   └── bottle.txt
├── cat
│   └── cat.txt
└── dog
    └── dog.txt
```

Where bird, boat, bottle, cat and dog are directories with the names of class labels and the corresponding images should be put in these directories instead of the text files. 

The text files contain imagenet class ids.


## Augmenting Images

This project uses different Augmentation algorithms to augment images.

Links for the corresponding respositories are provided from where augmentation algorithms were used from. 

1. ["Generalisation in humans and deep neural networks"][1]
	The following 
	- Greyscale
	- Constrast
	- Lowpass
	- Phase Noise
	- Salt & Pepper
	- Uniform Noise
2. ["Blind Geometric Distortion Correction on Images Through Deep Learning"][2]
	- Rotation
	- Barrel
	- Wave
	- Shear
	- Projective
	- Pincushion
3. ["Dual Adversarial Network: Toward Real-world Noise Removal and Noise Generation"][3]



## Converting to TFRecords

To convert images to TFRecord files, you can use the convert_tfrecords.py in the util folder.

```
python3 convert_tfrecords.py -image_directory 'Path to image directory' -output_filepath 'Output filepath of tfrecord'
```


## Training Neural Networks

Currently the following feature extractors are supported:
- 'mobilenetv2'
- 'nasnet_mobile'
- 'resnet50_v2'

To train a neural network, run train.py in the following file:
```
python3 train.py --train_fp 'Filepath to train tfrecord file' --valid_fp 'Filepath to validation tfrecord file' --cpkt_dir 'Path to store checkpoint file' -feature_extractor_name 'Use on the following keywords: mobilenetv2, nasnet_mobile, resnet50_v2' -batch_size 'Batch Size for training' -num_classes 'Number of classes of the dataset'
```


## Evaluating Neural Network

To evaluate the network, run evaluate.py
```
python3 evaluate.py -test_dir 'Path to image directory' -model_path 'Path to trained keras model'
```


[1]: https://github.com/rgeirhos/generalisation-humans-DNNs
[2]: https://github.com/xiaoyu258/GeoProj
[3]: https://github.com/zsyOAOA/DANet