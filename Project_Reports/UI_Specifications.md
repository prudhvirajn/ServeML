# UI Specifications

We provide the codebase used to run our experiments. Hence, we support a simple command line argument parsing with our project. To identify the commands needed, we identify what information the user must provide. In this repository, we support training neural networks and evaluating neural networks. 

Hence, we come up with a interface for these two.

## Training

Inputs needed: 
- Filepath to training set TFRecord file
- Filepath to validation set TFRecord file
- Filepath to checkpoint directory to save model checkpoints
- Name of neural network to use
- Batch Size for training
- Number of classes in dataset

## Testing

Inputs needed:
- Path to image directory 
- Path to saved model file