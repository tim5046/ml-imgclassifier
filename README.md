# Command line image classification neural network utility

This repository includes a python command line utility to build neural networks based on a few common PyTorch models. Train the neural network on any set of images, and use your trained model to predict new images.

## Usage - Training
**Training your model**
See help at any time:  `$ python train.py -h`
```
usage: train.py [-h] [-a ARCH] [-alpha LEARNING_RATE] [-e EPOCHS] [-gpu] [-hidden HIDDEN_LAYERS] [-o NUM_OUTPUTS] [-s SAVE_DIRECTORY] [-test TEST_DATA_DIRECTORY] [-train TRAIN_DATA_DIRECTORY]

arguments:
-h, --help  show this help message and exit
-a ARCH, --arch Pretrained model architecture
-alpha, --learning_rate Learning rate
-e, --epochs Number of epochs
-gpu, --gpu Use GPU if it's available
-hidden, --hidden_layers Number of model hidden layers
-o, --num_outputs Number of model outputs
-s, --save_directory Directory to save model
-test, --test_data_directory Testing data directory
-train, --train_data_directory Training data directory
```

#### Basic usage:
  python train.py -train path/to/training/images -test path/to/test/images -o number_of_outputs

**Note:** The testing and training folder structure should be formatted as a collection of image folders split apart by category name.  For example, if you were training a model with images of _dogs_ and _cats_, your folder structure might look like this:

    Training:
    assets/train/dogs/dog1.jpg
    assets/train/dogs/dog2.jpg
    assets/train/cats/cat1.jpg
    assets/train/cats/dog2.jpg

    Testing:
    assets/test/dogs/dog1.jpg
    assets/test/dogs/dog2.jpg
    assets/test/cats/cat1.jpg
    assets/test/cats/dog2.jpg

In this folder structure, your model would train to predict "dogs" or "cats" since that's what your folders are named. In this example, you should pass `-o 2` to your function, since the number of possible outputs based on your image dataset is 2 (cats or dogs). `

#### Advanced usage:

 - `-e num_epochs` - Define the number of training epochs. If none is provided, the model will default to 10.
 - `-a learning_rate` - Define the learning rate. If none is provided, the model will default to 0.001.
 - `-hidden layer1,layer2,layern` - Define of inputs into each hidden layer. If none is provided, the model will default to 2 hidden layers. The size of those hidden layers will be computed for you.
 - `-gpu` - Attempt to use a GPU for training, if your machine supports it.
 - `-a model_name` - Define the architecture to use for your model. Currently  `vgg11`, `vgg13`, `vgg16`, and `vgg19` are supported. If you don't pass in a model, you can select one later or use the default.

## Usage - Predicting
**Using your model to predict a new image**
See help at any time:  `$ python predict.py -h`

```
usage: predict.py [-h] [-i IMAGE] [-c CHECKPOINT] [-tk TOP_K] [--category_names CATEGORY_NAMES] [-gpu]

arguments:
-h, --help show this help message and exit
-i, --image Filepath to image to predict
-c, --checkpoint Saved model checkpoint to use for prediction.
-tk, --top_k Return the top K most likely predictions
--category_names Dictionary to map category indices to names
-gpu, --gpu Use GPU if it's available
 ```

 #### Basic usage:
  python predict.py -i path/to/image -c path/to/model_checkpoint
```
#######
Prediction: 52 with probability 64.07%

#######
Prediction: 62 with probability 33.06%

#######
Prediction: 47 with probability 2.32%

#######
Prediction: 2 with probability 0.31%

#######
Prediction: 38 with probability 0.23%
```

 #### Basic usage (with category names):
 `python predict.py -i sun.jpeg -c model_checkpoints/vgg16_75.pth --category_names cat_to_name.json`
```
#######
Prediction: primula with probability 64.07%

#######
Prediction: black-eyed susan with probability 33.06%

#######
Prediction: buttercup with probability 2.32%

#######
Prediction: canterbury bells with probability 0.31%

#######
Prediction: siam tulip with probability 0.23%
```