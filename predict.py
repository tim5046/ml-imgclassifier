import argparse
import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import time

from utils import IOUtils, TypeUtils

def parseInput():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-i', '--image', action='store',
        help="Path to image to predict")
    parser.add_argument('-c', '--checkpoint', action='store',
        help="Model checkpoint to use for prediction")
    parser.add_argument('-tk', '--top_k', action='store',
        help="Return number of most likely predictions")
    parser.add_argument('--category_names', action='store',
        help="Dictionary to map category indices to names")
    parser.add_argument('-gpu', '--gpu', action='store_true',
        help="Use GPU if it's available")

    args = parser.parse_args()
    required_arguments = ['image', 'checkpoint']

    for arg in required_arguments:
        if not vars(args)[arg]:
            IOUtils.notify(f"You are missing a required argument: {arg}")
            exit()

    return {
        'image': args.image,
        'checkpoint': args.checkpoint,
        'top_k': args.top_k,
        'category_names': args.category_names,
        'shouldTryGPU': args.gpu
    }

class Predictor:
    def __init__(self, *args, **kwargs):
        self.image = kwargs.get('image')
        self.checkpoint = kwargs.get('checkpoint')
        self.top_k = kwargs.get('top_k')
        self.category_names = kwargs.get('category_names')
        if kwargs.get('shouldTryGPU'):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            IOUtils.notify("ALERT: You are not predicting using GPU. Predictions may take a long time. To use GPU, call the function with the -gpu flag.")
            self.device = torch.device("cpu")

    def predictImage(self, *args, **kwargs):
      self.preparePredictor()
      self.loadModel()
      self.predict()

    def loadModel(self, *args, **kwargs):
      checkpoint = torch.load(self.checkpoint)
      self.model = checkpoint['architecture']

      for param in model.parameters():
        param.requires_grad = False

      self.model.classifier = checkpoint['classifier']
      self.model.load_state_dict(checkpoint['state_dict'])
      self.model.to(self.device)
      print("MODEL LOADED!", model)
      return model

    def preparePredictor(self, *args, **kwargs):
      if not self.category_names:
          IOUtils.notify("category_names dictionary not provided. Predictions will show indexes unless you pass in a valid category_names dict at runtime.")
      if not self.top_k:
          IOUtils.notify("top_k not provided. Showing top 5 category predictions by default.")
          self.top_k = 5


    def process_image(image):
      # Preprocess images to turn them into valid inputs into our model
      img = Image.open(image)
      width,height = img.size

      ## If width > height, resize height to 256. Else resize width to 256
      if width > height:
          img.thumbnail(size=(width,256))
      else:
          img.thumbnail(size=(256,height))

      ## Get 224x224 crop coordinates
      crop = {
        'left': (img.width / 2 - 112),
        'top': (img.height / 2 - 112),
        'right': (img.width / 2 + 112),
        'bottom': (img.height / 2 + 112)
      }

      img = img.crop((crop['left'], crop['top'], crop['right'], crop['bottom']))
      np_img = np.array(img)/255 # Convert to floats between 0 and 1
      mean = [0.485, 0.456, 0.406]
      std = [0.229, 0.224, 0.225]
      np_img = (np_img - mean)/std

      # Transpose because PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array.
      return np_img.transpose(2,0,1) #set color to first dimension

    def predict(self, *args, **kwargs):
      self.model.to(self.device)
      self.model.eval()

      processed_img = self.process_image(self.image)
      torch_img = torch.from_numpy(np.expand_dims(processed_img, axis=0)).type(torch.FloatTensor).to(self.device)

      with torch.no_grad():
          output = self.model.forward(torch_img)

      prob = torch.exp(output)
      probs, indexes = prob.topk(self.top_k)

      probs_list = np.array(probs)[0]
      predictions_list = np.array(indexes)[0]

      if self.category_names:
        #If they provided a category_names dict, use it to translate indexes into names now.
        classes_list = []
        category_indexes_dict = TypeUtils.cat_to_name(self.category_names)
        for idx in indexes_list:
          classes_list.append(category_indexes_dict[str(idx + 1)])
        predictions_list = classes_list

      print("PROBS LIST", probs_list)
      print("PREDICTIONS LIST", predictions_list)
      for i in range(len(probs_list)):
        IOUtils.notify(f"Prediction: {predictions_list[i]} with probability {probs_list[i]}.")

      return probs_list, predictions_list

predictorInputs = parseInput()
predictor = Predictor(**predictorInputs)

predictor.predictImage()