# Object_Classification_using_CNN
- Used VGG-16 and built an object classification CNN model with data augmentation for custom dataset
## Introduction
- The python file implements a CNN object classification algorthim using VGG-16 as pretrained model. The emphasis behind the program is the custom dataset and step by step process of data preparartion and labelling. A custom dataset is prepared by clicking pictures from the camera and by using google images. All the images are then manually segregated into differet class folders. The dataset folder follows the below shown directory path. Its a one super image folder that has subsets of folders for each class.     
  - Dataset    
          |_Class1  
          |_Class2  
          |_Class3  
          |_Class4  
- The python file follows step by step process of loading the images, resizing, labelling and train & test set preparation.

## Requirements
- The code is implemented on a GPU system, where it takes 30 mins to complete the training. The following libraries need to be installed for loading and running the code  
      - TensorFlow-GPU    
      - Keras  
      - OpenCV  
      - Scikit-learn  
      - Numpy  
      - Pandas  
      - Matplotlib  
      - tqdm  
  
## Running
- The python file includes classification for 4 classes, but the number of classes can be modified. The directory path needs to provided for each image class. Once the libraries are installed, the code can run on any GPU system. It can also run on CPU system, but can go into memory issues.
  
