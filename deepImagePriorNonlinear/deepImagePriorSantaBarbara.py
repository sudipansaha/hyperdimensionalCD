# -*- coding: utf-8 -*-
"""
Code Author: Sudipan Saha.

"""



import os
import sys
import glob
import torch

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from skimage.transform import resize
from skimage import filters
from skimage import morphology
import cv2


import random
import scipy.stats as sistats
import scipy
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import tifffile 

from utilities import  saturateSomePercentileBandwise, scaleContrast
from featureExtractionModule import deepPriorCd


import argparse



##Dataset details: https://citius.usc.es/investigacion/datasets/hyperspectral-change-detection-dataset

###The Santa Barbara scene, taken on the years 2013 and 2014 with the AVIRIS sensor over the Santa Barbara
## region (California) whose spatial dimensions are 984 x 740 pixels and includes 224 spectral bands.

### Santa Barbara: changed pixels: 52134   (label 1 in provided reference Map)
### Santa Barbara: unchanged pixels: 80418  (label 2 in provided reference Map)
### Santa Barbara: unknown pixels: 595608 (label 0 in reference Map)
### However we convert it in "referenceImageTransformed" and assign 0 to unchanged, 1 to changed and 2 to unknown pixels

### Imp link: https://aviris.jpl.nasa.gov/links/AVIRIS_for_Dummies.pdf

##Defining Parameters
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--manualSeed', type=int, default=40, help='manual seed')
opt = parser.parse_args()
manualSeed=opt.manualSeed
print('Manual seed is '+str(manualSeed))

outputLayerNumbers=[5]



nanVar=float('nan')

##setting manual seeds

torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
np.random.seed(manualSeed)



preChangeDataPath = '../datasets/santaBarbaraDatasets/santaBarbara/mat/barbara_2013.mat'
postChangeDataPath = '../datasets/santaBarbaraDatasets/santaBarbara/mat/barbara_2014.mat'
referencePath = '../datasets/santaBarbaraDatasets/santaBarbara/mat/barbara_gtChanges.mat'
resultPath = '../results/santaBarbara/santaBarbaraDeepImagePriorNonlinear.png'



##Reading images and reference
preChangeImageContents=sio.loadmat(preChangeDataPath)
preChangeImage = preChangeImageContents['HypeRvieW']

postChangeImageContents=sio.loadmat(postChangeDataPath)
postChangeImage = postChangeImageContents['HypeRvieW']

referenceContents=sio.loadmat(referencePath)
referenceImage = referenceContents['HypeRvieW']

##Transforming the reference image
referenceImageTransformed = np.zeros(referenceImage.shape)
### We assign 0 to unchanged, 1 to changed and 2 to unknown pixels
referenceImageTransformed[referenceImage==2] = 0 
referenceImageTransformed[referenceImage==1] = 1
referenceImageTransformed[referenceImage==0] = 2

del referenceImage 


###Pre-process/normalize the images
percentileToSaturate = 1
preChangeImage = saturateSomePercentileBandwise(preChangeImage,percentileToSaturate)
postChangeImage = saturateSomePercentileBandwise(postChangeImage,percentileToSaturate)

##Number of spectral bands
numSpectralBands = preChangeImage.shape[2]




## Getting normalized CD map (magnitude map)
detectedChangeMapNormalized, timeVector1FeatureAggregated, timeVector2FeatureAggregated = deepPriorCd(preChangeImage,postChangeImage, manualSeed, outputLayerNumbers)

## Saving features for visualization
# absoluteModifiedTimeVectorDifference=np.absolute(timeVector1FeatureAggregated-timeVector2FeatureAggregated) 
# print(absoluteModifiedTimeVectorDifference.shape)
# for featureIter in range(absoluteModifiedTimeVectorDifference.shape[2]):
#     detectedChangeMapThisFeature=absoluteModifiedTimeVectorDifference[:,:,featureIter]
#     detectedChangeMapNormalizedThisFeature=(detectedChangeMapThisFeature-np.amin(detectedChangeMapThisFeature))/(np.amax(detectedChangeMapThisFeature)-np.amin(detectedChangeMapThisFeature))
#     detectedChangeMapNormalizedThisFeature=scaleContrast(detectedChangeMapNormalizedThisFeature)
#     plt.imsave('./savedFeatures/santaBarbara'+'FeatureBest'+str(featureIter)+'.png',np.repeat(np.expand_dims(detectedChangeMapNormalizedThisFeature,2),3,2))



## Getting CD map from normalized CD maps
    

cdMap=np.zeros(detectedChangeMapNormalized.shape, dtype=bool)
otsuThreshold=filters.threshold_otsu(detectedChangeMapNormalized)
cdMap = detectedChangeMapNormalized>otsuThreshold
cdMap = morphology.binary_erosion(cdMap)
cdMap = morphology.binary_dilation(cdMap)
   



##Computing quantitative indices
referenceImageTo1DArray=(referenceImageTransformed).ravel()
cdMapTo1DArray=cdMap.astype(int).ravel()
confusionMatrixEstimated=confusion_matrix(y_true=referenceImageTo1DArray, y_pred=cdMapTo1DArray, labels=[0,1])

#getting details of confusion matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix
trueNegative,falsePositive,falseNegative,truePositive=confusionMatrixEstimated.ravel()
sensitivity=truePositive/(truePositive+falseNegative)
specificity=trueNegative/(trueNegative+falsePositive)
accuracy = (truePositive+trueNegative)/(truePositive+trueNegative+falsePositive+falseNegative)
print('Sensitivity is:' +str(sensitivity))
print('Specificity is:' +str(specificity))
print('Accuracy is:' +str(accuracy))
print('Missed alarm are:' +str(falseNegative))
print('False alarm are:' +str(falsePositive))


## ignoring label 2 while computing F1 score
referenceImageTo1DArrayInvalidIndices = np.argwhere(referenceImageTo1DArray==2)
referenceImageTo1DArrayValidIndices = np.setdiff1d(np.arange(len(referenceImageTo1DArray)),referenceImageTo1DArrayInvalidIndices)
f1Score = f1_score(y_true=referenceImageTo1DArray[referenceImageTo1DArrayValidIndices], y_pred=cdMapTo1DArray[referenceImageTo1DArrayValidIndices])
print('F1 score is:' +str(f1Score))
print('...')

#cv2.imwrite(resultPath,((1-cdMap)*255).astype('uint8'))












