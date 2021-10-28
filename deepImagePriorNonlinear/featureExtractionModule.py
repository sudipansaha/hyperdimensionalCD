# -*- coding: utf-8 -*-
"""
Code Author: Sudipan Saha.

"""


import torch
import numpy as np
from skimage.transform import resize




from networks import ModelDeepImagePriorHyperspectralNonLinear2Layer, ModelDeepImagePriorHyperspectralNonLinear3Layer,\
    ModelDeepImagePriorHyperspectralNonLinear4Layer, ModelDeepImagePriorHyperspectralNonLinear5Layer,\
         ModelDeepImagePriorHyperspectralNonLinear6Layer, ModelDeepImagePriorHyperspectralNonLinear7Layer, \
              ModelDeepImagePriorHyperspectralNonLinear8Layer

nanVar=float('nan')




def deepPriorCd(data1,data2,manualSeed,outputLayerNumbers):
    
    ##setting manual seeds
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    np.random.seed(manualSeed)
    
    ##Checking cuda availability
    useCuda = torch.cuda.is_available()
    
    ##Defining model
    numberOfImageChannels = data1.shape[2]
    nFeaturesIntermediateLayers = numberOfImageChannels*4
    modelInputMean = 0
    
   
    
   
    detectedChangeMapNormalized, timeVector1FeatureAggregated, timeVector2FeatureAggregated = detectChange(data1, data2, numberOfImageChannels,nFeaturesIntermediateLayers,modelInputMean,useCuda,outputLayerNumbers)
    return detectedChangeMapNormalized, timeVector1FeatureAggregated, timeVector2FeatureAggregated
    
    
    

   


def detectChange(preChangeImage, postChangeImage, numberOfImageChannels,nFeaturesIntermediateLayers,modelInputMean,useCuda,outputLayerNumbers):    
    
    
    preChangeImageOriginalShape = preChangeImage.shape
    if preChangeImageOriginalShape[0]<preChangeImageOriginalShape[1]: ##code is written in a way s.t. it expects row>col
        print('rotating')
        preChangeImage = np.swapaxes(preChangeImage,0,1)
        postChangeImage = np.swapaxes(postChangeImage,0,1)
    
    data1=np.copy(preChangeImage)  ##Order BGR to RGB
    data2=np.copy(postChangeImage)  ##VEG order remains unchanged
    
    nanVar=float('nan')
    sizeReductionTable=[nanVar,nanVar,1,1,1,1,1,1,1]
    featurePercentileToDiscardTable=[nanVar,nanVar,90,90,90,90,90,90,90] ##index starts from 0
    
    filterNumberTable=[nanVar,nFeaturesIntermediateLayers,nFeaturesIntermediateLayers,nFeaturesIntermediateLayers,\
                       nFeaturesIntermediateLayers,nFeaturesIntermediateLayers,
                       nFeaturesIntermediateLayers,nFeaturesIntermediateLayers,nFeaturesIntermediateLayers]
    if min(data1.shape[0],data1.shape[1])<(224+64):
        eachPatch = 128
    else:
        eachPatch = 256
    imageSize=data1.shape
    imageSizeRow=imageSize[0]
    imageSizeCol=imageSize[1]
    cutY=list(range(0,imageSizeRow,eachPatch))
    cutX=list(range(0,imageSizeCol,eachPatch))
    additionalPatchPixel=64
    
   
    modelLayer2= ModelDeepImagePriorHyperspectralNonLinear2Layer(numberOfImageChannels,nFeaturesIntermediateLayers)
    modelLayer3= ModelDeepImagePriorHyperspectralNonLinear3Layer(numberOfImageChannels,nFeaturesIntermediateLayers)
    modelLayer4= ModelDeepImagePriorHyperspectralNonLinear4Layer(numberOfImageChannels,nFeaturesIntermediateLayers)
    modelLayer5= ModelDeepImagePriorHyperspectralNonLinear5Layer(numberOfImageChannels,nFeaturesIntermediateLayers)
    modelLayer6= ModelDeepImagePriorHyperspectralNonLinear6Layer(numberOfImageChannels,nFeaturesIntermediateLayers)
    modelLayer7= ModelDeepImagePriorHyperspectralNonLinear7Layer(numberOfImageChannels,nFeaturesIntermediateLayers)
    modelLayer8= ModelDeepImagePriorHyperspectralNonLinear8Layer(numberOfImageChannels,nFeaturesIntermediateLayers)
    
   
    if useCuda:
        modelLayer2.cuda()
        modelLayer3.cuda()
        modelLayer4.cuda()
        modelLayer5.cuda()
        modelLayer6.cuda()
        modelLayer7.cuda()
        modelLayer8.cuda()
   
    modelLayer2.requires_grad=False
    modelLayer3.requires_grad=False
    modelLayer4.requires_grad=False
    modelLayer5.requires_grad=False
    modelLayer6.requires_grad=False
    modelLayer7.requires_grad=False
    modelLayer8.requires_grad=False
    
    modelLayer2.eval()
    modelLayer3.eval()
    modelLayer4.eval()
    modelLayer5.eval()
    modelLayer6.eval()
    modelLayer7.eval()
    modelLayer8.eval()
    
    
    
    
    
    layerWiseFeatureExtractorFunction=[nanVar,nanVar,modelLayer2,modelLayer3,modelLayer4,modelLayer5,\
                                       modelLayer6,modelLayer7,modelLayer8]
    
    
    
    #model=model.cpu()
    for outputLayerIter in range(0,len(outputLayerNumbers)):
        outputLayerNumber=outputLayerNumbers[outputLayerIter]
        filterNumberForOutputLayer=filterNumberTable[outputLayerNumber]
        featurePercentileToDiscard=featurePercentileToDiscardTable[outputLayerNumber]
        featureNumberToRetain=int(np.floor(filterNumberForOutputLayer*((100-featurePercentileToDiscard)/100)))
        sizeReductionForOutputLayer=sizeReductionTable[outputLayerNumber]
        patchOffsetFactor=int(additionalPatchPixel/sizeReductionForOutputLayer)
        print('Processing layer number:'+str(outputLayerNumber))
        
        timeVector1Feature=np.zeros([imageSizeRow,imageSizeCol,filterNumberForOutputLayer])
        timeVector2Feature=np.zeros([imageSizeRow,imageSizeCol,filterNumberForOutputLayer])
        for kY in range(0,len(cutY)):
            for kX in range(0,len(cutX)):
                          
                #extracting subset of image 1
                if (kY==0 and kX==0):
                    patchToProcessDate1=data1[cutY[kY]:(cutY[kY]+eachPatch+additionalPatchPixel),\
                                                   cutX[kX]:(cutX[kX]+eachPatch+additionalPatchPixel),:]
                elif (kY==0 and kX!=(len(cutX)-1)):
                    patchToProcessDate1=data1[cutY[kY]:(cutY[kY]+eachPatch+additionalPatchPixel),\
                                                   (cutX[kX]-additionalPatchPixel):(cutX[kX]+eachPatch),:]
                elif (kY!=(len(cutY)-1) and kX==(len(cutX)-1)):
                    patchToProcessDate1=data1[cutY[kY]:(cutY[kY]+eachPatch+additionalPatchPixel),\
                                                   (imageSizeCol-eachPatch-additionalPatchPixel):(imageSizeCol),:] 
                elif (kX==0 and kY!=(len(cutY)-1)):
                    patchToProcessDate1=data1[(cutY[kY]-additionalPatchPixel):\
                                              (cutY[kY]+eachPatch),\
                                                   cutX[kX]:(cutX[kX]+eachPatch+additionalPatchPixel),:]
                elif (kX!=(len(cutX)-1) and kY==(len(cutY)-1)):
                    patchToProcessDate1=data1[(imageSizeRow-eachPatch-additionalPatchPixel):\
                                              (imageSizeRow),\
                                                   cutX[kX]:(cutX[kX]+eachPatch+additionalPatchPixel),:]
                elif (kY==(len(cutY)-1) and kX==(len(cutX)-1)):
                    patchToProcessDate1=data1[(imageSizeRow-eachPatch-additionalPatchPixel):\
                                              (imageSizeRow),\
                                                   (imageSizeCol-eachPatch-additionalPatchPixel):(imageSizeCol),:]
                else:
                    patchToProcessDate1=data1[(cutY[kY]-additionalPatchPixel):\
                                              (cutY[kY]+eachPatch),\
                                              (cutX[kX]-additionalPatchPixel):(cutX[kX]+eachPatch),:]
                #extracting subset of image 2   
                if (kY==0 and kX==0):
                    patchToProcessDate2=data2[cutY[kY]:(cutY[kY]+eachPatch+additionalPatchPixel),\
                                                   cutX[kX]:(cutX[kX]+eachPatch+additionalPatchPixel),:]
                elif (kY==0 and kX!=(len(cutX)-1)):
                    patchToProcessDate2=data2[cutY[kY]:(cutY[kY]+eachPatch+additionalPatchPixel),\
                                                   (cutX[kX]-additionalPatchPixel):(cutX[kX]+eachPatch),:]
                elif (kY!=(len(cutY)-1) and kX==(len(cutX)-1)):
                    patchToProcessDate2=data2[cutY[kY]:(cutY[kY]+eachPatch+additionalPatchPixel),\
                                                   (imageSizeCol-eachPatch-additionalPatchPixel):(imageSizeCol),:] 
                elif (kX==0 and kY!=(len(cutY)-1)):
                    patchToProcessDate2=data2[(cutY[kY]-additionalPatchPixel):\
                                              (cutY[kY]+eachPatch),\
                                                  cutX[kX]:(cutX[kX]+eachPatch+additionalPatchPixel),:]
                elif (kX!=(len(cutX)-1) and kY==(len(cutY)-1)):
                    patchToProcessDate2=data2[(imageSizeRow-eachPatch-additionalPatchPixel):\
                                              (imageSizeRow),\
                                                   cutX[kX]:(cutX[kX]+eachPatch+additionalPatchPixel),:]
                elif (kY==(len(cutY)-1) and kX==(len(cutX)-1)):
                    patchToProcessDate2=data2[(imageSizeRow-eachPatch-additionalPatchPixel):\
                                              (imageSizeRow),\
                                                   (imageSizeCol-eachPatch-additionalPatchPixel):(imageSizeCol),:]    
                else:
                    patchToProcessDate2=data2[(cutY[kY]-additionalPatchPixel):\
                                              (cutY[kY]+eachPatch),\
                                              (cutX[kX]-additionalPatchPixel):(cutX[kX]+eachPatch),:]
               
                
                    
                #converting to pytorch varibales and changing dimension for input to net
                patchToProcessDate1=patchToProcessDate1-modelInputMean
                
                inputToNetDate1=torch.from_numpy(patchToProcessDate1)
                inputToNetDate1=inputToNetDate1.float()
                inputToNetDate1=np.swapaxes(inputToNetDate1,0,2)
                inputToNetDate1=np.swapaxes(inputToNetDate1,1,2)
                inputToNetDate1=inputToNetDate1.unsqueeze(0)
                
                
                patchToProcessDate2=patchToProcessDate2-modelInputMean
                
                inputToNetDate2=torch.from_numpy(patchToProcessDate2)
                inputToNetDate2=inputToNetDate2.float()
                inputToNetDate2=np.swapaxes(inputToNetDate2,0,2)
                inputToNetDate2=np.swapaxes(inputToNetDate2,1,2)
                inputToNetDate2=inputToNetDate2.unsqueeze(0)
                
                if useCuda:
                    inputToNetDate1 = inputToNetDate1.cuda()
                    inputToNetDate2 = inputToNetDate2.cuda()
                
                
                #running model on image 1 and converting features to numpy format
                with torch.no_grad():
                    obtainedFeatureVals1=layerWiseFeatureExtractorFunction[outputLayerNumber](inputToNetDate1)
                obtainedFeatureVals1=obtainedFeatureVals1.squeeze()
                obtainedFeatureVals1=obtainedFeatureVals1.data.cpu().numpy()
                
                #running model on image 2 and converting features to numpy format
                with torch.no_grad():
                    obtainedFeatureVals2=layerWiseFeatureExtractorFunction[outputLayerNumber](inputToNetDate2)
                obtainedFeatureVals2=obtainedFeatureVals2.squeeze()
                obtainedFeatureVals2=obtainedFeatureVals2.data.cpu().numpy()
                #this features are in format (filterNumber, sizeRow, sizeCol)
                
                
                ##clipping values to +1 to -1 range, be careful, if network is changed, maybe we need to modify this
                obtainedFeatureVals1=np.clip(obtainedFeatureVals1,-1,+1)
                obtainedFeatureVals2=np.clip(obtainedFeatureVals2,-1,+1)
                
                
                #obtaining features from image 1: resizing and truncating additionalPatchPixel
                if (kY==0 and kX==0):
                    for processingFeatureIter in range(0,filterNumberForOutputLayer):
                        timeVector1Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                       cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                       resize(obtainedFeatureVals1[processingFeatureIter,\
                                                                   0:int(eachPatch/sizeReductionForOutputLayer),\
                                                                   0:int(eachPatch/sizeReductionForOutputLayer)],\
                                                                   (eachPatch,eachPatch))
                    
                elif (kY==0 and kX!=(len(cutX)-1)):                                                    
                    for processingFeatureIter in range(0,filterNumberForOutputLayer):
                        timeVector1Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                       cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                       resize(obtainedFeatureVals1[processingFeatureIter,\
                                                                   0:int(eachPatch/sizeReductionForOutputLayer),\
                                                                   (patchOffsetFactor+1):\
                                                                   (int(eachPatch/sizeReductionForOutputLayer)+patchOffsetFactor+1)],\
                                                                   (eachPatch,eachPatch))
                elif (kY!=(len(cutY)-1) and kX==(len(cutX)-1)):
                    for processingFeatureIter in range(0,filterNumberForOutputLayer):                        
                        timeVector1Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                       cutX[kX]:imageSizeCol,processingFeatureIter]=\
                                       resize(obtainedFeatureVals1[processingFeatureIter,\
                                                                   0:int(eachPatch/sizeReductionForOutputLayer),\
                                                                   (obtainedFeatureVals1.shape[2]-1-int((imageSizeCol-cutX[kX])/sizeReductionForOutputLayer)):\
                                                                   (obtainedFeatureVals1.shape[2])],\
                                                                   (eachPatch,(imageSizeCol-cutX[kX])))
                elif (kX==0 and kY!=(len(cutY)-1)):
                    for processingFeatureIter in range(0,filterNumberForOutputLayer):
                        timeVector1Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                       cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                       resize(obtainedFeatureVals1[processingFeatureIter,\
                                                                   (patchOffsetFactor+1):\
                                                                   (int(eachPatch/sizeReductionForOutputLayer)+patchOffsetFactor+1),\
                                                                   0:int(eachPatch/sizeReductionForOutputLayer)],\
                                                                   (eachPatch,eachPatch))
                elif (kX!=(len(cutX)-1) and kY==(len(cutY)-1)):
                    for processingFeatureIter in range(0,filterNumberForOutputLayer):
                        timeVector1Feature[cutY[kY]:imageSizeRow,\
                                       cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                       resize(obtainedFeatureVals1[processingFeatureIter,\
                                                                   (obtainedFeatureVals1.shape[1]-1-int((imageSizeRow-cutY[kY])/sizeReductionForOutputLayer)):\
                                                                   (obtainedFeatureVals1.shape[1]),\
                                                                   0:int(eachPatch/sizeReductionForOutputLayer)],\
                                                                   ((imageSizeRow-cutY[kY]),eachPatch))
                elif (kX==(len(cutX)-1) and kY==(len(cutY)-1)):
                    for processingFeatureIter in range(0,filterNumberForOutputLayer):
                        timeVector1Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                       cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                       resize(obtainedFeatureVals1[processingFeatureIter,\
                                                                   (obtainedFeatureVals1.shape[1]-1-int((imageSizeRow-cutY[kY])/sizeReductionForOutputLayer)):\
                                                                   (obtainedFeatureVals1.shape[1]),\
                                                                   (obtainedFeatureVals1.shape[2]-1-int((imageSizeCol-cutX[kX])/sizeReductionForOutputLayer)):\
                                                                   (obtainedFeatureVals1.shape[2])],\
                                                                   ((imageSizeRow-cutY[kY]),(imageSizeCol-cutX[kX])))
                else:
                    for processingFeatureIter in range(0,filterNumberForOutputLayer):
                        timeVector1Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                       cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                       resize(obtainedFeatureVals1[processingFeatureIter,\
                                                                   (patchOffsetFactor+1):\
                                                                   (int(eachPatch/sizeReductionForOutputLayer)+patchOffsetFactor+1),\
                                                                   (patchOffsetFactor+1):\
                                                                   (int(eachPatch/sizeReductionForOutputLayer)+patchOffsetFactor+1)],\
                                                                   (eachPatch,eachPatch))
                #obtaining features from image 2: resizing and truncating additionalPatchPixel
                if (kY==0 and kX==0):
                    for processingFeatureIter in range(0,filterNumberForOutputLayer):
                        timeVector2Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                       cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                       resize(obtainedFeatureVals2[processingFeatureIter,\
                                                                   0:int(eachPatch/sizeReductionForOutputLayer),\
                                                                   0:int(eachPatch/sizeReductionForOutputLayer)],\
                                                                   (eachPatch,eachPatch))
                    
                elif (kY==0 and kX!=(len(cutX)-1)):
                    for processingFeatureIter in range(0,filterNumberForOutputLayer):
                        timeVector2Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                       cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                       resize(obtainedFeatureVals2[processingFeatureIter,\
                                                                   0:int(eachPatch/sizeReductionForOutputLayer),\
                                                                   (patchOffsetFactor+1):\
                                                                   (int(eachPatch/sizeReductionForOutputLayer)+patchOffsetFactor+1)],\
                                                                   (eachPatch,eachPatch))
                elif (kY!=(len(cutY)-1) and kX==(len(cutX)-1)):
                    for processingFeatureIter in range(0,filterNumberForOutputLayer):
                        timeVector2Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                       cutX[kX]:imageSizeCol,processingFeatureIter]=\
                                       resize(obtainedFeatureVals2[processingFeatureIter,\
                                                                   0:int(eachPatch/sizeReductionForOutputLayer),\
                                                                   (obtainedFeatureVals2.shape[2]-1-int((imageSizeCol-cutX[kX])/sizeReductionForOutputLayer)):\
                                                                   (obtainedFeatureVals2.shape[2])],\
                                                                   (eachPatch,(imageSizeCol-cutX[kX])))
                elif (kX==0 and kY!=(len(cutY)-1)):
                    for processingFeatureIter in range(0,filterNumberForOutputLayer):
                        timeVector2Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                       cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                       resize(obtainedFeatureVals2[processingFeatureIter,\
                                                                   (patchOffsetFactor+1):\
                                                                   (int(eachPatch/sizeReductionForOutputLayer)+patchOffsetFactor+1),\
                                                                   0:int(eachPatch/sizeReductionForOutputLayer)],\
                                                                   (eachPatch,eachPatch))
                elif (kX!=(len(cutX)-1) and kY==(len(cutY)-1)):
                    for processingFeatureIter in range(0,filterNumberForOutputLayer):
                        timeVector2Feature[cutY[kY]:imageSizeRow,\
                                       cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                       resize(obtainedFeatureVals2[processingFeatureIter,\
                                                                   (obtainedFeatureVals2.shape[1]-1-int((imageSizeRow-cutY[kY])/sizeReductionForOutputLayer)):\
                                                                   (obtainedFeatureVals2.shape[1]),\
                                                                   0:int(eachPatch/sizeReductionForOutputLayer)],\
                                                                   ((imageSizeRow-cutY[kY]),eachPatch))
                elif (kX==(len(cutX)-1) and kY==(len(cutY)-1)):
                    for processingFeatureIter in range(0,filterNumberForOutputLayer):
                        timeVector2Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                       cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                       resize(obtainedFeatureVals2[processingFeatureIter,\
                                                                   (obtainedFeatureVals2.shape[1]-1-int((imageSizeRow-cutY[kY])/sizeReductionForOutputLayer)):\
                                                                   (obtainedFeatureVals2.shape[1]),\
                                                                   (obtainedFeatureVals2.shape[2]-1-int((imageSizeCol-cutX[kX])/sizeReductionForOutputLayer)):\
                                                                   (obtainedFeatureVals2.shape[2])],\
                                                                   ((imageSizeRow-cutY[kY]),(imageSizeCol-cutX[kX])))
                else:
                    for processingFeatureIter in range(0,filterNumberForOutputLayer):
                        timeVector2Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                       cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                       resize(obtainedFeatureVals2[processingFeatureIter,\
                                                                   (patchOffsetFactor+1):\
                                                                   (int(eachPatch/sizeReductionForOutputLayer)+patchOffsetFactor+1),\
                                                                   (patchOffsetFactor+1):\
                                                                   (int(eachPatch/sizeReductionForOutputLayer)+patchOffsetFactor+1)],\
                                                                   (eachPatch,eachPatch))
                
                
                                                      
                                       
        timeVectorDifferenceMatrix=timeVector1Feature-timeVector2Feature
        
        nonZeroVector=[]
        stepSizeForStdCalculation=int(imageSizeRow/2)
        for featureSelectionIter1 in range(0,imageSizeRow,stepSizeForStdCalculation):
            for featureSelectionIter2 in range(0,imageSizeCol,stepSizeForStdCalculation):
                timeVectorDifferenceSelectedRegion=timeVectorDifferenceMatrix\
                                                   [featureSelectionIter1:(featureSelectionIter1+stepSizeForStdCalculation),\
                                                    featureSelectionIter2:(featureSelectionIter2+stepSizeForStdCalculation),
                                                    0:filterNumberForOutputLayer]
                stdVectorDifferenceSelectedRegion=np.std(timeVectorDifferenceSelectedRegion,axis=(0,1))
                featuresOrderedPerStd=np.argsort(-stdVectorDifferenceSelectedRegion)   #negated array to get argsort result in descending order
                nonZeroVectorSelectedRegion=featuresOrderedPerStd[0:featureNumberToRetain]
                nonZeroVector=np.union1d(nonZeroVector,nonZeroVectorSelectedRegion)
                
             
        modifiedTimeVector1=timeVector1Feature[:,:,nonZeroVector.astype(int)]
        modifiedTimeVector2=timeVector2Feature[:,:,nonZeroVector.astype(int)]
        
        
        ##Normalize the features (separate for both images)
        meanVectorsTime1Image=np.mean(modifiedTimeVector1,axis=(0,1))      
        stdVectorsTime1Image=np.std(modifiedTimeVector1,axis=(0,1))
        normalizedModifiedTimeVector1=(modifiedTimeVector1-meanVectorsTime1Image)/stdVectorsTime1Image
        
        meanVectorsTime2Image=np.mean(modifiedTimeVector2,axis=(0,1))      
        stdVectorsTime2Image=np.std(modifiedTimeVector2,axis=(0,1))
        normalizedModifiedTimeVector2=(modifiedTimeVector2-meanVectorsTime2Image)/stdVectorsTime2Image
        
        ##feature aggregation across channels
        if outputLayerIter==0:
            timeVector1FeatureAggregated=np.copy(normalizedModifiedTimeVector1)
            timeVector2FeatureAggregated=np.copy(normalizedModifiedTimeVector2)
        else:
            timeVector1FeatureAggregated=np.concatenate((timeVector1FeatureAggregated,normalizedModifiedTimeVector1),axis=2)
            timeVector2FeatureAggregated=np.concatenate((timeVector2FeatureAggregated,normalizedModifiedTimeVector2),axis=2)
        
     
        
        
    del obtainedFeatureVals1, obtainedFeatureVals2, timeVector1Feature, timeVector2Feature, inputToNetDate1, inputToNetDate2 
      
        
    absoluteModifiedTimeVectorDifference=np.absolute(timeVector1FeatureAggregated-timeVector2FeatureAggregated) 
    
        #take absolute value for binary CD
        
    detectedChangeMap=np.linalg.norm(absoluteModifiedTimeVectorDifference,axis=(2))
    detectedChangeMapNormalized=(detectedChangeMap-np.amin(detectedChangeMap))/(np.amax(detectedChangeMap)-np.amin(detectedChangeMap))
    
    if preChangeImageOriginalShape[0]<preChangeImageOriginalShape[1]: ##Conformity to row>col
            detectedChangeMapNormalized = np.swapaxes(detectedChangeMapNormalized,0,1)
            timeVector1FeatureAggregated = np.swapaxes(timeVector1FeatureAggregated,0,1)
            timeVector2FeatureAggregated = np.swapaxes(timeVector2FeatureAggregated,0,1)
            
    return detectedChangeMapNormalized, timeVector1FeatureAggregated, timeVector2FeatureAggregated
    
        