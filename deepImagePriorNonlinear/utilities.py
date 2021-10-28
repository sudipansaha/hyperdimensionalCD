# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 09:43:13 2021

@author: Sudipan
"""


import numpy as np

from skimage import filters


def scaleContrast(inputFeatureMap):
    topVal=np.percentile(inputFeatureMap,99)
    inputFeatureMapContrastEnhanced=np.copy(inputFeatureMap)
    inputFeatureMapScaled=2.2*inputFeatureMap
    inputFeatureMapScaled[inputFeatureMapScaled>1]=1
    inputFeatureMapContrastEnhanced[np.where(inputFeatureMapContrastEnhanced>topVal)]=inputFeatureMapScaled[np.where(inputFeatureMapContrastEnhanced>topVal)]
    return inputFeatureMapContrastEnhanced



def saturateSomePercentile(inputMap,percentileToSaturate):
    inputMapNormalized=(inputMap-np.amin(inputMap))/(np.percentile(inputMap,(100-percentileToSaturate))-np.amin(inputMap))
    inputMapNormalized[inputMapNormalized>1]=1
    return inputMapNormalized


def saturateSomePercentileBandwise(inputMapMultichannel,percentileToSaturate):
    for bandIter in range(inputMapMultichannel.shape[2]):
        inputMap = inputMapMultichannel[:,:,bandIter]
        inputMapNormalized=saturateSomePercentile(inputMap,percentileToSaturate)
        inputMapMultichannel[:,:,bandIter] = inputMapNormalized
    return inputMapMultichannel
       






def findReliableTrainingSamples(preChangeImage, postChangeImage):
    cvaAbsDifference = np.absolute(postChangeImage-preChangeImage)    
    detectedChangeMap=np.linalg.norm(cvaAbsDifference,axis=(2))
    detectedChangeMapNormalized=(detectedChangeMap-np.amin(detectedChangeMap))/(np.amax(detectedChangeMap)-np.amin(detectedChangeMap))
    
    cdMapWithReliableUnchangedSamples=np.zeros(detectedChangeMapNormalized.shape, dtype=bool)
    otsuThreshold=filters.threshold_otsu(detectedChangeMapNormalized)
    cdMapWithReliableUnchangedSamples = detectedChangeMapNormalized>otsuThreshold*0.3
    reliableUnchangedSamples = np.argwhere(cdMapWithReliableUnchangedSamples==False) ##seeking unchanged samples, hence False
    
    
    cdMapWithReliableChangedSamples=np.zeros(detectedChangeMapNormalized.shape, dtype=bool)
    otsuThreshold=filters.threshold_otsu(detectedChangeMapNormalized)
    cdMapWithReliableChangedSamples = detectedChangeMapNormalized>otsuThreshold*2
    reliableChangedSamples = np.argwhere(cdMapWithReliableChangedSamples==True)  ##seeking changed samples, hence True
    
        
    return reliableUnchangedSamples, reliableChangedSamples


def matchResultToOriginalLabel(resultMap, referenceMap):
    
        
    ##Adding 1 to both resultMap and referenceMap, so that number starts from 1
    resultMap = resultMap+1
    referenceMap = referenceMap+1
    
    ##Finding unique values
    resultMapUniqueVals = np.unique(resultMap)
    
    referenceMapUniqueVals,referenceMapUniqueCounts = np.unique(referenceMap, return_counts=True)
    referenceSortingIndices = np.argsort(-referenceMapUniqueCounts)
    referenceMapUniqueVals = referenceMapUniqueVals[referenceSortingIndices]
    
    
    resultToReferenceRelationMatrix = np.zeros((len(resultMapUniqueVals),len(referenceMapUniqueVals)))
    
    totalIntersection = 0
    for resultIndex, resultUniqueVal in enumerate(resultMapUniqueVals):
        resultUniqueValIndicator = np.copy(resultMap)
        resultUniqueValIndicator[resultUniqueValIndicator!=resultUniqueVal] = 0
        for referenceIndex,referenceUniqueVal in enumerate(referenceMapUniqueVals):
            referenceUniqueValIndicator = np.copy(referenceMap)
            referenceUniqueValIndicator[referenceUniqueValIndicator!=referenceUniqueVal] = 0
            resultReferenceIntersection = resultUniqueValIndicator*referenceUniqueValIndicator
            numIntersection = len(np.argwhere(resultReferenceIntersection))
            totalIntersection = totalIntersection+numIntersection
            resultToReferenceRelationMatrix[resultIndex,referenceIndex] = numIntersection
            
    
   
    resultMapReassigned = np.zeros(resultMap.shape)
    
    for referenceIndex,referenceUniqueVal in enumerate(referenceMapUniqueVals):
        matchesCorrespondingToThisVal = resultToReferenceRelationMatrix[:,referenceIndex]
        if np.sum(matchesCorrespondingToThisVal)==0: ##this check is important, other python finds a max even in a all-zero column
            continue
        maximizingIndex = np.argsort(matchesCorrespondingToThisVal)[-1]
        resultMapOptimumMatch = resultMapUniqueVals[maximizingIndex]
        resultMapReassigned[resultMap==resultMapOptimumMatch] = referenceUniqueVal
        resultToReferenceRelationMatrix[maximizingIndex,:] = 0
        
    ##Subtracting 1 to keep values as it were
    resultMapReassigned = resultMapReassigned-1
    
       
    return resultMapReassigned.astype(int)
        