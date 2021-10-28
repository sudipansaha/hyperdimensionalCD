# Hyperdimensional CD using untrained models

## Change detection in hyperspectral images
## Change detection in any other hyperdimensional images

The proposed method uses an untrained lightweight deep model, initialized with some weight initialization strategy for feature extraction from bi-temporal hyperdimensional images.

DCVA processes pre-change and post-change images through a pre-trained network and extracts bi-temporal deep features for subsequent processing in CD framework. For details, please read the paper: <br/>
Saha, S., Bovolo, F. and Bruzzone, L., 2019. Unsupervised deep change vector analysis for multiple-change detection in VHR images. IEEE Transactions on Geoscience and Remote Sensing, 57(6), pp.3677-3693.
Please cite the paper if you find the code useful.

The main part of the algorithm is in dcva.py. Other files provide supporting functionalities. <br/>
Input arguments are defined in options.py. Some of them are: <br/>
**dataPath**: a .mat file in which pre-change and post-change images are saved as 2 variables (preChangeImage and postChangeImage) <br/>
**inputChannels**: either 4 channel RGBNIR or 3 channel RGBIR, in that order <br/>
**layersToProcess**: the layers from which deep features are to extracted. Choose values from 2,5,8,10,11,23. Recommended value for 
a quasi-urban area is 2,5,8. If your analyzed scene is spatially less complex (e.g., agricultural land with less spatial variation),
more shallower layers can be used, e.g., 2,5. <br/>
**thresholding**: "adaptive" (for complex quasi-urban areas) or "otsu" (for spatially less complex areas) or "scaledOtsu" (otsu scaled by a factor to better address imbalance between number of changed and unchanged pixels). A comparison of result with Adaptive and Otsu methods can be found in the abovementioned paper. <br/>
**multipleCDBool** (True/False)whether multiple CD is needed. If false only binary CD is performed. By default, it is set as False.
**clusterNumber** number of clusters (if multiple CD is performed)

**To run the code** (if input images are square), use command, python dcva.py --dataPath <dataPath> (other arguments are optional)<br/>
Before running, download the trained model as instructed in "trainedNet" directory. <br/>
The output (a .png file and a .mat file) is stored in 'result' directory. There will be 2 more files in 'result' if multiple CD is performed.
  
 **However, if your input images are not square** (row size is not equal to column size), use command,
 python dcvaUnequalRowColumn.py --dataPath <dataPath> (other details are as discussed in previous case). Note that
  this case has not been extensively tested.<br/>
  
 **Remember to download** the pre-trained models as instructed in the trainedNet folder. There are two different pre-trained model depending on whether your input is RGB or RGB-NIR
 
 **About input format** the code expects a .mat file in which pre-change and post-change images are saved as 2 variables (preChangeImage and postChangeImage). However, if it is not so in your case, you can easily modify few sentences where .mat file is read to accommodate your input format.
  
Please note the method is not an exact replication of the abovementioned paper. The original code was implemented in Matlab and is not maintained/distributed anymore.

### Citation
If you find this code useful, please consider citing:
```[bibtex]
@article{saha2021change,
  title={Change Detection in Hyperdimensional Images using Untrained Modelss},
  author={Saha, Sudipan and Kondmann, Lukas and Song, Qian and Zhu, Xiao Xiang},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}
  year={2021},
  publisher={IEEE}
}
```
