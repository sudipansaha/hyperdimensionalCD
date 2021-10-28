# -*- coding: utf-8 -*-
"""
Code Author: Sudipan Saha.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F







# CNN model 4 channel
class ModelDeepImagePriorHyperspectralNonLinear2Layer(nn.Module):
    def __init__(self,numberOfImageChannels,nFeaturesIntermediateLayers):
        super(ModelDeepImagePriorHyperspectralNonLinear2Layer, self).__init__()
        
       
        
        kernelSize=3
        paddingSize=int((kernelSize-1)/2)
        self.conv1 = nn.Conv2d(numberOfImageChannels, nFeaturesIntermediateLayers, kernel_size=kernelSize, stride=1, padding=paddingSize )
        self.conv1.weight=torch.nn.init.kaiming_uniform_(self.conv1.weight)
        
        self.conv2 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.conv2.weight=torch.nn.init.kaiming_uniform_(self.conv2.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.conv2(x)
       
        
        return x#
    
    
    
class ModelDeepImagePriorHyperspectralNonLinear3Layer(nn.Module):
    def __init__(self,numberOfImageChannels,nFeaturesIntermediateLayers):
        super(ModelDeepImagePriorHyperspectralNonLinear3Layer, self).__init__()
        
        kernelSize=3
        paddingSize=int((kernelSize-1)/2)
        self.conv1 = nn.Conv2d(numberOfImageChannels, nFeaturesIntermediateLayers, kernel_size=kernelSize, stride=1, padding=paddingSize )
        self.conv1.weight=torch.nn.init.kaiming_uniform_(self.conv1.weight)
        
        self.conv2 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.conv2.weight=torch.nn.init.kaiming_uniform_(self.conv2.weight)
        
        self.conv3 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.conv3.weight=torch.nn.init.kaiming_uniform_(self.conv3.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.conv2(x)
        x = F.relu( x )
        x = self.conv3(x)
        
        return x
    
    
class ModelDeepImagePriorHyperspectralNonLinear4Layer(nn.Module):
    def __init__(self,numberOfImageChannels,nFeaturesIntermediateLayers):
        super(ModelDeepImagePriorHyperspectralNonLinear4Layer, self).__init__()
        
        kernelSize=3
        paddingSize=int((kernelSize-1)/2)
        self.conv1 = nn.Conv2d(numberOfImageChannels, nFeaturesIntermediateLayers, kernel_size=kernelSize, stride=1, padding=paddingSize )
        self.conv1.weight=torch.nn.init.kaiming_uniform_(self.conv1.weight)
        
        self.conv2 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.conv2.weight=torch.nn.init.kaiming_uniform_(self.conv2.weight)
        
        self.conv3 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.conv3.weight=torch.nn.init.kaiming_uniform_(self.conv3.weight)
        
        self.conv4 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.conv4.weight=torch.nn.init.kaiming_uniform_(self.conv4.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.conv2(x)
        x = F.relu( x )
        x = self.conv3(x)
        x = F.relu( x )
        x = self.conv4(x)
        
        return x
    
class ModelDeepImagePriorHyperspectralNonLinear5Layer(nn.Module):
    def __init__(self,numberOfImageChannels,nFeaturesIntermediateLayers):
        super(ModelDeepImagePriorHyperspectralNonLinear5Layer, self).__init__()
        
        kernelSize=3
        paddingSize=int((kernelSize-1)/2)
        self.conv1 = nn.Conv2d(numberOfImageChannels, nFeaturesIntermediateLayers, kernel_size=kernelSize, stride=1, padding=paddingSize )
        self.conv1.weight=torch.nn.init.kaiming_uniform_(self.conv1.weight)
        
        self.conv2 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.conv2.weight=torch.nn.init.kaiming_uniform_(self.conv2.weight)
        
        self.conv3 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.conv3.weight=torch.nn.init.kaiming_uniform_(self.conv3.weight)
        
        self.conv4 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.conv4.weight=torch.nn.init.kaiming_uniform_(self.conv4.weight)
        
        self.conv5 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.conv5.weight=torch.nn.init.kaiming_uniform_(self.conv5.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.conv2(x)
        x = F.relu( x )
        x = self.conv3(x)
        x = F.relu( x )
        x = self.conv4(x)
        x = F.relu( x )
        x = self.conv5(x)
        
        return x
        
class ModelDeepImagePriorHyperspectralNonLinear6Layer(nn.Module):
    def __init__(self,numberOfImageChannels,nFeaturesIntermediateLayers):
        super(ModelDeepImagePriorHyperspectralNonLinear6Layer, self).__init__()
        
        kernelSize=3
        paddingSize=int((kernelSize-1)/2)
        self.conv1 = nn.Conv2d(numberOfImageChannels, nFeaturesIntermediateLayers, kernel_size=kernelSize, stride=1, padding=paddingSize )
        self.conv1.weight=torch.nn.init.kaiming_uniform_(self.conv1.weight)
        
        self.conv2 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.conv2.weight=torch.nn.init.kaiming_uniform_(self.conv2.weight)
        
        self.conv3 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.conv3.weight=torch.nn.init.kaiming_uniform_(self.conv3.weight)
        
        self.conv4 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.conv4.weight=torch.nn.init.kaiming_uniform_(self.conv4.weight)
        
        self.conv5 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.conv5.weight=torch.nn.init.kaiming_uniform_(self.conv5.weight)
        
        self.conv6 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.conv6.weight=torch.nn.init.kaiming_uniform_(self.conv6.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.conv2(x)
        x = F.relu( x )
        x = self.conv3(x)
        x = F.relu( x )
        x = self.conv4(x)
        x = F.relu( x )
        x = self.conv5(x)
        x = F.relu( x )
        x = self.conv6(x)
        
        return x
    
    
class ModelDeepImagePriorHyperspectralNonLinear7Layer(nn.Module):
    def __init__(self,numberOfImageChannels,nFeaturesIntermediateLayers):
        super(ModelDeepImagePriorHyperspectralNonLinear7Layer, self).__init__()
        
        kernelSize=3
        paddingSize=int((kernelSize-1)/2)
        self.conv1 = nn.Conv2d(numberOfImageChannels, nFeaturesIntermediateLayers, kernel_size=kernelSize, stride=1, padding=paddingSize )
        self.conv1.weight=torch.nn.init.kaiming_uniform_(self.conv1.weight)
        
        self.conv2 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.conv2.weight=torch.nn.init.kaiming_uniform_(self.conv2.weight)
        
        self.conv3 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.conv3.weight=torch.nn.init.kaiming_uniform_(self.conv3.weight)
        
        self.conv4 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.conv4.weight=torch.nn.init.kaiming_uniform_(self.conv4.weight)
        
        self.conv5 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.conv5.weight=torch.nn.init.kaiming_uniform_(self.conv5.weight)
        
        self.conv6 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.conv6.weight=torch.nn.init.kaiming_uniform_(self.conv6.weight)
        
        self.conv7 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.conv7.weight=torch.nn.init.kaiming_uniform_(self.conv7.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.conv2(x)
        x = F.relu( x )
        x = self.conv3(x)
        x = F.relu( x )
        x = self.conv4(x)
        x = F.relu( x )
        x = self.conv5(x)
        x = F.relu( x )
        x = self.conv6(x)
        x = F.relu( x )
        x = self.conv7(x)
        
        return x
    
class ModelDeepImagePriorHyperspectralNonLinear8Layer(nn.Module):
    def __init__(self,numberOfImageChannels,nFeaturesIntermediateLayers):
        super(ModelDeepImagePriorHyperspectralNonLinear8Layer, self).__init__()
        
        kernelSize=3
        paddingSize=int((kernelSize-1)/2)
        self.conv1 = nn.Conv2d(numberOfImageChannels, nFeaturesIntermediateLayers, kernel_size=kernelSize, stride=1, padding=paddingSize )
        self.conv1.weight=torch.nn.init.kaiming_uniform_(self.conv1.weight)
        
        self.conv2 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.conv2.weight=torch.nn.init.kaiming_uniform_(self.conv2.weight)
        
        self.conv3 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.conv3.weight=torch.nn.init.kaiming_uniform_(self.conv3.weight)
        
        self.conv4 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.conv4.weight=torch.nn.init.kaiming_uniform_(self.conv4.weight)
        
        self.conv5 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.conv5.weight=torch.nn.init.kaiming_uniform_(self.conv5.weight)
        
        self.conv6 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.conv6.weight=torch.nn.init.kaiming_uniform_(self.conv6.weight)
        
        self.conv7 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.conv7.weight=torch.nn.init.kaiming_uniform_(self.conv7.weight)
        
        self.conv8 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.conv8.weight=torch.nn.init.kaiming_uniform_(self.conv8.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.conv2(x)
        x = F.relu( x )
        x = self.conv3(x)
        x = F.relu( x )
        x = self.conv4(x)
        x = F.relu( x )
        x = self.conv5(x)
        x = F.relu( x )
        x = self.conv6(x)
        x = F.relu( x )
        x = self.conv7(x)
        x = F.relu( x )
        x = self.conv8(x)
        
        return x



