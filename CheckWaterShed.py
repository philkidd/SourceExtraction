# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 19:12:09 2016

@author: Daniel
"""

from AuxilaryFunctions import GetFileName      
from Demo import GetDefaultParams
import numpy as np
from pylab import load
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_pdf import PdfPages
from AuxilaryFunctions import SplitComponents


params,params_dict=GetDefaultParams()
last_rep=params.repeats

SaveNames=[]
for rep in range(last_rep):
    name=GetFileName(params_dict,rep)
    SaveNames.append(name)

resultsName=SaveNames[rep]
results=load('NMF_Results/'+SaveNames[rep])
shapes=results['shapes']
activity=results['activity']

NumBKG=params_dict['Background_num']




new_shapes,new_activity,L,all_local_max=SplitComponents(shapes,activity,NumBKG)
#%% Plot stuff and check similarity between splitted components    
distance_threshold=4 #if peaks are closer than this..
corr_threshold=0.8 # and correlation is greater than this, then merge components
Z=np.shape(shapes)[-1]

#for kk in range(Z):
#    plt.subplot(1,Z,kk)
#    plt.imshow(all_markers[:,:,kk],interpolation='None',cmap='gray')
L=len(new_shapes)-NumBKG
distance_mat=np.zeros((L,L))
for ii in range(L):
    for jj in range(len(all_local_max)):
        distance_mat[ii,jj]=np.sqrt(np.sum((all_local_max[ii]-all_local_max[jj])**2))
        

activity_cov=np.dot(new_activity[:L],new_activity[:L].T)
activity_vars=np.diag(activity_cov).reshape(-1,1)
activity_corr=activity_cov/np.sqrt(np.dot(activity_vars,activity_vars.T))

merge=(activity_corr>corr_threshold)*(distance_mat<distance_threshold)

cmap='gray'        
plt.subplot(1,3,1)
im=plt.imshow(distance_mat, interpolation='none',cmap=cmap)
plt.colorbar(im)
plt.subplot(1,3,2)
im2=plt.imshow(activity_corr, interpolation='none',cmap=cmap)
plt.colorbar(im2)
plt.subplot(1,3,3)
im2=plt.imshow(merge, interpolation='none',cmap=cmap)
plt.colorbar(im2)



for kk in range(Z):
    plt.subplot(1,Z,kk)
    plt.imshow(new_shapes[-5,:,:,kk],interpolation='None',cmap='gray')