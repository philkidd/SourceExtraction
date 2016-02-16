# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 13:52:06 2015

@author: Daniel
"""
import numpy as np
    
def GetFileName(params_dict,rep):

    params_Name='mb'+str(params_dict['mbs'][0])+'ds'+str(params_dict['ds'])+'iters'
    params_Name+=str(params_dict['iters0'][0])+'_'+str(params_dict['iters'])+'intervals'+str(params_dict['updateLambdaIntervals'])
    if params_dict['TargetAreaRatio']!=[]: params_Name+='_Area'+str(params_dict['TargetAreaRatio'][0])+'_'+str(params_dict['TargetAreaRatio'][1])
    if params_dict['Background_num']>0: params_Name+='_Bkg'+str(params_dict['Background_num'])
    if params_dict['estimateNoise']: params_Name+='_Noise'
    if params_dict['PositiveError']: params_Name+='_PosErr'
    if params_dict['Connected']: params_Name+='_Connected'
    if params_dict['FixSupport']: params_Name+='_FixSupport'
    if params_dict['SuperVoxelize']: params_Name+='_SuperVoxelized'
    if params_dict['FinalNonNegative']: params_Name+='_FinalNonNegative'
    
    resultsName='NMF_results_'+ params_dict['data_name'] + '_Rep' + str(rep+1) +'of'+ str(params_dict['repeats']) +'_'+params_Name
    return resultsName
    
def GetRandColors(n):
    colors=[]
    for i in range(n):
        color=np.random.uniform(low=0,high=1,size=(1,3))
        colors.append(color/np.sum(color))
    
    return np.array(colors)
    
def max_intensity(x,axis):
    intensity=np.sum(x,axis=-1)
    ind=np.argmax(intensity,axis=axis)
    tup=np.indices(ind.shape)
    tup_ind=()
    for ii in range(len(tup)+1):
        if ii<axis:
            tup_ind+=(tup[ii],)
        elif ii>axis:
            tup_ind+=(tup[ii-1],)
        else:
            tup_ind+=(ind,)
    return x[tup_ind]
    
def PruneComponents(shapes,activity,params,L):
    
    deleted_indices=[]  # list of indices to remove  
    # If sparsity is too high or too low
    TargetAreaRatio=params.TargetAreaRatio
    for ll in range(L):
        cond1=np.mean(shapes[ll]>0)<TargetAreaRatio[0]
        cond2=np.mean(shapes[ll]>0)>TargetAreaRatio[1]
        if cond1 or cond2:
            deleted_indices.append(ll) 
        
    # If highly correlated with motion artifact
    
    # If shape overlaps with edge too much
    
    # constraint on L2 of shape?
    
    # constraint on Euler number?

    for ll in deleted_indices[::-1]:
        activity=np.delete(activity,(ll),axis=0)
        shapes=np.delete(shapes,(ll),axis=0)
    
    L=L-len(deleted_indices)
    return shapes,activity,L
    
#load mask
    
def GetDataFolder():
    import os
    
    if os.getcwd()[0]=='C':
        DataFolder='G:/BackupFolder/'
    else:
        DataFolder='Data/'
    
    return DataFolder

def SuperVoxelize(data):

    import scipy.io
    data=np.transpose(data,axes=[1,2,3,0])
    shape_data=data.shape
    data=np.reshape(data,(np.prod(shape_data[0:3]),shape_data[3]))
    #load mask
    DataFolder=GetDataFolder()
    
    temp=scipy.io.loadmat(DataFolder + 'oversegmentationForDaniel_daniel0_4e-4.mat')
    mask=temp["L"]            
    mask=np.transpose(mask,axes=[2,1,0])
    mask=np.reshape(mask,(1,np.size(mask)))
    ma=np.max(mask)
    mi=np.min(mask)
    for ii in range(mi,ma+1):
        ind=(mask==ii)
        trace=np.dot(ind,data)
        ind=np.ravel(ind)
        data[ind]=trace
        print ii
    
    data=np.reshape(data,shape_data)
    data=np.transpose(data,axes=[3,0,1,2])
    return data
    

    
    
def GetData(data_name):
    
    # Input - data_name - string of name of data to load
    # Output - data - Tx(XxYxZ) or Tx(XxY) numpy array 

    import h5py
    from pylab import load  
    from scipy.io import loadmat

    
    DataFolder=GetDataFolder()
    
    # Fetch experimental 3D data     
    if data_name=='HillmanSmall':
        data=load(DataFolder + 'Hillman/data_small')
    elif data_name=='Hillman':
        temp = h5py.File(DataFolder + 'Hillman/150724_mouseRH2d1_data_crop_zig_sm_ds.mat')
        data=temp["moviesub_sm"]
        data=np.asarray(data,dtype='float')
        data = data[10:-70,2:-2,2:-2,2:-2]   # bad values appear near the edges, and everything is moving at the few first and last frames
    elif data_name=='Sophie2D':
        temp=loadmat(DataFolder + 'Sophie2D_drosophila_lightfield/processed_data.mat')
        data=np.transpose(np.asarray(temp['data'],dtype='float'), [2, 0, 1])
        data=data-np.min(data) # takes care of negative values due to detrending
        temp=None
    elif data_name=='Sophie3D':# check data dimenstions
        import tifffile as tff
        img = tff.TiffFile(DataFolder + 'Sophie3D_drosophila_lightfield/Sophie3Ddata.tif')
        data=img.asarray()     
    elif data_name=='SophieVoltage3D':# check data dimenstions
        import tifffile as tff
        img = tff.TiffFile(DataFolder + 'Sophie3D_drosophila_lightfield/SophieVoltageData.tif')
        data=img.asarray()      
    elif data_name=='Sara19DEC2015_w1t1':
        temp = loadmat(DataFolder + 'Sara19DEC2015/processed_data.mat')
        data=temp["data"]
        data=np.asarray(data,dtype='float')  
        data=np.transpose(np.asarray(temp['data'],dtype='float'), [3,0,1, 2])              
    else:
        print 'unknown dataset name!'
    return data
    
def GetCentersData(data,data_name,NumCent):
    
    from numpy import  array,percentile    
    from BlockGroupLasso import gaussian_group_lasso, GetCenters
    from pylab import load
    import matplotlib.pyplot as plt
    import os
    import cPickle
        
    DataFolder=GetDataFolder()
    
    center_file_name=DataFolder + '/centers_'+data_name 
    if NumCent>0:
        if os.path.isfile(center_file_name)==False:
            if data.ndim==3:
                sig0=(2,2)
            else:
                sig0=(2,2,2)
                
            TargetRange = [0.1, 0.22]    
            lam = 35
            ds= 10 #downscale factor for group lasso        
            NonNegative=True
    
            downscaled_data=data[:int(len(data) / ds) * ds].reshape((-1, ds) + data.shape[1:]).max(1)
            x = gaussian_group_lasso(downscaled_data, sig0, lam/ds,NonNegative=NonNegative, TargetAreaRatio=TargetRange, verbose=True, adaptBias=True)
            pic_x = percentile(x, 95, axis=0)
            pic_data = percentile(data, 95, axis=0)
                ######
            # centers extracted from fista output using RegionalMax
            cent = GetCenters(pic_x)
                
            # Plot Results
            plt.figure(figsize=(12, 4. * data.shape[1] / data.shape[2]))
            ax = plt.subplot(131)
            ax.scatter(cent[1], cent[0],  marker='o', c='white')
            plt.hold(True)
            ax.set_title('Data + centers')
            ax.imshow(pic_data.max(1))
            ax2 = plt.subplot(132)
            ax2.scatter(cent[1], cent[0], marker='o', c='white')
            ax2.imshow(pic_x.max(1))
            ax2.set_title('Inferred x')
            ax3 = plt.subplot(133)
            ax3.scatter(cent[1], cent[0],   marker='o', c='white')
            ax3.imshow(pic_x.max(1))
            ax3.set_title('Denoised data')
            plt.show()
        
            f = file(center_file_name, 'wb')
        
            cPickle.dump(cent, f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()
        else:
            cent=load(center_file_name)
                
        new_cent=(array(cent)[:-1]).T                
        new_cent=new_cent[:NumCent] #just give strongest centers
    else:
        new_cent=np.reshape([],(0,data.ndim-1))
        
    return new_cent
    
    
    
## Depricated
    
# plot random projection of data
#    C=3 #RGB colors
#    dims=np.shape(data)
#    W=np.random.randn(C,dims[0])  
#    proj_data=np.dot(W,np.transpose(data,[1,2,0,3]))
#    proj_data=np.transpose(proj_data,[1,2,3,0])
#    pp = PdfPages('RandProj.pdf')
#    fig=plt.figure(figsize=(18 , 11))
#    for dd in range(dims[2]):
#        plt.subplot(8,8,dd+1)
#        plt.imshow(proj_data[:,dd],interpolation='none')
#    
#    pp.savefig(fig)
#    pp.close()
#    
    