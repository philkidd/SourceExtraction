# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 13:52:06 2015

@author: Daniel
"""
import numpy as np
from matplotlib import pyplot as plt
    
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
    
#load mask
    
def GetDataFolder():
    import os
    
    if os.getcwd()[0]=='C':
        DataFolder='G:/BackupFolder/'
    else:
        #DataFolder='Data/'
        DataFolder='D:/Documents/fijiproc/SourceExtraction-master/SourceExtraction-master'
    
    return DataFolder
    
    
def GetData(data_name):
    
    # Input - data_name - string of name of data to load
    # Output - data - Tx(XxYxZ) or Tx(XxY) numpy array 

    import h5py
    from pylab import load  
    from scipy.io import loadmat
    import tifffile as tff

    
    DataFolder=GetDataFolder()
    
    # Fetch experimental 3D data     
    if data_name=='HillmanSmall':
        data=load('data_small')
    elif data_name=='PhilConfocal':
        img= tff.TiffFile(DataFolder + 'Phil24FEB2016/confocal_stack.tif')
        data=img.asarray()
        data=np.transpose(data, [0,2,3,1]) 
        data=data-np.percentile(data, 0.1, axis=0)# takes care of negative values (ands strong positive values) in each pixel
    elif data_name=='PhilMFM':
        img= tff.TiffFile(DataFolder + 'Phil24FEB2016/Du_che2.tif')
        data=img.asarray()
#        data=np.asarray(data,dtype='float')  
        data=np.transpose(data, [0,2,3,1])  
        data=data-np.percentile(data, 0.1, axis=0)# takes care of negative values (ands strong positive values) in each pixel                  
    elif data_name=='Xin_MFM':
        img = tff.TiffFile('Xin_OTO_che2.tif')
        data=img.asarray()
        data = data[:,:,15:-20,15:-20]
        #data[:,3:6,:,:] = 150
        data=np.transpose(data,[0,2,3,1])
        #data=data-np.percentile(data, 0.1, axis=0)
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
                
            TargetRange = [0.1, 0.2]    
            lam = 500
            ds= 50 #downscale time for group lasso        
            NonNegative=True
    
            downscaled_data=data[:int(len(data) / ds) * ds].reshape((-1, ds) + data.shape[1:]).max(1)
            x = gaussian_group_lasso(downscaled_data, sig0, lam,NonNegative=NonNegative, TargetAreaRatio=TargetRange, verbose=True, adaptBias=False)
            pic_x = percentile(x, 95, axis=0)

                ######
            # centers extracted from fista output using RegionalMax
            cent = GetCenters(pic_x)
            print np.shape(cent)[0]
                
            # Plot Results
#            pic_data = np.percentile(data, 95, axis=0)
#            plt.figure(figsize=(12, 4. * data.shape[1] / data.shape[2]))
#            ax = plt.subplot(131)
#            ax.scatter(cent[1], cent[0],  marker='o', c='white')
#            plt.hold(True)
#            ax.set_title('Data + centers')
#            ax.imshow(pic_data.max(2))
#            ax2 = plt.subplot(132)
#            ax2.scatter(cent[1], cent[0], marker='o', c='white')
#            ax2.imshow(pic_x.max(2))
#            ax2.set_title('Inferred x')
#            ax3 = plt.subplot(133)
#            ax3.scatter(cent[1], cent[0],   marker='o', c='white')
#            ax3.imshow(pic_x.max(2))
#            ax3.set_title('Denoised data')
#            plt.show()
#        
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
    
#%% post processing
    
def PruneComponents(shapes,activity,TargetAreaRatio,L,deleted_indices=[]):
    
    if deleted_indices==[]:
        # If sparsity is too high or too low        
        cond1=0
        cond2=0
        for ll in range(L):
            cond1=np.mean(shapes[ll]>0)<TargetAreaRatio[0]
#            cond2=np.mean(shapes[ll]>0)>TargetAreaRatio[1]
            if cond1 or cond2:
                deleted_indices.append(ll) 
            
        # If highly correlated with motion artifact?
        
        # If shape overlaps with edge too much?
        
        # constraint on L2 of shape?
        
        # constraint on Euler number?

    for ll in deleted_indices[::-1]:
        activity=np.delete(activity,(ll),axis=0)
        shapes=np.delete(shapes,(ll),axis=0)
    
    L=L-len(deleted_indices)
    return shapes,activity,L

def SplitComponents(shapes,activity,NumBKG):
    # split components accoring to watershed around peaks
    # Inputs:
    # shapes - numpy array with all shape components - size (L,X,Y(,Z)) 
    # activity - numpy array with all activity components - size (L,T)
    # NumBKG - int, number of background components.
    # Outputs:
    # new_shapes- numpy array with new shape components - size (L,X,Y(,Z))
    # new_activity- numpy array with new activity components - size (L,T)
    # L - number of new components
    # all_local_max - location of (unique) smoothed local maxima of all components
    
    from scipy.ndimage.measurements import label
    from skimage.morphology import watershed
    from skimage.feature import peak_local_max
    from scipy.ndimage.filters import gaussian_filter

    sig=[1,1,1] # convolve with this gaussian, before finding peaks
    too_many_peaks=5 #how much is too many peaks? remove component with this many peaks
    split_background=False
    
    if split_background==False:
        L=len(shapes)-NumBKG
    else:
        L=len(shapes)
        
    new_shapes=np.zeros((0,)+np.shape(shapes)[1:])
    new_activity=np.zeros((0,)+np.shape(activity)[1:])
    all_local_max=np.zeros((0,3))
    all_markers=0
    
    for ll in range(L): 
        temp=np.copy(shapes[ll])
        temp=gaussian_filter(temp,sig)
        local_maxi = peak_local_max(temp, exclude_border=False, indices=False)
        local_maxi_loc = peak_local_max(temp, exclude_border=False, indices=True)
        markers,num_markers = label(local_maxi)
        
        all_markers=all_markers+local_maxi
    #    print ll,num_markers
        nonzero_mask=temp>0
        if np.sum(nonzero_mask)>9:
            labels = watershed(-temp, markers, mask=nonzero_mask)        #watershe regions
    #        for kk in range(Z):
    #            plt.subplot(L,Z,ll*Z+kk)
    #            plt.imshow(labels[:,:,kk])
            if num_markers<=too_many_peaks or ((ll>=L-NumBKG) and (NumBKG>0)): #throw away any component with too many peaks, except the background
                all_local_max=np.append(all_local_max,local_maxi_loc,axis=0)            
                for pp in range(num_markers):                
                    temp=np.copy(shapes[ll])
                    temp[labels!=(pp+1)]=0
                    new_shapes=np.append(new_shapes,np.reshape(temp,(1,)+np.shape(temp)),axis=0)
                    new_activity=np.append(new_activity,np.reshape(activity[ll],(1,)+np.shape(activity[ll])),axis=0)
    
    for pp in range(NumBKG):
        new_shapes=np.append(new_shapes,np.reshape(shapes[-pp-1],(1,)+np.shape(shapes[-pp-1])),axis=0)
        new_activity=np.append(new_activity,np.reshape(activity[-pp-1],(1,)+np.shape(activity[-pp-1])),axis=0)
    
    L=len(new_shapes)
    
    return new_shapes,new_activity,L,all_local_max
    
def ThresholdShapes(shapes,adaptBias,TargetAreaRatio,MaxRatio):
    #%% Threshold shapes
#    TargetAreaRatio - list with 2 components, 
#                      target area for the sparsity of largest connected component
#    MaxRatio - float in [0,1], 
#                if TargetAreaRatio =[], then we threshold according to value of MaxRatio*max(shapes[ll])
#    adaptBias - should we skip last component

    from scipy.ndimage.measurements import label
    rho=2 #exponential search parameter
    L=len(shapes)-adaptBias

    if TargetAreaRatio!=[]:
        for ll in range(L): 
            threshold=0.1
            threshold_high=-1
            threshold_low=-1
            while True:    
                temp=np.copy(shapes[ll])
                temp[temp<threshold]=0
                temp[temp>=threshold]=1
                # connected components target
        #        CC,num_CC=label(temp)
        #        sz=0
        #        ind_best=0
        #        for nn in range(num_CC):
        #            current_sz=np.count_nonzero(CC[CC==nn])
        #            if current_sz>sz:
        #                ind_best=nn
        #                sz=current_sz
        #        print threshold,sz/sz_all
        #        if ((sz/sz_all < TargetAreaRatio[0]) and (sz!=0)) or (np.sum(temp)==0):
        #            threshold_high = threshold
        #        elif (sz/sz_all > TargetAreaRatio[1]) or (sz==0):
        #            threshold_low = threshold
        #        else:
        #            temp[CC!=ind_best]=0
        #            shapes[ll]=np.copy(temp)
        #            break
                # sparsity target
                if (np.mean(temp) < TargetAreaRatio[0]):
                    threshold_high = threshold
                elif (np.mean(temp) > TargetAreaRatio[1]):
                    threshold_low = threshold
                else:
                    print np.mean(temp)
                    temp=np.copy(shapes[ll])
                    temp[temp<threshold]=0
                    shapes[ll]=np.copy(temp)
                    break
        
                if threshold_high == -1:
                    threshold = threshold * rho
                elif threshold_low == -1:
                    threshold = threshold / rho
                else:
                    threshold = (threshold_high + threshold_low) / 2
        
        for ll in range(L): 
            temp=np.copy(shapes[ll])
            CC,num_CC=label(temp)
            sz=0
            for nn in range(num_CC):
                current_sz=np.count_nonzero(CC[CC==nn])
                if current_sz>sz:
                    ind_best=nn
                    sz=current_sz
            temp[CC!=ind_best]=0
            shapes[ll]=np.copy(temp)
    else:
        for ll in range(L): 
            temp=np.copy(shapes[ll])
            threshold=MaxRatio*np.max(temp)
            temp[temp<threshold]=0
            shapes[ll]=temp
    return shapes

def shapesPlot(shapes,inds,fig,ax):
    
    from skimage.measure import label,regionprops
    from skimage import feature
    from skimage.morphology import binary_dilation
    from skimage.segmentation import find_boundaries
    import pylab as plt
    import numpy as np
    
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    sz = np.int32(shapes.shape)
    
    
    for i in inds:
        img = shapes[i,:,:]
        mx = img[:].max()
        test = img>0.4*mx
        test2 = binary_dilation(binary_dilation(test))
        lbls = label(test2)
        rgs = regionprops(lbls)
        if np.size(rgs)>0:
            szs = []
            for prop in rgs:
                szs.append(prop.area)
            ind = np.argmax(szs)
            if rgs[ind].area>100:
                pt = rgs[ind].centroid
                region = lbls==ind+1
                edges = find_boundaries(region)
                eln = edges.nonzero()
                ax.scatter(eln[1],eln[0],marker='.',color='r',linewidths=0.01)
                ax.text(pt[1]-4,pt[0]+4,'%i' % i,fontsize=14,color='k')
    
    return fig,ax
        
def stackPlot(data):
    
    sz = np.int32(data.shape)
    top = np.int32(np.sqrt(sz[2]))
    side = np.int32(np.ceil(np.double(sz[2])/top))
    fig = plt.figure()
    axdict = {}
    
    for i in range(1,sz[2]+1):
        axdict[i] = fig.add_subplot(top,side,i)
        axdict[i].imshow(data[:,:,i-1])
    
    return fig,axdict

def fullShapes(shapes,inds,fig,axdict):
    
    
    for i in inds:
        mx = shapes[i,:,:,:].max()
        shapes[i,:,:,:] = shapes[i,:,:,:]>0.4*mx
        
    for i in axdict.keys():
        fig, axdict[i] = shapesPlot(shapes[:,:,:,i-1],inds,fig,axdict[i])
    
    return fig, axdict

class Centers:
    
    def __init__(self, coords):
        self.coords = coords
        self.xs = []
        self.ys = []
        self.axes = []
        self.current = 1
        
    def connect(self):
        self.cid = self.coords.canvas.mpl_connect('button_press_event',self.onclick)
        self.cid2 = self.coords.canvas.mpl_connect('key_press_event',self.onkey)
        
    def onclick(self,event):
        
        if event.xdata==None: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.axes.append(self.current)
        print '###click###',event.xdata,event.ydata,self.current
    
    def onkey(self, event):
        
    
        if event.key=='n':
            self.current = self.current+1
        print '#### now on axis #',self.current
    
    def disconnect(self):
        
        self.coords.canvas.mpl_disconnect(self.cid)
        self.coords.canvas.mpl_disconnect(self.cid2)

def findCenters(data):
    
    fig,axdict = stackPlot(data.max(axis=0))
    cds = Centers(fig)
    cds.connect()
    
    
    return cds    


    
    
        
        
    
    


        
    
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
    