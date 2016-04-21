from numpy import min, max, asarray, percentile, zeros, ones, reshape, r_, ix_, arange, exp, nan_to_num, prod, mean, sqrt, repeat
from scipy.signal import welch
from scipy.ndimage.filters import median_filter
import numpy as np
from scipy.ndimage.measurements import label
from skimage.measure import regionprops
from skimage.morphology import watershed, binary_dilation
from skimage.feature import peak_local_max
from scipy.ndimage.filters import gaussian_filter
from scipy.sparse.csgraph import connected_components

def GetBox(centers, R, dims):
    D = len(R)
    box = zeros((D, 2), dtype=int)
    for dd in range(D):
        box[dd, 0] = max((centers[dd] - R[dd], 0))
        box[dd, 1] = min((centers[dd] + R[dd] + 1, dims[dd]))
    return box

def RegionAdd(Z, X, box):
    # Parameters
    #  Z : array, shape (T, X, Y[, Z]), dataset
    #  box : array, shape (D, 2), array defining spatial box to put X in
    #  X : array, shape (T, prod(diff(box,1))), Input
    # Returns
    #  Z : array, shape (T, X, Y[, Z]), Z+X on box region
    Z[[slice(len(Z))] + list(map(lambda a: slice(*a), box))
      ] += reshape(X, (r_[-1, box[:, 1] - box[:, 0]]))
    return Z


def RegionCut(X, box):
    # Parameters
    #  X : array, shape (T, X, Y[, Z])
    #  box : array, shape (D, 2), region to cut
    # Returns
    #  res : array, shape (T, prod(diff(box,1))),
    dims = X.shape
    return X[[slice(dims[0])] + list(map(lambda a: slice(*a), box))].reshape((dims[0], -1))
    
def DownScale(data,mb,ds):
    """
        Parameters
        ----------
        data : array, shape (T, X, Y[, Z])
            block of the data
        mbs : int
            minibatchsizes for temporal downsampling 
        ds : list/vector or int
            factor for spatial downsampling - must divide X,Y and Z! 
            if list/vector, length equal the number spatial dimensions in data
    
        Returns
        -------
        data0 : array, shape (T/mb, (X/ds[0])*(Y/ds[1])[*(Z/ds[2])])
            downscaled block of the data
        dims0 : array, vector
            original dimensions of the data0

        """
    dims = data.shape
    D = len(dims)
    if type(ds)==int:
        ds=ds*np.ones(D-1)
    elif (len(ds)!=D-1):
        print "either type(ds)==int, or len(ds)== the number of spatial dimensions in data"
        return
    data0 = data[:int(len(data) / mb) * mb].reshape((-1, mb) + data.shape[1:]).mean(1)
    if D == 4:
        data0 = data0[:,:int(dims[1] /ds[0]) *ds[0],:int(dims[2] /ds[1]) *ds[1],:int(dims[3] /ds[2]) *ds[2]].reshape(
            len(data0), dims[1] / ds[0], ds[0], dims[2] / ds[1], ds[1], dims[3] / ds[2], ds[2])\
            .mean(2).mean(3).mean(4)
    else:
        data0 = data0[:,:int(dims[1] /ds[0]) *ds[0],:int(dims[2] /ds[1]) *ds[1]].reshape(len(data0), dims[1] / ds[0], ds[0], dims[2] / ds[1], ds[1]).mean(2).mean(3)
    # for i,d in enumerate(dims[1:]):
    #     data0 = data0.reshape(data0.shape[:1+i] + (d / ds, ds, -1)).mean(2+i)
    dims0 = data0.shape
        
    return data0,dims0
    
def LargestConnectedComponent(shapes,dims,skipBias): 
    L=len(shapes)-skipBias     
    shapes=shapes.reshape((-1,) + dims[1:])    
    structure=np.ones(tuple(3*np.ones((np.ndim(shapes)-1,1))))
    for ll in range(L): 
        temp=np.copy(shapes[ll])
        CC,num_CC=label(temp,structure)
        sz=0
        ind_best=0
        for nn in range(num_CC):
            current_sz=np.count_nonzero(CC[CC==(nn+1)])
            if current_sz>sz:
                ind_best=nn+1
                sz=current_sz
        temp[CC!=ind_best]=0
        shapes[ll]=np.copy(temp)
    shapes=shapes.reshape((len(shapes),-1))
    return shapes
    
def LargestWatershedRegion(shapes,dims,skipBias): 
    L=len(shapes)-skipBias     
    shapes=shapes.reshape((-1,) + dims[1:]) 
    D=len(dims)
    num_peaks=4
#    structure=np.ones(tuple(3*np.ones((np.ndim(shapes)-1,1))))
    for ll in range(L): 
        temp=shapes[ll]
        local_maxi = peak_local_max(gaussian_filter(temp,[1]*(D-1)), exclude_border=False, indices=False, num_peaks=num_peaks)
        markers,junk = label(local_maxi)
        nonzero_mask=temp>0
        if np.sum(nonzero_mask)>(3**3)*num_peaks:
            labels = watershed(-temp, markers, mask=nonzero_mask) #watershed regions
            ind = 1
            temp2 = np.copy(temp)
            temp2[labels!=1]=0
            total_intensity = sum(temp2.reshape(-1,))
            for kk in range(2,labels.max()+1):
                temp2 = np.copy(temp)
                temp2[labels!=kk]=0
                total_intensity2 = sum(temp2.reshape(-1,))
                if total_intensity2>total_intensity:
                    ind = kk
                    total_intensity=total_intensity2
            temp[labels!=ind]=0
            shapes[ll]=temp
    shapes=shapes.reshape((len(shapes),-1))
    return shapes

def findROI(footprint):
    
    dims = footprint.shape
    footprint = footprint.reshape(dims[0],dims[1],-1)
    ROI = zeros(footprint.shape)
    
    mxs = []
    for i in range(np.size(footprint[0,0,:])):
        mxs.append(footprint.reshape(-1,1).max())
    
    for i in range(np.size(footprint[0,0,:])):
        img = footprint[:,:,i].reshape(dims[0],dims[1])
        mx = mxs[i]
        thresh = img>0.4*mx
        thresh2 = binary_dilation(binary_dilation(thresh))
        lbls,marks = label(thresh2)
        rgs = regionprops(lbls)
        if np.size(rgs)>0:
            szs = []
            for prop in rgs:
                szs.append(prop.area)
            ind = np.argmax(szs)
            if rgs[ind].area>100:
                region = lbls==ind+1
                out = zeros([dims[0],dims[1]])
                out[region] = 1
                ROI[:,:,i] = out
                
    ROI = ROI.reshape(dims)
    
    return ROI

def mergeOverlap(shapes,dims,skipBias,activity,mask,centers,boxes,ES):
    
    L = len(shapes)-skipBias
    shapes=shapes.reshape((-1,) + dims[1:])
    ROIs = zeros(shapes.shape)
    
    sizes = []
    for i in range(L):
        ROIs[i] = findROI(shapes[i])
        sizes.append(sum(ROIs[i].reshape(-1,1)))
        
    sizes = np.array(sizes)
    
    connMatrix = zeros([L,L])
    for jj in range(L):
        for kk in range(L-jj-1):
            overlap = sum(np.multiply(ROIs[jj],ROIs[kk+jj+1]).reshape(-1,1))
            if (overlap>0.6*sizes[kk+jj+1])&(overlap>0.6*sizes[jj]) or overlap>0.6*sizes[kk+jj+1] or overlap>0.6*sizes[jj]:
                connMatrix[jj,kk+jj+1] = 1
                connMatrix[kk+jj+1,jj] = 1
                
    num,labels = connected_components(connMatrix)
    shapesToDelete = []
    
    for ll in range(num):
        check = labels==ll
        if sum(check)>1:
            inds = check.nonzero()[0]
            mx = inds[np.argmax(sizes[inds])]
            shapesToMerge = zeros((len(inds),)+dims[1:])
            # activitiesToMerge = something
            # masks, centers, etc ??
            for jj,kk in enumerate(inds):
                shapesToMerge[jj] = shapes[kk]
                if kk != mx:
                    shapesToDelete.append(kk)
            shapes[mx] = shapesToMerge.max(axis=0)
            
            
    #shapes = np.delete(shapes,tuple(shapesToDelete),axis=0)
    
    for ll in shapesToDelete[::-1]:     
        shapes=np.delete(shapes,(ll),axis=0)
        activity=np.delete(activity,(ll),axis=0)
        del mask[ll]
        centers=np.delete(centers,(ll),axis=0)
        boxes=np.delete(boxes,(ll),axis=0)
        ES.delete(ll)
        
    L=len(shapes)-skipBias
    shapes=shapes.reshape((len(shapes),-1))
    # resizing, etc

    return shapes,activity,mask,centers,boxes,ES,L
    
    

def OldWatershed(shapes,dims,skipBias):
    
    L=len(shapes)-skipBias     
    shapes=shapes.reshape((-1,) + dims[1:]) 
    D=len(dims)
    num_peaks=2
#    structure=np.ones(tuple(3*np.ones((np.ndim(shapes)-1,1))))
    for ll in range(L): 
        temp=shapes[ll]
        local_maxi = peak_local_max(gaussian_filter(temp,[1]*(D-1)), exclude_border=False, indices=False, num_peaks=num_peaks)
        markers,junk = label(local_maxi)
        nonzero_mask=temp>0
        if np.sum(nonzero_mask)>(3**3)*num_peaks:
            labels = watershed(-temp, markers, mask=nonzero_mask)        #watershed regions
            temp[labels!=1]=0
            shapes[ll]=temp
    shapes=shapes.reshape((len(shapes),-1))
    return shapes
    
    return shapes
    
def SmoothBackground(shapes,dims,adaptBias,sig_filt): 
    num_peaks=2
    thresh=0.6
    if adaptBias==True:
        temp=gaussian_filter(shapes[-1].reshape(dims[1:]),sig_filt)
        local_maxi = peak_local_max(temp, exclude_border=False, indices=False, num_peaks=num_peaks)
        markers,num_markers = label(local_maxi)
        if num_markers>1:
            foo=gaussian_filter(1.0*(markers==1),sig_filt)
            nonzero_mask=(foo/np.max(foo))>thresh

            temp2=shapes[-1].reshape(dims[1:])
            temp2[nonzero_mask]=0
#            labels = watershed(-temp, markers, mask=nonzero_mask)        #watershed regions
#            temp2[labels==1]=0
            shapes[-1]=np.ndarray.flatten(temp2)
    return shapes

class ExponentialSearch:
    def __init__(self,lam,rho=2):
        # lam - an array of parameter values
        self.lam=lam
        self.lam_high=-np.ones_like(lam)
        self.lam_low=np.copy(self.lam_high)
        self.rho=rho #exponential search parameters
    
    def update(self,decrease,increase):
        ''' decrease - an array the sFize of lambda
                indicates which lam values should decrease
            increase - an array the size of lambda
                indicates which lam values should increase
        '''
        self.lam_high[decrease]=self.lam[decrease]
        self.lam_low[increase]=self.lam[increase]
        cond1=(self.lam_high==-1)
        cond2=(self.lam_low==-1)
        cond3=np.logical_not(np.logical_or(cond1,cond2))
        self.lam[cond1]=self.lam[cond1]*self.rho
        self.lam[cond2]=self.lam[cond2]/self.rho
        self.lam[cond3]=(self.lam_high[cond3]+self.lam_low[cond3])/2
        
    def delete(self,index):
        ''' delete lam,lam_high,lam_low for given index
        '''
        self.lam_high=np.delete(self.lam_high,(index),axis=0)
        self.lam_low=np.delete(self.lam_low,(index),axis=0)
        self.lam=np.delete(self.lam,(index),axis=0)
    
    def reorder(self,indices):
        ''' reorder lam,lam_high,lam_low accodring to given indices
        '''
        self.lam_high=self.lam_high[indices]
        self.lam_low=self.lam_low[indices]
        self.lam=self.lam[indices]
        
def LocalNMF(data, centers, sig, NonNegative=True,FinalNonNegative=True,verbose=False,adaptBias=True,TargetAreaRatio=[],estimateNoise=False,
             PositiveError=False,MedianFilt=False,Connected=False,FixSupport=False, WaterShed=False,SmoothBkg=False,
             updateLambdaIntervals=2,updateRhoIntervals=2,addComponentsIntervals=1,bkg_per=20,
             iters=10,iters0=[30], mbs=[1], ds=1,lam1_s=0,lam1_t=0,lam2_s=0,lam2_t=0):
    """
    Parameters
    ----------
    data : array, shape (T, X, Y[, Z])
        block of the data
    centers : array, shape (L, D)
        L centers of suspected neurons where D is spatial dimension (2 or 3)
    sig : array, shape (D,)
        size of the gaussian kernel in different spatial directions
    NonNegative : boolean
        if True, neurons activity should be considered as non-negative
    FinalNonNegative : boolean
        if False, last activity iteration is done without non-negativity constraint, even if NonNegative==True       
    verbose : boolean
        print progress and record MSE if true (about 2x slower)
    adaptBias : boolean
        subtract rank 1 estimate of bias (background)
    TargetAreaRatio : list of length 2
        Lower and upper bounds on sparsity of non-background components
    estimateNoise : boolean
        estimate noise variance and use it determine if to add components, and to modify sparsity by affecting lam1_s (does not work very well)
    PositiveError : boolean
        do not allow pixels in which the residual (summed over time) becomes negative, by increasing lam1_s in these pixels
    MedianFilt : boolean
        do median filter of spatial components 
    Connected: boolean
        impose connectedness of spatial component by keeping only the largest non-zero connected component in each iteration of HALS
    WaterShed: boolean
        impose that each spatial component has a single watershed region
    SmoothBkg: boolean
        Remove local peaks from background component
    FixSupport : boolean
        do not allow spatial components to be non-zero where sub-sampled spatial components are zero
    updateLambdaIntervals : int
        update lam1_s every this number of HALS iterations, to match contraints
    updateRhoIntervals : int
        decrease rho, update rate of lam1_s, every this number of updateLambdaIntervals HALS iterations (only active during main iterations)
    addComponentsIntervals : int
        add new component, if possible, every this number of updateLambdaIntervals HALS iterations (only active during sub-sampled iterations)
    bkg_per : float
        the background is intialized at this height (percentrilce image)
    iters : int
        number of final iterations on whole data
    iters0 : list
        numbers of initial iterations on subset
    mbs : list
        minibatchsizes for temporal downsampling 
    ds : int or list
        factor for spatial downsampling, can be an integer or a list of the size of spatial dimensions
    lam1_s : float
        L_1 regularization constant for sparsity of shapes
    lam2_s : float
        L_2 regularization constant for sparsity of shapes
    lam_t : float
        L_1 regularization constant for sparsity of activity
    lam2_t : float
        L_2 regularization constant for sparsity of activity

    Returns
    -------
    MSE_array : list (empty if verbose is False)
        Mean square error during algorithm operation
    shapes : array, shape (L+adaptBias, X, Y (,Z))
        the neuronal shape vectors (empty if no components found)
    activity : array, shape (L+adaptBias, T)
        the neuronal activity for each shape (empty if no components found)
    boxes : array, shape (L, D, 2)
        edges of the boxes in which each neuronal shapes lie (empty if no components found)
    """

    # Initialize Parameters
    dims = data.shape
    D = len(dims)
    R = 3 * asarray(sig)  # size of bounding box is 3 times size of neuron
    L = len(centers)
    inner_iterations=10 # number of iterations in inners loops
    shapes = []
    mask = []
    boxes = zeros((L, D - 1, 2), dtype=int)
    MSE_array = []
    mb = mbs[0] if iters0[0] > 0 else 1
    activity = zeros((L, dims[0] / mb))
    lam1_s0=np.copy(lam1_s)
    
    if iters0[0] == 0:
        ds = 1

### Function definitions ###
    # Estimate noise level for a time series
    def GetSnPSD(Y):
        L = len(Y)
        ff, psd_Y = welch(Y, nperseg=round(L / 8))
        sn = sqrt(mean(psd_Y[ff > .3] / 2))
        return sn
    noise = zeros(L)
    
    # Estimate noise level for an array of time series
    def GetSnPSDArray(Y,f_low=10,f_high=0.6):
        print "Calculating noise level..."
        N = len(Y)
        fmin=np.round(f_high*N/2)
        fmax=np.round(N/2) #maximal frequency is at N/2 - the rest is just symmetric
#        try:
#            psd_Y = (np.abs(np.fft.fft(Y, axis=0))**2)/N
#        except MemoryError:
        psd_Y=np.copy(Y)
        if np.ndim(Y)==2:
            M=Y.shape[1]
            for kk in range(M):
                psd_Y[:,kk] = (np.abs(np.fft.fft(Y[:,kk]))**2)/N
                counter=(kk/float(M))*100
                if (counter%10)==0:
                    print counter,'%'
#            else: 
#                raise
        sn=np.sqrt(psd_Y[fmin:fmax].mean(0))+np.sqrt(2*psd_Y[1:f_low].sum(0))/N # white noise + low freq stuff
        sn_std=0.5*sn/np.sqrt(N)
        print "Done"
        return sn,sn_std

    def HALS(data, S, activity, skip=[], check_skip=0, iters=1):
        idx = asarray(filter(lambda x: x not in skip, range(len(activity))))
        A = S[idx].dot(data.T)
        B = S[idx].dot(S.T)

        for ii in range(iters):
            for k, ll in enumerate(idx):
                if check_skip and ii == iters - 1:
                    a0 = activity[ll].copy()
                activity[ll] += nan_to_num((A[k] - np.dot(B[k], activity)-lam1_t-lam2_t*activity[ll]) / B[k, ll])
                if NonNegative:
                    activity[ll][activity[ll] < 0] = 0
            # skip neurons whose shapes already converged
                if check_skip and ll < L and ii == iters - 1:
                    if check_skip == 1:  # compute noise level only once
                        noise[ll] = GetSnPSD(a0) / a0.mean()
                    if np.allclose(a0, activity[ll] / activity[ll].mean(), 1e-4, noise[ll]):
                        skip += [ll]
        C = activity[idx].dot(data)
        D = activity[idx].dot(activity.T)

        for _ in range(iters):
            for k, ll in enumerate(idx):
                if ll == L:
                    S[ll] += nan_to_num((C[k] - np.dot(D[k], S)) / D[k, ll])
                else:
                    S[ll, mask[ll]] += nan_to_num((C[k, mask[ll]]
                                                   - np.dot(D[k], S[:, mask[ll]])-lam1_s[ll,mask[ll]]-lam2_s*S[ll, mask[ll]]) / D[k, ll])
                if NonNegative:
                    S[ll][S[ll] < 0] = 0

        return S, activity, skip

    def HALS4activity(data, S, activity,NonNegative, iters=1):
        A = S.dot(data.T)
        B = S.dot(S.T)

        for _ in range(iters):
            for ll in range(L + adaptBias):
                activity[ll] += nan_to_num((A[ll] - np.dot(B[ll].T, activity)-lam1_t-lam2_t*activity[ll]  ) / B[ll, ll]) #maybe multiply lam1_t by np.sign[activity[ll]?
                if NonNegative:
                    activity[ll][activity[ll] < 0] = 0
        return activity

    def HALS4shape(data, S, activity, iters=1):
        C = activity.dot(data)
        D = activity.dot(activity.T)
        for _ in range(iters):
            for ll in range(L + adaptBias):
                if ll == L:
                    S[ll] += nan_to_num((C[ll] - np.dot(D[ll], S)-lam1_s[ll]-lam2_s*S[ll]) / D[ll, ll])
                else:
                    S[ll, mask[ll]] += nan_to_num((C[ll, mask[ll]]
                                                   - np.dot(D[ll], S[:, mask[ll]])-lam1_s[ll,mask[ll]]-lam2_s*S[ll,mask[ll]])/ D[ll, ll])
#                if NonNegative:
                S[ll][S[ll] < 0] = 0
        # normalize/delete components

        return S 
        
    def HALS4lam(data, S, activity):
        C = activity.dot(data)
        D = activity.dot(activity.T)
        temp=np.copy(S[:L])*0
        for ll in range(L):
            temp[ll,mask[ll]] = C[ll, mask[ll]]- np.dot(D[ll], S[:, mask[ll]])-lam2_s*S[ll,mask[ll]]
            temp[temp<0]=0
        lam=temp.mean(0)+0.001

        return lam
        
        
    def RenormalizeDeleteSort( S, activity, mask,centers,boxes,ES):
        L=len(S)-adaptBias
        deleted_indices=[]
        
        ## Go over shapes
        for ll in range(L + adaptBias):
            if MedianFilt==True:
                S[ll]=median_filter(S[ll],3)
            S_normalization=np.sum(S[ll])
            A_normalization=np.sum(activity[ll])
            if A_normalization>0:
                activity[ll]=activity[ll]/A_normalization 
                S[ll]=S[ll]*A_normalization 
            if ll<L: # don't delete background component
                if ((A_normalization<=0) and (S_normalization<=0)):
                    deleted_indices.append(ll)      
    
        #delete components with zero activity AND zero shape (these will never become non-zero again)
        for ll in deleted_indices[::-1]:     
            S=np.delete(S,(ll),axis=0)
            activity=np.delete(activity,(ll),axis=0)
            del mask[ll]
            centers=np.delete(centers,(ll),axis=0)
            boxes=np.delete(boxes,(ll),axis=0)
            ES.delete(ll)
        L=len(S)-adaptBias
        
        #sort components according to magnitude
        magnitude=np.sum(S[:L],axis=1)*np.max(activity[:L],axis=1)
        sort_indices = np.argsort(magnitude)[::-1]
        centers=centers[sort_indices]
        boxes=boxes[sort_indices]
        mask=[mask[ii] for ii in sort_indices]      

        if adaptBias:
            sort_indices=np.append(sort_indices,L)
        activity=activity[sort_indices]
        S=S[sort_indices]
        ES.reorder(sort_indices)
                
        return  S, activity, mask,centers,boxes,ES,L
        
    def addComponent(new_cent,current_data,data_dim,box_size,S, activity, mask,centers,boxes):
        new_activity=current_data[:,new_cent]-np.dot(activity.T,S[:,new_cent])
#        new_activity=np.random.randn(data_dim[0]) # for testing purposes only
        activity=np.insert(activity,0,new_activity,axis=0)
        S=np.insert(S,0,0*current_data[0,:].reshape(1,-1),axis=0) 
        centers=np.insert(centers,0,np.unravel_index(new_cent,data_dim[1:]),axis=0)
        boxes=np.insert(boxes,0,GetBox(centers[0], box_size, data_dim[1:]),axis=0)
                
        temp = zeros(data_dim[1:])
        temp[map(lambda a: slice(*a), boxes[0])]=1
        temp2=np.where(temp.ravel())[0]
        mask.insert(0,temp2)

        L=len(S)-adaptBias
        
        return  S, activity, mask,centers,boxes,L
        
        
        
### Initialize shapes, activity, and residual ###
        
    
    data0,dims0=DownScale(data,mb,ds)
    if type(ds)==int:
        ds=ds*np.ones(D-1)

    if D == 4:
        activity = data0[:, map(int, centers[:, 0] / ds[0]), map(int, centers[:, 1] / ds[1]),
                         map(int, centers[:, 2] / ds[2])].T
    else:
        activity = data0[:, map(int, centers[:, 0] / ds[0]), map(int, centers[:, 1] / ds[1])].T
        
    data0 = data0.reshape(dims0[0], -1)
    Energy0=np.sum(data0**2,axis=0) #data energy per pixel
    data0sum=np.sum(data0,axis=0) # for sign check later

    data = data.astype('float').reshape(dims[0], -1)
    datasum=np.sum(data,axis=0)# for sign check later
    
    # float is faster than float32, presumable float32 gets converted later on
    # to float again and again
    Energy=np.sum((data**2),axis=0)
    
    
    for ll in range(L):
        boxes[ll] = GetBox(centers[ll] / ds, R / ds, dims0[1:])
        temp = zeros(dims0[1:])
        temp[map(lambda a: slice(*a), boxes[ll])]=1
        mask += np.where(temp.ravel())
        temp = [(arange(int(dims[i + 1] / ds[i])) -int( centers[ll][i] / ds[i])) ** 2 / (2 * (sig[i] / ds[i]) ** 2)
                for i in range(D - 1)]
        temp = exp(-sum(ix_(*temp)))
        temp.shape = (1,) + dims0[1:]
        temp = RegionCut(temp, boxes[ll])
        shapes.append(temp[0])
    S = zeros((L + adaptBias, prod(dims0[1:])))
    for ll in range(L):
        S[ll] = RegionAdd(
            zeros((1,) + dims0[1:]), shapes[ll].reshape(1, -1), boxes[ll]).ravel()
    if adaptBias:
        # Initialize background as bkg_per percentile
        S[-1] = percentile(data0, bkg_per, 0)
        activity = np.r_[activity, ones((1, dims0[0]))]
    
    lam1_s=lam1_s0*np.ones_like(S)
#    if adaptBias:
#        lam1_s[L]=0.1*lam1_s[L]

### Get shape estimates on subset of data ###
    if iters0[0] > 0:
        for it in range(len(iters0)):
            if estimateNoise:
                sn_target,sn_std= GetSnPSDArray(data0)#target noise level
            else:
                sn_target=np.zeros(prod(dims0[1:]))
                sn_std=sn_target
            MSE_target = np.mean(sn_target**2)
            ES=ExponentialSearch(lam1_s)
            lam1_s=ES.lam
            for kk in range(iters0[it]):
                if kk%updateLambdaIntervals==0:
                    # adjust lambda value 
#                    sn_square=(Energy0-2*np.sum(np.dot(activity,data0)*S,axis=0)+np.sum(np.dot(np.dot(activity,activity.T),S)*S,axis=0))/dims0[0] # efficient way to calcuate MSR per pixel
#                    sn=np.sqrt(np.nan_to_num(sn_square*(sn_square>0))) #not sure why, but I get numerical issues here without these precusions 
                    
                    sn=np.sqrt(Energy0-2*np.sum(np.dot(activity,data0)*S,axis=0)+np.sum(np.dot(np.dot(activity,activity.T),S)*S,axis=0))/dims0[0] # efficient way to calcuate MSR per pixel
        
#                    sn=np.sqrt(np.mean((residual**2),axis=0))
                    delta_sn=sn-sn_target # noise margin
                    signcheck=(data0sum-np.dot(np.sum(activity.T,axis=0),S))<0
                    if PositiveError:
                        delta_sn[signcheck]=-float("inf") # residual should not have negative pixels, so we increase lambda for these pixels
                    
                    if len(S)==0:
                        spars=0
                    else:
                        spars=np.mean(S>0,axis=1)
                        
                        temp=repeat(delta_sn.reshape(1,-1),L+adaptBias,axis=0) 
    
                        if TargetAreaRatio==[]:  
                            cond_decrease=temp>sn_std
                            cond_increase=temp<-sn_std
                        else:
                            if adaptBias:
                                spars[-1]=(TargetAreaRatio[1]+TargetAreaRatio[0])/2 # ignore sparsity target for background (bias) component  
                            temp2=repeat(spars.reshape(-1,1),len(S[0]),axis=1)
                            cond_increase=np.logical_or(temp2>TargetAreaRatio[1],temp<-sn_std)
                            cond_decrease=np.logical_and(temp2<TargetAreaRatio[0],temp>sn_std)
        
                        ES.update(cond_decrease,cond_increase)    
                        lam1_s=ES.lam
                    MSE = np.mean(sn**2)
#                    MSE = np.mean((data0-np.dot(activity.T,S))**2)
                    if verbose and L>0:
                        
#                        Norms=(np.sum(S*lam1_s)+lam1_t*np.sum(activity)+0.5*lam2_s*np.sum(S**2)+0.5*lam2_t*np.sum(activity**2))/ data.size
#                        Objective=MSE+Norms                        
                        
                        print(' MSE = {0:.6f}, Target MSE={1:.6f},Sparsity={2:.4f},lam1_s={3:.6f}'.format(MSE,MSE_target,np.mean(spars[:L]),np.mean(lam1_s)))

                    if (kk%addComponentsIntervals==0) and (kk!=iters0[it]-1):
                        delta_sn[signcheck]=-float("inf") # residual should not have negative pixels
                        new_cent=np.argmax(delta_sn) #should I smooth the data a bit first?
                        MSE_std=np.mean(sn_std**2)
                        checkNoZero= not((0 in np.sum(activity,axis=1)) and (0 in np.sum(S,axis=1)))
                        if ((MSE-MSE_target>2*MSE_std) and checkNoZero and (delta_sn[new_cent]>sn_std[new_cent])):                            
                            S, activity, mask,centers,boxes,L=addComponent(new_cent,data0,dims0,R/ds,S, activity, mask,centers,boxes)
                            if (L+adaptBias)>1:                            
                                new_lam=np.take(lam1_s,0,axis=0).reshape(1,-1)
                            else:
                                new_lam=lam1_s0*np.ones_like(data0[0,:]).reshape(1,-1)
                            lam1_s=np.insert(lam1_s,0,values=new_lam,axis=0)
                            ES=ExponentialSearch(lam1_s) #we need to restart exponential search each time we add a component
                        
                S = HALS4shape(data0, S, activity,inner_iterations)
                if Connected==True:
                    S=LargestConnectedComponent(S,dims0,adaptBias)
                if WaterShed==True:
                    S=LargestWatershedRegion(S,dims0,adaptBias)
                    S,activity,mask,centers,boxes,ES,L = mergeOverlap(shapes,dims,skipBias,activity,mask,centers,boxes,ES)
                activity = HALS4activity(data0, S, activity,NonNegative,inner_iterations)                
                S, activity, mask,centers,boxes,ES,L=RenormalizeDeleteSort(S, activity, mask,centers,boxes,ES)
                lam1_s=ES.lam
                if SmoothBkg==True:
                    S=SmoothBackground(S,dims0,adaptBias,tuple(np.array(sig)/np.array(ds)))
                
                print 'Subsampled iteration',kk,'it=',it,'L=',L
                
            if it < len(iters0) - 1:
                mb = mbs[it + 1]
                data0 = data[:len(data) / mb * mb].reshape(-1, mb, prod(dims[1:])).mean(1)
                if D==4:
                    data0 = data0.reshape(len(data0), int(dims[1] / ds[0]), ds[0], int(dims[2] / ds[1]), ds[1],
                                          int(dims[3] / ds[2]), ds[2]).mean(-1).mean(-2).mean(-3)                    
                else:
                    data0 = data0.reshape(len(data0), int(dims[1] / ds[0]), ds[0], int(dims[2] / ds[1]),
                                          ds[1]).mean(-1).mean(-2)
                data0.shape = (len(data0), -1)
                
                activity = ones((L + adaptBias, len(data0))) * activity.mean(1).reshape(-1, 1)
                lam1_s=lam1_s#*mbs[it]/mbs[it+1]
                activity = HALS4activity(data0, S, activity,NonNegative,inner_iterations)
                S, activity, mask,centers,boxes,ES,L=RenormalizeDeleteSort(S, activity, mask,centers,boxes,ES)
                lam1_s=ES.lam

    ### Back to full data ##
        if L==0: #if no non-background components found, return empty arrays
            return [], [], [], []
            
        activity = ones((L + adaptBias, dims[0])) * activity.mean(1).reshape(-1, 1)
        if D==4:
            S = repeat(repeat(repeat(S.reshape((-1,) + dims0[1:]), ds[0], 1), ds[1], 2), ds[2], 3)
            lam1_s= repeat(repeat(repeat(lam1_s.reshape((-1,) + dims0[1:]), ds[0], 1), ds[1], 2), ds[2], 3)
        else:
            S = repeat(repeat(S.reshape((-1,) + dims0[1:]), ds[0], 1), ds[1], 2)
            lam1_s= repeat(repeat(lam1_s.reshape((-1,) + dims0[1:]), ds[0], 1), ds[1], 2)
        for dd in range(1,D):
            while S.shape[dd]<dims[dd]:
                shape_append=np.array(S.shape)
                shape_append[dd]=1
                S=np.append(S,values=np.take(S,-1,axis=dd).reshape(shape_append),axis=dd)
                lam1_s=np.append(lam1_s,values=np.take(lam1_s,-1,axis=dd).reshape(shape_append),axis=dd)
        S=S.reshape(L + adaptBias, -1)
        lam1_s=lam1_s.reshape(L+ adaptBias,-1)
        for ll in range(L):
            boxes[ll] = GetBox(centers[ll], R, dims[1:])
            temp = zeros(dims[1:])
            temp[map(lambda a: slice(*a), boxes[ll])] = 1
            mask[ll] = np.where(temp.ravel())[0]
        
        if FixSupport:
            for ll in range(L):
                lam1_s[ll,S[ll]==0]=float("inf")
#                lam1_s[ll,S[ll]>0]=0
            
        
        ES=ExponentialSearch(lam1_s)
        activity = HALS4activity(data, S, activity,NonNegative, inner_iterations)
        S, activity, mask,centers,boxes,ES,L=RenormalizeDeleteSort(S, activity, mask,centers,boxes,ES)
        lam1_s=ES.lam
        
        if estimateNoise:
            sn_target,sn_std= GetSnPSDArray(data)#target noise level
        else:
            sn_target=np.zeros(prod(dims[1:]))
            sn_std=sn_target
        MSE_target = np.mean(sn_target**2)
        MSE_std=np.mean(sn_std**2)
#        MSE = np.mean((data0-np.dot(activity.T,S))**2)
        
#### Main Loop ####
  
    print 'starting main NMF loop'
    for kk in range(iters):
        S = HALS4shape(data, S, activity,inner_iterations)
        if Connected==True:            
            S=LargestConnectedComponent(S,dims,adaptBias)
        if WaterShed==True:
            S=LargestWatershedRegion(S,dims,adaptBias)
            S,activity,mask,centers,boxes,ES,L = mergeOverlap(shapes,dims,skipBias,activity,mask,centers,boxes,ES)
        if kk==iters-1:
            if FinalNonNegative==False:
                NonNegative=False
        activity = HALS4activity(data, S, activity,NonNegative,inner_iterations)
#       S=LargestConnectedComponent(S)   
        S, activity, mask,centers,boxes,ES,L=RenormalizeDeleteSort(S, activity, mask,centers,boxes,ES)
        lam1_s=ES.lam

        # Recenter
        # if kk % 30 == 20:
        #     for ll in range(L):
        #         shp = shapes[ll].reshape(np.ravel(np.diff(boxes[ll])))
        #         com = boxes[ll][:, 0] + round(center_of_mass(shp))
        #         newbox = GetBox(com, R, dims[1:])
        #         if any(newbox != boxes[ll]):
        #             newshape = zeros(np.ravel(np.diff(newbox)))
        #             lower = vstack([newbox[:, 0], boxes[ll][:, 0]]).max(0)
        #             upper = vstack([newbox[:, 1], boxes[ll][:, 1]]).min(0)
        #             newshape[lower[0] - newbox[0, 0]:upper[0] - newbox[0, 0],
        #                      lower[1] - newbox[1, 0]:upper[1] - newbox[1, 0]] =
        #                 shp[lower[0] - boxes[ll][0, 0]:upper[0] - boxes[ll][0, 0],
        #                     lower[1] - boxes[ll][1, 0]:upper[1] - boxes[ll][1, 0]]
        #             shapes[ll] = newshape.reshape(-1)
        #             boxes[ll] = newbox

        # Measure MSE
        print 'main iteration kk=',kk,'L=',L
        if (kk+1)%updateLambdaIntervals==0:            
            sn=np.sqrt((Energy-2*np.sum(np.dot(activity,data)*S,axis=0)+np.sum(np.dot(np.dot(activity,activity.T),S)*S,axis=0))/dims[0])
            delta_sn=sn-sn_target
            MSE = np.mean(sn**2)
            
            signcheck=(datasum-np.dot(np.sum(activity.T,axis=0),S))<0
            if PositiveError:
                delta_sn[signcheck]=-float("inf") # residual should not have negative pixels, so we increase lambda for these pixels
            
            if S==[]:
                spars=0
            else:
                spars=np.mean(S>0,axis=1)
                
            temp=repeat(delta_sn.reshape(1,-1),L+adaptBias,axis=0) 

            if TargetAreaRatio==[]:  
                cond_decrease=temp>sn_std
                cond_increase=temp<-sn_std
            else:
                if adaptBias:
                    spars[-1]=(TargetAreaRatio[1]+TargetAreaRatio[0])/2 # ignore sparsity target for background (bias) component  
                temp2=repeat(spars.reshape(-1,1),len(S[0]),axis=1)
                cond_increase=np.logical_or(temp2>TargetAreaRatio[1],temp<-sn_std)
                cond_decrease=np.logical_and(temp2<TargetAreaRatio[0],temp>sn_std)

            ES.update(cond_decrease,cond_increase)
#            print spars
#            print lam1_s[:,0]
            lam1_s=ES.lam
            if kk<iters/3: #restart exponential search unless enough iterations have passed
                ES=ExponentialSearch(lam1_s)                
            elif L+adaptBias>1: # if we have more then one component just keep exponitiated grad descent instead
                if (kk+1)%updateRhoIntervals==0: #update rho every updateRhoIntervals if we are still not converged
                    if np.any(spars[:L]>TargetAreaRatio[1]) or np.any(spars[:L]>TargetAreaRatio[1]):
                        ES.rho=2-1/(ES.rho)
                        print 'rho=',ES.rho
                ES=ExponentialSearch(lam1_s,rho=ES.rho)
                    
            if verbose:             
#                        Norms=(np.sum(S*lam1_s)+lam1_t*np.sum(activity)+0.5*lam2_s*np.sum(S**2)+0.5*lam2_t*np.sum(activity**2))/ data.size
#                        Objective=MSE+Norms        
                print(' MSE = {0:.6f}, Target MSE={1:.6f},Sparsity={2:.4f},lam1_s={3:.6f}'.format(MSE,MSE_target,np.mean(spars[:L]),np.mean(lam1_s)))
                if kk == (iters - 1):
                    print('Maximum iteration limit reached')
                MSE_array.append(MSE)

    return asarray(MSE_array), S.reshape((-1,) + dims[1:]), activity, boxes


# example


#T = 1000
#X = 201
#Y = 101
#data = np.random.randn(T, X, Y)
#centers = asarray([[40, 30]])
#data[:, 30:45, 25:33] += 2*np.sin(np.array(range(T))/200).reshape(-1,1,1)*np.ones([T,15,8])
#sig = [300, 300]
#
#MSE_array, shapes, activity, boxes = LocalNMF( 
#    data, centers, sig, NonNegative=True, verbose=True,lam1_s=0.1,adaptBias=True)
#
#
#import matplotlib.pyplot as plt
#plt.imshow(shapes[0])
#
#for ll in range(len(shapes)):
#    print np.mean(shapes[ll]>0)