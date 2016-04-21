from __future__ import division

def GetDefaultParams():
    # Get parameters for: Data type, NMF algorithm and intialization
    
    # choose dataset name (function GetData will use this to fetch the correct dataset)
    data_name_set=['Xin_MFM','HillmanSmall','PhilConfocal','PhilMFM']
    data_name=data_name_set[0]
    
    # "default" parameters - for additional information see "LocalNMF" in BlockLocalNMF
    
    NumCent=0 # Max number of centers to import from Group Lasso intialization - if 0, we don't run group lasso
    mbs=[1] # temporal downsampling of data in intial phase of NMF
    ds=1 # spatial downsampling of data in intial phase of NMF. Ccan be an integer or a list of the size of spatial dimensions
    TargetAreaRatio=[0.005,0.02] # target sparsity range for spatial components
    repeats=1 # how many repeations to run NMF algorithm
    iters0=[10] # number of intial NMF iterations, in which we downsample data and add components
    iters=30 # number of main NMF iterations, in which we fine tune the components on the full data
    lam1_s=0 # l1 regularization parameter initialization (for increased sparsity). If zero, we have no l1 sparsity penalty
    updateLambdaIntervals=2 # update sparsity parameter every updateLambdaIntervals iterations
    addComponentsIntervals=1 # in initial NMF phase, add new component every updateLambdaIntervals*addComponentsIntervals iterations
    updateRhoIntervals=1 # in main NMF phase, add new component every updateLambdaIntervals*updateRhoIntervals iterations
    Background_num=1 #number of background components - one of which at every repetion
    bkg_per=0.2 # intialize of background shape at this percentile (over time) of video
    sig=(500,500,500) # estiamte size of neuron - bounding box is 3 times this size. If larger then data, we have no bounding box.
    
    NonNegative=True # should we constrain activity and shapes to be non-negative?
    FinalNonNegative=True # should we constrain activity to be non-negative at final iteration?
    Connected=False # should we constrain all spatial component to be connected?
    WaterShed=False # should we constrain all spatial component to have only one watershed component?
    
    # experimental stuff - don't use for now
    estimateNoise=False # should we tune sparsity and number of neurons to reach estimated noise level?
    PositiveError=False # should we tune sparsity and number of neurons to have only positive residual error?
    FixSupport=False # should we fix non-zero support at main NMF iterations?
    SmoothBackground=False # Should we cut out out peaks from background component?

    SuperVoxelize=False # should we supervoxelize data (does not work now)

    # change parameters for other datasets    
    if data_name=='Xin_MFM':
        NumCent=0
        mbs=[2]
        #ds=[2,2,1]
        ds=[2,2]
        #TargetAreaRatio=[0.005,0.03]
        TargetAreaRatio = [0.3,0.8]
        iters=20
        iters0=[60]
        repeats=1
        updateLambdaIntervals=4
        addComponentsIntervals=1
        updateRhoIntervals=1
        lam1_s=1
        Background_num=1 #number of background components
        bkg_per=0.02

        Connected=True
        WaterShed=True

        FinalNonNegative=True
        #sig=(500,500,3)
        sig=(30,30)
    elif data_name=='PhilConfocal':
        NumCent=50
        mbs=[1]
        ds=[2,2,1]
        TargetAreaRatio=[0.005,0.05]
        repeats=1
        iters0=[50]
        iters=20
        updateLambdaIntervals=2
        updateRhoIntervals=1
        lam1_s=1
        Background_num=1 #number of background components
        bkg_per=0.2

        Connected=True
        WaterShed=True
        FinalNonNegative=False
        sig=(20,20,2)
    elif data_name=='PhilMFM':
        NumCent=0
        mbs=[2]
        ds=[2,2,1]
        TargetAreaRatio=[0.005,0.03]
        repeats=1
        iters0=[60]
        iters=20        
        updateLambdaIntervals=2
        addComponentsIntervals=1
        updateRhoIntervals=1
        lam1_s=1
        Background_num=1 #number of background components
        bkg_per=0.02

        Connected=True
        WaterShed=True

        FinalNonNegative=False
        sig=(500,500,3)
        
        

    params_dict=dict([['data_name',data_name],['SuperVoxelize',SuperVoxelize],['NonNegative',NonNegative],
                      ['FinalNonNegative',FinalNonNegative],['mbs',mbs],['TargetAreaRatio',TargetAreaRatio],
                     ['iters',iters],['iters0',iters0],['lam1_s',lam1_s]
                     ,['updateLambdaIntervals',updateLambdaIntervals],['updateRhoIntervals',updateRhoIntervals],['addComponentsIntervals',addComponentsIntervals],
                     ['estimateNoise',estimateNoise],['PositiveError',PositiveError],['sig',sig],['NumCent',NumCent],
                    ['bkg_per',bkg_per],['ds',ds],['sig',sig],['Background_num',Background_num],['Connected',Connected],['WaterShed',WaterShed],
                    ['SmoothBackground',SmoothBackground],['FixSupport',FixSupport],['repeats',repeats]])
    class Bunch(object):
        def __init__(self, adict):
            self.__dict__.update(adict)

    params=Bunch(params_dict)
    
    return params,params_dict

#%% Main script for running NMF

if __name__ == "__main__":
        
    import matplotlib
    matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
        
    import numpy as np
    from BlockLocalNMF import LocalNMF
    import matplotlib.pyplot as plt
    from AuxilaryFunctions import GetFileName,SuperVoxelize,GetData,GetCentersData
    import cPickle
    
    plt.close('all')
    
    plot_all=True
    do_NMF=True
        
    params,params_dict=GetDefaultParams()
    
    data=GetData(params.data_name)
    if params.SuperVoxelize==True:
        data=SuperVoxelize(data)        

    cent=GetCentersData(data,params.data_name,params.NumCent)
            
    if do_NMF==True:        
        for rep in range(params.repeats):  
            if rep>=params.Background_num:
                adaptBias=False
            else:
                adaptBias=True
            MSE_array, shapes, activity, boxes = LocalNMF(
                data, cent, params.sig,TargetAreaRatio=params.TargetAreaRatio,updateLambdaIntervals=params.updateLambdaIntervals,addComponentsIntervals=params.addComponentsIntervals,
                PositiveError=params.PositiveError,NonNegative=params.NonNegative,FinalNonNegative=params.FinalNonNegative, verbose=True,lam1_s=params.lam1_s, adaptBias=adaptBias,estimateNoise=params.estimateNoise,
                Connected=params.Connected,SmoothBkg=params.SmoothBackground,FixSupport=params.FixSupport,bkg_per=params.bkg_per,iters0=params.iters0,iters=params.iters,mbs=params.mbs, ds=params.ds)
            
            L=len(shapes)
            if L<=adaptBias:
                break
            saveName=GetFileName(params_dict,rep)        
            f = file('NMF_Results/'+saveName, 'wb')
            
            results=dict([['MSE_array',MSE_array], ['shapes',shapes], 
                         ['activity',activity],['boxes',boxes],['cent',cent],
                         ['params',params_dict]])
            cPickle.dump(results, f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()
            print 'rep #',str(rep+1), ' finished'  

            data=data- activity.T.dot(shapes.reshape(len(shapes), -1)).reshape(np.shape(data)) #subtract this iteration components from data        
        
    #%% PLotting
    if plot_all==True:
        from PlotResults import PlotAll
        SaveNames=[] 
        for rep in range(params.repeats):
            SaveNames+=[GetFileName(params_dict,rep)]
        PlotAll(SaveNames,params)
        
    #    L=len(activity)   
    #    for ll in range(L):
    #        print 'Sparsity=',np.mean(shapes[ll]>0)
    
