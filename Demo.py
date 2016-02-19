from __future__ import division

def GetDefaultParams():
    
    NumCent=0
    NonNegative = True
    data_name_set=['Hillman','HillmanSmall','Sophie2D','Sophie3D','SophieVoltage3D','Sara19DEC2015_w1t1']
    data_name=data_name_set[4]
    SuperVoxelize=False
    
    #default parameters
    mbs=[2]
    ds=2
    TargetAreaRatio=[0.005,0.02]
    repeats=1
    iters0=[10]
    iters=50
    updateLambdaIntervals=2
    updateRhoIntervals=1
    lam1_s=0.1
    Background_num=1 #number of background components
    estimateNoise=False
    PositiveError=False
    FixSupport=False
    Connected=True
    NonNegative=True
    FinalNonNegative=True
    sig=(500,500,500)
        
    if data_name=='Sophie2D':
        mbs=[2]
        ds=2
        TargetAreaRatio=[0.03,0.15]
        iters=30
        iters0=[30]
        repeats=10
        updateLambdaIntervals=2
        updateRhoIntervals=1
        lam1_s=0.01
        Background_num=0 #number of background components
        estimateNoise=False
        PositiveError=False
        FixSupport=False
        Connected=True
        sig=(500,500)
    elif data_name=='Sophie3D':
        mbs=[1]
        ds=1
        TargetAreaRatio=[0.002,0.02]
        iters=50
        iters0=[10]
        repeats=100
        Background_num=5 #number of background components
    elif data_name=='SophieVoltage3D':
        mbs=[1]
        ds=1
        TargetAreaRatio=[0.002,0.02]
        iters=50
        iters0=[10]
        repeats=100
        Background_num=0 #number of background components
        FinalNonNegative=False
    elif data_name=='Sara19DEC2015_w1t1':
        mbs=[1]
        ds=1
        TargetAreaRatio=[0.005,0.02]
        repeats=1
        iters0=[40]
        iters=120
        updateLambdaIntervals=2
        updateRhoIntervals=1
        lam1_s=1
        Background_num=1 #number of background components
        FixSupport=False
        Connected=True
        FinalNonNegative=False
        sig=(30,30,3)
        

    params_dict=dict([['data_name',data_name],['SuperVoxelize',SuperVoxelize],['NonNegative',NonNegative], ['FinalNonNegative',FinalNonNegative],['mbs',mbs],['TargetAreaRatio',TargetAreaRatio],
                 ['iters',iters],['iters0',iters0],['lam1_s',lam1_s],['updateLambdaIntervals',updateLambdaIntervals],['updateRhoIntervals',updateRhoIntervals],
                 ['estimateNoise',estimateNoise],['PositiveError',PositiveError],['sig',sig],['NumCent',NumCent],
                 ['ds',ds],['sig',sig],['Background_num',Background_num],['Connected',Connected],['FixSupport',FixSupport],['repeats',repeats]])
    class Bunch(object):
        def __init__(self, adict):
            self.__dict__.update(adict)

    params=Bunch(params_dict)
    
    return params,params_dict

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
                data, cent, params.sig,TargetAreaRatio=params.TargetAreaRatio,updateLambdaIntervals=params.updateLambdaIntervals,PositiveError=params.PositiveError,
                NonNegative=params.NonNegative,FinalNonNegative=params.FinalNonNegative, verbose=True,lam1_s=params.lam1_s, adaptBias=adaptBias,estimateNoise=params.estimateNoise,
                Connected=params.Connected,FixSupport=params.FixSupport,iters0=params.iters0,iters=params.iters,mbs=params.mbs, ds=params.ds)
            
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
    
