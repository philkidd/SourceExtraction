def PlotAll(SaveNames,params):
    from numpy import  min, max, percentile,asarray,mean,ceil,sqrt
    import numpy as np
    from pylab import load
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.backends.backend_pdf import PdfPages
    from scipy.ndimage.measurements import label    
    from AuxilaryFunctions import GetRandColors, max_intensity,SuperVoxelize,GetData,PruneComponents

    ## plotting params 
    # what to plot 
    KeepCC=False #threshold shapes in the end and keep only connected components
    plot_activities=True
    plot_shapes=True
    plot_activityCorrs=False
    plot_clustered_shape=False
    plot_residual=False
    make_video_shapes=False
    make_video_residual=True
    # what to save
    save_video=True
    save_plot=True
    close_figs=True#close all figs right after saving (to avoid memory overload)
    
    IncludeBackground=False #should we include the background as an extracted component?
    
    # how to plot
    scale=0.5 #scale colormap to enhance colors
    dpi=500 #for videos
    restrict_support=True #in shape video, zero out data outside support of shapes
    C=4 #number of components to show in shape videos (if larger then number of shapes L, then we automatically set C=L)
    
    # Fetch experimental 3D data 
    data=GetData(params.data_name)
    if params.SuperVoxelize==True:
        data=SuperVoxelize(data)     
    
    
    dims=np.shape(data)
    denoised_data=0
    residual=data
    detrended_data=data
    
    
    for rep in range(len(SaveNames)): 
        resultsName=SaveNames[rep]
        try:
            results=load('NMF_Results/'+SaveNames[rep])
        except IOError:
            if rep==0:
                print 'results file not found!!'              
            else:
                break            
        shapes=results['shapes']
        activity=results['activity']

        if rep>params.Background_num:
            adaptBias=False
        else:
            adaptBias=True
        if IncludeBackground==True:
            adaptBias=False                        
        
        L=len(activity)-adaptBias 

        if L==0: #Stop if we encounter a file with zero components
            break
        
#        shapes,activity,L=PruneComponents(shapes,activity,params,L)
        
        activity_NonNegative=np.copy(activity)
        activity_NonNegative[activity_NonNegative<0]=0
        colors=GetRandColors(L)
        color_shapes=np.transpose(shapes[:L].reshape(L, -1,1)*colors,[1,0,2]) #weird transpose for tensor dot product next line
        denoised_data = denoised_data + (activity_NonNegative[:L].T.dot(color_shapes)).reshape(tuple(dims)+(3,))        
        residual = residual - activity_NonNegative.T.dot(shapes.reshape(L+adaptBias, -1)).reshape(dims)        
        detrended_data= detrended_data - adaptBias*((activity_NonNegative[-1].reshape(-1, 1)).dot(shapes[-1].reshape(1, -1))).reshape(dims)
        
#        denoised_data =np.asarray(denoised_data ,dtype='float')
    #    plt.close('all')
        
        #%% plotting params
        a=ceil(sqrt(L+adaptBias))
        b=ceil((L+adaptBias)/a)
        left  = 0.05 # the left side of the subplots of the figure
        right = 0.95   # the right side of the subplots of the figure
        bottom = 0.05   # the bottom of the subplots of the figure
        top = 0.95      # the top of the subplots of the figure
        wspace = 0.05   # the amount of width reserved for blank space between subplots
        hspace = 0.05  # the amount of height reserved for white space between subplots        
              
        #%% ###### Plot Individual neurons' activities
        if plot_activities:
            fig0=plt.figure(figsize=(dims[1] , dims[2]))
    
            for ii in range(L+adaptBias):
                ax = plt.subplot(a,b,ii+1)
                plt.plot(activity[ii])
                plt.setp(ax,xticks=[],yticks=[])
            plt.subplots_adjust(left, bottom, right, top, wspace, hspace)            
            
            if save_plot==True:
                pp = PdfPages('Activities'+resultsName+'.pdf')
                pp.savefig(fig0)
                pp.close()
                if close_figs:
                    plt.close('all')
                
        #%% ###### Plot Individual neurons' area which is correlated with their activities
        if plot_activityCorrs:
            if save_plot==True:
                pp = PdfPages('CorrelationWithActivity'+resultsName+'.pdf')
            for dd in range(len(shapes[0].shape)):
                fig0=plt.figure(figsize=(11,18))
        
                for ii in range(L+adaptBias):
                    ax = plt.subplot(a,b,ii+1)
                    corr_imag=np.dot(activity[ii],np.transpose(data,[1,2,0,3]))/np.sqrt(np.sum(data**2,axis=0)*np.sum(activity[ii]**2))
                    plt.imshow(np.abs(corr_imag).max(dd),cmap='gnuplot')
                    plt.setp(ax,xticks=[],yticks=[])
                plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
            
                if save_plot==True:
                    pp.savefig(fig0)
            pp.close()
            if close_figs:
                plt.close('all')
       
        #%% Threshold shapes
        TargetAreaRatio=[0.005,0.02]  #size of largest connected component
        rho=2 #exponential search parameters
        
        if KeepCC==True:
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
                    if (mean(temp) < TargetAreaRatio[0]):
                        threshold_high = threshold
                    elif (mean(temp) > TargetAreaRatio[1]):
                        threshold_low = threshold
                    else:
                        print mean(temp)
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
            
            for ll in range(L+adaptBias): 
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
        
        #%%  All Shapes projections
        if plot_shapes:
            if save_plot==True:
                pp = PdfPages('Shapes'+resultsName+'.pdf')

            for dd in range(len(shapes[0].shape)):
                fig=plt.figure(figsize=(18 , 11))
                for ll in range(L+adaptBias):
                    ax = plt.subplot(a,b,ll+1)  
                    temp=shapes[ll].max(dd)
                    if dd==2:
                        temp=temp.T
                    mi=np.min(temp)
                    ma=np.max(temp)
                    plt.imshow(temp,vmin=mi,vmax=ma,cmap='gnuplot')
                    plt.setp(ax,xticks=[],yticks=[])
                    #sparsity
                    spar_str=str(np.round(np.mean(shapes[ll]>0)*100,2))+'%'
                    ax.text(0.02, 0.02, spar_str,
                    verticalalignment='bottom', horizontalalignment='left',
                    transform=ax.transAxes,
                    color='white',weight='bold', fontsize=13)
                    #L^p
                    for p in range(2,6,2):
                        Lp=(np.sum(shapes[ll]**p))**(1/float(p))/np.sum(shapes[ll])
                        Lp_str=str(np.round(Lp*100,2))+'%' #'L'+str(p)+'='+
                        ax.text(0.02+p*0.2, 0.02, Lp_str,
                        verticalalignment='bottom', horizontalalignment='left',
                        transform=ax.transAxes,
                        color='yellow',weight='bold', fontsize=13)
                plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
                if save_plot==True:
                    pp.savefig(fig)            
            pp.close()
            if close_figs:
                plt.close('all')
        #for ll in range(L+adaptBias):
        #    print 'Sparsity=',np.mean(shapes[ll]>0)
                
        #%% ###### Plot Individual neurons' shape with clustering
        if plot_clustered_shape:
            from sklearn.cluster import spectral_clustering
            pp = PdfPages('ClusteredShapes'+resultsName+'.pdf')
            figs=[]
            for dd in range(len(shapes[0].shape)):
                figs.append(plt.figure(figsize=(18 , 11)))
            for ll in range(L):              
                ind=np.reshape(shapes[ll],(1,)+tuple(dims[1:]))>0
                temp=data[np.repeat(ind,dims[0],axis=0)].reshape(dims[0],-1)
                delta=1 #affinity trasnformation parameter
                clust=3 #number of cluster
                similarity=np.exp(-np.corrcoef(temp.T)/delta)                    
                labels = spectral_clustering(similarity, n_clusters=clust, eigen_solver='arpack')
                ind2=np.array(np.nonzero(ind.reshape(-1))).reshape(-1)
                temp_shape=np.repeat(np.zeros_like(shapes[ll]).reshape(-1,1),clust,axis=1)
                for cc in range(clust):
                    temp_shape[ind2[labels==cc],cc]=1
                temp_shape=temp_shape.reshape(tuple(dims[1:])+(clust,))

                for dd in range(len(shapes[0].shape)):
                    current_fig=figs[dd]
                    ax = current_fig.add_subplot(a,b,ll+1)
                    if dd==2:
                        temp_shape=np.transpose(temp_shape,axes=[1,0,2,3])
                    ax.imshow(temp_shape.max(dd))

                    plt.setp(ax,xticks=[],yticks=[])
                    plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
            
            if save_plot==True:
                for dd in range(len(shapes[0].shape)):
                    current_fig=figs[dd]
                    pp.savefig(current_fig)
            pp.close()
            if close_figs:
                plt.close('all')

        #%% #####  Video Shapes
        if make_video_shapes:
            components=range(min(asarray([C,L])))
            C=len(components)
            if restrict_support==True:
                shape_support=shapes[components[0]]>0            
                for cc in range(C):
                    shape_support=np.logical_or(shape_support,shapes[components[cc]]>0)
                detrended_data=shape_support.reshape((1,)+tuple(dims[1:]))*detrended_data
            
            fig = plt.figure(figsize=(16,7))
            mi = 0
            ma = max(detrended_data)*scale
            #mi2 = 0
            #ma2 = max(shapes[ll])*max(activity[ll])
            
            ii=0
            #import colormaps as cmaps
            #cmap=cmaps.viridis
            cmap='gnuplot'
            a=3
            b=1+C
            
            ax1 = plt.subplot(a,b,1)
            im1 = ax1.imshow(detrended_data[ii].max(0), vmin=mi, vmax=ma,cmap=cmap)
            title=ax1.set_title('Data')
            #plt.colorbar(im1)
            ax2=[] 
            ax4=[] 
            ax6=[]
            im2=[]
            im4=[]
            im6=[]
            
            for cc in range(C):
                ax2.append(plt.subplot(a,b,2+cc))
                comp=shapes[components[cc]].max(0)*activity_NonNegative[components[cc],ii]
                ma2=max(shapes[components[cc]].max(0))*max(activity_NonNegative[components[cc]])*scale
                im2.append(ax2[cc].imshow(comp,vmin=0,vmax=ma2,cmap=cmap))
            #ax2[0].set_title('Shape')
            #    plt.colorbar(im2)
            
            ax3 = plt.subplot(a,b,1+b)
            im3 = ax3.imshow(detrended_data[ii].max(1), vmin=mi, vmax=ma,cmap=cmap)
            
            #plt.colorbar(im3)
            
            for cc in range(C):
                ax4.append(plt.subplot(a,b,2+b+cc))
                comp=shapes[components[cc]].max(1)*activity_NonNegative[components[cc],ii]
                ma2=max(shapes[components[cc]].max(1))*max(activity_NonNegative[components[cc]])*scale
                im4.append(ax4[cc].imshow(comp,vmin=0,vmax=ma2,cmap=cmap))
            
            #plt.colorbar(im4)
            
            ax5 = plt.subplot(a,b,1+2*b)
            im5 = ax5.imshow(np.transpose(detrended_data[ii].max(2)), vmin=mi, vmax=ma,cmap=cmap)
            
            #plt.colorbar(im5)
            for cc in range(C):
                ax6.append(plt.subplot(a,b,2+2*b+cc))
                comp=np.transpose(shapes[components[cc]].max(2))*activity_NonNegative[components[cc],ii]
                ma2=max(shapes[components[cc]].max(2))*max(activity_NonNegative[components[cc]])*scale
                im6.append(ax6[cc].imshow(comp,vmin=0,vmax=ma2,cmap=cmap))
            
            #plt.colorbar(im6)
            
            fig.tight_layout()
            ComponentsActive=np.array([])
            for cc in range(C):
                ComponentsActive=np.append(ComponentsActive,np.nonzero(activity_NonNegative[components[cc]]))
            ComponentsActive=np.unique(ComponentsActive)
            
            def update(tt):
                ii=ComponentsActive[tt]
                im1.set_data(detrended_data[ii].max(0))        
                im3.set_data(detrended_data[ii].max(1))        
                im5.set_data(np.transpose(detrended_data[ii].max(2)))
            
                for cc in range(C): 
                    im2[cc].set_data(shapes[components[cc]].max(0)*activity_NonNegative[components[cc],ii])
                    im4[cc].set_data(shapes[components[cc]].max(1)*activity_NonNegative[components[cc],ii])
                    im6[cc].set_data(np.transpose(shapes[components[cc]].max(2))*activity_NonNegative[components[cc],ii])
                title.set_text('Data, time = %.1f' % ii)
            
            if save_video==True:
                writer = animation.writers['ffmpeg'](fps=10)
                ani = animation.FuncAnimation(fig, update, frames=len(ComponentsActive), blit=True, repeat=False)
                if restrict_support==True:
                    ani.save('Shapes_Restricted'+resultsName+'.mp4',dpi=dpi,writer=writer)
                else:                        
                    ani.save('Shapes_'+resultsName+'.mp4',dpi=dpi,writer=writer)
            else:
                ani = animation.FuncAnimation(fig, update, frames=len(ComponentsActive), blit=True, repeat=False)
                plt.show()
    
    
    #%% normalize denoised data range
#    denoised_data=10*denoised_data/np.max(denoised_data)/scale
    denoised_data=denoised_data/np.percentile(denoised_data,99)
    denoised_data[denoised_data>1]=1
    
    #%% ##### Plot denoised Results

    if plot_residual==True:
        
        dims=data.shape
        cmap='gnuplot'         
        
        pic_residual=percentile(residual, 95, axis=0)
        pic_denoised = max_intensity(denoised_data, axis=0)
        pic_data=percentile(data, 95, axis=0)
        
        left  = 0.05 # the left side of the subplots of the figure
        right = 0.95   # the right side of the subplots of the figure
        bottom = 0.05   # the bottom of the subplots of the figure
        top = 0.95      # the top of the subplots of the figure
        wspace = 0.05   # the amount of width reserved for blank space between subplots
        hspace = 0.05  # the amount of height reserved for white space between subplots
        
        
        fig1=plt.figure(figsize=(11,18))
        mi=min(pic_data)
        ma=max(pic_data)
        ax = plt.subplot(311)
        im=ax.imshow(pic_data.max(0),vmin=mi,vmax=ma,cmap=cmap)
        ax.set_title('Data')
        plt.colorbar(im)
        plt.setp(ax,xticks=[],yticks=[])
        ax2 = plt.subplot(312)
        im2=ax2.imshow(max_intensity(pic_denoised,0),interpolation='None')
        ax2.set_title('Denoised')
        plt.setp(ax,xticks=[],yticks=[])
        plt.colorbar(im2)
        ax3 = plt.subplot(313)
        im3=ax3.imshow(pic_residual.max(0),cmap=cmap)
        ax3.set_title('Residual')
        plt.setp(ax,xticks=[],yticks=[])
        plt.colorbar(im3)
        plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
        
        fig2=plt.figure(figsize=(11,18))
        mi=min(pic_data)
        ma=max(pic_data)
        ax = plt.subplot(311)
        im=ax.imshow(pic_data.max(1),vmin=mi,vmax=ma,cmap=cmap)
        ax.set_title('Data')
        plt.colorbar(im)
        plt.setp(ax,xticks=[],yticks=[])
        ax2 = plt.subplot(312)
        im2=ax2.imshow(max_intensity(pic_denoised,1),interpolation='None')
        ax2.set_title('Denoised')
        plt.colorbar(im2)
        plt.setp(ax,xticks=[],yticks=[])
        ax3 = plt.subplot(313)
        im3=ax3.imshow(pic_residual.max(1),cmap=cmap)
        ax3.set_title('Residual')
        plt.colorbar(im3)
        plt.setp(ax,xticks=[],yticks=[])
        plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
        
        fig3=plt.figure(figsize=(11,18))
        mi=min(pic_data)
        ma=max(pic_data)
        ax = plt.subplot(311)
        im=ax.imshow(pic_data.max(2).T,vmin=mi,vmax=ma,cmap=cmap)
        ax.set_title('Data')
        plt.colorbar(im)
        plt.setp(ax,xticks=[],yticks=[])
        ax2 = plt.subplot(312)
        im2=ax2.imshow(np.transpose(max_intensity(pic_denoised,2),[1,0,2]),interpolation='None')
        ax2.set_title('denoised')
        plt.setp(ax,xticks=[],yticks=[])
        plt.colorbar(im2)
        ax3 = plt.subplot(313)
        im3=ax3.imshow(np.transpose(pic_residual.max(2)),cmap=cmap)
        ax3.set_title('Residual')
        plt.colorbar(im3)
        plt.setp(ax,xticks=[],yticks=[])
        plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
    
        if save_plot==True:
            pp = PdfPages('Data_Denoised_Residual_'+resultsName+'.pdf')
            pp.savefig(fig1)
            pp.savefig(fig2)
            pp.savefig(fig3)
            pp.close()
    
    
    #fig = plt.figure()
    #plt.plot(MSE_array)
    #plt.xlabel('Iteration')
    #plt.ylabel('MSE')
    #plt.show()
    
    #%% #####  Video Residual    
    if make_video_residual:
        fig = plt.figure(figsize=(16,7))
        mi = 0
        ma = max(detrended_data)*scale
        mi3 = 0
        ma3 = max(residual)*scale

        ii=0
        #import colormaps as cmaps
        #cmap=cmaps.viridis
        cmap='gnuplot'
        a=3
        b=3
        
        
        D=len(np.shape(data))-1 #number of spatial dimensions
        im_array=[]
        transpose_flags=[]
        for kk in range(D):
            transpose_flags+= [False]
            temp=np.shape(detrended_data[ii].max(kk))
            if temp[0]>temp[1]:
                transpose_flags[kk]=True    
                
        for kk in range(D):
            ax1 = plt.subplot(a,b,D*kk+1)
            if transpose_flags[kk]==False:
                pic=detrended_data[ii].max(kk)
            else:
                pic=np.transpose(detrended_data[ii].max(kk))
                
            im_array += [ax1.imshow(pic, vmin=mi, vmax=ma,cmap=cmap)]
            title=ax1.set_title('Data')
            plt.colorbar(im_array[-1])
            ax2 = plt.subplot(a,b,D*kk+2)
            
            if transpose_flags[kk]==False:
                pic=residual[ii].max(kk)
            else:
                pic=np.transpose(residual[ii].max(kk))        
            im_array += [ax2.imshow(pic, vmin=mi3, vmax=ma3,cmap=cmap)]
            ax2.set_title('Residual')
            plt.colorbar(im_array[-1])
            ax3 = plt.subplot(a,b,D*kk+3)
            
            if transpose_flags[kk]==False:
                pic=max_intensity(denoised_data[ii],kk)
            else:
                pic=np.transpose(max_intensity(denoised_data[ii],kk),[1,0,2])  
            im_array += [ax3.imshow(pic,interpolation='None')]
            ax3.set_title('Denoised')
            plt.colorbar(im_array[-1])

        fig.tight_layout()
            
        def update(ii):
            for kk in range(D):
                if transpose_flags[kk]==False:
                    im_array[kk*D].set_data(detrended_data[ii].max(kk))        
                    im_array[kk*D+1].set_data(residual[ii].max(kk)) 
                    im_array[kk*D+2].set_data(max_intensity(denoised_data[ii],kk))
                else:
                    im_array[kk*D].set_data(np.transpose(detrended_data[ii].max(kk)))        
                    im_array[kk*D+1].set_data(np.transpose(residual[ii].max(kk))) 
                    im_array[kk*D+2].set_data(np.transpose(max_intensity(denoised_data[ii],kk),[1,0,2]))
            
            title.set_text('Data, time = %.1f' % ii)
        
        if save_video==True:
            writer = animation.writers['ffmpeg'](fps=10)
            ani = animation.FuncAnimation(fig, update, frames=len(data), blit=False, repeat=False)
            ani.save('Data_Denoised_Residual'+resultsName+'.mp4',dpi=dpi,writer=writer)
        else:
            ani = animation.FuncAnimation(fig, update, frames=len(data), blit=False, repeat=False)
            plt.show()  
    #
    #fig = plt.figure(figsize=(dims[1] , dims[2]))
    #mi = min(data)
    #ma = max(data)
    #dpi=100
    #
    #
    #ax = plt.subplot(131)
    #im = ax.imshow(data[ii], vmin=mi, vmax=ma)
    #ax2 = plt.subplot(132)
    #im2 = ax2.imshow(residual[ii], vmin=mi, vmax=ma)
    #ax3 = plt.subplot(133)
    #im3 = ax3.imshow(denoised_data[ii], vmin=mi, vmax=ma)
    #def update(ii):
    #    im.set_data(data[ii])
    ##    ax.set_title('Data t='+str(ii))
    #    im2.set_data(residual[ii])
    ##    ax2.set_title('Residual')
    #    im3.set_data(denoised_data[ii])
    ##    ax3.set_title('Denoised')
    #
    #
    #ani = animation.FuncAnimation(fig, update, frames=len(data), blit=False, interval=30, repeat=False)
    #ani.save('MaxProjections'+str(dd)+'.mp4',writer=writer,dpi=dpi)
    
     
        #%%  Figure for grant
        #left  = 0.05 # the left side of the subplots of the figure
        #right = 0.95   # the right side of the subplots of the figure
        #bottom = 0.1   # the bottom of the subplots of the figure
        #top = 0.95      # the top of the subplots of the figure
        #wspace = 0.01   # the amount of width reserved for blank space between subplots
        #hspace = 0.01  # the amount of height reserved for white space between subplots
        #font = {'family' : 'normal',
        #        'weight' : 'bold',
        #        'size'   : 22}
        #
        #matplotlib.rc('font', **font)
        #
        #pp = PdfPages('GrantFigure.pdf')
        #components=[20,3,26,24]
        #a=len(components)
        #b=4
        #fig=plt.figure(figsize=(16 , 10.5))
        #fs=10.0 #sampling rate
        #t=array(range(len(activity[0])))/fs
        #for dd in range(3):
        #    for ll in range(a):
        #        ax = plt.subplot(b,a,ll+1+dd*a)  
        #        temp=shapes[components[ll]].max(dd)
        #        temp=temp[:,:-20]
        #        if dd==2:
        #            temp=transpose(temp)
        #        mi=min(temp[temp>0])
        #        ma=max(temp)
        #        plt.imshow(temp,vmin=mi,vmax=ma,cmap='gnuplot')
        #        plt.setp(ax,xticks=[],yticks=[])
        #        if dd==0:
        #            ax.set_title('Component #'+str(ll+1))
        #        if ll==0:
        #            if dd==0:
        #                plt.ylabel('x-projection')
        #            elif dd==1:
        #                plt.ylabel('y-projection')
        #            elif dd==2:
        #                    plt.ylabel('z-projection')
        #                    
        #for ii in range(a):
        #    ax = plt.subplot(a,b,a*(b-1)+ii+1)
        #    plt.plot(t,activity[components[ii]],color='black',linewidth=2)
        #    plt.xlim([t[0],t[-1]])
        #    plt.setp(ax,yticks=[],xticks=[0,20,40,60,80])
        #    plt.xlabel('t [sec]')
        #    if ii==0:
        #        plt.ylabel('activity [AU]')
        #plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
        #
        #pp.savefig(fig)
        #pp.close()
        
        
        
        
        #%%  Overlay Shapes
        #cmaps =  ['BuGn', 'BuPu','GnBu', 'Greens',
        #          'Greys', 'Oranges', 'OrRd','PuBu', 'PuBuGn',
        #          'PuRd', 'Purples', 'RdPu','Reds', 'YlGn', 
        #          'YlGnBu', 'YlOrBr', 'YlOrRd','Blues']
        #cmaps =  [ 'afmhot', 'bone', 'copper', 'pink']
        #pp = PdfPages('Overlay_Shapes_Thresholded.pdf')
        ##components=[12,20,3,27]
        #for dd in range(len(dims)):
        #    fig=plt.figure(figsize=(dims[1] , dims[2]))
        #    first=True
        #    ind=0
        #    for ll in range(48):            
        #        temp=shapes[ll].max(dd)
        #        mi=min(temp[temp>0])
        #        ma=max(temp)
        #        if first==True:
        #            first=False
        #        else:
        #            temp=np.ma.masked_where(temp==0,temp)            
        ##        plt.imshow(temp,interpolation='bilinear',cmap=plt.get_cmap(cmaps[ind]),vmin=mi,vmax=ma,alpha=1-0.2*ind)
        #        plt.imshow(temp,interpolation='nearest',vmin=mi,vmax=ma,alpha=0.2)
        #        ind+=1
        #    pp.savefig(fig)
        #
        #pp.close()
        
        #def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
#    '''
#    Function to offset the "center" of a colormap. Useful for
#    data with a negative min and positive max and you want the
#    middle of the colormap's dynamic range to be at zero
#
#    Input
#    -----
#      cmap : The matplotlib colormap to be altered
#      start : Offset from lowest point in the colormap's range.
#          Defaults to 0.0 (no lower ofset). Should be between
#          0.0 and `midpoint`.
#      midpoint : The new center of the colormap. Defaults to 
#          0.5 (no shift). Should be between 0.0 and 1.0. In
#          general, this should be  1 - vmax/(vmax + abs(vmin))
#          For example if your data range from -15.0 to +5.0 and
#          you want the center of the colormap at 0.0, `midpoint`
#          should be set to  1 - 5/(5 + 15)) or 0.75
#      stop : Offset from highets point in the colormap's range.
#          Defaults to 1.0 (no upper ofset). Should be between
#          `midpoint` and 1.0.
#    '''
#    cdict = {
#        'red': [],
#        'green': [],
#        'blue': [],
#        'alpha': []
#    }
#    import matplotlib
#    matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
#
#    import numpy as np
#    import matplotlib.pyplot as plt
#
#    # regular index to compute the colors
#    reg_index = np.linspace(start, stop, 257)
#
#    # shifted index to match the data
#    shift_index = np.hstack([
#        np.linspace(0.0, midpoint, 128, endpoint=False), 
#        np.linspace(midpoint, 1.0, 129, endpoint=True)
#    ])
#
#    for ri, si in zip(reg_index, shift_index):
#        r, g, b, a = cmap(ri)
#
#        cdict['red'].append((si, r, r))
#        cdict['green'].append((si, g, g))
#        cdict['blue'].append((si, b, b))
#        cdict['alpha'].append((si, a, a))
#
#    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
#    plt.register_cmap(cmap=newcmap)
#
#    return newcmap





#%% Main    
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!

    from AuxilaryFunctions import GetFileName      
    from Demo import GetDefaultParams
    
    params,params_dict=GetDefaultParams()
    last_rep=params.repeats
    
    SaveNames=[]
    for rep in range(last_rep):
        name=GetFileName(params_dict,rep)
        SaveNames.append(name)
        
    PlotAll(SaveNames,params)    