#myPlots.py
import os
import matplotlib.pyplot as plt 
from lmfit import minimize, Parameters
import scipy
#import myDataFuns as dataFuns
import seaborn as sns   #plot options - ease and beauty 
import pandas as pd  #data frames
import numpy as np   #arrays/matrixes - matlab
import math as math
import myUtils as utils
#import myFitFuns as fitFuns
import initialize as initialize


def plot_sims(parsDf,lArray,index_pat, myModel, modelSpecs, my_fit, dataDf,boolTinput,fitting,tx,solver_kws={}, ax=None, **kwargs):
    print("plotting....")
    dataFit=fitting[0]
    modelFit=fitting[1]
    fitWeight=fitting[2]

    #[stateVarsNames,stateVarsType,pops0Input,pops0Output,pops0Type]=initialize.getModelSpecs(modelSpecs)
    [plots,plotsData,plotsIndex,plotsScale,plotsTransform] = initialize.getPlotSpecs(modelSpecs)
    
    # varsNames=vars0[0]
    # varsTypes=vars0[1]
    # varsV=[] 
    # varsC=[]
    # for index,name in enumerate(varsNames): 
    #     if(varsTypes[index]=='lV'):
    #         varsV.append(name)
    #     else:
    #         varsC.append(name)
       
    #setup plots
    popCount = 1 if (myModel.stateVarsTypeList.count("Pop")>=1 or myModel.stateVarsTypeList.count("Pop1")>=1) else 0
    otherCount = len(np.unique(plotsIndex))-popCount
    numPlots = 1 + popCount + otherCount
    totalHeight = 1 + 3*popCount + 3*otherCount
    height_ratios = [.5]
    for ps in range(0,popCount):
        height_ratios.append(1.5)
    for ps in range(0,otherCount):
        height_ratios.append(1.5)
    fig, ax = plt.subplots(numPlots, 1, figsize=kwargs.get("figsize",(10,totalHeight)), gridspec_kw={'height_ratios': height_ratios})
    fig.tight_layout()
    
    # Simulate top 10
    topDf=[]
    topDf=pd.DataFrame(topDf)
    minT=[]
    maxT=[]
    #tx = utils.ExtractTxsFromDf(dataDf,timeColumn=timeCol,txCols=txCols)
    params = Parameters() 
    [consSet,consSplit,outsSplit] = utils.getSplits(myModel)

        #params.add(cset0, value=float(lArray[cset0][index_pat]),vary=False) 
    #parSet=myga.sampleParsLHS(parsAll,dictionary,nPop)
    for i in range(0,parsDf.shape[0]):
        #params=fitFuns.setConParamsFromParSet(parsDf,vars0,params,i)
        params=utils.setParams(params,myModel.varsC,myModel.varsV,parsDf,myModel.dictionary,i) ##clean up        
        params = utils.setConstants(consSet,lArray,params,index_pat,myModel.dictionary)
        params = utils.setSplits(consSplit,lArray,index_pat,params,outsSplit)

        myModel.SetParams(**params.valuesdict())
        if(boolTinput):
            nanT=np.isnan(dataDf['Testosterone'])
            dataDf['Tlog']=np.log10(dataDf['Testosterone'])
            #f1T = scipy.interpolate.interp1d(data.Days[~nanT],data['Testosterone'][~nanT],fill_value="extrapolate",kind="linear")#"extrapolate"
            f1T = scipy.interpolate.interp1d(dataDf.Days[~nanT],dataDf['Tlog'][~nanT],fill_value="extrapolate",kind="linear")#"extrapolate"
            teval=range(0,dataDf['Days'][len(dataDf['Days'])-1])
            testo=f1T(teval)
            testo=10**testo
            i=0
            tx=[]
            for d in teval:
                tx0=[]
                tx0.append(d)
                tx0.append(d+1)
                for col in myModel.txCols:
                    tx0.append(dataDf[col].loc[i])
                tx.append(tx0)
                #tx.append([d,d+1,dataDf['Lup'].loc[i],dataDf['Abi'].loc[i],dataDf['Enza'].loc[i],dataDf['Apa'].loc[i]])
                if(d>=dataDf['Days'].loc[i+1]):
                    i=i+1 

            myModel.SimulateWithT(testo,treatmentScheduleList=tx,max_step=1,**solver_kws)
            # x_S=myModel.getxS(myModel.resultsDf)
            # myModel.resultsDf['Pi']=myModel.RunInstaPSA(x_S,myModel.resultsDf)
        else:
            myModel.Simulate(treatmentScheduleList=tx,max_step=1,**solver_kws)
        #myModel.Trim(dt=dt)
        myModel.resultsDf['id']=i
        #myModel.resultsDf[myModel.timeCol]=myModel.resultsDf['Days']
        topDf = pd.concat([topDf, myModel.resultsDf],ignore_index=True)
        l1=myModel.resultsDf[modelFit[0]].values
        l1=l1[l1 < 1E308]
        l1=l1[l1 > -1E308]
        minT.append(np.min(l1))
        maxT.append(np.max(l1))
    #update these with topDf instead of just resultsDf, which is just last one
    xmin=0
    xmax=max(myModel.resultsDf['Days'])
    xmarg=1 

    #plotting
    #drugs
    axi=0
    axx=ax[axi]
    plot_dose_over_time(axx,myModel,tx,dataDf,xmin,xmax,xmarg)
    
    for j in range(0,numPlots-1):
        axi=axi+1
        axx=ax[axi]
        p = [i for i, x in enumerate(plotsIndex) if x==j]
        #print(p)
        for pi in p:
            # print(plots0[0][pi])
            if(plots[pi]=='TumorSize'):
                plot_tumor_over_time(axx,xmin,xmax,topDf,kwargs,dataDf,xmarg,myModel,modelFit,dataFit,plots,plotsData,plotsScale,plotsIndex,pi)
            else:
                metric=plots[pi]
                plot_metric_over_time(axx,xmin,xmax,topDf,kwargs,dataDf,xmarg,metric,myModel,modelFit,dataFit,plots,plotsData,plotsScale,plotsTransform)
            
        
####################
# Plot types 
####################

def plot_dose_over_time(axx,myModel,tx,dataDf,xmin,xmax,xmarg):
    cols=['ti','tf']+myModel.txCols
    tx=pd.DataFrame(tx,columns=cols)
    for i,drug in enumerate(myModel.txCols):
        drugBarPosition = 1-.5*i
        # timeVec = dataDf[timeCol].values
        # tt = {'ti':[tx.tf[len(tx)-1]], 'tf':[timeVec[len(timeVec)-1]]}
        # for txName in txCols:
        #     tt[txName]=[0]
        # tt=pd.DataFrame(tt,columns=cols)
        # tx = pd.concat([tx,tt],ignore_index=True)
        plot_drug(tx,drug,drugBarPosition,axx,color=myModel.colorPalette[drug])
    axx.set_xticks([])
    axx.set_xlim((xmin-xmarg, xmax+xmarg))

def plot_drug(tx,drug,drugBarPosition,ax,color):  
    drugConcentrationVec=tx[drug]
    drugConcentrationVec = np.array([x/(np.max(drugConcentrationVec)+1e-12) for x in drugConcentrationVec])
    drugConcentrationVec = 0.5*drugConcentrationVec + drugBarPosition
    axName=ax
    axName.fill_between(tx['ti'], drugConcentrationVec, drugBarPosition, 
                        step="post", color=color, alpha=0.6, label=drug)

def plot_tumor_over_time(axx,xmin,xmax,topDf,kwargs,dataDf,xmarg,myModel,modelFit,dataFit,plots,plotsData,plotsScale,plotsIndex,pi):
    popName=plots[pi]
    #print(popName)
    plotMods(topDf,['Days'],[popName],5,axx,myModel.colorPalette,0.5,**kwargs)
    tumorPopsList = []
    #tPopPlot=''
    tmp=popName.replace('TumorSize','')
    popName='Pop'+tmp
    sizeName='TumorSize'+tmp
    for p,pops in enumerate(myModel.stateVarsTypeList):
        if(pops == 'Pop' or pops==popName):
            tumorPopsList.append(myModel.stateVars[p])
            #tPopPlot=myModel.stateVarsPlotList[pi]
    for pop in tumorPopsList:
        plotMods(topDf,['Days'],[pop],3,axx,myModel.colorPalette,0.5,**kwargs)

    ind2=plots.index(sizeName)
    if(plotsData[ind2]!='Non'):
        #ind1 = modelFit.index(sizeName)
        sns.scatterplot(data=dataDf,x='Days', y=plotsData[ind2], color=myModel.colorPalette[sizeName],edgecolor="black",ax=axx)
    plt.tight_layout()
    plt.ylabel("Tumor size") 
    axx.set_xlim((xmin-xmarg, xmax+xmarg))
    #axx.set_ylim(bottom=0)
    #plt.ylim(bottom=0.0001)
    axx.get_legend().remove()
    if(plotsScale[ind2]=='log'):
    #if(tPopPlot=='log'):
        axx.set_yscale('log',base=10)

def plotMods(df,tCol,modFits,lw0,ax0,palette,alpha):
    modelFitsDf = pd.melt(df, id_vars=tCol, value_vars=modFits)
    # print(modelFitsDf)
    # print(tCol)
    sns.lineplot(x=tCol[0],y="value", hue="variable", style="variable",lw=lw0, palette=palette, alpha=alpha,legend='full', data=modelFitsDf, ax=ax0)

def plot_metric_over_time(axx,xmin,xmax,topDf,kwargs,dataDf,xmarg,metric,myModel,modelFit,dataFit,plots,plotsData,plotsScale,plotsTransform):
    #print(metric)
    # if(len(metric)>1):
    #     for mi,met in enumerate(metric):
    #         if(mi==0):
    #             topDf0=topDf[met]
    #         else:
    #             topDf0  = [topDf0[i] + topDf[met][i] for i in range(len(topDf0))]
    #     topDf['split'] = topDf0
    # else:
    #     topDf0=topDf[met]
    ind2=plots.index(metric)
    if(plotsTransform[ind2]=="norm" and (metric in modelFit)):
        ind = modelFit.index(metric)
        normName=metric+"norm"  
        data1=dataDf[dataFit[ind]].interpolate(method ='linear', limit_direction ='backward')  
        topDf[normName] = topDf[metric]*data1.iloc[0]/topDf[metric][0]
        plotMods(topDf,['Days'],[normName],5,axx,myModel.colorPalette,0.5,**kwargs)
    elif(plotsTransform[ind2]=="splitnorm" and (metric in modelFit)):
        ind = modelFit.index(metric)
        normName="splitnorm"  
        data1=dataDf[dataFit[ind]].interpolate(method ='linear', limit_direction ='backward')  
        topDf[normName] = topDf['split']*data1.iloc[0]/topDf['split'][0]
        plotMods(topDf,['Days'],[normName],5,axx,myModel.colorPalette,0.5,**kwargs)
    else:
        plotMods(topDf,['Days'],[metric],5,axx,myModel.colorPalette,0.5,**kwargs)
    if(plotsData[ind2]!='None'):
        sns.scatterplot(data=dataDf,x='Days', y=plotsData[ind2], color=myModel.colorPalette[plotsData[ind2]],edgecolor="black",ax=axx)
    # if(metric in modelFit):
    #     ind = modelFit.index(metric)
    #     sns.scatterplot(data=dataDf,x=myModel.timeCol, y=dataFit[ind], color=myModel.colorPalette[dataFit[ind]],edgecolor="black",ax=axx)
    plt.tight_layout()
    plt.ylabel(metric) 
    axx.set_xlim((xmin-xmarg, xmax+xmarg))
    #axx.set_ylim((-.01, plots0[4][ind]))
    if(plotsScale[ind2]=='log'):
        axx.set_yscale('log',base=10)

def plot_pars(patsdf,lPars,dictionary):
    ncolMax=6
    colSize=2
    if(len(lPars)<ncolMax):
        ncol=len(lPars)
    else:
        ncol=ncolMax
    nrow=math.floor((len(lPars)-1)/ncol)+1

    x1=colSize*ncolMax*2
    x2=colSize*nrow
    fig, axis = plt.subplots(nrow, ncol,figsize=(x1,x2))

    ax1=0
    ax2=0
    for i in range(0,len(lPars)): 
        ff=dictionary.get(lPars[i])
        if(nrow==1):
            axis[ax2].hist(patsdf[lPars[i]], bins=10) #density=True, 
            axis[ax2].set_title(lPars[i])
            #axis[ax2].set_xlim([ff.min,ff.max])
        else:  
            axis[ax1, ax2].hist(patsdf[lPars[i]], bins=10) #density=True, 
            axis[ax1, ax2].set_title(lPars[i])
            #axis[ax1, ax2].set_xlim([ff.min,ff.max])
        ax2=ax2+1
        if(ax2>=ncolMax):
            ax2=0
            ax1=ax1+1
        
def plot_pars_over_time(pars,path,dictionary,nIterations): ##make error bars!!!!!
    meanTime=[]
    stdTime=[]
    xTime=[]
    for i in range(1,nIterations+1):
        dd=path + str(i) + '.xlsx'
        parsI = pd.read_excel(dd,index_col=[0])
        tmp = parsI.mean(axis=0)
        tmp2 = parsI.std(axis=0)
        meanTime.append(tmp.values.tolist())
        stdTime.append(tmp2.values.tolist())
        xTime.append(i)
    meanTime=pd.DataFrame(meanTime)
    stdTime=pd.DataFrame(stdTime)

    ncolMax=6
    colSize=2
    if(len(pars)<ncolMax):
        ncol=len(pars)
    else:
        ncol=ncolMax
    nrow=math.floor((len(pars)-1)/ncol)+1

    x1=colSize*ncolMax*2
    x2=colSize*nrow
    fig, axis = plt.subplots(nrow, ncol,figsize=(x1,x2))

    ax1=0
    ax2=0
    for i, par in enumerate(pars):
        if(nrow==1):
            axis[ax2].fill_between(xTime,meanTime[i]-stdTime[i], meanTime[i]+stdTime[i],alpha=0.5)
            axis[ax2].plot(xTime,meanTime[i])
            axis[ax2].set_title(pars[i])
        else: 
            axis[ax1, ax2].fill_between(xTime,meanTime[i]-stdTime[i], meanTime[i]+stdTime[i],alpha=0.5) 
            axis[ax1, ax2].plot(xTime,meanTime[i])        
            axis[ax1, ax2].set_title(pars[i])
        ax2=ax2+1
        if(ax2>=ncolMax):
            ax2=0
            ax1=ax1+1
        
    # for i in range(1,len(pars)-1): 
    #     ff=dictionary.get(pars[i])
    #     if(nrow==1):
    #         axis[ax2].fill_between(xTime,meanTime[i]-stdTime[i], meanTime[i]+stdTime[i],alpha=0.5)
    #         axis[ax2].plot(xTime,meanTime[i])
    #         axis[ax2].set_title(pars[i])
    #     else: 
    #         axis[ax1, ax2].fill_between(xTime,meanTime[i]-stdTime[i], meanTime[i]+stdTime[i],alpha=0.5) 
    #         axis[ax1, ax2].plot(xTime,meanTime[i])        
    #         axis[ax1, ax2].set_title(pars[i])
    #     ax2=ax2+1
    #     if(ax2>=ncolMax):
    #         ax2=0
    #         ax1=ax1+1

def plot_pars_dists(pars,path,dictionary,nIterations,lesionId,path_to_file,n_wins): ##make error bars!!!!!, and bounds with dic
    dd=path_to_file
    parsI = pd.read_excel(dd,index_col=[0])

    ncolMax=6
    colSize=2
    if(len(pars)<ncolMax):
        ncol=len(pars)
    else:
        ncol=ncolMax
    nrow=math.floor((len(pars)-1)/ncol)+1

    x1=colSize*ncolMax*2
    x2=colSize*nrow
    fig, axis = plt.subplots(nrow, ncol,figsize=(x1,x2))

    ax1=0
    ax2=0
    for par in pars:
        if(len(pars)==1):
            axis.set_title(par)
            axis.hist(parsI[par][:n_wins].values,range=[dictionary.get(par).min,dictionary.get(par).max])
        elif(nrow==1):
            axis[ax2].set_title(par)
            axis[ax2].hist(parsI[par][:n_wins].values,range=[dictionary.get(par).min,dictionary.get(par).max])
        else: 
            axis[ax1, ax2].set_title(par)
            axis[ax1, ax2].hist(parsI[par][:n_wins].values,range=[dictionary.get(par).min,dictionary.get(par).max])       
        ax2=ax2+1
        if(ax2>=ncolMax):
            ax2=0
            ax1=ax1+1

    pf_name="pars"+"%s.pdf"%lesionId
    path_to_fig=os.path.join(path,pf_name)    
    plt.savefig(path_to_fig)
    plt.close()