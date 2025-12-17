# ====================================================================================
# Various functions that I found useful in this project
# ====================================================================================
import numpy as np
import pandas as pd
import sys
import scipy
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
from tqdm import tqdm
from lmfit import Parameters
from scipy.stats import qmc

sys.path.append("./local/")
import odeFuns as odeFuns
import initialize as initialize
import myPlots as plots
import models as mods


solver_kws={'scaleTumourVolume':True, 'method':'DOP853', 'absErr':1.0e-6, 'relErr':1.0e-6, 'suppressOutputB':True}
optimiser_kws = {'method':'leastsq', 'xtol':1e-6, 'ftol':1e-6,'max_nfev':10,'epsfcn':0.001}#'nan_policy':'omit',

##################################
# Loop/fit through patient data
##################################
def patient_loop(patientIdDic,dataDir,lArray,modelName,txCols,colorPalette,outputCols,boolTinput,my_fit,outDir):
    modDir = os.path.join(outDir,modelName)
    mkdir(modDir)

    # setup model
    modelSpecs= pd.read_excel('./Parameters/Pars_'+modelName+'.xlsx','modelSpecs')
    [stateVarsNames,stateVarsType,pops0Input,pops0Output,pops0Type]=initialize.getModelSpecs(modelSpecs) 
    [model,outputFunction]=mods.pickModel(modelName)
    tmpModel = odeFuns.setupModel(stateVarsNames,stateVarsType,pops0Input,pops0Output,pops0Type,txCols,model,modelName,colorPalette,outputCols,outputFunction,dt=1)   

    #patient loop  
    for index_pat,p in enumerate(patientIdDic):
        patientId = patientIdDic[p]
        print(patientId)

        dataDf = initialize.loadData(dataDir,patientId)
        tx = ExtractTxsFromDf(dataDf,tmpModel.txCols)
        parSet=samplePars(tmpModel,my_fit.n_sample)

        params = Parameters() #initialize parameters
        [consSet,consSplit,outsSplit] = getSplits(tmpModel)

        parsDf=[]
        parsDf=pd.DataFrame(parsDf)
        for si in tqdm(range(0,my_fit.n_sample)):
            # setup parameters
            params=setParams(params,tmpModel.varsC,tmpModel.varsV,parSet,tmpModel.dictionary,si)
            params = setConstants(consSet,lArray,params,index_pat,tmpModel.dictionary)#set constants ---check!!!
            params = setSplits(consSplit,lArray,index_pat,params,outsSplit)#set splits
            currParams = params.copy()

            # Fit 
            fitObj = residualPCa(currParams, dataDf, my_fit.fitting, tx, boolTinput,tmpModel)
            
            #add pars and err to list
            tmpRow = {}
            for name in tmpModel.varsV: #varsList
                tmpRow[name]=float(currParams[name])      
            resultList = list(tmpRow.items())
            resultList=np.array(resultList)
            df = pd.DataFrame(data=resultList.T,columns = tmpModel.varsV)
            df=df.iloc[1:,:]
            if(type(fitObj)==float): df['err']=float(fitObj)
            else: df['err']=float(sum(fitObj))
            parsDf = pd.concat([parsDf,df])#,ignore_index=True)

        #sort err ascending and save
        parsDf=parsDf.sort_values(by=['err'], ascending=True,ignore_index=True)
        path_to_par_all=os.path.join(modDir, "parAll"+tmpModel.modelName+"%s.xlsx" % (patientId))
        parsDf.to_excel(path_to_par_all)
        
        #save winners
        parsDfShort=parsDf.iloc[0:my_fit.n_wins,0:len(tmpModel.varsV)]
        path_to_par_wins=os.path.join(modDir, "parWins"+tmpModel.modelName+"%s.xlsx" % (patientId))
        parsDfShort.to_excel(path_to_par_wins)
        
        #plotting
        plots.plot_sims(parsDfShort,lArray,index_pat,tmpModel,modelSpecs,my_fit,dataDf,boolTinput,my_fit.fitting,tx)
        pf_name="patientFit"+tmpModel.modelName+"%s.pdf"%patientId
        path_to_fig=os.path.join(modDir,pf_name)    
        plt.savefig(path_to_fig)
        plt.close()
        
        path_to_file=os.path.join(modDir, "parWins"+tmpModel.modelName+"%s.xlsx" % (patientId))#include pat#?
        plots.plot_pars_dists(tmpModel.varsV,modDir,tmpModel.dictionary,my_fit.nIterations,patientId,path_to_file,my_fit.n_wins)

# ====================================================================================
# Functions for dealing with treatment schedules
# ====================================================================================
# # Helper function to obtain treatment schedule from calibration data
def ExtractTxsFromDf(dataDf,txCols):
    timeVec = dataDf['Days'].values
    txDf = pd.DataFrame()
    for i,t in enumerate(txCols):
        txDf[txCols[i]] = dataDf[txCols[i]].values

    return ConvertTDToTSFormat(timeVec, txDf)

# # Helper function to extract the treatment schedule from the data
def ConvertTDToTSFormat(timeVec,txDf):
    txScheduleList = [] 
    tStart = timeVec[0]
    diffDf = txDf.diff()
    #sumDiff = diffDf.sum(axis=1)
    sumDiff=abs(diffDf).sum(axis=1)
    for i,t in enumerate(timeVec):#1:
        flatList = []
        if (sumDiff.iloc[i]!=0):
            tmp = txDf.iloc[i-1].values
            flatList.append(tStart)
            flatList.append(t)
            for j in range(0,tmp.size):
                flatList.append(tmp[j])
            txScheduleList.append(flatList)
            tStart = t
    flatList = []
    flatList.append(tStart)
    flatList.append(timeVec[-1]+(tStart==timeVec[-1])*1)
    tmp = txDf.iloc[-1].values
    for j in range(0,tmp.size):
        flatList.append(tmp[j])
    txScheduleList.append(flatList)
    return txScheduleList


# Turns a treatment schedule in list format (i.e. [tStart, tEnd, DrugConcentration]) into a time series
def TreatmentListToTS(treatmentList,tVec):
    drugConcentrationVec = np.zeros_like(tVec)
    for drugInterval in treatmentList:
        drugConcentrationVec[(tVec>=drugInterval[0]) & (tVec<=drugInterval[1])] = drugInterval[2]
    return drugConcentrationVec


# ====================================================================================
# Misc
# ====================================================================================   
    
def mkdir(dirName):
    dirToCreateList = [dirName] if type(dirName) is str else dirName
    for directory in dirToCreateList:
        currDir = ""
        for subdirectory in directory.split("/"):
            currDir = os.path.join(currDir, subdirectory)
            try:
                os.mkdir(currDir)
            except:
                pass
        return True    

def SplitPop2(stateVars, inputs): #maybe only need inputs and do +1 for length
    F = 1
    F0 = 1
    outputs=[]
    sumOut=0
    if(len(stateVars)>1):
        for si,sV in enumerate(range(0,len(stateVars)-1)):
            outputs.append(inputs[si])
            sumOut=sumOut+inputs[si]
        td=1-sumOut
        outputs=[td]+outputs
    else:
        outputs.append(1)
    return outputs

def samplePars(tmpModel,n_sample):
    xlimits=np.array([[0,0]])
    
    sample_scaled=pd.DataFrame()
    if(len(tmpModel.varsList)>0):
        for name in tmpModel.varsList: 
            ff=tmpModel.dictionary.get(name)
            xlimits=np.append(xlimits,[[ff.min,ff.max]],axis=0)                     
        xlimits=xlimits[1:,:]
        sampler = qmc.LatinHypercube(d=len(tmpModel.varsList))
        sample = sampler.random(n=n_sample)
        l_bounds = xlimits[:,0]
        u_bounds = xlimits[:,1]
        sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
        sample_scaled = pd.DataFrame(sample_scaled,columns=tmpModel.varsList)
    for vC in tmpModel.varsC:
        ff=tmpModel.dictionary.get(vC)
        sample_scaled[vC]=ff.value
    return sample_scaled

def getSplits(tmpModel):
    consSet=[]
    consSplit=[]
    outsSplit=[]
    for ppindex,pType in enumerate(tmpModel.pops0Type): 
        if(pType=='split'):
            consSplit.append(tmpModel.pops0Input[ppindex])
            outsSplit.append(tmpModel.pops0Output[ppindex])
        else:
            consSet.append(tmpModel.pops0Input[ppindex])
    return [consSet,consSplit,outsSplit]

def setSplits(consSplit,lArray,index_pat,params,outsSplit):
    inputs1=[]
    for index_sp,name in enumerate(consSplit):
        if(index_sp==0 and name in lArray):
            inputs1.append(lArray[name][index_pat])
        else:
            ff=params.get(name)
            inputs1.append(ff.value) 
    outputs1 = SplitPop2(outsSplit, inputs1[1:])
    for index_sp,op in enumerate(outsSplit):
        params.add(op,value=inputs1[0]*outputs1[index_sp],vary = False)
    return params

def setParams(params,varsC,varsV,parSet,dictionary,si):
    params=setConParamsFromDictionary(varsC,params,dictionary)
    params=setVarParamsFromParSet(parSet,varsV,params,dictionary,si) 
    return params

def setVarParamsFromParSet(parSet,vars0,params,dictionary,i):
    for index,name in enumerate(vars0):
        ff=dictionary.get(name)
        params.add(ff.name, value=float(parSet[name][i]),min=ff.min,max=ff.max,vary=True) 
    return params

def setConParamsFromDictionary(varsNames,params,dictionary):
    for name in varsNames:
        ff=dictionary.get(name)
        params.add(name, value=ff.value,vary=False)  
    return params

def setConstants(consSet,lArray,params,index_pat,dictionary):
    for cset0 in consSet:
        if(cset0 in lArray.columns):
            params.add(cset0, value=float(lArray[cset0][index_pat]),vary=False) 
        elif(cset0 in params):
            params.add(cset0,value=params[cset0],vary=False) 
        else:
            params.add(cset0, value=dictionary.get(cset0).value,vary=False) 
    return params

def residualPCa(params, data, fitting, tx, boolTinput,model):
    model.SetParams(**params.valuesdict())
    converged = False
    max_step = solver_kws.get('max_step',np.inf)
    currSolver_kws = solver_kws.copy()
    k=0
    dataFit=fitting[0]
    modelFit=fitting[1]
    fitWeight=fitting[2]
    detectThreshold=fitting[3]

    if(boolTinput):
        nanT=np.isnan(data['Testosterone'])
        data['Tlog']=np.log10(data['Testosterone'])
        f1T = scipy.interpolate.interp1d(data.Days[~nanT],data['Tlog'][~nanT],fill_value="extrapolate",kind="linear")#"extrapolate"
        teval=range(0,data['Days'].iloc[len(data['Days'])-1])
        testo=f1T(teval)
        testo=10**testo
        i=0
        tx=[]
        for d in teval:
            tx0=[]
            tx0.append(d)
            tx0.append(d+1)
            for col in model.txCols:
                tx0.append(data[col].loc[i])
            tx.append(tx0)
            if(d>=data['Days'].loc[i+1]):
                i=i+1 

    while not converged:
        if(boolTinput):
            model.SimulateWithT(testo,treatmentScheduleList=tx,**currSolver_kws)
        else:
            model.Simulate(treatmentScheduleList=tx,**currSolver_kws)
        converged = model.successB & k<50
        max_step = 0.75*max_step if max_step < np.inf else 100
        currSolver_kws['max_step'] = max_step
        k=k+1

    # Interpolate to the data time grid & calculate error
    t_eval = data['Days']
    residual=0
    for i in range(0,len(dataFit)):
        dF1=dataFit[i]
        mF1=modelFit[i]
        f1 = scipy.interpolate.interp1d(model.resultsDf.Days,model.resultsDf[mF1],fill_value="extrapolate")
        modelPredictionLin = f1(t_eval)
        dataLin=data[dF1]
        dataLog=np.log10(dataLin)
        errLin = np.abs(dataLin-modelPredictionLin)
        errLin[np.isnan(errLin)] = np.nanmax(errLin)
        if(np.all(modelPredictionLin>0)):
            modelPredictionLog=np.log10(modelPredictionLin)
            modelPredictionLog[np.isinf(modelPredictionLog)] = 500
            errLog = 10**(np.abs((dataLog-modelPredictionLog)))
            errLog[np.isnan(errLog)] = np.nanmax(errLog)
        else:
            errLog=1000*errLin
        err = errLog
        residual = residual+fitWeight[i]*err    
    return residual

    

    