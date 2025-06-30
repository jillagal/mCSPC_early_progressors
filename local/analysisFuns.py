import numpy as np
import pandas as pd
import os
import scipy
import math
import random
import sys  #system talk
from lmfit import minimize, Parameters
import matplotlib.pyplot as plt 
import seaborn as sns

sys.path.append("./local")
import myUtils as utils
#import myFitFuns as fitFuns
import odeFuns as odeFuns

#from my_fitting import Fitting


def loadData(pathToData,patientId,cycle):
    patientDataDf = pd.read_excel(pathToData, sheet_name=str(patientId),header=1)
    # tempDD=patientDataDf.Date-patientDataDf.Date[0]
    # patientDataDf['Date']=pd.to_numeric(tempDD.dt.days, downcast='integer')
    dropCols=['Notes','Gleason','primary Tx','etc','cfDNA','cfRNA','AR CNV','AR-V7','AR T878A']
    for col in dropCols:
        patientDataDf.drop(col, axis=1, inplace=True)
    mapping = {patientDataDf.columns[0]:'Date',patientDataDf.columns[1]:'Cycle', patientDataDf.columns[2]: 'PSA', patientDataDf.columns[3]:'Testosterone', patientDataDf.columns[4]: 'Lup',patientDataDf.columns[5]:'Abi', 
            patientDataDf.columns[6]: 'Enza',patientDataDf.columns[7]:'Apa', patientDataDf.columns[8]: 'BMs',patientDataDf.columns[9]:'LNs', patientDataDf.columns[10]: 'other'}
    patientDataDf.rename(columns=mapping,inplace=True) 
    #patientDataDf['Time'] = patientDataDf['Days'].astype(float)
    patientDataDf['PSA_normalised'] = patientDataDf['PSA'].astype(float)/float(patientDataDf.PSA.iloc[0])
    patientDataDf['PSA_logN'] = np.log10(patientDataDf.PSA_normalised)
    patientDataDf['PSA_log'] = np.log10(patientDataDf['PSA'].astype(float))
    patientDataDf['PSA'] = patientDataDf['PSA'].astype(float)
    patientDataDf['Testosterone'] = patientDataDf['Testosterone'].astype(float)
    patientDataDf['Testosterone_normalized'] = patientDataDf['Testosterone'].astype(float)/float(patientDataDf.Testosterone.iloc[0])
    patientDataDf['Testosterone_log'] = np.log10(patientDataDf['Testosterone'].astype(float))#/T_norm
    patientDataDf['Lup'] = np.nan_to_num(patientDataDf['Lup'], copy=True, nan=0.0, posinf=None, neginf=None)
    patientDataDf['Abi'] = np.nan_to_num(patientDataDf['Abi'], copy=True, nan=0.0, posinf=None, neginf=None)
    patientDataDf['BMs'] = patientDataDf['BMs']
    patientDataDf['LNs'] = patientDataDf['LNs']
    patientDataDf['other'] = patientDataDf['other']
    patientDataDf['mets'] =  patientDataDf['BMs']+ patientDataDf['LNs']+ patientDataDf['other']

    ppp=patientDataDf[patientDataDf['Cycle']==1]
    patientDataDf['DaysOG']=patientDataDf['Date']-ppp['Date'].iloc[0]

    patientDataDfC=patientDataDf.copy()
    patientDataDfC2=patientDataDf.copy()
    if(cycle!='all'):
        patientDataDfC=patientDataDfC[patientDataDfC['Cycle']==int(cycle)]
        patientDataDfC = patientDataDfC.reset_index()
        if(int(cycle)+1 in patientDataDfC2['Cycle'].values):
            patientDataDfC2=patientDataDfC2[patientDataDfC2['Cycle']==int(cycle)+1]
            patientDataDfC.loc[len(patientDataDfC['Date'])] = patientDataDfC2.iloc[0]
        elif(99 in patientDataDfC2['Cycle'].values and int(cycle)!=99):
            patientDataDfC2=patientDataDfC2[patientDataDfC2['Cycle']==99]
            patientDataDfC.loc[len(patientDataDfC['Date'])] = patientDataDfC2.iloc[0]
        else:
            patientDataDfC2=[]
        patientDataDfC = patientDataDfC.reset_index()

    startDate=patientDataDfC['Date'].iloc[0]
    patientDataDfC['Days']=patientDataDfC.apply(lambda x: (x['Date']-startDate).days,axis=1)
    patientDataDfC['Time']=patientDataDfC['Days']
    #initial T
    if(np.isnan(patientDataDfC['Testosterone'].iloc[0])):
        if(0 in patientDataDf['Cycle'].values):
            dataDf0=patientDataDf.copy()
            dataDf0=dataDf0[dataDf0['Cycle']==0]
            tmpT=np.nanmax(dataDf0['Testosterone'])
            if(np.isnan(np.nanmax(dataDf0['Testosterone']))):
                tmpT=np.nanmax(patientDataDfC['Testosterone'])
        else:
            tmpT=np.nanmax(patientDataDfC['Testosterone'])
    else:
        tmpT=patientDataDfC['Testosterone'].iloc[0]
    T0=tmpT
    patientDataDfC.loc[0,'Testosterone']=T0
    tmpP=0
    if(np.isnan(patientDataDfC['PSA'].iloc[0])):
        if(0 in patientDataDf['Cycle'].values):
            dataDf0=patientDataDf.copy()
            dataDf0=dataDf0[dataDf0['Cycle']==0]
            tmpP=np.nanmean(dataDf0['PSA'])
            if(np.isnan(tmpP)):
                tmpP=np.nanmean(patientDataDf['PSA'])
            # else:
            #     tmp=np.nanmean(patientDataDf['PSA'])
    else:
        tmpP=patientDataDfC['PSA'].iloc[0]
        
    PSA0=tmpP
    patientDataDfC.loc[0,'PSA']=PSA0

    bstr=[]
    for i in range(0,len(patientDataDfC['Lup'])):
        b=str(patientDataDfC['Lup'][i])+str(patientDataDfC['Abi'][i])+str(patientDataDfC['Enza'][i])+str(patientDataDfC['Apa'][i])
        bstr.append(int(b,2))
    patientDataDfC['DrugIndex']=bstr

    # if(index0>0):
    #     patientDataDf = patientDataDf.drop(range(0,index0))
    #     patientDataDf = patientDataDf.reset_index()

    return patientDataDfC

def defineInitValues(patientIdDic,dataDir,dataName,testoOnly,psaOnly,outDir,prevFits):
    tmp=[]
    for p in patientIdDic.keys():
        patientId = patientIdDic[p]
        if(dataName == 'mCSPC'):
            patientDataDf = pd.read_excel(dataDir, sheet_name=str(patientId), header=1)
            dropCols=['Notes','Gleason','primary Tx','etc','cfDNA','cfRNA','AR CNV','AR-V7','AR T878A']
            for col in dropCols:
                patientDataDf.drop(col, axis=1, inplace=True)
            mapping = {patientDataDf.columns[0]:'Date', patientDataDf.columns[1]: 'Cycle',patientDataDf.columns[2]: 'PSA', patientDataDf.columns[3]:'Testosterone', patientDataDf.columns[4]: 'Lup',patientDataDf.columns[5]:'Abi', 
                patientDataDf.columns[6]: 'Enza',patientDataDf.columns[7]:'Apa', patientDataDf.columns[8]: 'BMs',patientDataDf.columns[9]:'LNs', patientDataDf.columns[10]: 'other'}
            patientDataDf.rename(columns=mapping,inplace=True) 
            fact=1
        else:
            path_to_file = os.path.join(dataDir, "patient%.3d.txt" % patientId)
            patientDataDf = pd.read_csv(path_to_file, header=None)
            patientDataDf.rename(columns={0: "PatientId", 1: "Date", 2: "CPA", 3: "LEU", 4: "PSA", 5: "Testosterone",
                                  6: "CycleId", 7: "Abi"}, inplace=True)
            fact=28.8

        patientDataDf1=patientDataDf.copy()
        patientDataDf1=patientDataDf1[patientDataDf1['Cycle']==1]
        patientDataDf1 = patientDataDf1.reset_index()
        startDate=patientDataDf1['Date'].iloc[0]
        patientDataDf1['Days']=patientDataDf1.apply(lambda x: (x['Date']-startDate).days,axis=1)

        #initial T
        if(np.isnan(patientDataDf1['Testosterone'].iloc[0])):
            if(0 in patientDataDf['Cycle'].values):
                dataDf0=patientDataDf.copy()
                dataDf0=dataDf0[dataDf0['Cycle']==0]
                tmpT=np.nanmax(dataDf0['Testosterone'])
                if(np.isnan(np.nanmax(dataDf0['Testosterone']))):
                    tmpT=np.nanmax(patientDataDf1['Testosterone'])
            else:
                tmpT=np.nanmax(patientDataDf1['Testosterone'])
        else:
            tmpT=patientDataDf1['Testosterone'].iloc[0]
        T0=tmpT
        #patientDataDf.loc[0,'Testosterone']=T0
        tmpP=0
        if(np.isnan(patientDataDf1['PSA'].iloc[0])):
            if(0 in patientDataDf['Cycle'].values):
                dataDf0=patientDataDf.copy()
                dataDf0=dataDf0[dataDf0['Cycle']==0]
                tmpP=np.nanmean(dataDf0['PSA'])
                if(np.isnan(tmpP)):
                    tmpP=np.nanmean(patientDataDf1['PSA'])
                # else:
                #     tmpP=np.nanmean(patientDataDf1['PSA'])
        else:
            tmpP=patientDataDf1['PSA'].iloc[0]
        
        PSA0=tmpP
        #patientDataDf.loc[0,'PSA']=PSA0

        # if (np.isnan(float(patientDataDf.Testosterone.iloc[0]))):
        #     T_norm = max(patientDataDf.Testosterone.dropna())
        # else:
        #     T_norm = patientDataDf.Testosterone.iloc[0]
        # T_norm = fact*T_norm
        # T_min = fact*min(patientDataDf.Testosterone)
        # PSA0 = patientDataDf.PSA.iloc[0]
        #list values...
        tmp0=[]
        if(testoOnly):
            #tmp0=[T_norm]
            tmp0=[T0]
        elif(psaOnly):
            tmp0=[PSA0]
        else:
            tmp0=[T0,PSA0,PSA0]
            #tmp0=[T0,PSA0]
            #tmp0=[T_norm,PSA0,PSA0]
        #if(prevFits[0]==0):
            #tmp.append(tmp0)
        if(prevFits[0]==3):
            path1='initPars/pars'+patientId+'.xlsx'
            parsPat = pd.read_excel(path1,header=0)
            parsPat.columns=['par','value']
            for pi,p in enumerate(parsPat['par']):
                tmp0=tmp0+[parsPat['value'].iloc[pi]]
        else:
            if(prevFits[0]>0):
                path1='initPars/parsT'+patientId+'.xlsx'
                parsPatT = pd.read_excel(path1,header=0)
                parsPatT.columns=['par','value']
                for pi,p in enumerate(parsPatT['par']):
                    tmp0=tmp0+[parsPatT['value'].iloc[pi]]
                #tmp.append(tmp0)
            if(prevFits[1]>0):
                path1='initPars/parsT2'+patientId+'.xlsx'
                parsPatT2 = pd.read_excel(path1,header=0)
                parsPatT2.columns=['par','value']
                for pi,p in enumerate(parsPatT2['par']):
                    tmp0=tmp0+[parsPatT2['value'].iloc[pi]]
                #tmp.append(tmp0)
            if(prevFits[2]>0):
                path1='initPars/parsP'+patientId+'.xlsx'
                parsPatP = pd.read_excel(path1,header=0)
                parsPatP.columns=['par','value']
                for pi,p in enumerate(parsPatP['par']):
                    tmp0=tmp0+[parsPatP['value'].iloc[pi]]
            #tmp.append(tmp0)
        tmp.append(tmp0)

    # # list columns/names
    cols=[]
    if(testoOnly):
        cols=cols + ['T0']
    elif(psaOnly):
        cols=cols + ['P0']
    else:
        cols=cols + ['T0','P0','C0'] #['T0','P0']#
    if(prevFits[0]==3):
        for p in parsPat['par']:
            cols=cols+[p]
    else:
        if(prevFits[0]>0):
            for p in parsPatT['par']:
                cols=cols+[p]
        if(prevFits[1]>0):
            for p in parsPatT2['par']:
                cols=cols+[p]
        if(prevFits[2]>0):
            for p in parsPatP['par']:
                cols=cols+[p] 
    lArray = pd.DataFrame(tmp,columns=cols)
    return lArray

def defineInitValuesNew(patientIdDic,dataDir,dataName,testoOnly,psaOnly,outDir,prevFits,my_fit):
    tmp=[]
    for p in patientIdDic.keys():
        patientId = patientIdDic[p]
        if(dataName == 'mCSPC'):
            patientDataDf = pd.read_excel(dataDir, sheet_name=str(patientId), header=1)
            dropCols=['Notes','Gleason','primary Tx','etc','cfDNA','cfRNA','AR CNV','AR-V7','AR T878A']
            for col in dropCols:
                patientDataDf.drop(col, axis=1, inplace=True)
            mapping = {patientDataDf.columns[0]:'Date', patientDataDf.columns[1]: 'Cycle',patientDataDf.columns[2]: 'PSA', patientDataDf.columns[3]:'Testosterone', patientDataDf.columns[4]: 'Lup',patientDataDf.columns[5]:'Abi', 
                patientDataDf.columns[6]: 'Enza',patientDataDf.columns[7]:'Apa', patientDataDf.columns[8]: 'BMs',patientDataDf.columns[9]:'LNs', patientDataDf.columns[10]: 'other'}
            patientDataDf.rename(columns=mapping,inplace=True) 
            fact=1
        else:
            path_to_file = os.path.join(dataDir, "patient%.3d.txt" % patientId)
            patientDataDf = pd.read_csv(path_to_file, header=None)
            patientDataDf.rename(columns={0: "PatientId", 1: "Date", 2: "CPA", 3: "LEU", 4: "PSA", 5: "Testosterone",
                                  6: "CycleId", 7: "Abi"}, inplace=True)
            fact=28.8

        patientDataDf1=patientDataDf.copy()
        patientDataDf1=patientDataDf1[patientDataDf1['Cycle']==1]
        patientDataDf1 = patientDataDf1.reset_index()
        startDate=patientDataDf1['Date'].iloc[0]
        patientDataDf1['Days']=patientDataDf1.apply(lambda x: (x['Date']-startDate).days,axis=1)

        #initial T
        if(np.isnan(patientDataDf1['Testosterone'].iloc[0])):
            if(0 in patientDataDf['Cycle'].values):
                dataDf0=patientDataDf.copy()
                dataDf0=dataDf0[dataDf0['Cycle']==0]
                tmpT=np.nanmax(dataDf0['Testosterone'])
                if(np.isnan(np.nanmax(dataDf0['Testosterone']))):
                    tmpT=np.nanmax(patientDataDf1['Testosterone'])
            else:
                tmpT=np.nanmax(patientDataDf1['Testosterone'])
        else:
            tmpT=patientDataDf1['Testosterone'].iloc[0]
        T0=tmpT
        #patientDataDf.loc[0,'Testosterone']=T0
        tmpP=0
        if(np.isnan(patientDataDf1['PSA'].iloc[0])):
            if(0 in patientDataDf['Cycle'].values):
                dataDf0=patientDataDf.copy()
                dataDf0=dataDf0[dataDf0['Cycle']==0]
                tmpP=np.nanmean(dataDf0['PSA'])
                if(np.isnan(tmpP)):
                    tmpP=np.nanmean(patientDataDf1['PSA'])
                # else:
                #     tmpP=np.nanmean(patientDataDf1['PSA'])
        else:
            tmpP=patientDataDf1['PSA'].iloc[0]
        
        PSA0=tmpP
        #patientDataDf.loc[0,'PSA']=PSA0

        # if (np.isnan(float(patientDataDf.Testosterone.iloc[0]))):
        #     T_norm = max(patientDataDf.Testosterone.dropna())
        # else:
        #     T_norm = patientDataDf.Testosterone.iloc[0]
        # T_norm = fact*T_norm
        # T_min = fact*min(patientDataDf.Testosterone)
        # PSA0 = patientDataDf.PSA.iloc[0]
        #list values...
        tmp0=[]
        if('T' in my_fit.fitting[0]):
            tmp0=tmp0+T0
        if('PSA' in my_fit.fitting[0]):
            tmp0=tmp0+PSA0

        if(prevFits[0]==3):
            path1='initPars/pars'+patientId+'.xlsx'
            parsPat = pd.read_excel(path1,header=0)
            parsPat.columns=['par','value']
            for pi,p in enumerate(parsPat['par']):
                tmp0=tmp0+[parsPat['value'].iloc[pi]]
        else:
            if(prevFits[0]>0):
                path1='initPars/parsT'+patientId+'.xlsx'
                parsPatT = pd.read_excel(path1,header=0)
                parsPatT.columns=['par','value']
                for pi,p in enumerate(parsPatT['par']):
                    tmp0=tmp0+[parsPatT['value'].iloc[pi]]
                #tmp.append(tmp0)
            if(prevFits[1]>0):
                path1='initPars/parsT2'+patientId+'.xlsx'
                parsPatT2 = pd.read_excel(path1,header=0)
                parsPatT2.columns=['par','value']
                for pi,p in enumerate(parsPatT2['par']):
                    tmp0=tmp0+[parsPatT2['value'].iloc[pi]]
                #tmp.append(tmp0)
            if(prevFits[2]>0):
                path1='initPars/parsP'+patientId+'.xlsx'
                parsPatP = pd.read_excel(path1,header=0)
                parsPatP.columns=['par','value']
                for pi,p in enumerate(parsPatP['par']):
                    tmp0=tmp0+[parsPatP['value'].iloc[pi]]
            #tmp.append(tmp0)
        tmp.append(tmp0)
    print(tmp)

    # # list columns/names
    cols=[]
    if('T' in my_fit.fitting[0]):
        cols=cols + ['T0']
    if('PSA' in my_fit.fitting[0]):
        cols=cols + ['P0']

    if(prevFits[0]==3):
        for p in parsPat['par']:
            cols=cols+[p]
    else:
        if(prevFits[0]>0):
            for p in parsPatT['par']:
                cols=cols+[p]
        if(prevFits[1]>0):
            for p in parsPatT2['par']:
                cols=cols+[p]
        if(prevFits[2]>0):
            for p in parsPatP['par']:
                cols=cols+[p] 
    print(cols)
    lArray = pd.DataFrame(tmp,columns=cols)
    return lArray

def defineInitValuesNewNew(patientIdDic,dataDir,dataName,outDir,my_fit,parName,modelName):
    tmp=[]
    for p in patientIdDic.keys():
        patientId = patientIdDic[p]
        if(dataName == 'mCSPC'):
            patientDataDf = pd.read_excel(dataDir, sheet_name=str(patientId), header=1)
            dropCols=['Notes','Gleason','primary Tx','etc','cfDNA','cfRNA','AR CNV','AR-V7','AR T878A']
            for col in dropCols:
                patientDataDf.drop(col, axis=1, inplace=True)
            mapping = {patientDataDf.columns[0]:'Date', patientDataDf.columns[1]: 'Cycle',patientDataDf.columns[2]: 'PSA', patientDataDf.columns[3]:'Testosterone', patientDataDf.columns[4]: 'Lup',patientDataDf.columns[5]:'Abi', 
                patientDataDf.columns[6]: 'Enza',patientDataDf.columns[7]:'Apa', patientDataDf.columns[8]: 'BMs',patientDataDf.columns[9]:'LNs', patientDataDf.columns[10]: 'other'}
            patientDataDf.rename(columns=mapping,inplace=True) 
            fact=1
        else:
            path_to_file = os.path.join(dataDir, "patient%.3d.txt" % patientId)
            patientDataDf = pd.read_csv(path_to_file, header=None)
            patientDataDf.rename(columns={0: "PatientId", 1: "Date", 2: "CPA", 3: "LEU", 4: "PSA", 5: "Testosterone",
                                  6: "CycleId", 7: "Abi"}, inplace=True)
            fact=28.8

        patientDataDf1=patientDataDf.copy()
        patientDataDf1=patientDataDf1[patientDataDf1['Cycle']==1]
        patientDataDf1 = patientDataDf1.reset_index()
        startDate=patientDataDf1['Date'].iloc[0]
        patientDataDf1['Days']=patientDataDf1.apply(lambda x: (x['Date']-startDate).days,axis=1)

        #initial T
        if(np.isnan(patientDataDf1['Testosterone'].iloc[0])):
            if(0 in patientDataDf['Cycle'].values):
                dataDf0=patientDataDf.copy()
                dataDf0=dataDf0[dataDf0['Cycle']==0]
                tmpT=np.nanmax(dataDf0['Testosterone'])
                if(np.isnan(np.nanmax(dataDf0['Testosterone']))):
                    tmpT=np.nanmax(patientDataDf1['Testosterone'])
            else:
                tmpT=np.nanmax(patientDataDf1['Testosterone'])
        else:
            tmpT=patientDataDf1['Testosterone'].iloc[0]
        T0=tmpT
        #patientDataDf.loc[0,'Testosterone']=T0
        tmpP=0
        if(np.isnan(patientDataDf1['PSA'].iloc[0])):
            if(0 in patientDataDf['Cycle'].values):
                dataDf0=patientDataDf.copy()
                dataDf0=dataDf0[dataDf0['Cycle']==0]
                tmpP=np.nanmean(dataDf0['PSA'])
                if(np.isnan(tmpP)):
                    tmpP=np.nanmean(patientDataDf1['PSA'])
                # else:
                #     tmpP=np.nanmean(patientDataDf1['PSA'])
        else:
            tmpP=patientDataDf1['PSA'].iloc[0]
        
        PSA0=tmpP
        #patientDataDf.loc[0,'PSA']=PSA0

        # if (np.isnan(float(patientDataDf.Testosterone.iloc[0]))):
        #     T_norm = max(patientDataDf.Testosterone.dropna())
        # else:
        #     T_norm = patientDataDf.Testosterone.iloc[0]
        # T_norm = fact*T_norm
        # T_min = fact*min(patientDataDf.Testosterone)
        # PSA0 = patientDataDf.PSA.iloc[0]
        #list values...
        tmp0=[]
        if('Testosterone' in my_fit.fitting[0]):
            tmp0=tmp0+[T0]
        if('PSA' in my_fit.fitting[0]):
            tmp0=tmp0+[PSA0]

        if(len(parName)>0):
            parsFile=os.path.join(outDir,modelName,parName+modelName+str(patientId)+'.xlsx')
            parsPat = pd.read_excel(parsFile,header=0)
            # print(parsPat.columns)
            if('Unnamed: 0' in parsPat.columns):
                parsPat=parsPat.drop('Unnamed: 0',axis=1)
            # parsPat=parsPat.drop('Unnamed: 0')
            #print(parsPat)
            # parsPat.columns=['par','value']
            # for pi,p in enumerate(parsPat['par']):
            #     tmp0=tmp0+[parsPat['value'].iloc[pi]]
            for p in parsPat.columns:
                #print(parsPat[p].iloc[0])
                tmp0=tmp0+[parsPat[p].iloc[0]]
        tmp.append(tmp0)

    # # list columns/names
    cols=[]
    if('Testosterone' in my_fit.fitting[0]):
        cols=cols + ['T0']
    if('PSA' in my_fit.fitting[0]):
        cols=cols + ['P0']
    if(len(parName)>0):
        # for p in parsPat['par']:
        #     cols=cols+[p]
        for p in parsPat.columns:
            cols=cols+[p]

    lArray = pd.DataFrame(tmp,columns=cols)
    return lArray


def getParamsFromFile(patientIdDic,paramFile):
    tmp=[]
    for p in patientIdDic.keys():
        patientId = patientIdDic[p]

        tmp0=[]
        path1='initPars/'+paramFile+patientId+'.xlsx'
        parsPat = pd.read_excel(path1,header=0)
        parsPat.columns=['par','value']
        for pi,p in enumerate(parsPat['par']):
            tmp0=tmp0+[parsPat['value'].iloc[pi]]
        tmp.append(tmp0)

    # # list columns/names
    cols=[]
    for p in parsPat['par']:
        cols=cols+[p]
    lArray = pd.DataFrame(tmp,columns=cols)
    return lArray

def residualPCa(params, data, fitting, tx, txCols, boolTinput,model, solver_kws={}):#threshold,
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
        #f1T = scipy.interpolate.interp1d(data.Days[~nanT],data['Testosterone'][~nanT],fill_value="extrapolate",kind="linear")#"extrapolate"
        f1T = scipy.interpolate.interp1d(data.Days[~nanT],data['Tlog'][~nanT],fill_value="extrapolate",kind="linear")#"extrapolate"
        teval=range(0,data['Days'][len(data['Days'])-1])
        testo=f1T(teval)
        testo=10**testo
        i=0
        tx=[]
        for d in teval:
            tx.append([d,d+1,data['Lup'].loc[i],data['Abi'].loc[i],data['Enza'].loc[i],data['Apa'].loc[i]])
            if(d>=data['Days'].loc[i+1]):
                i=i+1 

    while not converged:
        model.Simulate(txCols,treatmentScheduleList=tx,**currSolver_kws)
        converged = model.successB & k<50
        max_step = 0.75*max_step if max_step < np.inf else 100
        currSolver_kws['max_step'] = max_step
        k=k+1
    # Interpolate to the data time grid & calculate error
    t_eval = data.Time

    residual=0
    for i in range(0,len(dataFit)):
        dF1=dataFit[i]
        mF1=modelFit[i]
        f1 = scipy.interpolate.interp1d(model.resultsDf.Time,model.resultsDf[mF1],fill_value="extrapolate")
        modelPrediction1 = f1(t_eval)
        data1=data[dF1]

        pp=0.2*np.ones(len(data1))
        pp[0]=0.8
        pp[len(pp)-1]=0.8
        pp[data1==np.nanmin(data1)]=0.8

        result = np.where(modelPrediction1 > 0.0000001, modelPrediction1, -100)
        modelPrediction1=np.log10(result, out=result, where=result > 0)
        modelPrediction1[np.isinf(modelPrediction1)] = -100

        data1=np.log10(data1)
        # print(data1[0])
        # print(data1[len(data1)-1])
        # #print(data1[-1])
        # print(np.nanmin(data1))

        # print(data1)
        # print(modelPrediction1)
        err1 = np.abs((data1-modelPrediction1)/data1) 
        #err1[data1==np.nanmin(data1)]=0
        ######err1[data1<detectThreshold[i]]=0.2*np.abs((data1-modelPrediction1)/data1) 
        #err1[modelPrediction1<.00001]=10*err1[modelPrediction1<.00001]
        #err1[np.isnan(err1)] = 0.5*err1[np.isnan(err1)]
        err1[np.isnan(err1)] = np.nanmax(err1)
        err1 = err1*pp
        #print(err1)
        residual = residual+fitWeight[i]*err1      

    return residual

def plot_dose_over_time(axx,txCols,tx,dataDf,timeCol,xmin,xmax,xmarg):
    cols=['ti','tf']+txCols
    tx=pd.DataFrame(tx,columns=cols)
    for i,drug in enumerate(txCols):
        drugBarPosition = 1-.5*i
        # timeVec = dataDf[timeCol].values
        # tt = {'ti':[tx.tf[len(tx)-1]], 'tf':[timeVec[len(timeVec)-1]]}
        # for txName in txCols:
        #     tt[txName]=[0]
        # tt=pd.DataFrame(tt,columns=cols)
        # tx = pd.concat([tx,tt],ignore_index=True)
        plot_drug(tx,drug,drugBarPosition,axx)#,color=palette[drug])
    # axx.set_xlim((xmin-xmarg, xmax+xmarg))

def plot_dose_ADT_over_time(axx,txCols,tx,dataDf,timeCol,xmin,xmax,xmarg,ymax):
    cols=['ti','tf']+txCols
    tx=pd.DataFrame(tx,columns=cols)
    for i,drug in enumerate(txCols):
        drugBarPosition = (1-.5*i)*ymax
        # timeVec = dataDf[timeCol].values
        # tt = {'ti':[tx.tf[len(tx)-1]], 'tf':[timeVec[len(timeVec)-1]]}
        # for txName in txCols:
        #     tt[txName]=[0]
        # tt=pd.DataFrame(tt,columns=cols)
        # tx = pd.concat([tx,tt],ignore_index=True)
        plot_drug(tx,drug,drugBarPosition,axx)#,color=palette[drug])
    # axx.set_xlim((xmin-xmarg, xmax+xmarg))

def plot_drug(tx,drug,drugBarPosition,ax):  #,color
    drugConcentrationVec=tx[drug]
    drugConcentrationVec = np.array([x/(np.max(drugConcentrationVec)+1e-12) for x in drugConcentrationVec])
    drugConcentrationVec = 0.5*drugConcentrationVec + drugBarPosition
    axName=ax
    axName.fill_between(tx['ti'], drugConcentrationVec, drugBarPosition, 
                        step="post", alpha=0.6, label=drug)#color=color, 
    
def runSim(tmpModel,patientId,patInd,lArray,parDicAdd,tx,my_fit,boolTinput,testo,solver_kws):

    print(patientId)

    params = Parameters() #initialize parameters
    #print(vars0[0])
    params=utils.setConParamsFromDictionary(tmpModel.varsC,params,tmpModel.dictionary)
    #print(params)
    for par in lArray.columns:
        params.add(par, value=lArray[par][patInd],vary=False)

    for par in parDicAdd.keys():
        params.add(par, value=parDicAdd[par],vary=False)

    #if(modelName!='modT' and modelName!='modT1C'):
    ####set splits####
    [consSet,consSplit,outsSplit] = utils.getSplits(tmpModel)
    #set splits
    params = utils.setSplits(consSplit,lArray,patInd,params,outsSplit)
    print(params)


    tmpModel.SetParams(**params.valuesdict())

    #tx=[[0,tx[len(tx)-1][0],1,0,0,0],[tx[len(tx)-1][0],tx[len(tx)-1][0]+1,0,0,0,0]]
    if(boolTinput):
        tmpModel.SimulateWithT(testo,treatmentScheduleList=tx,**solver_kws)
    else:
        tmpModel.Simulate(treatmentScheduleList=tx,**solver_kws)

    topDf = tmpModel.resultsDf
    return topDf


def plotmodSX(dataDf,topDf,txCols,tx,colorPalette,modelName,logBool,topDfCT,CTbool):
    fig, axes = plt.subplots(3, 1, figsize=(10,8),gridspec_kw={'height_ratios': [3,3,3]})
    #plot_dose_over_time(axes[0],txCols,tx,dataDf,timeCol,[],[],[])

    currDrugBarPosition = dataDf['Testosterone'].min()
    drugBarHeight = dataDf['Testosterone'].max()-dataDf['Testosterone'].min()
    txDf=pd.DataFrame(tx,columns=['t0','tf','GnRH','Abi','ARSI'])
    txDf['Tdrug']=txDf['GnRH']
    #dataDf['Tdrug']=dataDf['Lup']
    for ti,tt in enumerate(txDf['t0']):
        if(txDf['Abi'][ti]==1):
            txDf.loc[ti,'Tdrug']=2
    # for drug in txCols[0:2]:
    #     drugConcentrationVec=txDf[drug]
    #     #drugConcentrationVec[drugConcentrationVec < 0] = 0
    #     drugConcentrationVec = drugConcentrationVec * drugBarHeight + currDrugBarPosition
    #     if(drug=='Lup'):
    #         axes[0].fill_between(txDf['t0'], currDrugBarPosition, drugConcentrationVec, step="post",color='blue', alpha=0.2)
    #     else:
    #         axes[0].fill_between(txDf['t0'], currDrugBarPosition, drugConcentrationVec, step="post",color='red', alpha=0.2)

    pp=sns.lineplot(x='Days',y='T',lw=3, alpha=0.8,legend='full',color='k', data=topDf,ax=axes[0])
    axes[0].scatter(x=dataDf['Days'],y=dataDf['Testosterone'],s=100,color='k')
    #axes[0].set_xlim([0,tMax])
    if(logBool):
        axes[0].set(yscale='log')
        axes[0].set_ylim([1,1.1*np.max(dataDf['Testosterone'])])
    else:
        axes[0].set_ylim([-10,1.1*np.max(dataDf['Testosterone'])])
    #axes[0].set_xlim(-1,dataDf['Time'].max())

    qq=sns.lineplot(x='Days',y='PSAi',lw=3, alpha=0.8,color='k',legend='full', data=topDf, linestyle='--',ax=axes[1])#
    axes[1].scatter(x=dataDf['Days'],y=dataDf['PSA'],s=100,color='k')#,ax=axes[0]
    if(logBool):
        axes[1].set(yscale='log')


    pp=sns.lineplot(x='Days',y='TumorSize',lw=3, alpha=0.5,legend='full',color=colorPalette['TumorSize'], data=topDf,ax=axes[2])
    pp=sns.lineplot(x='Days',y='S',lw=3, alpha=0.5,legend='full',color=colorPalette['S'], data=topDf,ax=axes[2])
    if(logBool):
        axes[2].set(yscale='log')

# def plotData(dataDf,topDf,txCols,tx,timeCol,colorPalette,modelName,logBool,topDfCT,CTbool):
#     fig, axes = plt.subplots(2, 1, figsize=(5,4),gridspec_kw={'height_ratios': [3,3]})
#     #plot_dose_over_time(axes[0],txCols,tx,dataDf,timeCol,[],[],[])

#     currDrugBarPosition = dataDf['Testosterone'].min()
#     drugBarHeight = dataDf['Testosterone'].max()-dataDf['Testosterone'].min()
#     txDf=pd.DataFrame(tx,columns=['t0','tf','Lup','Abi','Enza','Apa'])
#     txDf['Tdrug']=txDf['Lup']
#     #dataDf['Tdrug']=dataDf['Lup']
#     for ti,tt in enumerate(txDf['t0']):
#         if(txDf['Abi'][ti]==1):
#             txDf.loc[ti,'Tdrug']=2
#     # for ti,tt in enumerate(dataDf[timeCol]):
#     #     if(dataDf['Abi'][ti]==1):
#     #         dataDf.loc[ti,'Tdrug']=2
#     #         #dataDf['Tdrug'][ti]=2  
#     # for drug in txCols[0:2]:
#     #     drugConcentrationVec=dataDf[drug]
#     #     #drugConcentrationVec[drugConcentrationVec < 0] = 0
#     #     drugConcentrationVec = drugConcentrationVec * drugBarHeight + currDrugBarPosition
#     #     if(drug=='Lup'):
#     #         axes[0].fill_between(dataDf[timeCol], currDrugBarPosition, drugConcentrationVec, step="post",color='blue', alpha=0.2)
#     #     else:
#     #         axes[0].fill_between(dataDf[timeCol], currDrugBarPosition, drugConcentrationVec, step="post",color='red', alpha=0.2)
#     for drug in txCols[0:2]:
#         drugConcentrationVec=txDf[drug]
#         #drugConcentrationVec[drugConcentrationVec < 0] = 0
#         drugConcentrationVec = drugConcentrationVec * drugBarHeight + currDrugBarPosition
#         if(drug=='Lup'):
#             axes[0].fill_between(txDf['t0'], currDrugBarPosition, drugConcentrationVec, step="post",color='blue', alpha=0.2)
#         else:
#             axes[0].fill_between(txDf['t0'], currDrugBarPosition, drugConcentrationVec, step="post",color='red', alpha=0.2)

#     pp=sns.lineplot(x='Time',y='Testosterone',lw=3, alpha=0.8,legend='full',color=colorPalette['T'], data=dataDf,ax=axes[0])
#     axes[0].scatter(x=dataDf['Time'],y=dataDf['Testosterone'],s=100,color=colorPalette['Testosterone'])
#     #axes[0].set_xlim([0,tMax])
#     axes[0].set_ylim([-10,1.1*np.max(dataDf['Testosterone'])])
#     if(logBool):
#         axes[0].set(yscale='log')
#     #axes[0].set_xlim(-1,dataDf['Time'].max())

#     qq=sns.lineplot(x='Time',y='PSA',lw=3, alpha=0.8,color='k',legend='full', data=dataDf, linestyle='--',ax=axes[1])#
#     axes[1].scatter(x=dataDf['Time'],y=dataDf['PSA'],s=100,color='k')#,ax=axes[0]
#     if(logBool):
#         axes[1].set(yscale='log')


################
# Data analysis
################

def analyzeData(dataFileName,patientIdDic):

    for pi,patientId in enumerate(patientIdDic.keys()):
        dataDf=pd.read_excel(dataFileName,sheet_name=patientIdDic[patientId], header=1)
        tdata = pd.DataFrame()
        cols=['Date','PSA','testosterone','GnRH','Abiraterone','Enzalutamide','Apalutamide']
        for col in cols:
            tdata[col]=dataDf[col]
        index0 = (dataDf['GnRH'] == 1).idxmax()
        startDate=dataDf['Date'].iloc[index0]
        tdata['Days']=dataDf.apply(lambda x: (x['Date']-startDate).days,axis=1)
        psa0=10*(math.log(tdata['PSA'].iloc[index0])+1)


         #adjust for initial conditions################
        inds=range(index0,len(dataDf['GnRH']))
        tdata=tdata.iloc[inds].reset_index()
        tchange=np.zeros(len(tdata['testosterone']))
        tdata['tchange']=tchange
        if(math.isnan(tdata['testosterone'][0])):
            tdata['testosterone'][0]=np.nanmax(tdata['testosterone'])
        nanP=np.isnan(tdata['PSA'])
        nanT=np.isnan(tdata['testosterone'])
        f1P = scipy.interpolate.interp1d(tdata.Days[~nanP],tdata['PSA'][~nanP],fill_value="extrapolate",kind="previous")
        f1T = scipy.interpolate.interp1d(tdata.Days[~nanT],tdata['testosterone'][~nanT],fill_value="extrapolate",kind="previous")#"extrapolate"
        teval=tdata['Days']
        tdata['TestoInterp']=f1T(teval)
        tdata['PSAInterp']=f1P(teval)

        bstr=[]
        bTstr=[]
        indCol=[]
        for i in range(0,len(tdata['GnRH'])):
            b=str(int(tdata['GnRH'][i]))+str(int(tdata['Abiraterone'][i]))
            bT=str(tdata['GnRH'][i])+str(tdata['Abiraterone'][i])+str(tdata['Enzalutamide'][i])+str(tdata['Apalutamide'][i])
            bstr.append(int(b,2))
            bTstr.append(int(bT,2))
        tdata['DrugIndex']=bstr
        indOn0=np.nonzero(bstr)[0][0]
        drugOnTimes=[0]
        drugOffTimes=[]
        indOff=0
        indOn=0
        #indOn0=0
        bstr0=bstr
        while len(bstr0)>1:
            if(0 in bstr0):
                indOff0=bstr0.index(0)
                indOff=indOff+indOff0+indOn0
                drugOffTimes.append(indOff)
                bstr0=bstr0[indOff0:]
                if(sum(bstr0)>0):
                    indOn0=np.nonzero(bstr0)[0][0]
                    indOn=indOn0+indOff
                    drugOnTimes.append(indOn)
                    bstr0=bstr0[indOn0:]
                else:
                    break
            else:
                break
        ton=[]
        toff=[]
        sizes=[]
        for i,cyc in enumerate(drugOffTimes):
            if(len(drugOffTimes)>=i+1 and len(drugOnTimes)>=i+2):
                ton.append(tdata['Days'][drugOffTimes[i]]-tdata['Days'][drugOnTimes[i]])
                toff.append(tdata['Days'][drugOnTimes[i+1]]-tdata['Days'][drugOffTimes[i]])
                sizes.append((i+1)*10)
        #over time - 1st cycle
        ind1=np.where(tdata['Days']==ton[0])[0][0]+1
        ind2=np.where(tdata['Days']==ton[0]+toff[0])[0][0]+1
        tmp3=np.where(tdata['testosterone'].iloc[0:ind2]==np.nanmin(tdata['testosterone'].iloc[0:ind2]))
        if(len(tmp3[0])>=1):
            ind3=tmp3[0][0]
        elif(len(tmp3[0])==1):
            ind3=tmp3[0]
        else:
            ind3=tmp3
        tmp4=np.where(tdata['PSA'].iloc[0:ind2]==np.nanmin(tdata['PSA'].iloc[0:ind2]))
        if(len(tmp4[0])>=1):
            ind4=tmp4[0][0]
        elif(len(tmp4[0])==1):
            ind4=tmp4[0]
        else:
            ind4=tmp4

        ttdata=pd.DataFrame()
        ttdata['Days']=tdata['Days']
        ttdata['testosterone']=tdata['testosterone']
        ttdata['PSA']=tdata['PSA']
        ttdata=ttdata.interpolate()


        # evaluate metrics#############################################
        tnorm=ttdata['testosterone']/ttdata['testosterone'][0]
        tdiff=np.diff(tnorm)
        indi=[i for i,x in enumerate(tdiff) if x>=0][0]
        daysTresp=ttdata['Days'][indi+1]
        daysTlow=ttdata['Days'][ind1]-ttdata['Days'][indi]
        tmpi2=[i for i,x in enumerate(tdiff[ind1-1:ind2]) if (np.abs(x)<.01 and ttdata['testosterone'][i]>50)]
        if(len(tmpi2)>0):
            indi2=0
        else:
            indi2=i
        #indi2=[i for i,x in enumerate(tdiff[ind1-1:ind2]) if np.abs(x)<.01][0]#and tdata['testosterone'][i]>50*np.nanmin(tdata['testosterone'])/tdata['testosterone'][0]
        daysTregrow=(ttdata['Days'][ind1+indi2]-ttdata['Days'][ind1-1])
        daysThigh=ttdata['Days'][ind2]-ttdata['Days'][ind1+indi2-1]
        daysPresp=ttdata['Days'][ind1]
        daysPregrow=ttdata['Days'][ind2]-ttdata['Days'][ind1-1]

        DT_Tresp=daysTresp*np.log(2)/(ttdata['testosterone'][indi+1]-ttdata['testosterone'][0])
        #DT_Tregrow=daysTregrow*np.log(2)/(ttdata['testosterone'][ind1+indi2]-ttdata['testosterone'][ind1-1])
        DT_Presp=daysPresp*np.log(2)/(ttdata['PSA'][ind1]-ttdata['PSA'][0])
        DT_Pregrow=daysPregrow*np.log(2)/(ttdata['PSA'][ind2]-ttdata['PSA'][ind1-1])

        DT_pc_Tresp=ttdata['testosterone'][0]*daysTresp*np.log(2)/(ttdata['testosterone'][indi+1]-ttdata['testosterone'][0])
        #DT_pc_Tregrow=ttdata['testosterone'][ind1-1]*daysTregrow*np.log(2)/(ttdata['testosterone'][ind1+indi2]-ttdata['testosterone'][ind1-1])
        DT_pc_Presp=ttdata['PSA'][0]*daysPresp*np.log(2)/(ttdata['PSA'][ind1]-ttdata['PSA'][0])
        DT_pc_Pregrow=ttdata['PSA'][ind1-1]*daysPregrow*np.log(2)/(ttdata['PSA'][ind2]-ttdata['PSA'][ind1-1])

        rate_Tresp=np.abs(ttdata['testosterone'][indi+1]-ttdata['testosterone'][0])/daysTresp
        rate_Tregrow=np.abs(ttdata['testosterone'][ind1+indi2]-ttdata['testosterone'][ind1-1])/daysTregrow
        rate_Presp=np.abs(ttdata['PSA'][ind1]-ttdata['PSA'][0])/daysPresp
        rate_Pregrow=np.abs(ttdata['PSA'][ind2]-ttdata['PSA'][ind1-1])/daysPregrow



def calculateMetrics(pat,pi,dataDf,dataVals,modVals,fitVals):
    colsDf=[]
    valsDf=[]
    PSA0=dataDf['PSA'].loc[0]
    colsDf.append('PSA0')
    valsDf.append(PSA0)
    nanP=np.isnan(dataDf['PSA'])
    colsDf.append('aveP')
    valsDf.append(np.nanmean(dataDf['PSA']))

    tmpD=dataDf['Days'][len(dataDf['Days'])-1]
    indF=tmpD-1
    teval=range(0,tmpD)
    if(len(dataDf.Days[~nanP])>1):
        f1P = scipy.interpolate.interp1d(dataDf.Days[~nanP],dataDf['PSA'][~nanP],fill_value="extrapolate",kind="linear")
        tmpPSA=f1P(teval)


    #calculated time point indexes
    indFinal=len(dataDf['DrugIndex'])-1
    if(0 in dataDf['DrugIndex'].values and len(dataDf['Days'])>1):
        indStmp=np.where(dataDf['DrugIndex']==0)[0]
        if(len(indStmp)>1):
            indSwitch=indStmp[0]
            indS=dataDf['Days'].iloc[indSwitch]
        elif(len(indStmp)==1):
            if isinstance(indStmp, np.ndarray):
                indSwitch=indStmp[0]
            else:
                indSwitch=indStmp
            indS=dataDf['Days'].iloc[indSwitch]
        else:
            indSwitch=0
            indS=0
    else:
        indSwitch=0
        indS=0

    #cycle data
    if(indSwitch!=0):
        onT=dataDf['Days'][indSwitch]# indent?
        colsDf.append('OnTime')
        valsDf.append(onT/30)
        offT=dataDf['Days'][indFinal]-dataDf['Days'][indSwitch-1]###indSwithc -1????!!!!
        colsDf.append('OffTime')
        valsDf.append(offT/30)#in months
        colsDf.append('On/Off')
        valsDf.append(onT/offT)
    colsDf.append('CycleTime')
    valsDf.append(dataDf['Days'][indFinal]/30)# in months
    
    #PSA metrics
    if(indSwitch!=0):
        P0=tmpPSA[0]
        PS=tmpPSA[indS]
        DS=teval[indS]/30
        colsDf.append('PSA_switch')
        valsDf.append(PS)
        #drug on
        colsDf.append('diffPOnRate')
        valsDf.append((PS-P0)/(DS))
        colsDf.append('diffPOn')
        valsDf.append(PS-P0)
        colsDf.append('diffPOnRatePer')
        valsDf.append((PS-P0)/(P0*DS))
        #valsDf.append((dataDf['PSA'].iloc[indSwitch]-dataDf['PSA'].iloc[0])/(dataDf['PSA'].iloc[0]*dataDf['Days'].iloc[indSwitch]))
        colsDf.append('diffPOnPer')
        valsDf.append((PS-P0)/P0)
        colsDf.append('diffPOnPerPos')
        valsDf.append(100*np.abs(PS-P0)/P0)
        PS=tmpPSA[indS]#dataDf['PSA'].iloc[indSwitch]
        PF=tmpPSA[indF]#dataDf['PSA'].iloc[indFinal]
        delD=(teval[indF]-teval[indS])/30#dataDf['Days'].iloc[indFinal]-dataDf['Days'].iloc[indSwitch]
        #drug off
        colsDf.append('diffPOffRate')
        valsDf.append((PF-PS)/delD)
        colsDf.append('diffPOff')
        valsDf.append(PF-PS)
        colsDf.append('diffPOffRatePer')
        valsDf.append((PF-PS)/(P0*delD))#######was PS in denominator!!!
        colsDf.append('diffPOffPer')
        valsDf.append((PF-PS)/P0)
    P0=tmpPSA[0]#dataDf['PSA'].iloc[0]
    PF=tmpPSA[indF]#dataDf['PSA'].iloc[indFinal]
    delD=(teval[indF]-teval[0])/30#dataDf['Days'].iloc[indFinal]-dataDf['Days'].iloc[0]
    #total cycle
    colsDf.append('diffPTotRate')
    valsDf.append((PF-P0)/delD)
    colsDf.append('diffPTot')
    valsDf.append(PF-P0)
    colsDf.append('diffPTotRatePer')
    valsDf.append((PF-P0)/(P0*delD))
    colsDf.append('diffPTotPer')
    valsDf.append((PF-P0)/P0)

    # other metrics
    indNP=[i for i,x in enumerate(tmpPSA) if x<=1]#get indexes if >1
    if(len(indNP)>0):
        colsDf.append('TimePHigh1')# time high during Tx
        valsDf.append(teval[indNP[0]]/30) # in months
        colsDf.append('TimePHigh2')# time high after Tx
        valsDf.append((dataDf['Days'][indFinal] - teval[indNP[len(indNP)-1]])/30)# in months
        colsDf.append('TimePHighTotal')# total time high
        valsDf.append((teval[indNP[0]]+dataDf['Days'][indFinal] - teval[indNP[len(indNP)-1]])/30)# in months
    colsDf.append('TimePLow')
    valsDf.append(len(indNP)/30)# in months
    colsDf.append('PSA_nadir')
    valsDf.append(np.nanmin(dataDf['PSA']))

    #testosterone metrics
    if('Testosterone' in dataDf.columns):
        T0=dataDf['Testosterone'].loc[0]
        colsDf.append('T0')
        valsDf.append(T0)
        colsDf.append('aveT')
        valsDf.append(np.nanmean(dataDf['Testosterone']))
        nanT=np.isnan(dataDf['Testosterone'])
        if(len(dataDf.Days[~nanT])>1):
            f1T = scipy.interpolate.interp1d(dataDf.Days[~nanT],dataDf['Testosterone'][~nanT],fill_value="extrapolate",kind="linear")#"extrapolate"
            tmpTest=f1T(teval)

        if(indSwitch!=0):
            T0=tmpTest[0]
            TS=tmpTest[indS]
            DS=teval[indS]/30
            colsDf.append('T_switch')
            valsDf.append(TS)
            #drug on
            colsDf.append('diffTOnRate')
            valsDf.append((TS-T0)/(DS))
            colsDf.append('diffTOn')
            valsDf.append((TS-T0))
            colsDf.append('diffTOnRatePer')
            valsDf.append((TS-T0)/(DS*T0))
            colsDf.append('diffTOnPer')
            valsDf.append((TS-T0)/T0)
            TF=tmpTest[indF]
            TS=tmpTest[indS]
            delD=(teval[indF]-teval[indS])/30
            #drug off
            colsDf.append('diffTOffRate')
            valsDf.append((TF-TS)/(delD))
            colsDf.append('diffTOff')
            valsDf.append((TF-TS))
            colsDf.append('diffTOffRatePer')
            valsDf.append((TF-TS)/(TS*delD))
            colsDf.append('diffTOffPer')
            valsDf.append((TF-TS)/TS)
        T0=tmpTest[0]
        TF=tmpTest[indF]
        delD=(teval[indF]-teval[0])/30
        colsDf.append('diffTTotRate')
        valsDf.append((TF-T0)/(delD))
        colsDf.append('diffTTot')
        valsDf.append((TF-T0))
        colsDf.append('diffTTotRatePer')
        valsDf.append((TF-T0)/(T0*delD))
        colsDf.append('diffTTotRatePerPos')
        valsDf.append(100*np.abs(TF-T0)/(T0*delD))
        colsDf.append('diffTTotPer')
        valsDf.append((TF-T0)/T0)
        colsDf.append('diffTTotPerPos')
        valsDf.append(100*np.abs(TF-T0)/T0)

        #other metrics
        indNT=[i for i,x in enumerate(tmpTest) if x<=100]
        if(len(indNT)>0):
            colsDf.append('TimeTHigh1')# time high during Tx
            valsDf.append(teval[indNT[0]]/30)# in months
            colsDf.append('TimeTHigh2')# time high after Tx
            valsDf.append((dataDf['Days'][indFinal] - teval[indNT[len(indNT)-1]])/30)# in months
            colsDf.append('TimeTHighTotal')# total time high
            valsDf.append((teval[indNT[0]]+dataDf['Days'][indFinal] - teval[indNT[len(indNT)-1]])/30)# in months
            colsDf.append('PerTimeTHighTotal')# total time high
            valsDf.append(100*(teval[indNT[0]]+dataDf['Days'][indFinal] - teval[indNT[len(indNT)-1]])/(teval[indF]-teval[0]))# in months
        indCT=[i for i,x in enumerate(tmpTest) if x<=50]
        if(len(indCT)>0):
            colsDf.append('TimeToCastrate')# time to castrate levels ## need more castrate while on/off separate??
            valsDf.append(indCT[0]/30)# in months
            colsDf.append('TimeCastrate')# time castrate levels
            valsDf.append(len(indCT)/30)# in months
        colsDf.append('T_nadir')
        valsDf.append(np.nanmin(dataDf['Testosterone']))
        colsDf.append('TE_nadir')
        valsDf.append(np.nanmin(dataDf['Testosterone'])/T0)


        #P/T
        fact=.01
        colsDf.append('P0/T0')
        valsDf.append(fact*dataDf['PSA'].iloc[0]/dataDf['Testosterone'].iloc[0])
        if(indSwitch!=0):
            colsDf.append('Pswitch/Tswitch')
            valsDf.append(fact*tmpPSA[indSwitch]/tmpTest[indSwitch])
        colsDf.append('PF/TF')
        valsDf.append(fact*dataDf['PSA'].iloc[indFinal]/dataDf['Testosterone'].iloc[indFinal])
        colsDf.append('aveP/T')
        valsDf.append(fact*np.mean(tmpPSA/tmpTest))## back to interpolated values
        colsDf.append('maxP/T')
        valsDf.append(fact*np.nanmax(tmpPSA/tmpTest))## back to interpolated values
        
        if(indSwitch!=0):
            P0=tmpPSA[0]
            T0=tmpTest[0]
            PS=tmpPSA[indS]
            TS=tmpTest[indS]
            delD=teval[indS]-teval[0]
            #drug on
            colsDf.append('diffOnRate')
            valsDf.append(fact*(PS/TS-P0/T0)/(delD))
            colsDf.append('diffOn')
            valsDf.append(fact*(PS/TS-P0/T0))
            colsDf.append('diffOnRatePer')
            valsDf.append(fact*(PS/TS-P0/T0)/((P0/T0)*delD))
            colsDf.append('diffOnPer')
            valsDf.append(fact*(PS/TS-P0/T0)/((P0/T0)))
            colsDf.append('diffOnPer2')
            valsDf.append(100*fact*(PS/TS-P0/T0)/((P0/T0)))
            PS=tmpPSA[indS]
            TS=tmpTest[indS]
            PF=tmpPSA[indF]
            TF=tmpTest[indF]
            delD=teval[indF]-teval[indS]
            #drug off
            colsDf.append('diffOffRate')
            valsDf.append(fact*(PF/TF-PS/TS)/(delD))
            colsDf.append('diffOff')
            valsDf.append(fact*(PF/TF-PS/TS))
            colsDf.append('diffOffRatePer')
            valsDf.append(fact*(PF/TF-PS/TS)/((PS/TS)*delD))
            colsDf.append('diffOffPer')
            valsDf.append(fact*(PF/TF-PS/TS)/((PS/TS)))
        P0=tmpPSA[indS]
        T0=tmpTest[indS]
        PF=tmpPSA[indF]
        TF=tmpTest[indF]
        delD=teval[indF]-teval[0]
        #drug off
        colsDf.append('diffTotRate')
        valsDf.append(fact*(PF/TF-P0/T0)/(delD))
        colsDf.append('diffTot')
        valsDf.append(fact*(PF/TF-P0/T0))
        colsDf.append('diffTotRatePer')
        valsDf.append(fact*(PF/TF-P0/T0)/((P0/T0)*delD))
        colsDf.append('diffTotPer')
        valsDf.append(fact*(PF/TF-P0/T0)/((P0/T0)))

        colsDf.append('Gleason')
        valsDf.append(dataVals['GS'][pi])
        colsDf.append('BMs')
        valsDf.append(dataVals['BMs'][pi])
        colsDf.append('LNs')
        valsDf.append(dataVals['LNs'][pi])
        colsDf.append('TMs')
        valsDf.append(dataVals['TMs'][pi])
        colsDf.append('risk')
        valsDf.append(dataVals['risk'][pi])
        colsDf.append('dNM1')
        valsDf.append(dataVals['dNM1'][pi])

        colsDf.append('mrD')
        valsDf.append(modVals['rD'][pi])
        colsDf.append('mdD')
        valsDf.append(modVals['dD'][pi])
        colsDf.append('mrd')
        valsDf.append(modVals['dD'][pi]*modVals['rD'][pi]*(T0/(T0+100)))
        colsDf.append('rPSA')
        valsDf.append(modVals['rPSA'][pi])
        colsDf.append('xS')
        valsDf.append(modVals['xS'][pi])

        #fitVals - check if unit changes are needed????
        colsDf.append('dRP')
        valsDf.append(fitVals['dRP'][pi])
        colsDf.append('maxP')
        valsDf.append(fitVals['maxP'][pi])
        colsDf.append('nadP')
        valsDf.append(fitVals['nadP'][pi])
        colsDf.append('maxPP')
        valsDf.append(fitVals['maxPP'][pi])
        colsDf.append('gRP')
        valsDf.append(fitVals['gRP'][pi])
        colsDf.append('tdelayP')
        valsDf.append(fitVals['tdelayP'][pi])
        colsDf.append('minT')
        valsDf.append(fitVals['minT'][pi])
        colsDf.append('dRT')
        valsDf.append(fitVals['dRT'][pi])
        colsDf.append('maxTT')
        valsDf.append(fitVals['maxTT'][pi])
        colsDf.append('gRT')
        valsDf.append(fitVals['gRT'][pi])
        colsDf.append('tdelayT')
        valsDf.append(fitVals['tdelayT'][pi])
        colsDf.append('gRP2')
        valsDf.append(fitVals['gRP2'][pi])

        colsDf.append('id')
        valsDf.append(pat)
        colsDf.append('prog1')
        valsDf.append(dataVals['prog1'][pi])
        colsDf.append('prog2')
        valsDf.append(dataVals['prog2'][pi])
        colsDf.append('prog3')
        valsDf.append(dataVals['prog3'][pi])
        colsDf.append('prog4')
        valsDf.append(dataVals['prog4'][pi])
        colsDf.append('dAbi')
        valsDf.append(dataVals['drug'][pi])
        colsDf.append('TTP')
        valsDf.append(dataVals['TTP'][pi])

        pData=pd.DataFrame()
        pData['names']=colsDf
        pData['values']=valsDf
        pData['id']=pat
        pData['progressed1']=dataVals['prog1'][pi]
        pData['progressed2']=dataVals['prog2'][pi]
        pData['progressed3']=dataVals['prog3'][pi]
        pData['progressed4']=dataVals['prog4'][pi]
        pData['dAbi']=dataVals['drug'][pi]
        return pData

    
print("done import aFuns")
