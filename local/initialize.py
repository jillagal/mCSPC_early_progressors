import pandas as pd
import numpy as np
from lmfit import Parameters

###############
# Load specs
###############
def getFittingSpecs(FittingSpecs):
    fitting0=FittingSpecs.DataFit.tolist()
    fitting0 = [x for x in fitting0 if str(x) != 'nan']
    fitting1=FittingSpecs.ModelFit.tolist()
    fitting1 = [x for x in fitting1 if str(x) != 'nan']
    fitting2=FittingSpecs.Weight.tolist()
    fitting2 = [x for x in fitting2 if str(x) != 'nan']
    fitting3=FittingSpecs.DetectionLimit.tolist()
    fitting3 = [x for x in fitting3 if str(x) != 'nan']
    fitting=[fitting0,fitting1,fitting2,fitting3]
    return [fitting]

def getPlotSpecs(modelSpecs):
    plotsIndex=modelSpecs.PlotsIndex.tolist()
    plotsIndex = [x for x in plotsIndex if str(x) != 'nan']
    plots=modelSpecs.Plots.tolist()
    plots = [x for x in plots if str(x) != 'nan']
    plotsData=modelSpecs.PlotsData.tolist()
    plotsData = [x for x in plotsData if str(x) != 'nan']
    plotsScale=modelSpecs.PlotsScale.tolist()
    plotsScale = [x for x in plotsScale if str(x) != 'nan']
    plotsTransform=modelSpecs.PlotsTransform.tolist()
    plotsTransform = [x for x in plotsTransform if str(x) != 'nan']
    return [plots,plotsData,plotsIndex,plotsScale,plotsTransform]

def getModelSpecs(modelSpecs):
    stateVarsNames=modelSpecs.StateVars.tolist()
    stateVarsNames = [x for x in stateVarsNames if str(x) != 'nan']
    stateVarsType=modelSpecs.StateVarsType.tolist()
    stateVarsType = [x for x in stateVarsType if str(x) != 'nan']
    pops0Input=modelSpecs.Pops0Input.tolist()
    pops0Input = [x for x in pops0Input if str(x) != 'nan']
    pops0Output=modelSpecs.Pops0Output.tolist()
    pops0Output = [x for x in pops0Output if str(x) != 'nan']
    pops0Type=modelSpecs.Pops0Type.tolist()
    pops0Type = [x for x in pops0Type if str(x) != 'nan']
    return [stateVarsNames,stateVarsType,pops0Input,pops0Output,pops0Type]

class Fitting2():
    def __init__(self,nIterations,n_sample,n_wins,fitting, **kwargs):
        self.nIterations=nIterations
        self.n_sample=n_sample
        self.n_wins=n_wins
        self.fitting=fitting

def defineInitValues(patientIdDic,dataDir,my_fit):
    tmp=[]
    for p in patientIdDic.keys():
        patientId = patientIdDic[p]
        patientDataDf = loadData(dataDir,patientId)
        T0=patientDataDf['Testosterone'].iloc[0]
        PSA0=patientDataDf['PSA'].iloc[0]

        tmp0=[]
        if('Testosterone' in my_fit.fitting[0]):
            tmp0=tmp0+[T0]
        if('PSA' in my_fit.fitting[0]):
            tmp0=tmp0+[PSA0]
        tmp.append(tmp0)

    # # list columns/names
    cols=[]
    if('Testosterone' in my_fit.fitting[0]):
        cols=cols + ['T0']
    if('PSA' in my_fit.fitting[0]):
        cols=cols + ['P0']

    lArray = pd.DataFrame(tmp,columns=cols)
    return lArray

def loadData(pathToData,patientId):
    patientDataDf = pd.read_excel(pathToData, sheet_name=str(patientId),header=1)
    dropCols=['Notes']
    for col in dropCols:
        patientDataDf.drop(col, axis=1, inplace=True)

    patientDataDfC=patientDataDf[patientDataDf['Cycle']>0]
    patientDataDfC = patientDataDfC.reset_index()

    bstr=[]
    for i in range(0,len(patientDataDfC['GnRH'])):
        b=str(patientDataDfC['GnRH'].iloc[i])+str(patientDataDfC['Abi'].iloc[i])+str(patientDataDfC['ARSI'].iloc[i])
        bstr.append(int(b,2))
    patientDataDfC['DrugIndex']=bstr

    if(np.isnan(patientDataDfC['Testosterone'].iloc[0])):
        if(0 in patientDataDf['Cycle'].values):
            dataDf0=patientDataDf.copy()
            dataDf0=dataDf0[dataDf0['Cycle']==0]
            tmpT=np.nanmax(dataDf0['Testosterone'])
            if(np.isnan(np.nanmax(dataDf0['Testosterone']))):
                tmpT=np.nanmax(patientDataDf['Testosterone'])
        else:
            tmpT=np.nanmax(patientDataDf['Testosterone'])
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
    else:
        tmpP=patientDataDfC['PSA'].iloc[0]
        
    PSA0=tmpP
    patientDataDfC.loc[0,'PSA']=PSA0
    
    return patientDataDfC


def loadDictionary(filename):
    fullDictionary= pd.read_excel(filename)
    dictionary = Parameters()
    for i in range(0,len(fullDictionary)):
        dictionary.add(fullDictionary.Name[i], value=fullDictionary.Value[i], min=fullDictionary.Min[i], max=fullDictionary.Max[i], vary = True)
    print("done: initialize parameters")
    return dictionary

def loadFullDictionary(modelName): 
    dicFile = './Parameters/Pars_'+modelName+'.xlsx'
    fullDictionary= pd.read_excel(dicFile,'Pars')
    parDic = Parameters()
    for i in range(0,len(fullDictionary)):
        if(fullDictionary.Type[i]=='C'):
            parDic.add(fullDictionary.Name[i], value=fullDictionary.Value[i], min=fullDictionary.Min[i], max=fullDictionary.Max[i], vary = False)
        else:
            parDic.add(fullDictionary.Name[i], value=fullDictionary.Value[i], min=fullDictionary.Min[i], max=fullDictionary.Max[i], vary = True)
    return [fullDictionary,parDic]


