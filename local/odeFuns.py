# ====================================================================================
# ODE models
# ====================================================================================
from modelClass import ODEModel
import numpy as np  
import sys

sys.path.append("./local/")
import initialize as initialize

# ------------------------------------------
class Drugs():
    def Structure(nTxs,nVars,uVec,drugInds):
        D = np.zeros(nTxs)
        for j in range(0,nTxs):
            ind=j+nVars
            D[j] = uVec[ind]
        return D
    
# ------------------------------------------
class setupModel(ODEModel): 
    def __init__(self,stateVarsNames,stateVarsType,pops0Input,pops0Output,pops0Type,txCols,model,modelName,colorPalette,outputCols,outputFunction,dt, **kwargs):
        [fullDictionary,dictionary]=initialize.loadFullDictionary(modelName)

        self.model=model
        self.name = "model0"
        self.modelName = modelName
        self.stateVarsList=stateVarsNames
        self.stateVarsTypeList=stateVarsType
        #self.stateVarsPlot=stateVars0[2]
        self.pops0Input=pops0Input
        self.pops0Output=pops0Output 
        self.pops0Type=pops0Type
        self.colorPalette=colorPalette
        self.dictionary = dictionary
        self.txCols = txCols
        self.outputCols=outputCols
        self.outputFunction=outputFunction

        #setup variable and constant parameter lists
        varsV=[] 
        varsC=[]
        for name in fullDictionary.Name:
            entry=fullDictionary[fullDictionary.Name==name]
            if(entry.Type.iloc[0]=='V'):
                varsV.append(name)
            else:
                varsC.append(name)
        varsNames=varsC+varsV
        self.varsList=varsNames
        self.varsV=varsV
        self.varsC=varsC
        
        super().__init__(modelName=modelName,dt=dt,**kwargs)
        self.SetParamsVars()     

    # Set the parameters
    def SetParamsVars(self):
        self.paramDic = {**self.paramDic}
        for name in self.varsList:
            ff=self.dictionary.get(name)
            self.paramDic[ff.name] = ff.value
            self.stateVars = self.stateVarsList
        
    # The governing equations
    def ModelEqns(self, t, uVec):
        nTxs = len(self.txCols)
        drugInds = range(0,nTxs-1)
        nVars=len(self.stateVars)
        D = Drugs.Structure(nTxs,nVars,uVec,drugInds) 
        
        dudtVec=self.model(self,t,uVec,D)
    
        #drugs
        for i in drugInds:
            dudtVec[i+len(self.stateVars)] = 0#D[i]#0
        return (dudtVec)
    
print("done: set up models")