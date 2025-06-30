import numpy as np  

##############
# models
#############

def modPCaSX(self, t, uVec, D):
    if(D[0]>0 or D[1]>0):
        if(D[0]>0 and D[1]>0):
            D_death=1
        elif(D[0]>0):
            D_death=.8
        else:
            D_death=.5
    else:
        D_death=0
    D_pro=self.paramDic['d_AA']*(D[1]+D[2]) 

    dudtVec = np.zeros_like(uVec)
    dudtVec[0] = 0
    dudtVec[1] = uVec[1]*(self.paramDic['r_S']*((uVec[0]/self.paramDic['T0'])*self.paramDic['X0']*(1-D_pro))/((uVec[0]/self.paramDic['T0'])*self.paramDic['X0']*(1-D_pro)+1) - D_death*self.paramDic['d_S'])

    return dudtVec

##############
# choose model
##############

def pickModel(modelName):
    if(modelName=='modPCaSX'):
        model = modPCaSX
        outputFunction=getPSAiSX

    return [model,outputFunction] 

def getPSAiSX(paramDic,df):
    dAA=paramDic.get('d_AA', 1)
    T0=paramDic.get('T0', 1)
    X0=paramDic.get('X0', 1)
    TT0X=X0*df['T'].values/T0
    adj=1-dAA*(df['ARSI'].values+df['Abi'].values)

    return df['S'].values*(TT0X)*adj/(TT0X*adj+1)

