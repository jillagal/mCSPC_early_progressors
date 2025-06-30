# ====================================================================================
# Abstract model class
# ====================================================================================
import numpy as np
import scipy.integrate
import pandas as pd
import sys

import seaborn as sns
sns.set(style="white")
sys.path.append("./")


class ODEModel():
    def __init__(self,dt= 1e-1, **kwargs):
        # Initialize parameters
        self.paramDic = {}
        self.stateVars = []
        for i in range(1,len(self.stateVarsList)+1):
            self.stateVars.append('P'+str(i))
        self.resultsDf = None

        # Set the parameters
        self.SetParams(**kwargs)

        # Configure the solver
        self.dt = kwargs.get('dt', dt)  # Time resolution to return the model prediction on
        self.absErr = kwargs.get('absErr', 1.0e-8)  # Absolute error allowed for ODE solver
        self.relErr = kwargs.get('relErr', 1.0e-6)  # Relative error allowed for ODE solver
        self.solverMethod = kwargs.get('method', 'DOP853')  # ODE solver used
        self.suppressOutputB = kwargs.get('suppressOutputB', False)  # If true, suppress output of ODE solver (including warning messages)
        self.successB = False  # Indicate successful solution of the ODE system

    # =========================================================================================
    # Function to set the parameters
    def SetParams(self, **kwargs):
        if len(self.paramDic.keys()) > 1:
            for key in self.paramDic.keys():
                self.paramDic[key] = float(kwargs.get(key, self.paramDic[key]))
            self.initialStateList = [self.paramDic[self.stateVars[var] + "0"] for var in range(0,len(self.stateVars))]
        
# =========================================================================================

    # Function to simulate the model
    def SimulateWithT(self, testo,treatmentScheduleList, **kwargs): 
        self.dt = float(kwargs.get('dt', self.dt))  # Time resolution to return the model prediction on
        self.absErr = kwargs.get('absErr', self.absErr)  # Absolute error allowed for ODE solver
        self.relErr = kwargs.get('relErr', self.relErr)  # Relative error allowed for ODE solver
        self.solverMethod = kwargs.get('method', self.solverMethod)  # ODE solver used
        self.successB = False  # Indicate successful solution of the ODE system
        self.suppressOutputB = kwargs.get('suppressOutputB', self.suppressOutputB) # If true, suppress output of ODE solver (including warning messages)
        
        # Set initial state
        self.treatmentScheduleList = treatmentScheduleList
        if self.resultsDf is None or treatmentScheduleList[0][0] == 0:  # 
            currStateVec = self.initialStateList
            for j in range(0,len(self.txCols)):
                currStateVec.append(0)
            self.resultsDf = None
        else:
            currStateVec = []
            for j in range(0,len(self.initialStateList)):
                currStateVec.append(self.resultsDf[self.stateVars[j]].iloc[-1])
            for j in range(0,len(self.txCols)):
                currStateVec.append(0)
                #currStateVec.append(self.resultsDf[self.stateVars[j]].iloc[-1])

        # Solve over time
        resultsDFList = []
        encounteredProblemB = False
        for intervalId, interval in enumerate(treatmentScheduleList):
            if(interval[1]-interval[0]>self.dt):
                tVec = np.arange(interval[0], interval[1], self.dt)
            else:
                tVec = [interval[0],interval[1]]
            if intervalId == (len(treatmentScheduleList) - 1):
                tVec = np.arange(interval[0], interval[1] + self.dt, self.dt)
            for i, txi in enumerate(self.txCols):
                currStateVec[len(self.stateVars)+i] = interval[2+i]
            currStateVec[0]=testo[intervalId]
            solObj = scipy.integrate.solve_ivp(self.ModelEqns, y0=currStateVec,
                                                   t_span=(tVec[0], tVec[-1] + self.dt), t_eval=tVec,
                                                   method=self.solverMethod,
                                                   atol=self.absErr, rtol=self.relErr,
                                                   max_step=kwargs.get('max_step', np.inf))
            
            # Check that the solver converged
            if not solObj.success or np.any(solObj.y < 0):
                self.errMessage = solObj.message
                encounteredProblemB = True
                if not self.suppressOutputB: print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                if not solObj.success:
                    if not self.suppressOutputB: print(self.errMessage)
                else:
                    if not self.suppressOutputB: print(
                        "Negative values encountered in the solution. Make the time step smaller or consider using a stiff solver.")
                    if not self.suppressOutputB: print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                self.solObj = solObj
                break
            # Save results
            resApp=pd.DataFrame({"Days": tVec})
            for i in range(0,len(self.stateVars)):
                resApp[self.stateVars[i]]=solObj.y[i, :]
            for i,t in enumerate(self.txCols):
                resApp[self.txCols[i]] = solObj.y[len(self.stateVars)+i, :]
            resultsDFList.append(resApp)
            currStateVec = solObj.y[:, -1]

        # If the solver diverges in the first interval, it can't return any solution. 
        #Catch this here, and in this case
        # replace the solution with all zeros.
        if len(resultsDFList) > 0:
            resultsDf = pd.concat(resultsDFList)
        else:
            resApp=pd.DataFrame({"Days": tVec})
            for i in range(0,len(self.stateVars)):
                resApp[self.stateVars[i]]=np.zeros_like(tVec)

            for i,t in enumerate(self.txCols):
                resApp[self.txCols[i]] = np.zeros_like(tVec)
            resultsDf = resApp

        tumorPopsList = []
        for pi,pops in enumerate(self.stateVarsTypeList):
            if(pops=='Pop'):
                tumorPopsList.append(self.stateVars[pi])

        if(len(tumorPopsList)>1):
            resultsDf['TumorSize'] = resultsDf[tumorPopsList].sum(axis=1)
        elif(len(tumorPopsList)==1):
            resultsDf['TumorSize'] = resultsDf[tumorPopsList]
        if(len(self.outputCols)>0):
            resultsDf=self.addOutputCol(self.paramDic,resultsDf)
           
        if self.resultsDf is not None:
            resultsDf = pd.concat([self.resultsDf, resultsDf])
        self.resultsDf = resultsDf
        self.successB = True if not encounteredProblemB else False
    
    def addOutputCol(self,paramDic,resultsDf):
        for oC in self.outputCols:
            resultsDf[oC]=self.outputFunction(paramDic,resultsDf)
        return resultsDf

    # =========================================================================================
    # Interpolate to specific time resolution (e.g. for plotting)
    def Trim(self, t_eval=None, dt=1):
        t_eval = np.arange(0, self.resultsDf.Days.max(), dt) if t_eval is None else t_eval
        tmpDfList = []
        trimmedResultsDic = {'Days': t_eval}
        vals = []
        for i in range(0,len(self.stateVars)):
            vals.append(self.stateVars[i])
        for drug in self.txCols:
            vals.append(drug)
        if(any('Pop' in self.stateVarsTypeList for item in self.stateVarsTypeList)): 
            vals.append('TumorSize')
        for variable in vals: 
            f = scipy.interpolate.interp1d(self.resultsDf.Time, self.resultsDf[variable])
            trimmedResultsDic = {**trimmedResultsDic, variable: f(t_eval)}
        tmpDfList.append(pd.DataFrame(trimmedResultsDic))
        self.resultsDf = pd.concat(tmpDfList)
 
