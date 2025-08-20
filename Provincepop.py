# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 18:01:49 2023

@author: citiz
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize,linprog
from scipy import sparse

pop=pd.read_excel(r"C:\Users\citiz\Downloads\Chinapop\forecast.xlsx",sheet_name="pop")
pop.columns=['date']+list(map(str,np.arange(0,91)))
pop['Total']=pop.iloc[:,1:].sum(axis=1).astype(int)
pop['Total']=pop['Total']/1000
province=pd.read_excel(r"C:\Users\citiz\Downloads\Chinapop\Province Population.xlsx")
province.columns=['province','2010','2019','2020']
#province['s2010']=province.iloc[:,1]/province.iloc[-1,1]*1000
province['s2020']=province.iloc[:,3]/province.iloc[-1,3]*1000
province['gr']=(province['2020']/province['2010']).pow(1/10)
province['g2020']=(province['2020']/province['2019'])
#province['index']=province['s2020']>province['s2010']
#province['index']=province['index'].astype(int)
provincer=province.iloc[:-1,:].sort_values(by=['g2020'],ascending=False)
args1=provincer['2020'].to_numpy()
ini_value=provincer['g2020'].to_numpy().round(4)
temp=provincer['gr'].to_numpy()
#the date can be replaced by any year
args2=pop.loc[pop['date']==2021].iloc[:,-1].to_numpy()
#number of province
nprovince=province.shape[0]-1
#matrix to keep growth rates ordered
sortmatrix=np.zeros((nprovince,nprovince-1))
for i in range(nprovince-1):
	sortmatrix[i:i+2,i]=np.array([1,-1])
#ssortmatrix=sparse.csr_matrix(sortmatrix).astype(int)
#formalize a nonlinear programming optimization problem
#args1 would be 2020 province population and args2 would be total population
#estimate in 2030 or whichever year you'd like 

def fun(args1,args2):
	a,b=args1,args2
	v=lambda x:np.square(np.sum(a*x)-b)
	return v
constraint1=[{'type':'ineq','fun':f} for f in [lambda x,y=i:np.sum(x*sortmatrix[:,y]) for i in range(nprovince-1)]]
constraint2=[{'type':'ineq','fun':f} for f in [lambda x,y=i:temp[y]-x[y] for i in range(nprovince)]]
#minimize will get you a local minimum
for i in range(pop.shape[0]):
	#args1=provincer[str(2020+i)]
	args2=pop.loc[pop['date']==2020+i+1].iloc[:,-1].to_numpy()
	fgr=minimize(fun(args1,args2),ini_value,method='COBYLA',constraints=constraint1+constraint2)
	ini_value=fgr.x
	args1=args1*fgr.x
	provincer[str(2020+i+1)]=args1

#formulate the problem as a linear programming problem
def lp(args1,args2,ini_value):
	args1m=np.array(args1,ndmin=2)
	v1=np.append(-1,args1m)
	v2=np.append(-1,-args1m)
	v3=np.zeros((nprovince,nprovince+1))
	v3[1:,1:]=-sortmatrix.T
	v4=np.zeros((nprovince+1,nprovince+1))
	v4[1:,1:]=np.identity(nprovince)
	lmatrix=np.vstack((v1,v2,v3,v4))
	rmatrix=np.zeros(lmatrix.shape[0])
	rmatrix[0:2]=np.append(args2,-args2)
	rmatrix[-nprovince:]=ini_value
	return lmatrix, rmatrix
lmatrix,rmatrix=lp(args1,args2,ini_value) #rmatrix should be of 1d
cmatrix=np.append(1,np.zeros(nprovince)) #cmatrix should be of 1d
#when growth rate is taken as the minimizer, the lp program doesn't work very well
for i in range(3):
	#args1=provincer[str(2020+i)]
	#args2=pop.loc[pop['date']==2020+i+1].iloc[:,-1].to_numpy()
	#args2m=np.array(args2,ndmin=2)
	fgr=linprog(cmatrix,A_ub=lmatrix,b_ub=rmatrix) #need to add bounds to constraints bc tuples are immutable
	ini_value=fgr.x[1:]
	args1=args1*fgr.x[1:]
	provincer[str(2020+i+1)]=args1
	args2=pop.loc[pop['date']==2021+i+1].iloc[:,-1].to_numpy()
	lmatrix,rmatrix=lp(args1,args2,ini_value)
	print(fgr.x[1:])
#alter the constraints or the minimizer to check if a better solution can be obtained

