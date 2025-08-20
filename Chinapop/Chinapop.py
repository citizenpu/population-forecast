# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 14:09:27 2023
This exercise is based on the Carter-Lee model in which the sex ratio at birth is set to be at 0.4734 and
K^t is following a random walk process for both mortality and fertility.
formortality is the function to predict future mortality and forfertility is to predict future fertility
@author: citiz
"""

import numpy as np
import pandas as pd
from pandas import ExcelWriter
from statsmodels.tsa.arima.model import ARIMA

class Chinapop():
	def __init__(self,mortality,fertility,ini_population,period):
		self.morality=morality[:,1:]/1000
		self.fertility=fertility[:,1:]/1000
		self.T=mortality.shape[0]
		self.ini_population=ini_population[:,1:]
		self.period=period

	def formortality(self):
		alpha=np.zeros((1,91))
		for i in range(91):
			alpha[0,i]=np.sum(np.log(self.mortality),0)[i]/self.T
		self.alpha=alpha
		key=np.sum(np.log(self.mortality)-self.alpha,1)
		rsmodel=ARIMA(key,order=(1,0,0)).fit()
		y=rsmodel.forecast(self.period)
		self.m=y
		self.keym=key
		yh=np.array(y,ndmin=2).reshape(self.period,1)
		lhv=np.log(self.mortality)-self.alpha
		denominator=np.array(key,ndmin=2) @ np.array(key,ndmin=2).T
		beta=np.array(key,ndmin=2) @ lhv/denominator
		tempm=np.exp(np.kron(self.alpha,np.ones((self.period,1)))+np.kron(beta,yh))
		return tempm
	
	def forfertility(self):
		alphaf=np.zeros((1,35))
		for i in range(35):
			alphaf[0,i]=np.sum(np.log(self.fertility),0)[i]/self.T
		self.alphaf=alphaf
		keyf=np.sum(np.log(self.fertility)-self.alphaf,1)
		rsmodel=ARIMA(keyf,order=(1,0,0)).fit()
		y=rsmodel.forecast(self.period)
		self.f=y
		self.keyf=keyf
		yh=np.array(y,ndmin=2).reshape(self.period,1)
		lhv=np.log(self.fertility)-self.alphaf
		denominator=np.array(keyf,ndmin=2) @ np.array(keyf,ndmin=2).T
		betaf=np.array(keyf,ndmin=2) @ lhv/denominator
		tempf=np.exp(np.kron(self.alphaf,np.ones((self.period,1)))+np.kron(betaf,yh))
		return tempf
	
	def fortotpop(self):
		data=np.zeros((2,self.period,91))
		fertility=np.zeros((self.period,91))
		bd=np.zeros((2,self.period,91))
		fertility[:,15:50]=self.forfertility()
		bd[0,0,:]=self.ini_population[:,1]*fertility[0,:]
		bd[1,0,:]=self.ini_population[:,0]*self.formortality()[0,:]
		v=self.ini_population-self.formortality()[0,:].reshape(91,1)*self.ini_population
		data[:,0,0]=np.array([np.sum(bd[0,0,:]),0.4734*np.sum(bd[0,0,:])])
		data[:,0,1:-1]=v[:-2,:].T
		data[:,0,90]=np.sum(v[-2:,:],0)
		for i in list(range(self.period))[1:]:
			bd[0,i,:]=data[1,i-1,:]*fertility[i,:]
			bd[1,i,:]=data[0,i-1,:]*self.formortality()[i,:]
			data[:,i,0]=np.array([np.sum(bd[0,i,:]),0.4734*np.sum(bd[0,i,:])])
			v=data[:,i-1,:]-self.formortality()[i,:].reshape(1,91)*data[:,i-1,:]
			data[:,i,1:-1]=v[:,:-2]
			data[:,i,90]=np.sum(v[:,-2:],1)
		pop=data[0,:,:]
		female=data[1,:,:]
		birth=bd[0,:,:]
		death=bd[1,:,:]
		return pop, female, birth, death, fertility

if __name__ == '__main__':
	mortality=pd.read_excel(r"C:\Users\citiz\Downloads\Chinapop\mortality.xlsx",sheet_name="ALLwo")
	mortality=morality.to_numpy()
	fertility=pd.read_excel(r"C:\Users\citiz\Downloads\Chinapop\fertility.xlsx",sheet_name="ALLwo")
	fertility=fertility.to_numpy()
	ini_population=pd.read_excel(r"C:\Users\citiz\Downloads\Chinapop\Total Population.xlsx",sheet_name="ALLwo")
	ini_population=ini_population.to_numpy()
	eiu=Chinapop(mortality,fertility,ini_population,30)
	head=list(map(str,np.arange(0,91)))
	date=list(map(str,np.arange(2021,2051)))
	name=list(['pop','female','birth','death','fertility'])
	w = ExcelWriter(r"C:\Users\citiz\Downloads\Chinapop\forecast.xlsx")
	for i in range(len(eiu.fortotpop())):
		df=pd.DataFrame(eiu.fortotpop()[i],index=date,columns=head)
		df.to_excel(w,sheet_name=name[i])
	w.save()
		
	
