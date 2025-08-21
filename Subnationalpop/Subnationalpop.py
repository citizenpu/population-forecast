# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 14:04:13 2025

@author: citiz
"""

import pandas as pd 
import numpy as np 
from scipy.optimize import minimize  
import openpyxl
# Load dataset
df = pd.read_excel('C:/Users/citiz/Downloads/raw data.xlsx', header=0)  
df=df.rename(columns={'Unnamed: 0':"province",'Unnamed: 1':"city"})
df.iloc[:,2:] = df.iloc[:,2:].apply(pd.to_numeric, errors='coerce')
# Compute 2035 growth rates
df['g2035'] = (df[2035] - df[2034]) / df[2034] 
df.dropna(subset=[2035,"g2035"],inplace=True)
df['totpop2035']=np.nan
df['totpop2036']=np.nan
df['2036']=np.nan


# === Define province targets ===

# Calculate province targets (replace with your actual targets)
province_targets = {}
for province, group in df.groupby('province'):
    df.loc[df['province']==province,'totpop2035'] = group[2035].sum()
    province_targets[province] = group[2035].sum()*(group[2035].sum()/group[2034].sum()-0.0005)

# === Process each province separately ===
results = []

for province, province_df in df.groupby('province'):
    if province in ['Hainan','Tibet']:
        continue
    print(f"\n=== Processing province: {province} ===")
    print(f"Number of cities: {len(province_df)}")
    print(f"Province target: {province_targets[province]:,.2f}")
    
    # Reset index for current province. The reset will impact a step further down in line 138.
    province_df = province_df.reset_index(drop=True)
    n = len(province_df)
    
    # === New Objective Function: Minimize squared gap from province target ===
    def objective(x):
        total_pop = np.sum(x)
        return (total_pop - province_targets[province]) ** 2

    # === Constraints (within province) ===
    constraints = []  
    
    # 1. Total population constraint (<= target)
    constraints.append({
        'type': 'ineq', 
        'fun': lambda x: province_targets[province] - np.sum(x)
    })  
    
    # 2. Decreasing growth for each city (g2036 < g2035)
    for i in range(n):
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: province_df["g2035"].iloc[i] - ((x[i] - province_df[2035].iloc[i]) / province_df[2035].iloc[i]) - 0.0005
        })
    
    # 3. Ranking preservation (within province)
    tol = 1e-6  # Tolerance for floating-point comparisons
    g2035 = province_df["g2035"].values
    for i in range(n):
        for j in range(n):
            if i != j and (g2035[i] > g2035[j] + tol):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: ((x[i] - province_df[2035].iloc[i]) / province_df[2035].iloc[i]) - \
                                               ((x[j] - province_df[2035].iloc[j]) / province_df[2035].iloc[j])
                })
    
    # 4. Growth gap constraint (g2036 > g2035 - 0.02)
    for i in range(n):
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: ((x[i] - province_df[2035].iloc[i]) / province_df[2035].iloc[i]) - (province_df["g2035"].iloc[i] - 0.02)
        })
    
    # === Initial guess: 2035 * (1 + g2035) ===
    x0 = province_df[2035].values * (1 + province_df['g2035'].values)  
    
    # === Run optimization ===
    result = minimize(objective, x0, constraints=constraints, method='SLSQP', options={'maxiter': 1000})
    
    # === Collect results ===
    if result.success:
        province_df["2036"] = result.x
        province_df["g2036"] = (province_df["2036"] - province_df[2035]) / province_df[2035]
        province_df["gap_%"] = 100 * (province_df["g2036"] - province_df["g2035"])
        province_df['totpop2036']=province_df["2036"].sum()
        
        
        print(f"✅ Optimization successful for {province}!")
        print(f"Province 2036 population: {province_df['2036'].sum():,.2f}")
        print(f"Deviation from target: {province_df['2036'].sum() - province_targets[province]:,.2f}")
        
        # === Post-check ===
        print("\n=== Constraint Verification ===")
        
        # 1. Total population
        total_pop = province_df["2036"].sum()
        total_ok = total_pop <= province_targets[province]
        print(f"Total pop ≤ {province_targets[province]:,.2f}: {total_ok} ({total_pop:,.2f})")
        
        # 2. Decreasing growth
        decreasing_ok = all(province_df["g2036"] < province_df["g2035"])
        print("Strictly decreasing growth for all cities:", decreasing_ok)
        
        # 3. Ranking preservation
        ranking_ok = True
        g36 = province_df["g2036"].values
        for i in range(n):
            for j in range(n):
                if i != j and (g2035[i] > g2035[j] + tol):
                    if g36[i] <= g36[j]:
                        ranking_ok = False
                        print(f"Ranking violation: City {i} had higher growth than {j} in 2035 but not in 2036")
        print("Ranking preserved:", ranking_ok)
        
        # 4. Growth gap
        gap_ok = all((province_df["g2035"] - province_df["g2036"]) <= 0.02 + tol)
        print("Growth gap within 2%:", gap_ok)
        
        results.append(province_df)
    else:
        print(f"❌ Optimization failed for {province}: {result.message}")
        # Fallback: use projected growth
        province_df["2036"] = np.round(province_df[2035] * (1 + province_df['g2035'])).astype(int)
        results.append(province_df)
    # change the raw data set. Note here we use df. values instead of df, this is because if we did this way, the match will take place based on the index of the two dataframes (absolute positions), whereas by using the atrribute of values, we match by their relative positions. 
    df.loc[df["province"]== province,"2036"] = province_df["2036"].values
    df.loc[df["province"]== province,"g2036"] = province_df["g2036"].values
    df.loc[df["province"]== province,"gap_%"] = province_df["gap_%"].values
    df.loc[df["province"]== province,"totpop2036"] = province_df['totpop2036'].values

# Combine all province results
#final_df = pd.concat(results, ignore_index=True)

# Save results
#final_df.to_excel("C:/Users/citiz/Downloads/final_df.xlsx")
df.to_excel("C:/Users/citiz/Downloads/df_final.xlsx")

