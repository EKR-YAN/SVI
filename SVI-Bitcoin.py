#!/usr/bin/env python
# coding: utf-8

# # Surface de volatilité sur crypto-actifs

# # Clean

# In[31]:


def clearall():
    all = [var for var in globals() if var[0] != "_"]
    for var in all:
        del globals()[var]

clearall()


# ## Importations

# In[32]:


import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import quad
import sympy as sp 
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from py_vollib.black_scholes.implied_volatility import implied_volatility as implVola
import matplotlib.pyplot as plt
from pandas_datareader.data import Options
from dateutil.parser import parse
from datetime import datetime
from matplotlib import cm
from matplotlib.colors import LogNorm
#from implied_vol import BlackScholes
from functools import partial
from scipy.optimize import fsolve
import scipy.optimize as optimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint


# ## Données 

# In[33]:


implVola(0.1365*22869.45, 22869.45, 20000, 9.5/365.25, 0.0, 'c')


# In[34]:


implVola(0.0133*20630, 20630, 21500, 2/365.25, 0.0, 'c')


# In[35]:


implVola(0.0125*21500, 20630, 21500, 2/365.25, 0.0, 'c')


# In[36]:


implVola(0.0140*21500, 20630, 21500, 2/365.25, 0.0, 'c')


# In[37]:


DBO= pd.read_excel('drbt.xlsx') #Data Bitcoin Options (DBO)
DBO


# In[38]:


Liste_total_K=np.array(DBO.loc[:,["Strike"]].sort_values(by = 'Strike'))
DBO1=DBO.sort_values(by = 'Strike')
DBO1


# In[39]:


Liste_total_K


# In[40]:


dfT=DBO.loc[:,["Maturity.1"]]

DBO1.rename(columns = {'Maturity.1':'Mat'}, inplace = True) 
 


DBO_ordered=DBO.drop(DBO.index[[i for i in range (len(DBO))] ] )

dfT.drop_duplicates(keep = 'first', inplace=True)

    
dfT.reset_index(inplace=True, drop=True)
dfT
DBO1


# In[41]:


for k in range(len(dfT)):
    T=float(dfT.loc[k,["Maturity.1"]])
    df=DBO1[ DBO1['Mat'] == T ]
    DBO_ordered=pd.concat([ DBO_ordered,df ])


DBO_ordered.drop(DBO_ordered.columns[[3]], axis = 1, inplace = True)

DBO_ordered


# In[44]:


ListeT=[]
ListeK=[]
ListeIVMID=[]
ListePrice=[]




# In[45]:


Daysperyear=365
ListeMat=DBO_ordered.loc[:,["Mat"]]
ListeMat=np.unique(np.array(ListeMat))
ListeLogMoneyness=DBO_ordered.loc[:,["Strike"]]
ListeLogMoneyness=np.unique(np.array(ListeLogMoneyness))
ListeLogMoneyness=np.log(ListeLogMoneyness/S)
Listevoltot=((np.array(DBO_ordered.loc[:,["IV (Bid)"]]))+np.array(DBO_ordered.loc[:,["IV (Ask)"]]))*0.5


ListeT=ListeMat/Daysperyear
ListeK=[]
ListeIVMID=[]
ListeIVASK=[]
ListeIVBID=[]
ListePrice=[]


for idx in range(len(ListeT)):
    ListeK+=[np.array(DBO_ordered.loc[:,["Strike"]][ DBO_ordered['Mat'] == ListeMat[idx]])]
    ListeIVASK+=[np.array(DBO_ordered.loc[:,["IV (Ask)"]][ DBO_ordered['Mat'] == ListeMat[idx]])/100]
    ListeIVBID+=[np.array(DBO_ordered.loc[:,["IV (Bid)"]][ DBO_ordered['Mat'] == ListeMat[idx]])/100]
    ListePrice+=[np.array(DBO_ordered.loc[:,["Mark"]][ DBO_ordered['Mat'] == ListeMat[idx]])]
    ListeIVMID+=[(ListeIVASK[-1]+ListeIVBID[-1])*0.5]
    

    
vartot=[]
vartot=np.array(vartot)
for T in ListeT:
    for k in range(len(Listevoltot)):
        w=((Listevoltot[k]/100)**2)*T
        vartot=np.concatenate((vartot, w))
        
max(vartot)
        
    



# In[46]:


Daysperyear=365
ListeMat=DBO_ordered.loc[:,["Mat"]]
ListeMat=np.unique(np.array(ListeMat))
ListeLogMoneyness=DBO_ordered.loc[:,["Strike"]]
ListeLogMoneyness=np.unique(np.array(ListeLogMoneyness))
ListeLogMoneyness=np.log(ListeLogMoneyness/S)
Listevoltot=((np.array(DBO_ordered.loc[:,["IV (Bid)"]]))+np.array(DBO_ordered.loc[:,["IV (Ask)"]]))*0.5


ListeT=ListeMat/Daysperyear
ListeK=[]
ListeIVMID=[]
ListeIVASK=[]
ListeIVBID=[]
ListePrice=[]


for idx in range(len(ListeT)):
    ListeK+=[np.array(DBO_ordered.loc[:,["Strike"]][ DBO_ordered['Mat'] == ListeMat[idx]])]
    ListeIVASK+=[np.array(DBO_ordered.loc[:,["IV (Ask)"]][ DBO_ordered['Mat'] == ListeMat[idx]])/100]
    ListeIVBID+=[np.array(DBO_ordered.loc[:,["IV (Bid)"]][ DBO_ordered['Mat'] == ListeMat[idx]])/100]
    ListePrice+=[np.array(DBO_ordered.loc[:,["Mark"]][ DBO_ordered['Mat'] == ListeMat[idx]])]
    ListeIVMID+=[(ListeIVASK[-1]+ListeIVBID[-1])*0.5]
    


# ## Extraction de numpy array à partir du dataframe

# In[43]:


S=float(DBO_ordered.loc[0,["Spot"]])
S


# n=len(DBO_ordered)
# 
# ar=np.zeros(n)
# 
# Liste_T=DBO_ordered.loc[:,["Mat"]]/365
# Liste_K=DBO_ordered.loc[:,["Strike"]]
# 
# Liste_price=pd.DataFrame(ar, index = [str(i) for i in range(n)], columns = ["Price"])
# 
# 
# 
# Liste_total_var=pd.DataFrame(ar, index = [str(i) for i in range(n)], columns = ['total variance'])
# Liste_vol_imp=pd.DataFrame(ar, index = [str(i) for i in range(n)], columns = ['vol_imp'])
# 
# 
# for k in range(0,n):
#     Liste_price.loc[k,"Price"] = (float(DBO_ordered.loc[k,"Bid"])+float(DBO_ordered.loc[k,"Ask"]))*S/2
#     
# 
# for k in range(0,n):
#     Liste_vol_imp.loc[k,"vol_imp"] = compute_IV(Liste_price.loc[k,"Price"], S, float(DBO_ordered.loc[k,["Strike"]]), float(DBO_ordered.loc[k,["Mat"]])/365, 0, "c")
#     Liste_total_var.loc[k,"total variance"] = float(Liste_vol_imp.loc[k,"vol_imp"])*float(Liste_vol_imp.loc[k,"vol_imp"])*float(Liste_T.loc[k,"Mat"])
# 
# 
# 

# # Fonctions de base

# ## Fonction de répartition d'une loi gausienne 

# In[47]:


def CND(X):
    return norm.cdf(X)


# ## Fonction pour pricer une option européenne avec le modèle de Black and Scholes

# In[48]:


def callBS(S, K, T, r, sigma):
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1-sigma*np.sqrt(T)
    call = S*CND(d1)-CND(d2)*K*np.exp(-r*T)
    return call

def putBS(S, K, T, r, sigma):
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1-sigma*np.sqrt(T)
    put = -S*CND(-d1)+CND(-d2)*K*np.exp(-r*T)
    return put
    


# ## Vega

# In[49]:


def vegaBS(S, K, T, r, sigma):
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    vega = S*np.sqrt(T)*CND(d1)
    return vega


# ## Fonctions pour calculer la volatilité implicite pour un strike et une maturité donnée

# In[50]:


def compute_IV(target_value, S, K, T, r, o):
    MAX_ITERATIONS = 200
    PRECISION = 1.0e-5
    sigma = 0.25
    for i in range(0, MAX_ITERATIONS):
        if(o == "c"):
            price = callBS(S, K, T, r, sigma)
        elif(o == "p"):
            price = putBS(S, K, T, r, sigma)
        vega = vegaBS(S, K, T, r, sigma)
        diff = target_value - price
        if (abs(diff) < PRECISION):
            return sigma
        sigma = sigma + diff/vega
    return sigma


# # Modèle SVI

#  On va calibrer un modèle de type SVI pour obtenir la variance totale $w_{total}(k,\chi _R)$ du sous-jacent Bitcoin en fonction du strike pour plusieur maturités fixées. 
#  $w_{total}^{}(k,\chi _R)=a+b\left \{ \rho (k-m)+\sqrt{(k-m)^{2}+\sigma ^{2}} \right \} $
#  Où $ \chi_R=\left \{ a,b,\rho ,m,\sigma  \right \} $ sont les paramètres à calibrer.
#  On précise que $k=log(K/F_T)$ avec $F_T=S_0exp(\int_{0}^{T}r(t)dt)$.

# In[51]:


def SVI(x,sviParams):
    a, b, rho, m, sigma=sviParams[0],sviParams[1],sviParams[2],sviParams[3],sviParams[4]
    
    return a + b*(rho*(x-m)+((x-m)**2+sigma**2)**0.5)


# In[52]:


def SVI1(x,sviParams):
    # First derivative with respect to x
    a, b, rho, m, sigma=sviParams[0],sviParams[1],sviParams[2],sviParams[3],sviParams[4]
    sig2=sigma*sigma
    return b*(rho +(x-m)/(np.sqrt((x-m)*(x-m)+sig2)))
    


# In[53]:


def SVI2(x,sviParams):
    # Second derivative with respect to x
    a, b, rho, m, sigma=sviParams[0],sviParams[1],sviParams[2],sviParams[3],sviParams[4]
    sig2=sigma*sigma
    return (b/(np.sqrt((x-m)*(x-m)+sig2)))-(b*(x-m)*(x-m))/(((x-m)*(x-m)+sig2))**1.5



# In[54]:


def g(x,sviParams):
    w=SVI(x,sviParams)
    w1=SVI1(x,sviParams)
    w2=SVI2(x,sviParams)
    return (1. - 0.5 * x * w1 / w) * (1.0 - 0.5 * x * w1 / w) - 0.25 * w1 * w1 * (0.25 + 1. / w)


# In[55]:


#Pour un modèle où le taux sans risque est constant
def Forward_price(So,T,r):
    return So*np.exp(r*T)
    
    


# In[56]:


## input T Liste_K pour avoir une fonction générale

def globalOptimization(y,StrikeArray,IVArray,T,Spot,r): 
    n=len(StrikeArray)
    sviParams = y[0], y[1], y[2], y[3], y[4]
    x=np.log(StrikeArray/Spot)
    
    vol=IVArray
    vartot=(IVArray**2)*T
    vegaArr=vegaBS(1,StrikeArray/Spot,T,r,IVArray)
    #vegaArr=vegaBS(Spot,StrikeArray,T,r,IVArray)
    #print(np.mean(vegaArr*(SVI(x,sviParams) - vartot)**2))
    return np.mean((np.sqrt(SVI(x,sviParams)/T) - vol)**2)
#vegaArr*
#vegaArr*
    #vegaArr*
    


# # Tracé des smiles sur un exemple joué

# ### Contraintes

# In[57]:



epsilon=0.001


a_min=10**-5
a_max=max(vartot)

b_min=0.001
b_max=1000

rho_min=-1
rho_max=1

m_min=2*min(ListeLogMoneyness)
m_max=2*max(ListeLogMoneyness)

sigma_min=0.01
sigma_max=1

#bounds = Bounds([a_min, a_max], [b_min, b_max],[rho_min,rho_max] ,[m_min,m_max] ,[sigma_min,sigma_max ])

bounds = ((a_min, a_max),(b_min, b_max),(rho_min,rho_max),(m_min,m_max),(sigma_min,sigma_max))
#bounds = [(-512, 512), (-512, 512)]
#linear_constraint = LinearConstraint([[1,0,0,0,0], [0,1,0,0,0 ],[0,0,1,0,0 ],[0,0,0,1,0 ],[0,0,0,0,1 ]], [a_min,b_min,rho_min,m_min,sigma_min ], [a_max,b_max,rho_max,m_max,sigma_max ])

#def cons1(x):
    #a,b,m,rho,sigma=x[0],x[1],x[2],x[3],x[4]
    #(a - m *b*(rho + 1))*(4 - a + m*b*(rho + 1))- b*b* (rho + 1)*(rho + 1)

#def cons2(x):
    #a,b,m,rho,sigma=x[0],x[1],x[2],x[3],x[4]
    #(a - m *b*(rho - 1))*(4 - a + m*b*(rho + 1))- b*b*(rho - 1)*(rho - 1)


con = lambda x: (x[0] -x[2] *x[1]*(x[3] + 1))*(4 - x[0] + x[2]*x[1]*(x[3] + 1))- x[1]*x[1]* (x[3] + 1)*(x[3] + 1)-epsilon
nlc = NonlinearConstraint(con, 0, +np.inf)

con2 = lambda x: (x[0] -x[2] *x[1]*(x[3] - 1))*(4 - x[0] + x[2]*x[1]*(x[3] - 1))- x[1]*x[1]* (x[3] - 1)*(x[3] - 1)-epsilon
nlc2 = NonlinearConstraint(con, 0, +np.inf)

#con2 = lambda x: cons2(x)-epsilon
#nlc2 = NonlinearConstraint(con2, 0, +np.inf)



#cons={con,con2}
    

#À définir sur une liste de logmoneyness plus étendue [k_min,k_max] 
#Pour les bornes +-  3 standard dev avec la vol impli ATM (méthode en deux temps soit vol impli la plus proche de la vol ATM)
def cons_f(x):
    l=[]
    std=3
    #k_min=-std*vol_atm*sqrt(T)
    #k_max=std*vol_atm*sqrt(T)
    svi=np.array([x[0],x[1],x[2],x[3],x[4]])
    LLM=np.linspace(-1,1,100)
    #LLM=np.linspace(k_min,k_max,100)
    for k in range (len(LLM)):
        l+=[g(LLM[k],svi)]
        
    return l

nonlinear_constraint = NonlinearConstraint(cons_f, epsilon, +np.inf)





# # Traitement de l'ensemble des données 

# In[58]:


Spot,r=S,0
ListeSVIPARAMS=[]
for i in range(len(ListeT)):
    T=ListeMat[i]
    StrikeArray=ListeK[i]
    IVArray=ListeIVMID[i]
    a, b, rho, m, sigma = 0.5*min(IVArray**2*T),0.1,-0.5,0.1,0.1
    y0 = np.array([a, b, rho, m, sigma])
    Result1 =optimize.minimize(lambda X: globalOptimization(X,StrikeArray,IVArray,T,Spot,r), y0,bounds=bounds,constraints=[nlc,nlc2] ,method='SLSQP')#options={'disp': True},constraints=[ nonlinear_constraint])#bounds=bounds #boundaries page 29/52)
    a, b, rho, m, sigma = Result1.x[0], Result1.x[1], Result1.x[2], Result1.x[3], Result1.x[4]
    sviParams =np.array([ a, b, rho, m, sigma])
    ListeSVIPARAMS.append(sviParams)
    
    

    


# In[ ]:


for idx in range(len(ListeSVIPARAMS)):
    
    sviparams=ListeSVIPARAMS[idx]
    StrikeArray=ListeK[idx]
    IVArray=ListeIVMID[idx]
    T=ListeT[idx]
    print(sviparams)
    #sviParams = params[0],params[1],params[2],params[3],params[4]
    impliedVol =np.sqrt(SVI(np.log(StrikeArray/Spot), sviparams)/T)
    plt.plot(StrikeArray*100/Spot, IVArray*100, 'ro', label='Market')
    plt.plot(StrikeArray*100/Spot, impliedVol*100, 'bo', label='SVI')
    plt.xlabel('Strike (%)')
    plt.ylabel('IV (%)')
    plt.grid(linestyle='-')
    plt.legend()
    plt.show()

#plt.plot(StrikeArray*100/Spot, vegaArr, 'go', label='Vega')
#[ 0.66756257  0.06332579 -0.48020885 -0.01890211  0.1521707 ]   
#[ 0.66756257  0.06332579 -0.48020885 -0.01890212  0.1521707 ]


# In[ ]:


idx=4
print(ListeIVMID[idx])
print(ListeK[idx])
print(ListeT[idx])
print(ListeSVIPARAMS[idx])
#params=ListeSVIPARAMS[idx]
StrikeArray=ListeK[idx]
IVArray=ListeIVMID[idx]
TT=ListeT[idx]
#a=0.5*min(IVArray*IVArray*TT)
    
#a, b, rho, m, sigma = 0.5*min(IVArray**2*TT),0.1,-0.5,0.1,0.1
#y0 = np.array([a, b, rho, m, sigma])
#Result1 =optimize.minimize(lambda X: globalOptimization(X,StrikeArray,IVArray,TT,Spot,r), y0, method='SLSQP')
#a, b, rho, m, sigma = Result1.x[0], Result1.x[1], Result1.x[2], Result1.x[3], Result1.x[4]
#sviParams =np.array([ a, b, rho, m, sigma])
sviParams=ListeSVIPARAMS[idx]
impliedVol =np.sqrt(SVI(np.log(StrikeArray/Spot), sviParams)/TT)
plt.plot(StrikeArray*100/Spot, IVArray*100, 'ro', label='Market')
plt.plot(StrikeArray*100/Spot, impliedVol*100, 'bo', label='SVI')
plt.xlabel('Strike (%)')
plt.ylabel('IV (%)')
plt.grid(linestyle='-')
plt.legend()
plt.show()

print(sviParams)


# In[81]:


idx=1
r=0
print(ListeIVMID[idx])
print(ListeK[idx])
print(ListeT[idx])

#params=ListeSVIPARAMS[idx]
StrikeArray=ListeK[idx]
IVArray=ListeIVMID[idx]
TT=ListeT[idx]
a=0.5*min(IVArray*IVArray*TT)



a, b, rho, m, sigma = 0.5*min(IVArray**2*TT),0.1,-0.5,0.1,0.1
y0 = np.array([a, b, rho, m, sigma])
Result1 =optimize.minimize(lambda X: globalOptimization(X,StrikeArray,IVArray,TT,Spot,r), y0,bounds=bounds,constraints=[nlc,nlc2], method='SLSQP')
a, b, rho, m, sigma = Result1.x[0], Result1.x[1], Result1.x[2], Result1.x[3], Result1.x[4]
sviParams=np.array([ a, b, rho, m, sigma])

impliedVol =np.sqrt(SVI(np.log(StrikeArray/Spot), sviParams)/TT)
plt.plot(StrikeArray*100/Spot, IVArray*100, 'ro', label='Market')
plt.plot(StrikeArray*100/Spot, impliedVol*100, 'b', label='SVI')
print("Le score de performance est de " + str(model_perf_SVI(ListeK,ListeT,ListeIVMID,r,Spot,sviParams)))
plt.xlabel('Strike (%)')
plt.ylabel('IV (%)')
plt.grid(linestyle='-')
plt.legend()
plt.show()

print(sviParams)


# In[ ]:


idx=3
print(ListeIVMID[idx])
print(ListeK[idx])
print(ListeT[idx])

#params=ListeSVIPARAMS[idx]
StrikeArray=ListeK[idx]
IVArray=ListeIVMID[idx]
TT=ListeT[idx]
a=0.5*min(IVArray*IVArray*TT)



a, b, rho, m, sigma = 0.5*min(IVArray**2*TT),0.1,-0.5,0.1,0.1
y0 = np.array([a, b, rho, m, sigma])
Result1 =optimize.minimize(lambda X: globalOptimization(X,StrikeArray,IVArray,TT,Spot,r), y0,bounds=bounds ,method='SLSQP',constraints=[ nonlinear_constraint])
a, b, rho, m, sigma = Result1.x[0], Result1.x[1], Result1.x[2], Result1.x[3], Result1.x[4]
sviParams=np.array([ a, b, rho, m, sigma])

impliedVol =np.sqrt(SVI(np.log(StrikeArray/Spot), sviParams)/TT)
plt.plot(StrikeArray*100/Spot, IVArray*100, 'ro', label='Market')
plt.plot(StrikeArray*100/Spot, impliedVol*100, 'b', label='SVI')

plt.xlabel('Strike (%)')
plt.ylabel('IV (%)')
plt.grid(linestyle='-')
plt.legend()
plt.show()

print(sviParams)


# In[60]:


print(ListeK[7])


# # Y'a t'il des arbitrages ? 

# In[61]:


def Butterfly_Arbitrage(ListeK,ListeIVMID,ListeT,S,r):
    n=len(ListeT)
    M=[]
    for i in range(n):
        k=ListeK[i]
        vol=ListeIVMID[i]
        T=ListeT[i]
        nn=len(k)
        L=np.array([])
        for j in range(1,nn-1):
            ButterflyPrice=-2*callBS(S, k[j], T, r, vol[j])+callBS(S, k[j-1], T, r, vol[j-1])+callBS(S, k[j+1], T, r, vol[j+1])
            if ButterflyPrice<=0:
                L=np.append(L,np.array([j-1,j,j+1]))
                
                
                
        if len(L)>0:
            M.append([i,L])
        
    
    return M
            
#Pour une maturité donnée on boucle sur les strikes 
#for i in range(1,len(StrikeArray)):  
  #cs=prixC_BS(K[i-1])-prixC_BS(K[i])
  #if cs<0:
     #Arbitrage existant => on ajoute K[i-1] et K[i]
#for i in range(1,len(StrikeArray)-1):
   #Butterfly=(prixC_BS(K[i-1])-prixC_BS(K[i]))/(K[i]-K[i-1]) - (prixC_BS(K[i])-prixC_BS(K[i+1]))/(K[i+1]-K[i])
    #if Butterfly<0:
     #Arbitrage existant => on ajoute K[i-1] et K[i] et K[i+1]
            
            
def Butterfly_Arbitrage_Bool(ListeK,ListeIVMID,ListeT,S,r):
    M=Butterfly_Arbitrage(ListeK,ListeIVMID,ListeT,S,r)
    if M==[]:
        return False
    else: 
        return True 
    
            
        
            
    
    
    


# In[62]:


print(ListeK)


# In[63]:


Butterfly_Arbitrage(ListeK,ListeIVMID,ListeT,S,r)


# In[64]:


Butterfly_Arbitrage_Bool(ListeK,ListeIVMID,ListeT,S,r)


# In[65]:


A=np.array([])
A=np.append(A,np.array([1,2,2]))
A
np.unique(A)


# In[66]:


# output désiré dans cette exemple  [0,[0,...,6]] ajouté les strike j-1 et j+1 si j est problematique
Butterfly_Arbitrage(ListeK,ListeIVMID,ListeT,S,r)


# In[67]:


def Calendar_arbitrage(ListeK,ListeIVMID,ListeT,S,r):
    n=len(ListeT)
    for k in range(n-1):
        
        idx1=k
        idx2=k+1
        vol1=ListeIVMID[idx1]
        vol2=ListeIVMID[idx2]
        T1=ListeT[idx1]
        T2=ListeT[idx2]
        K1=ListeK[idx1]
        K2=ListeK[idx2]
        L_K=[]
        for k1 in range(len(K1)): 
            for k2 in range(len(K2)):
                if K1[k1]==K2[k2]:
                    L_K.append([K1[k1],k1,k2])
        for k3 in range(len(L_K)):
            Price=callBS(S, L_K[k3][0], T1, r,vol1[L_K[k3][1]])-callBS(S, L_K[k3][0], T2, r,vol2[L_K[k3][2]] )
            if Price>0:
                return (True,L_K[k3])
            
        return False 
        
        
        
        
    
    
        
    
    
                


# In[68]:


Calendar_arbitrage(ListeK,ListeIVMID,ListeT,S,r)


# #  Modèle SABR 

# SABR est un modèle paramétrique pour approcher la volatilité implicite. On a quatre paramètres $\left \{ \alpha ,\beta ,\rho ,\sigma_0\left.  \right \} \right.$ Le modèle est une bonne approximation sous des conditions de marchés qui sont telles que $\varepsilon=\alpha T^{2}$ est petit.
# Ainsi à l'odre 0 d'approximation on a $\sigma =\alpha \frac{log(\frac{F_0}{K})}{D(\zeta )}$ où $\zeta= \frac{\alpha}{\sigma_0(1-\beta )}(F_0^{1-\beta }-K^{1-\beta })$  et  $D(\zeta)=log(\frac{\sqrt{1-2\rho \zeta +\zeta ^{2}}+\zeta -\rho }{1-\rho })$

# In[69]:


def SABR(SABR_Params,Strike,Spot):
    y=Spot/Strike
    alpha, beta, rho,sigma=SABR_Params[0],SABR_Params[1],SABR_Params[2],SABR_Params[3]
    khi=(alpha/(sigma*(1-beta)))*(Spot**(1-beta))*(1-(1/y)**(1-beta))
    
    return alpha*np.log(y)/(np.log((np.sqrt(1-2*rho*khi+khi**2)+khi-rho)/(1-rho)))


# À présent nous allons générer des volatilités implicites avec le modèle SABR.

# In[70]:


IVArray_SABR=[]
Spot=S
epsilon=0
for idx in range(len(ListeK)):
    alpha, beta, rho,sigma =0.5+epsilon,0.9+epsilon,-0.66+epsilon,0.9+epsilon
    SABR_Params = np.array([alpha, beta, rho,sigma])
    StrikeArray=ListeK[idx]
    vol_SABR=SABR(SABR_Params,StrikeArray,Spot)
    IVArray_SABR.append(vol_SABR)
    
    


# In[71]:


IVArray_SABR


# À présent on calibre un modèle SVI sur nos données de volatilité SABR.

# In[72]:


r=0
idx=3
print(IVArray_SABR[idx])
print(ListeK[idx])
print(ListeT[idx])

#params=ListeSVIPARAMS[idx]
StrikeArray=ListeK[idx]
IVArray=IVArray_SABR[idx]
TT=ListeT[idx]

a, b, rho, m, sigma = 0.5*min(IVArray**2*TT),0.1,-0.5,0.1,0.1
y0 = np.array([a, b, rho, m, sigma])
Result1 =optimize.minimize(lambda X: globalOptimization(X,StrikeArray,IVArray,TT,Spot,r), y0,method="SLSQP",bounds=bounds ,constraints=[nlc,nlc2])
a, b, rho, m, sigma = Result1.x[0], Result1.x[1], Result1.x[2], Result1.x[3], Result1.x[4]
sviParams=np.array([ a, b, rho, m, sigma])

impliedVol =np.sqrt(SVI(np.log(StrikeArray/Spot), sviParams)/TT)
plt.plot(StrikeArray*100/Spot, IVArray*100, 'ro', label='SABR')
plt.plot(StrikeArray*100/Spot, impliedVol*100, 'b', label='SVI')

plt.xlabel('Strike (%)')
plt.ylabel('IV (%)')
plt.grid(linestyle='-')
plt.legend()
plt.show()

print(sviParams)


# In[1]:


r=0
idx=4
print(IVArray_SABR[idx])
print(ListeK[idx])
print(ListeT[idx])

#params=ListeSVIPARAMS[idx]
StrikeArray=ListeK[idx]
IVArray=IVArray_SABR[idx]
TT=ListeT[idx]

a, b, rho, m, sigma = 0.5*min(IVArray**2*TT),0.1,-0.5,0.1,0.1
y0 = np.array([a, b, rho, m, sigma])
Result1 =optimize.minimize(lambda X: globalOptimization(X,StrikeArray,IVArray,TT,Spot,r), y0,method="SLSQP")
a, b, rho, m, sigma = Result1.x[0], Result1.x[1], Result1.x[2], Result1.x[3], Result1.x[4]
sviParams=np.array([ a, b, rho, m, sigma])

impliedVol =np.sqrt(SVI(np.log(StrikeArray/Spot), sviParams)/TT)
plt.plot(StrikeArray*100/Spot, IVArray*100, 'ro', label='SABR')
plt.plot(StrikeArray*100/Spot, impliedVol*100, 'b', label='SVI')

plt.xlabel('Strike (%)')
plt.ylabel('IV (%)')
plt.grid(linestyle='-')
plt.legend()
plt.show()

print(sviParams)


# # Calibration SABR sur des données marché

# In[74]:


def globalOptimization_SABR(y,StrikeArray,IVArrayMarket,Spot,r): 
    sabrParams = y[0], y[1], y[2], y[3]
    vol=IVArrayMarket
    #vegaArr=vegaBS(1,StrikeArray/Spot,T,r,IVArray)
    #print(np.mean(vegaArr*(SVI(x,sviParams) - vartot)**2))
    return np.mean((SABR(sabrParams,StrikeArray,Spot) - vol)**2)
#vegaArr*
#vegaArr*
    #vegaArr*


# In[82]:


idx=1
print(ListeIVMID[idx])
print(ListeK[idx])
print(ListeT[idx])
r=0
#params=ListeSVIPARAMS[idx]
StrikeArray=ListeK[idx]
IVArrayMarket=ListeIVMID[idx]
TT=ListeT[idx]


alpha, beta, rho,sigma =0.5,0.9,0.8,0.9
y0 = np.array([alpha, beta, rho,sigma])
Result1 =optimize.minimize(lambda X: globalOptimization_SABR(X,StrikeArray,IVArrayMarket,Spot,r), y0)
alpha, beta, rho,sigma = Result1.x[0], Result1.x[1], Result1.x[2], Result1.x[3]
sabrParams=np.array([ alpha, beta, rho,sigma])

impliedVolSABR =SABR(sabrParams,StrikeArray,Spot)
plt.plot(StrikeArray*100/Spot, IVArrayMarket*100, 'ro', label='Market')
plt.plot(StrikeArray*100/Spot, impliedVolSABR*100, 'b', label='SABR')
print("Le score de performance est de " + str(model_perf_SABR(ListeK,ListeT,ListeIVMID,r,Spot,sabrParams)))
plt.xlabel('Strike (%)')
plt.ylabel('IV (%)')
plt.grid(linestyle='-')
plt.legend()
plt.show()


# # Calibration d'un modèle SVI sur un SABR avec opportunité d'arbitrage

# In[94]:


IVArray_SABR_bis=[]
Spot=S
alpha, beta, rho,sigma =1.2,0.99,-0.6,0.15

SABR_Params = np.array([alpha, beta, rho,sigma])
for idx in range(len(ListeK)):
    StrikeArray=ListeK[idx]
    vol_SABR_bis=SABR(SABR_Params,StrikeArray/Spot,1)
    IVArray_SABR_bis.append(vol_SABR_bis)
    



# In[107]:



StrikeArray=np.exp(np.linspace(-0.8,0.8,25))
Spot=1
IVArray=SABR(SABR_Params,StrikeArray/Spot,1)

TT=3


a, b, rho, m, sigma = 0.5*min(IVArray**2*TT),0.1,-0.5,0.1,0.1
y0 = np.array([a, b, rho, m, sigma])
Result1 =optimize.minimize(lambda X: globalOptimization(X,StrikeArray,IVArray,TT,Spot,r), y0,method="SLSQP",bounds=bounds,constraints=[ nonlinear_constraint])
a, b, rho, m, sigma = Result1.x[0], Result1.x[1], Result1.x[2], Result1.x[3], Result1.x[4]
sviParams=np.array([ a, b, rho, m, sigma])

impliedVol =np.sqrt(SVI(np.log(StrikeArray/Spot), sviParams)/TT)
plt.plot(StrikeArray*100/Spot, IVArray*100, 'ro', label='SABR')
plt.plot(StrikeArray*100/Spot, impliedVol*100, 'b', label='SVI')

plt.xlabel('Strike (%)')
plt.ylabel('IV (%)')
plt.grid(linestyle='-')
plt.legend()
plt.show()

print(sviParams)


# In[96]:


TT


# # Métrique de performance des modèles

# In[79]:


def model_perf_SABR(ListeK,ListeT,IVMarket,r,Spot,params):
    n=len(ListeT)
    Somme=0
    SommeVega=0
    SABR_Params=params[0],params[1],params[2],params[3]
    
    for idx in range(n):
        vol_SABR=SABR(SABR_Params,ListeK[idx],Spot)
        vegaArr=vegaBS(Spot,ListeK[idx],ListeT[idx],r,IVMarket[idx])
        Somme+=np.mean(vegaArr*abs(vol_SABR - IVMarket[idx]))
        SommeVega+=np.mean(vegaArr)
        
    return Somme/SommeVega
        
            
    
        
            
        
    
        
    


# In[80]:


def model_perf_SVI(ListeK,ListeT,IVMarket,r,Spot,params):
    n=len(ListeT)
    Somme=0
    SommeVega=0
    SVI_Params=params[0],params[1],params[2],params[3],params[4]
    
    for idx in range(n):
        vol_SVI=np.sqrt(SVI(np.log(ListeK[idx]/Spot), SVI_Params)/ListeT[idx])
        vegaArr=vegaBS(Spot,ListeK[idx],ListeT[idx],r,IVMarket[idx])
        Somme+=np.mean(vegaArr*abs(vol_SVI - IVMarket[idx]))
        SommeVega+=np.mean(vegaArr)
        
    return Somme/SommeVega
        


# Tâches à faire: 
# 
# -Résoudre le problème au niveau du code: les paramètres en sortie de l'optimisation sont systématiquement les paramètres en entrée : OK
# 
# -Ajouter les contraintes:  
# 
# -Utiliser la méthode SLSQP: OK
# 
# 
# -remplacer les data frame par des numpy array pour la calibration: OK
# 
# 
# -réécrire globaloptization avec comme imput une maturity, npy arr de strike, de vol le spot et le taux r: OK  
# 
# 
# 
# -mieux comprendre les données derebit pour comprendre l'écart entre vol calculé et vol de derebit: pas d'infos sur derebit mais avec un taux sans risque r=-0.8 on retombe sur les bonne vol impli
# 
# 
# -une fonction qui vérifient qu'il y'a pas d'arbitrage : *
# 
# -en maturité interpolation linéaire pour obtenir une surface: *  
# 
# -La fonction qui pour chaque maturité associe les strikes ordonnés et les vols , prix etc ordonné également: ***
# 
# 
# -Voir le problèle avec les contraintes: 
# 
# -ajouter la jacobienne de la fonction coût quand on optimise et g(k): 
# 
# 
# 
# 
# -comprendre ce qu'il se passe pour idx=0: OK mais problème avec idx=7 ???
# -check si les données sont bien ordonnées dans ListeK, ListeIVMID etc ...:  OK 
# 
# -SviParams: t-ulple--->np.array: OK 
# 
# 
# 
# -Comprendre pourquoi ça ne fonctionne pas pour des short-term maturity: RBF interpolation 
# 
# 
# - initial guess: prendre le bon pour a: OK
# 
# 
# 
# - Ajouter la contrainte sur g à la lumière du call avec M.ferhati ok  
# 
# 0- Tester l'autre méthode de contraintes pour annuler les butterfly 
# 
# 1-Commencer à implemter les contraintes temporelles: slice par slice à discuter 
# -butterflyspread price à discuter 
# 
# 
# 2-Calibrer un SABR sur nos données : sigma0 est-ce un paramètre? comment le gérer ? F0=Spot ? 
# 
# 
# 
# 3-Générer avec le modèle SABR et calibrer un modèle un modèle SVI deçu: 
# -introduire des arbitrages sur SABR et voir comment le SVI calibré par dessus réagit 
# 
# 
# 
# 
# 
# priorité: 
# 
# -calendar contraintes à ajouter
# 
# -redefinir constraintes sur g avec k_min k_max (volatm): faire en prenant vol_atm SABR 
# 
# TO DO: 
# -SABR court terme idx=1,2 régler le pb: To do
# 
# -Tester la calibration d'un SVI sur un SABR arbitrable (cf param bastien): PB
# 
# -comparer SVI/SABR: (somme sur T(somme sur K vega*abs(volmodele-volmarket)))/somme sur vega : OK
# 
# -Butterfly_Arbitrage et Calendar_arbitrage avec un spread bid/ask: to do  
# 
# -data binance websocket (en cas de problème compte de client):
# 
# -SABR arbitrage: loss-function y introduire les arbitrages
# 
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




