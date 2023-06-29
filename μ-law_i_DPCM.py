#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import sounddevice as sd
import soundfile as sf
from scipy.interpolate import interp1d


def kwantyzacja(data, koncowy_format):
    data = data.copy()
    m_pocz = -1
    n_pocz = 1
    if np.issubdtype(data.dtype,np.integer):
        m_pocz = np.iinfo(data.dtype).min
        n_pocz = np.iinfo(data.dtype).max
    
   
    do_bitow = int(''.join(i for i in koncowy_format if i.isdigit()))
    do_typu = str(''.join(i for i in koncowy_format if i.isalpha()))
    
    m_konc = -1
    n_konc = 1
    if do_typu == 'int':
        m_konc = int(2**do_bitow/2 * -1)
        n_konc = int(2**do_bitow/2 - 1)
    if do_typu == 'uint':
        m_konc = 0 
        n_konc = int(2**do_bitow-1)
        
    
    
    for i in range(len(data)):    
        
        Z_A = (data[i] - m_pocz)/(n_pocz - m_pocz)
        Z_C = np.floor(Z_A*(n_konc-m_konc)+m_konc)

        Z_A = (Z_C - m_konc)/(n_konc - m_konc)
        if np.issubdtype(data.dtype,np.integer):
            data[i]=int(np.round(Z_A*(n_pocz-m_pocz)+m_pocz))
        else:
            data[i]=Z_A*(n_pocz-m_pocz)+m_pocz
    return data


def kompresja_mu_law(x):
    x = x.copy()
    mask = np.logical_or(x <= 1, x >= -1)
    x[mask] = np.sign(x[mask])*(np.log(1+255*np.abs(x[mask])))/(np.log(1+255))
    return x

def dekompresja_mu_law(x):
    x = x.copy()
    mask = np.logical_or(x <= 1, x >= -1)
    x[mask] = np.sign(x[mask])*(1/255)*((1+255)**np.abs(x[mask])-1)
    return x


def DPCM_koder(x, do_bitow):
    e = np.zeros(x.shape, dtype='float32')
    y = np.zeros(x.shape, dtype='float32')
    yp = np.zeros(x.shape, dtype='float32')

    for i in range(len(x)):
        if i == 0:
            yp[i] = x[i] - 0
            y[i] = kwantyzacja(np.array([yp[i], 0.2], dtype='float32'), do_bitow)[0]
            e[i] = y[i] - 0

        else:
            yp[i] =x[i]-e[i-1]
            y[i] = kwantyzacja(np.array([yp[i],0.2], dtype='float32'), do_bitow)[0]
            e[i] = y[i] + e[i-1]
    return y

def DPCM_dekoder(y):
    xp = np.zeros(y.shape, dtype='float32')
    for i in range(len(y)):
        if i == 0 :
            xp[i] = y[i] + 0
        else:
            xp[i] = y[i] + xp[i-1]
    return xp


# In[31]:


data, fs = sf.read('sing_high1.wav', dtype='float32')# MOLIWE PRZEKSZTALCENIA:
                                                       # z int16 na [int32;int2]   #z float32 na [int32;int2]
sf.write(r'C:\Users\antoni\Downloads\WYKRESY\org.wav', data, fs)
for i in [8,7,6,5,4,3,2]:
    kompr = kompresja_mu_law(data)
    kompr_kwant = kwantyzacja(kompr, 'int'+str(i))
    dekompr_kwant = dekompresja_mu_law(kompr_kwant)
    sf.write(r'C:\Users\antoni\Downloads\WYKRESY\sing_low1_mu'+str(i)+str('.wav'), dekompr_kwant, fs)
    plt.plot(np.linspace(-1,1,data.shape[0]), dekompr_kwant)
    plt.title("mu-law kwantyzacja do " + str(i) + "-bitow")
    plt.xlim(-0.8,-0.795)
    plt.ylim(-0.2, 0.2)
    plt.show()


# In[50]:


data, fs = sf.read('sing_high1.wav', dtype='float32')# MOLIWE PRZEKSZTALCENIA:
                                                       # z int16 na [int32;int2]   #z float32 na [int32;int2]
sf.write(r'C:\Users\antoni\Downloads\WYKRESY\org.wav', data, fs)
bity = np.arange(2,13)[::-1]
for i in bity:
    y = DPCM_koder(data, 'int'+str(i))
    xp = DPCM_dekoder(y)
    sf.write(r'C:\Users\antoni\Downloads\WYKRESY\sing_low1_mu'+str(i)+str('.wav'), xp, fs)
    plt.plot(np.linspace(-1,1,data.shape[0]), xp)
    plt.title("DPCM kwantyzacja do " + str(i) + "-bitow")
    plt.xlim(-0.8,-0.795)
    plt.ylim(-0.25, 0.25)
    plt.show()


# In[ ]:




