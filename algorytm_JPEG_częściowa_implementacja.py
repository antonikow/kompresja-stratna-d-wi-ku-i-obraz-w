#!/usr/bin/env python
# coding: utf-8



import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import scipy.fftpack


def dekoder(data):
    rozmiar = 1
    start = 0 
    for el in data:
        if el == np.inf:
            break
        else:
            start += 1
            rozmiar *= int(el)
    start += 1 # pomijamy np.inf
    d = np.zeros(rozmiar)
    
    idx_zakod = np.arange(start,len(data))
    zakodowany = data[idx_zakod].astype(int)
    occurences = np.arange(len(zakodowany))
    occurences = zakodowany[occurences[occurences % 2 == 0]]
    numbers = np.arange(len(zakodowany))
    numbers = zakodowany[numbers[numbers % 2 == 1]]
    it = 0 
    for i in range(len(numbers)): 
        num = numbers[i]
        occur = occurences[i]
        for j in range(occur):
            d[it] = num
            it += 1
            
    d = d.reshape(data[0:start-1].astype(int))
    return d            
    
def koduj(data): # tylko dla danych nie bedacych -inf w celu unikniecia rozszerzania tablicy zakodowanej co iteracje
    k = np.zeros((data.flatten().shape[0]*5))
    k[:]=-np.inf
    for i in range(len(data.shape)):
        k[i] = data.shape[i]
    k[len(data.shape)] = np.inf # koniec fragmentu przechowujacego wymiary
    prev = np.inf
    licznik = 0
    idx_k = len(data.shape)+1
    data = data.flatten()
    for i in range(len(data)):
        if i == 0: # pierwsza iteracja
            prev = data[0]
            licznik += 1 
            if i == len(data)-1: # jesli pierwsza iteracja jest ostatnia
                k[idx_k] = 1 
                k[idx_k+1] = data[i]
                
        elif prev == data[i]: # powtorzenie
            licznik += 1
            if i == len(data)-1:
                k[idx_k] = licznik 
                k[idx_k+1] = prev
        else:                  # natrafiono na inny element
            k[idx_k] = licznik 
            k[idx_k+1] = prev
            idx_k += 2
            prev = data[i]
            licznik = 1
            if i == len(data)-1: # natrafiono na inny element, a ostatnia iteracja wiec zapisz 
                k[idx_k] = 1 
                k[idx_k+1] = data[i]
    return k[0:np.argmin(k)]


def rgb2(RGB):
    RGB = RGB.copy()
    YCrCb=cv2.cvtColor(RGB,cv2.COLOR_RGB2YCrCb).astype(np.uint8)
    return YCrCb

def irgb2(YCrCb):
    YCrCb = YCrCb.copy()
    RGB=cv2.cvtColor(YCrCb.astype(np.uint8),cv2.COLOR_YCrCb2RGB)
    return RGB

def subsampling(x, text):
    x = x.copy()
    S=list(map(int,text.split(':')))
    if S == [4,2,2]:
        x[:,1::2,:]=0
    return x

def resubsampling(x, text):
    x=x.copy()
    S=list(map(int,text.split(':')))
    if S == [4,2,2]:
        x[:,1::2,:]=x[:,::2,:]
    return x

def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a.astype(float), axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a.astype(float), axis=0 , norm='ortho'), axis=1 , norm='ortho')


def img2blocks(RGB):
    bloki = []
    for i in range(0, RGB.shape[0], 8):
        for j in range(0, RGB.shape[1], 8):
            bloki.append(RGB[i:i+8,j:j+8])
            
    return np.array(bloki, dtype=int) 

def blocks2img(bloki, wymiary): # dostaje liste 
    bloki = np.array(bloki, dtype=int)
    wiersze = []
    for i in range(0,len(bloki),int(wymiary[1]/8)):
        wiersze.append(np.hstack(bloki[i:i+int(wymiary[1]/8)]))
    wiersze = np.array(wiersze)
    
    return np.vstack(wiersze)

def kwant_wzor(d, Q):
    return np.round(d/Q).astype(int)

def dekwant_wzor(qd, Q):
    return qd*Q

def kwantyzacja(threeblocksDCT, Q, lum=True): #kwantyzacja domyslnie luminancji
    threeblocksQuant = threeblocksDCT.copy().astype(int) #uzyskanie ksztaltu macierzy
    if lum: #kwantyzacja luminancji
        for i in [0]: # iteracja po wymiarach
            for j in range(threeblocksDCT.shape[2]): # iteracja po blokach
                threeblocksQuant[i][j] = kwant_wzor(threeblocksDCT[i][j], Q)
    else:
        print("na chrom")
        for i in [1, 2]: # iteracja po wymiarach
            for j in range(threeblocksDCT.shape[2]): # iteracja po blokach
                threeblocksQuant[i][j] = kwant_wzor(threeblocksDCT[i][j], Q)
    return threeblocksQuant

def dekwantyzacja(threeblocksQuant, Q, lum=True): #kwantyzacja domyslnie luminancji
    threeblocksDeQuant = threeblocksQuant.copy().astype(int) #uzyskanie ksztaltu macierzy
    if lum: #kwantyzacja luminancji
        for i in [0]: # iteracja po wymiarach
            for j in range(threeblocksQuant.shape[2]): # iteracja po blokach
                threeblocksDeQuant[i][j] = dekwant_wzor(threeblocksQuant[i][j], Q)
    else:
        print("na chrom")
        for i in [1, 2]: # iteracja po wymiarach
            for j in range(threeblocksQuant.shape[2]): # iteracja po blokach
                threeblocksDeQuant[i][j] = dekwant_wzor(threeblocksQuant[i][j], Q)
    return threeblocksDeQuant

def zigzag(A):
    template= n= np.array([
            [0,  1,  5,  6,  14, 15, 27, 28],
            [2,  4,  7,  13, 16, 26, 29, 42],
            [3,  8,  12, 17, 25, 30, 41, 43],
            [9,  11, 18, 24, 31, 40, 44, 53],
            [10, 19, 23, 32, 39, 45, 52, 54],
            [20, 22, 33, 38, 46, 51, 55, 60],
            [21, 34, 37, 47, 50, 56, 59, 61],
            [35, 36, 48, 49, 57, 58, 62, 63],
            ])
    if len(A.shape)==1:
        B=np.zeros((8,8))
        for r in range(0,8):
            for c in range(0,8):
                B[r,c]=A[template[r,c]]
    else:
        B=np.zeros((64,))
        for r in range(0,8):
            for c in range(0,8):
                B[template[r,c]]=A[r,c]
    return B

def block2zigzag(wej, wymiary):
    wektor = []
    for i in range(3):
        for j in range(wej.shape[1]):
            wektor.append(zigzag(wej[i][j]))
    wektor = np.array(wektor).flatten().astype(int)
    return wektor

def zigzag2block(wej):
    blokownawymiar = int(wej.shape[0]/(3*64))
    unzigzagged = np.zeros((3,blokownawymiar,8,8))
    pocz = 0
    for i in range(3):
        for j in range(blokownawymiar):
            unzigzagged[i][j] = zigzag(wej[pocz:pocz+64])
            pocz += 64
    return unzigzagged.astype(int)

def threeblocks2img(wej, imgwymiary):
    obraz = []
    for i in range(3):
        obraz.append(blocks2img(wej[i],imgwymiary[0:2]))
    obraz = np.array(obraz)
    return obraz

def img2threeblocks(wej):
    threeblocks = []
    for i in range(3):
        threeblocks.append(img2blocks(wej[:,:,i]))
    threeblocks = np.array(threeblocks)
    
    de = []
    for i in range(3):
        de.append(blocks2img(threeblocks[i],wej.shape))

    odtw = np.dstack(de)
    print("poprawnie zblokowano: ", np.all(wej==odtw))
    
    return threeblocks


def wyswietl1(tytul, IMG, YCrCb, subs, poIDCT, resubs, IMGfinal):  
    fig, axs = plt.subplots(6, 1 , sharey=True   )
    fig.set_size_inches(int(44/3), int(68/3))
    fig.suptitle(tytul)
    fig.subplots_adjust(top=0.96)
    axs[0].imshow(IMG)
    axs[0].set_title('oryginal RGB')
    axs[1].imshow(YCrCb)
    axs[1].set_title('YCrCb')
    axs[2].imshow(subs)
    axs[2].set_title('subsampling')
    axs[3].imshow(poIDCT)
    axs[3].set_title('odwrotna DCT')
    axs[4].imshow(resubs)
    axs[4].set_title('resubsampling')
    axs[5].imshow(IMGfinal) 
    axs[5].set_title('odtworzony RGB')
    plt.show()
    
def wyswietl2(tytul, IMG, YCrCb, subs, poIDCT, resubs, IMGfinal):  
    fig, axs = plt.subplots(6, 1 , sharey=True   )
    fig.set_size_inches(int(44/3), int(68/3))
    
    axs[0].imshow(YCrCb[:,:,0],cmap=plt.cm.gray)
    axs[0].set_title('Y YCrCb')
    axs[1].imshow(YCrCb[:,:,1],cmap=plt.cm.gray)
    axs[1].set_title('Cr YCrCb')
    axs[2].imshow(YCrCb[:,:,2],cmap=plt.cm.gray)
    axs[2].set_title('Cb YCrCb')
    
    axs[3].imshow(subs[:,:,0],cmap=plt.cm.gray)
    axs[3].set_title('Y subsampling')
    axs[4].imshow(subs[:,:,1],cmap=plt.cm.gray)
    axs[4].set_title('Cr subsampling')
    axs[5].imshow(subs[:,:,2],cmap=plt.cm.gray)
    axs[5].set_title('Cb subsampling')
    
    plt.show()
    
    
def wyswietl3(tytul, IMG, YCrCb, subs, poIDCT, resubs, IMGfinal):  
    fig, axs = plt.subplots(6, 1 , sharey=True   )
    fig.set_size_inches(int(44/3), int(68/3))
    
    
    axs[0].imshow(poIDCT[:,:,0],cmap=plt.cm.gray)
    axs[0].set_title('Y poIDCT')
    axs[1].imshow(poIDCT[:,:,1],cmap=plt.cm.gray)
    axs[1].set_title('Cr poIDCT')
    axs[2].imshow(poIDCT[:,:,2],cmap=plt.cm.gray)
    axs[2].set_title('Cb poIDCT')
    
    axs[3].imshow(resubs[:,:,0],cmap=plt.cm.gray)
    axs[3].set_title('Y resubsampling')
    axs[4].imshow(resubs[:,:,1],cmap=plt.cm.gray)
    axs[4].set_title('Cr resubsampling')
    axs[5].imshow(resubs[:,:,2],cmap=plt.cm.gray)
    axs[5].set_title('Cb resubsampling')
    
    plt.show()


class ver:
    def __init__(self, RGB):
        self.RGB = RGB
        self.shape = RGB.shape
        self.YCrCb = None
        self.subs = None
        self.threeblocks = None
        self.threeblocksDCT = None
        self.resubs = None
        self.threeblocksQuant = None
        
def JPEG(sciezka, tytul, chrominancja, Q, kwantyzacja_na_luminancji):
    RGB = plt.imread(sciezka)
    RGB = RGB[:,:,0:3].copy()
    height = RGB.shape[0]
    width = RGB.shape[1]
    RGB = RGB[0:height-height%16, 0:width-width%16, :]
    data = ver(RGB)
    data.YCrCb = rgb2(data.RGB)


    text = chrominancja

    data.subs = subsampling(data.YCrCb, text)
    data.threeblocks = img2threeblocks(data.subs)
    data.threeblocksDCT = dct2(data.threeblocks-128)


    if kwantyzacja_na_luminancji:
        data.threeblocksQuant = kwantyzacja(data.threeblocksDCT, Q) # kwantyzacja na Y
    else:
        data.threeblocksQuant = kwantyzacja(data.threeblocksDCT, Q, False) # kwantyzacja na Cr i Cb
        
    data.zigzagged = block2zigzag(data.threeblocksQuant, data.threeblocksQuant.shape)
    przeslane = koduj(data.zigzagged)
    odebrane = dekoder(przeslane) 
    data.unzigzagged = zigzag2block(odebrane)
    if kwantyzacja_na_luminancji:
        data.threeblocksDeQuant = dekwantyzacja(data.unzigzagged, Q) # kwantyzacja na Y
    else:
        data.threeblocksDeQuant = dekwantyzacja(data.unzigzagged, Q, False) # kwantyzacja na Cr i Cb
    data.threeblocksIDCT = idct2(data.threeblocksDeQuant)
    data.poIDCT = np.dstack(threeblocks2img(data.threeblocksIDCT, data.shape)[0:3])+128
    data.resubs = resubsampling(data.poIDCT, text) 
    data.iRGB = irgb2(data.resubs)
    wyswietl1(tytul, RGB, data.YCrCb, data.subs, data.poIDCT, data.resubs, data.iRGB)
    wyswietl2(tytul, RGB, data.YCrCb, data.subs, data.poIDCT, data.resubs, data.iRGB)
    wyswietl3(tytul, RGB, data.YCrCb, data.subs, data.poIDCT, data.resubs, data.iRGB)
    
    
    
    
QY = np.array([
            [16, 11, 10, 16, 24,  40,  51,  61],
            [12, 12, 14, 19, 26,  58,  60,  55],
            [14, 13, 16, 24, 40,  57,  69,  56],
            [14, 17, 22, 29, 51,  87,  80,  62],
            [18, 22, 37, 56, 68,  109, 103, 77],
            [24, 36, 55, 64, 81,  104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
            ])
QC = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        ])
Q1 = np.ones((8,8)).astype(int)

sciezka = 'images.jpg'
JPEG(sciezka, "redukcja chrominancji 4:4:4", "4:4:4", Q1, True)
JPEG(sciezka, "redukcja chrominancji 4:2:2", "4:2:2", Q1, True)

JPEG(sciezka, "kwantyzacja luminancji", "4:4:4", QY, True)
JPEG(sciezka, "kwantyzacja chrominancji", "4:4:4", QC, False)


# In[ ]:





