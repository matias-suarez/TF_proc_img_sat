# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 22:12:45 2018

@author: Sergio
"""
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage

def ajuste_lineal_trozos(vo, vt, v):
    '''
  Realiza un ajuste lineal a trozos para un array 2D
  Transforma mediante ajuste lineales el valor de entrada a una salida entre 0 y 1
  Inputs:
  v0 : valores que definen los umbrales de corte
       tienen que ser valores ordenados Reales
  vt : valores a los que se transforman los valores de corte,
       tienen que estar en el intervalo [0,1]
  v : array 2 D de entrada   
  Ouput:
  vv : array 2 D transformado
    '''
    vv=np.copy(v)
    for j in range(len(vo)-1):
            i=np.argwhere((vo[j] <= v)  & (v <= vo[j+1]))
            if np.size(i) >0:
                vv[i[:,0],i[:,1]]=(vt[j+1]-vt[j])/(vo[j+1]-vo[j])*(v[i[:,0],i[:,1]]-vo[j])+vt[j]
    vv[vv<0] = 0.
    vv[vv>1] = 1.
    return vv  

def realce_p(imag, p = 2):
  '''
  input:
    imag : imagen a realzar linealmente
    p : porcentaje de puntos a derecha e izquierda a eliminar para reescalar. Valor maximo 50 
  '''
  if (p>=50):
      print('Valor máximo excedido: p < 50 ')
      return
  else:
      aux = np.sort(imag.flatten()) #ordena de mayor a menor
      imin = int(len(aux) * p / 100)
      imax = int(len(aux) * (100 - p) / 100) -1 #Para que sea válido para p=0
      vmin = float(aux[imin]) # para no tener problemas con uint8
      vmax = aux[imax]
      rimag = (imag - vmin) / (vmax - vmin)
      rimag[rimag < 0] = 0  
      rimag[rimag > 1] = 1
      return rimag

def realce_ecualiza(imag, n = 10):
    '''
  input:
    imag : imagen a realzar linealmente
    n : número de puntos para uniformizar
    '''
    vt = np.linspace(0, 1, n+1)#valores de salida
    vo = np.zeros(n+1)
    
    frec, edges = np.histogram(imag, bins=256, range=[0, 1])
    frec = frec/imag.size

    suma = np.cumsum(frec)
    for i in range(1, n+1):
        j = np.argmin(abs(suma - i/n))
        vo[i] = edges[j+1]
        vt[i] = i/n

    return ajuste_lineal_trozos(vo, vt, imag)

def realce_gama(img, vmin= 0, vmax= 1, gama = 1):
    '''
  input:
    img : imagen a realzar con exponente gama
    vmin = valor mínimo de la imagen que irá a 0 
    vmin = valor mínimo de la imagen que irá a 1
    gama : exponente
    '''
    imgaux = (img - vmin) / (vmax - vmin)
    imgaux[img <= vmin] = 0
    imgaux[img >= vmax] = 1
    return imgaux**gama


def rgb2hsi(imagen):
    R=imagen[:,:,0]
    G=imagen[:,:,1]
    B=imagen[:,:,2]
    dRG, dRB, dGB = R - G, R - B, G - B
    aux=np.arccos(.5*(dRG+dRB)/((dRG)**2+dRB*dGB)**.5)
    aux[np.isnan(aux)] = 0.0
    H1=np.copy(aux)
    H2=2*np.pi-aux
    H1[G<B]=0
    H2[G>=B]=0
    H=H1+H2
    I=(R+G+B)/3.
    S=1-np.min(imagen,axis=2)/I
    S[np.isnan(S)] = 0.0
    S[S<0]=0.0
    return H,S,I

def hsi2rgb(H,S,I):
    B1=I*(1-S)
    B21=I*(1+(S*np.cos(H))/np.cos(np.pi/3-H))
    B22=I*(1+(S*np.cos(H-np.pi*2/3.))/np.cos(np.pi/3-H+np.pi*2/3.))
    B23=I*(1+(S*np.cos(H-np.pi*4/3.))/np.cos(np.pi/3-H+np.pi*4/3.))
    B31=3*I-B1-B21
    B32=3*I-B1-B22
    B33=3*I-B1-B23
    iaux1=np.argwhere(H<=np.pi*2/3.)
    iaux2=np.argwhere((H>np.pi*2/3.)*(H<=np.pi*4/3.))
    iaux3=np.argwhere(H>np.pi*4/3.)
    R=np.zeros(H.shape)
    G=np.zeros(H.shape)
    B=np.zeros(H.shape)
    
    R[iaux1[:,0],iaux1[:,1]]=B21[iaux1[:,0],iaux1[:,1]]
    R[iaux2[:,0],iaux2[:,1]]=B1[iaux2[:,0],iaux2[:,1]]
    R[iaux3[:,0],iaux3[:,1]]=B33[iaux3[:,0],iaux3[:,1]]

    G[iaux1[:,0],iaux1[:,1]]=B31[iaux1[:,0],iaux1[:,1]]
    G[iaux2[:,0],iaux2[:,1]]=B22[iaux2[:,0],iaux2[:,1]]
    G[iaux3[:,0],iaux3[:,1]]=B1[iaux3[:,0],iaux3[:,1]]

    B[iaux1[:,0],iaux1[:,1]]=B1[iaux1[:,0],iaux1[:,1]]
    B[iaux2[:,0],iaux2[:,1]]=B32[iaux2[:,0],iaux2[:,1]]
    B[iaux3[:,0],iaux3[:,1]]=B23[iaux3[:,0],iaux3[:,1]]

    R[R<0]=0
    G[G<0]=0
    B[B<0]=0

    R[R>1]=1
    G[G>1]=1
    B[B>1]=1

    return R,G,B,iaux1,iaux2,iaux3


def hsi2rgbPunto(H,S,I):
    B1=I*(1-S)
    B21=I*(1+(S*np.cos(H))/np.cos(np.pi/3-H))
    B22=I*(1+(S*np.cos(H-np.pi*2/3.))/np.cos(np.pi/3-H+np.pi*2/3.))
    B23=I*(1+(S*np.cos(H-np.pi*4/3.))/np.cos(np.pi/3-H+np.pi*4/3.))
    B31=3*I-B1-B21
    B32=3*I-B1-B22
    B33=3*I-B1-B23
    
    if (H<=np.pi*2/3.):
        R=B21
        G=B31
        B=B1
        iaux=1
    elif (H>np.pi*2/3. and H<=np.pi*4/3.):
        R=B1
        G=B22
        B=B32 
        iaux=2
    else:    
        R=B33
        G=B1
        B=B23
        iaux=3

    if (R<0): R=0
    if (G<0): G=0
    if (B<0): B=0

    return R,G,B,iaux

def rgb2hsiPunto(R,G,B):
    dRG, dRB, dGB = R - G, R - B, G - B
    aux=(dRG**2+dRB*dGB)
    if aux==0.:
        H1=0.
    else:
        H1=np.arccos(.5*(dRG+dRB)/aux**.5)
    H2=2*np.pi-aux
    if G<B: H1=0
    if G>=B:H2=0
    H=H1+H2
    I=(R+G+B)/3.
    S=1-np.min([R,G,B])/I
    if np.isnan(S) or S<0: S= 0.0
    return H,S,I

def clasific0(imagen,c1): #clasifica por umbrales solamente
    ic1=((imagen[:,:,0] >= c1[0][0])*(imagen[:,:,0] <= c1[0][1])*
         (imagen[:,:,1] >= c1[1][0])*(imagen[:,:,1] <= c1[1][1])*
         (imagen[:,:,2] >= c1[2][0])*(imagen[:,:,2] <= c1[2][1]))   
    return ic1


def filtro0(imagen,ker): #Filtro
    fil,col=np.shape(imagen)
    sL=np.shape(ker)[0]/2#semilado del filtro
    imagenF=np.copy(imagen)
    for j in range(sL,fil-sL):
        for i in range(sL,col-sL):
            imagenF[j,i]=(imagen[j-sL:j+sL+1,i-sL:i+sL+1]*ker).sum()
    
    return imagenF

#Pasa al centro los corners de una imagen
def centF(Mat):
    fil,col=np.shape(Mat)
    MatC=np.zeros([fil,col])
    fil_2 = int(fil/2)
    col_2 = int(col/2)
    MatC[fil_2:,col_2:]=Mat[:fil_2,:col_2]
    MatC[:fil_2,col_2:]=Mat[fil_2:,:col_2]
    MatC[:fil_2,:col_2:]=Mat[fil_2:,col_2:]
    MatC[fil_2:,:col_2]=Mat[:fil_2,col_2:]
    return MatC


#Funciรณn para crear una mascara de un rectangulo orientado y centrado dentro
# de un cuadrado
def rectangulo(b, h, ang):#angulo en sexagesimal
    lado0 = int(np.max([b, h]))
    Mat0 = np.zeros([lado0, lado0])
    
    if h > b:
        Mat0[:h, int((h-b)/2): int((h+b)/2)]=1
    else:
        Mat0[int((b-h)/2): int((h+b)/2), :b]=1

    Mat=ndimage.rotate(Mat0,ang,reshape='False')
    Mat[Mat>=.1]=1
    Mat[Mat<.1]=0

    return Mat

def genbanda(ancho,dist,ang,L): #generador de imagen con banda
    L1=2*L
    im0=np.zeros([L1,L1])
    for i in range(int(L1/dist)-1):
        im0[i*dist:i*dist+ancho,:]=1
    im1=ndimage.rotate(im0,ang,reshape='False')
    im1[im1>=.1]=1
    im1[im1<.1]=0
    
    nx,ny = np.shape(im1)
    
    nx1 = int(nx/2-L/2)
    ny1 = int(ny/2-L/2)
    return im1[nx1:nx1+L, ny1:ny1+L]
