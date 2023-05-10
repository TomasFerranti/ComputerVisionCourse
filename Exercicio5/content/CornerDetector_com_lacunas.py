#%%
import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('..\\Imagens\\frutas.jpg')
cv2.imshow('Frutas', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
impb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
features = cv2.goodFeaturesToTrack(image=impb, maxCorners=100, 
            qualityLevel=0.01, minDistance = 10, blockSize=3, useHarrisDetector=True, k=0.06)
cv2.imshow('Frutas', impb)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
# Iterate over the corners and draw a circle at that location
for i in features:
    x,y = i.ravel()
    cv2.circle(img = img,center = (int(x),int(y)),radius = 5,color=(255,0,0),thickness = -1)
cv2.imshow('Frutas', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
# usando opencv
harris = cv2.cornerHarris(impb, 2, 3, 0.04)
# %%
def HarrisCornerDetector(image, blockSize, ksize, k=0.06):
    #image é uma imagem em tons de cinza
    #blockSize é o tamanho da vizinhança considerada para a detecção de cada canto
    #ksize é o tamanho do filtro de Sobel usado para calcular os gradientes horizontal e vertical
    #k é um parâmetro livre do detector de Harris na equação
    #retorna uma imagem binária com o score de cada pixel
    nlin, ncol = image.shape
    image = np.float32(image)
    image = image/255.0
    Ix = 
    Iy = 
    Ix2 = 
    Iy2 = 
    Ixy = 
    Rresult = np.zeros((nlin, ncol))
    cont = 0
    for lin in range(blockSize, nlin-blockSize):
        for col in range(blockSize, ncol-blockSize):
            Ix2_block = 
            Iy2_block = 
            Ixy_block = 
            Sxx = 
            Syy = 
            Sxy = 
            det = Sxx*Syy - Sxy**2
            trace = Sxx + Syy
            R = 
            Rresult[lin, col] = R
            if R > 0.01:
                cont += 1
    print(cont)
    return Rresult

#%%
Rresult = HarrisCornerDetector(impb, 1, 3, 0.05)
RresultCV = cv2.cornerHarris(impb, 1, 3, 0.05)


#for i in range(npoints):
#    PointsInterest[ind] = 1
cv2.imshow('Harris', cv2.normalize(Rresult, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
npoints = 1000
PointsInterest = np.zeros(impb.shape)
top_indices = np.argpartition(Rresult.flatten(), -npoints)[-npoints:]
top_row_indices, top_col_indices = np.unravel_index(top_indices, impb.shape)
for i in range(npoints):
    PointsInterest[top_row_indices[i], top_col_indices[i]] = 1

PointsInterestcv = np.zeros(impb.shape)
top_indicescv = np.argpartition(Rresult.flatten(), -npoints)[-npoints:]
top_row_indicescv, top_col_indicescv = np.unravel_index(top_indicescv, impb.shape)
for i in range(npoints):
    PointsInterestcv[top_row_indicescv[i], top_col_indicescv[i]] = 1
cv2.imshow('Harris', PointsInterest)
cv2.imshow('HarrisCV', PointsInterestcv)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
# Encontre os pontos de máximo usando Non Maximal Supression (máximos em janelas 3x3)
def NonMaximalSupression(im, window_size=3):
    nl = im.shape[0]
    nc = im.shape[1]
    maximos = []
    for lin in range(window_size, nl-window_size):
        for col in range(window_size, nc-window_size):
            window = 
    
    return maximos