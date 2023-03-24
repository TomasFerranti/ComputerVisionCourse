# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2 as cv
import sys
import numpy as np

#%%  **********  você pode usar esta solução, mas não neste exercício!!! ****************
img = cv.imread('..\\Imagens\\1200px-Palazzo_Farnese_Fassade.jpg')

# Obter o centro da imagem
height, width = img.shape[:2]
center = (width/2, height/2)

# Definir a matriz de rotação
M = cv.getRotationMatrix2D(center, 30, 1)

# Aplicar a rotação na imagem
rotated_img = cv.warpAffine(img, M, (width, height))

#cv.imshow('Display window', rotated_img)
#cv.waitKey(0); cv.destroyAllWindows()

#%%

def my_rotation(img, angulo, centro):
    
    # nessa primeira parte, vamos definir a transformação que leva a posicao dos pixels da imagem original
    # para a posicao dos pixels do imagem rotacionada.
    
    # a primeira matriz de translação muda a origem das coordenadas do canto da imagem para o centro da imagem
    matriz_translacao = ?
    # a matriz de rotacao aplica a rotacao em torno da origem
    matriz_rotacao = ?
    # a composicao coloca todas as matrizes em uma só: aplica a translacao (muda a origem), rotaciona, volta para a origem anterior
    matriz_composicao = ?
    
    # criar imagem rotacionada em preto, com mesmas dimensões da original
    height, width = img.shape[:2]
    rotated_image = np.zeros((height,width,3), np.uint8)
    # o próximo passo é percorrer cada pixel da nova imagem e verificar qual é o pixel correspondente na imagem original
    m_comp_inv = matriz_composicao.I
    for linr in range(height):
        for colr in range(width):
            pos_rot = np.matrix([linr, colr, 1]).T
            pos_orig = ?
            lin = round(pos_orig[0,0]); col = round(pos_orig[1,0]);
            if (lin >=0 and lin < height) and (col >= 0 and col < width):
                #opa, é um pixel pertencente à imagem original...
                rotated_image[linr, colr] = ?
                
    return rotated_image
#%%
img = cv.imread('..\\Imagens\\1200px-Palazzo_Farnese_Fassade.jpg')
height, width = img.shape[:2]
centro = (height/2, width/2)
angulo = np.pi/6
my_rotated_image =  my_rotation(img, angulo, centro)

#%%
cv.imshow('Display window', my_rotated_image)
cv.waitKey(0); cv.destroyAllWindows()


