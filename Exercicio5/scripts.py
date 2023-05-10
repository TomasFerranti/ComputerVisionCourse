"""
funcoes questao1(), questao2(), questao4() implementam os codigos de cada questao
main() é chamada na execução e executa os codigos de todas as questoes
"""
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# Arquivos para as questões
img1Filename = "content/keble_a.jpg"
img2Filename = "content/keble_b.jpg"
img3Filename = "content/keble_c.jpg"

# ----------------------------------------------------------------
# Questão 1
def sobelKernel(kSize, horizontal):
    #ksize é o tamanho do filtro de Sobel usado para calcular os gradientes horizontal e vertical
    #horizontal é booleano indicando se é gradiente horizontal, caso contrario vertical
    #retorna um filtro de sobel tamanho 2*kSize + 1
    x = np.array([1, 2, 1]).reshape(-1, 1)
    y = np.array([1, 0, -1]).reshape(-1, 1)
    
    if horizontal:
        sobel = x @ y.T
    else:
        sobel = y @ x.T

    kernel = x @ x.T
    for i in range(1, kSize):
        kernelBase = np.zeros((kernel.shape[0] + 2, kernel.shape[1] + 2))
        kernelBase[1:-1, 1:-1] = kernel
        sobel = convolve2d(kernel, sobel, mode="same")
        kernel = kernelBase
   
    return sobel / np.sum(np.abs(sobel))

def HarrisCornerDetector(image, blockSize=1, ksize=1, k=0.06):
    #image é uma imagem em tons de cinza
    #blockSize é o tamanho da vizinhança considerada para a detecção de cada canto
    #ksize é o tamanho do filtro de Sobel usado para calcular os gradientes horizontal e vertical
    #k é um parâmetro livre do detector de Harris na equação
    #retorna uma imagem binária com o score de cada pixel
    nlin, ncol = image.shape
    image = np.float32(image)
    image = image/255.0

    sobelX = sobelKernel(ksize, True)
    sobelY = sobelKernel(ksize, False)

    Ix = convolve2d(image, sobelX, mode="same")
    Iy = convolve2d(image, sobelY, mode="same")

    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy
    Rresult = np.zeros((nlin, ncol))
    for lin in range(blockSize, nlin - blockSize):
        for col in range(blockSize, ncol - blockSize):
            xi, xf = lin-blockSize, lin+blockSize+1
            yi, yf = col-blockSize, col+blockSize+1
            Ix2_block = Ix2[xi:xf, yi:yf]
            Iy2_block = Iy2[xi:xf, yi:yf]
            Ixy_block = Ixy[xi:xf, yi:yf]
            Sxx = np.sum(Ix2_block)
            Syy = np.sum(Iy2_block)
            Sxy = np.sum(Ixy_block)
            det = Sxx*Syy - Sxy**2
            trace = Sxx + Syy
            R = det - k * (trace ** 2)
            Rresult[lin, col] = R            
    return Rresult

def NonMaximalSupression(Rresult, threshold=0.01, window_size=11):
    #Rresult é a matriz de scores de Harris
    #threshold é o valor corte para considerar possíveis cantos
    #window_size é o tamanho da janela que este deve ser o máximo
    #retorna uma imagem binária indicando quais pontos são de destaque
    nlin, ncol = Rresult.shape
    halfWindowSize = window_size // 2

    masks = Rresult > threshold
    featMask = np.zeros_like(Rresult)
    for lin in range(window_size, nlin-window_size):
        for col in range(window_size, ncol-window_size):
            xi, xf = lin-halfWindowSize, lin+halfWindowSize+1
            yi, yf = col-halfWindowSize, col+halfWindowSize+1
            if masks[lin, col] != 0:
                values = (Rresult[xi:xf, yi:yf])[masks[xi:xf, yi:yf]]
                if Rresult[lin,col] == np.max(values):
                    featMask[lin, col] = 1
    return featMask

def questao1():
    # Meu algoritmo
    img = cv.imread(img1Filename)
    imgG = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    harris = HarrisCornerDetector(imgG)
    featMask = NonMaximalSupression(harris)

    yInd, xInd = np.where(featMask == 1)
    for ind in range(xInd.shape[0]):
        y, x = yInd[ind], xInd[ind]
        cv.circle(img=img, center=(int(x),int(y)), radius=5, color=(255,0,0), thickness=-1)

    # OpenCV
    features = cv.goodFeaturesToTrack(image=imgG, maxCorners=100, 
                                      qualityLevel=0.01, minDistance = 10, blockSize=3, useHarrisDetector=True, k=0.06)
    imgOpenCV = img.copy()
    for i in features:
        x,y = i.ravel()
        cv.circle(img=imgOpenCV,center = (int(x),int(y)),radius = 5,color=(255,0,0),thickness = -1)

    # Plot
    img = img[:, :, np.array([2, 1, 0])]
    imgOpenCV = imgOpenCV[:, :, np.array([2, 1, 0])]

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)    
    
    ax[0].imshow(img, vmin=0, vmax=255)    
    ax[1].imshow(imgOpenCV, vmin=0, vmax=255)

    ax[0].set_title("Meu script")
    ax[1].set_title("Função OpenCV")

    plt.suptitle("Comparação entre os pontos obtidos pelo script com o OpenCV")
    plt.savefig("output/img1.png", dpi=300)
    plt.show()

# ----------------------------------------------------------------
# Questão 2
def getFeatures(img):
    #img é uma imagem em tons RGB
    #retorna uma lista com elementos [x, y] indicando coordenadas dos pontos de destaque
    imgG = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    harris = HarrisCornerDetector(imgG)
    featMask = NonMaximalSupression(harris)

    features = []
    xInd, yInd = np.where(featMask == 1)
    for ind in range(xInd.shape[0]):
        x, y = xInd[ind], yInd[ind]
        features.append(np.array([x, y]))
    return features

def boundingBoxRadius(x, y, width, height, kernelSize):
    #x, y são as coordenadas centrais
    #width, height são as dimensões da imagem
    #kernelSize é o tamanho do kernel centrado
    #retorna índices válidos referentes a um kernel de tamanho no máximo kernelSize
    halfKernel = kernelSize // 2
    xi, xf = x - halfKernel, x + halfKernel + 1
    yi, yf = y - halfKernel, y + halfKernel + 1
    xi, yi = max(0, xi), max(0, yi)
    xf, yf = min(width, xf), min(height, yf)
    dxl, dxr = x - xi, xf - x
    dyl, dyr = y - yi, yf - y
    return dxl, dxr, dyl, dyr

def featureMatch(img1, img2, patchSize=10, corrThreshold=0.8):
    #img1 é uma imagem em tons RGB
    #img2 é uma imagem em tons RGB
    #patchSize é o tamanho dos patches para calcular correlação
    #corrThreshold é o valor de corte para os casamentos
    #retorna uma lista com elementos [p0,p1] onde p0, p1 são coordenadas (x,y) dos casamentos na img1, img2
    features1 = getFeatures(img1)
    features2 = getFeatures(img2)

    corrMatrix = np.zeros((len(features1), len(features2)))

    # Preencher matriz de correlação
    for i, p1 in enumerate(features1):
        x1, y1 = p1
        dxl_1, dxr_1, dyl_1, dyr_1 = boundingBoxRadius(x1, y1, img1.shape[0], img1.shape[1], patchSize)

        for j, p2 in enumerate(features2):
            x2, y2 = p2
            dxl_2, dxr_2, dyl_2, dyr_2 = boundingBoxRadius(x2, y2, img2.shape[0], img2.shape[1], patchSize)
            dxl, dxr, dyl, dyr = min(dxl_1, dxl_2), min(dxr_1, dxr_2), min(dyl_1, dyl_2), min(dyr_1, dyr_2)

            patch1 = img1[x1-dxl:x1+dxr, y1-dyl:y1+dyr].ravel()
            patch2 = img2[x2-dxl:x2+dxr, y2-dyl:y2+dyr].ravel()

            corrMatrix[i, j] = np.abs(np.corrcoef(patch1, patch2)[0, 1])
            
    # Para cada ponto da primeira imagem, seleciona o ponto da segunda imagem com maior correlação
    matches = []
    for i, p1 in enumerate(features1):
        bestMatch = np.argmax(corrMatrix[i, :])
        if np.argmax(corrMatrix[:, bestMatch]) == i:
            if corrMatrix[i, bestMatch] > corrThreshold:
                matches.append([features1[i], features2[bestMatch]])
    
    return matches

def questao2():
    # Pegar pontos de destaque
    img1 = cv.imread(img1Filename)
    img2 = cv.imread(img2Filename)

    matches = featureMatch(img1, img2)

    imgDraw = np.concatenate([img1[:, :, np.array([2, 1, 0])], img2[:, :, np.array([2, 1, 0])]], axis=1)

    # Plot
    width = img1.shape[1]
    plt.imshow(imgDraw, vmin=0, vmax=255)
    for k in range(len(matches)):
        feat1, feat2 = matches[k]
        feat2[1] += width
        plt.plot([feat1[1], feat2[1]], [feat1[0], feat2[0]])
    
    plt.title(f"Correspondência entre as {len(matches)} matches obtidas nas duas imagens")
    plt.tight_layout()
    plt.savefig("output/img2.png", dpi=300)
    plt.show()

# ----------------------------------------------------------------
# Questão 4
def computeHomography(points):
    #points é um numpy array de tamanho [n,4]
    #retorna a homografia entre [n,:2] e [n,2:]
    A = []
    for i in range(points.shape[0]):
        x, y, u, v = points[i, 0], points[i, 1], points[i, 2], points[i, 3]
        A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1, :].reshape(3, 3)
    return H

def ransacHomography(matchesArr, numIte=1000, errorThreshold=10):
    #matchesArr é um array de coordenadas de casamentos de tamanho [n,4]
    #numIte é o numero de iteracoes do RANSAC
    #errorThreshold é o parametro de distancia dos inliers na projeção
    #retorna a melhor homografia entre aqueles casamentos

    # Numero de pontos para calcular homografia
    k = 4

    # Melhores parametros encontrados
    maxInliers = 0
    bestH = None

    # Iniciar RANSAC
    for i in range(numIte):
        # Aleatoriamente amostrar
        sample = matchesArr[np.random.choice(matchesArr.shape[0], k, replace=False)]
        
        # Estimar homographia
        H = computeHomography(sample)

        # Computar erros de projeção
        nMatches = matchesArr.shape[0]
        coordsHomo = np.column_stack((matchesArr[:, 0:2], np.ones(nMatches)))
        coordsProj = np.dot(H, coordsHomo.T).T
        coordsProj /= coordsProj[:, 2].reshape(-1, 1)
        projError = np.sqrt(np.sum((matchesArr[:, 2:4] - coordsProj[:, 0:2])**2, axis=1))
        
        # Encontrar pontos válidos
        inliers = matchesArr[projError < errorThreshold]
        n_inliers = inliers.shape[0]
        
        # Atualizar o melhor caso seja uma melhora
        if n_inliers > maxInliers:
            maxInliers = n_inliers
            bestH = computeHomography(inliers)
    return bestH, maxInliers

def questao4():
    # Calcular homografia
    np.random.seed(0)

    img1 = cv.imread(img1Filename)
    img2 = cv.imread(img2Filename)

    matches = featureMatch(img1, img2)
    matches = [list(f1)+list(f2) for f1, f2 in matches]

    matchesArr = np.array(matches).reshape(-1, 4)

    H, nInliers = ransacHomography(matchesArr)

    # Plot
    pts1 = matchesArr[:, 0:2]
    pts2 = matchesArr[:, 2:4]
    pts2Transformed = np.dot(H, np.column_stack((pts2, np.ones(pts2.shape[0]))).T).T
    pts2Transformed /= pts2Transformed[:, 2].reshape(-1, 1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img1[:, :, np.array([2, 1, 0])])
    ax[0].scatter(pts1[:, 1], pts1[:, 0], s=10, c='r')
    ax[0].set_title('Imagem 1')

    ax[1].imshow(img2[:, :, np.array([2, 1, 0])])
    ax[1].scatter(pts2Transformed[:, 1], pts2Transformed[:, 0], s=10, c='r')
    ax[1].set_title('Imagem 2 (com pontos projetados)')
    
    plt.suptitle(f"Homografia RANSAC com {nInliers} Pontos Projetados Corretamente")
    plt.tight_layout()
    plt.savefig("output/img4.png", dpi=300)
    plt.show()


# ----------------------------------------------------------------
# Enquadrar imagens
def removeImgBorders():
    img_filenames = os.listdir("output/")
    offset = 10
    for img_filename in img_filenames:
        if ".png" not in img_filename and ".jpg" not in img_filename:
            continue
        # Caminho para o diretório
        img_filename = "output/" + img_filename

        # Carregar imagem
        img = cv.imread(img_filename)
        width, height = img.shape[0], img.shape[1]

        # Converter para escala de cinza
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Achar os indexes dos primeros pixels nao brancos de cada direção
        width, height = gray.shape
        for i in range(0, int(width/2)):
            if np.unique(gray[i, :]).shape[0] > 1:
                xL = i - offset
                xL = max(0, xL)
                break
        for i in range(width - 1, int(width/2), -1):
            if np.unique(gray[i, :]).shape[0] > 1:
                xR = i + offset
                xR = min(xR, width)
                break
        for i in range(0, int(height/2)):
            if np.unique(gray[:, i]).shape[0] > 1:
                yL = i - offset
                yL = max(0, yL)
                break
        for i in range(height - 1, int(height/2), -1):
            if np.unique(gray[:, i]).shape[0] > 1:
                yR = i + offset
                yR = min(yR, height)
                break

        # Cortar a imagem usando os indices
        cropped_image = img[xL:xR, yL:yR, :]

        # Salvar imagem
        cv.imwrite(img_filename, cropped_image)

if __name__ == "__main__":
    print("Executando os códigos das questões em sequência.")
    print("\tExecutando código da questão 1...")
    questao1()
    print("\tExecutando código da questão 2...")
    questao2()
    print("\tExecutando código da questão 4...")
    questao4()
    print("Finalizando cortando as imagens")
    removeImgBorders()
    print("Finalizado!")