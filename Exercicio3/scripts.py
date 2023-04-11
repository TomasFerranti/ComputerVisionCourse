"""
funcoes questao1(), questao2(), questao3(), questao4() implementam os codigos de cada questao
main() é chamada na execução e executa os codigos de todas as questoes
"""
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Arquivo para questão 1 e 2
img1_filename = 'content/PoucoContraste.png'

# Arquivos para questão 3 e 4
img2_filename = 'content/Chess_Board.svg.png'

# ----------------------------------------------------------------
# Questão 1
def changeImg(img, brightness=0, contrast=1, convertToColor=True):
  # Convert to grey
  imgGrey = (np.mean(img, axis=2) / 255)

  # Brightness
  imgGrey = imgGrey + brightness
  imgGrey = np.clip(imgGrey, 0, 1)

  # Contrast
  mean = np.mean(imgGrey)
  imgGrey = contrast * (imgGrey - mean) + mean
  imgGrey = np.clip(imgGrey, 0, 1)

  if not convertToColor:
    return imgGrey

  # Convert to color
  width, height = imgGrey.shape
  imgColor = (255 * imgGrey).astype("uint8")
  imgColor = np.concatenate(3 * [imgColor.reshape(width, height, 1)], axis = 2)
  return imgColor

def questao1():
    # Ler imagem e executar
    img = cv.imread(img1_filename)
    img = changeImg(img, 0, 0.5)

    # Display
    fig, ax = plt.subplots(1, 1, figsize=(10, 15))
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    fig.savefig("figs/img1.png", dpi=300)

# ----------------------------------------------------------------
# Questão 2
def questao2():
    brightness = 0.5
    contrast = 0.5

    img = cv.imread(img1_filename)
    imgGrey = changeImg(img, convertToColor=False)
    imgGreyB = changeImg(img, brightness=brightness, convertToColor=False)
    imgGreyC = changeImg(img, contrast=contrast, convertToColor=False)

    imgGreyB_cv = cv.convertScaleAbs(img, beta=brightness)
    imgGreyC_cv = cv.convertScaleAbs(img, alpha=contrast)

    imgGreyB_cv = changeImg(imgGreyB_cv, convertToColor=False)
    imgGreyC_cv = changeImg(imgGreyC_cv, convertToColor=False)

    rows, cols = 2, 3
    fig, ax = plt.subplots(rows, cols, figsize=(20,10))
    ax[1][0].axis('off')

    nBins = 20

    hists = [imgGrey.ravel(), imgGreyB.ravel(), imgGreyC.ravel(), None, imgGreyB_cv.ravel(), imgGreyC_cv.ravel()]
    titles = ["Original distribution", f"My brightness={brightness} dist", f"My contrast={contrast} dist",
            None, f"CV brightness={brightness} dist", f"CV contrast={contrast} dist"]
    for row in range(rows):
        for col in range(cols):
            hist = hists[row * cols + col]
            title = titles[row * cols + col]
            if type(hist) != type(None):
                cAx = ax[row][col]
                cAx.hist(hist, bins=nBins)
                cAx.set_xlabel("Gray value")
                cAx.set_ylabel("Pixels count")
                cAx.set_title(title)
    fig.savefig("figs/img2.png", dpi=300)
    
# ----------------------------------------------------------------
# Questão 3
def applyFilter(img, filter, convertToColor=True):
    # Convert to grey
    width, height = img.shape
    widthf, heightf = filter.shape

    # Filter
    imgFiltered = np.zeros((width - widthf, height - heightf))
    for i in range(0, width - widthf):
        for j in range(0, height - heightf):
            imgFiltered[i, j] = np.sum(img[i:i+widthf, j:j+heightf] * filter)
    imgFiltered = np.clip(imgFiltered, 0, 1)

    if not convertToColor:
        return imgFiltered

    # Convert to color
    width, height = imgFiltered.shape
    imgColor = (255 * imgFiltered).astype("uint8")
    imgColor = np.concatenate(3 * [imgColor.reshape(width, height, 1)], axis = 2)
    return imgColor

def questao3():
    filter1 = np.ones((3,3))
    filter2_ver = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    filter2_hor = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    filter3_ver = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    filter3_hor = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])    
    filter4 = np.array([1, 4, 6, 4, 1]).reshape(-1, 1)
    filter4 = filter4 @ filter4.T
    filters = [filter1, filter2_ver, filter2_hor, filter3_ver, filter3_hor, filter4]
    filters = [f / np.sum(np.abs(f)) for f in filters]

    img = cv.imread(img2_filename)
    imgGrey = (np.mean(img, axis=2) / 255)
    imgs = [applyFilter(imgGrey, f, convertToColor=False) for f in filters]
    imgs.append(np.round(np.sqrt(imgs[3]**2 + imgs[4]**2)))

    cols = 4
    rows = int(np.ceil(len(filters) / cols))
    
    fig, ax = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    imgCounter = 0
    titles = ["Constant", "Vertical Derivative", "Horizontal Derivative", None, "Vertical Sobel", "Horizontal Sobel", "Gaussian Blur 5x5", "Gradient Sobel"]
    for i in range(len(titles)):
        cAx = ax[i//cols][i%cols]
        if type(titles[i]) != type(None):
            cAx.imshow(imgs[imgCounter], cmap='gray', vmin=0, vmax=1)
            cAx.set_title(titles[i])
            imgCounter += 1
        else:
            cAx.imshow(np.zeros(imgs[0].shape), cmap='gray', vmin=0, vmax=1)
            
    fig.savefig("figs/img3.png", dpi=300)

# ----------------------------------------------------------------
# Questão 4
def halfImage(img):
    width, height = img.shape
    imgH = np.zeros((int(width/2),int(height/2)))
    for i in range(imgH.shape[0]):
        for j in range(imgH.shape[0]):
            imgH[i, j] = img[2*i, 2*j]
    return imgH

def questao4():
    img = cv.imread(img2_filename)
    img = (np.mean(img, axis=2) / 255)

    imgH = halfImage(img)

    filter = np.array([1, 4, 6, 4, 1]).reshape(-1, 1)
    filter = filter @ filter.T
    filter = filter / np.sum(np.abs(filter))

    imgHS = halfImage(applyFilter(img, filter, convertToColor=False))

    fig, ax = plt.subplots(1, 2, figsize=(10,15))
    ax[0].imshow(imgH, cmap='gray', vmin=0, vmax=1)
    ax[1].imshow(imgHS, cmap='gray', vmin=0, vmax=1)
    ax[0].set_title("Image halved without suavization")
    ax[1].set_title("Image halved with suavization (Gaussian Blur 5x5)")
    fig.savefig("figs/img4.png", dpi=300)

# ----------------------------------------------------------------
# Enquadrar imagens
def removeImgBorders():
    img_filenames = os.listdir("figs/")
    for img_filename in img_filenames:
        # Path to folder
        img_filename = "figs/" + img_filename

        # Load the image
        img = cv.imread(img_filename)

        # Convert it to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find indexes of first non white in 4 directions
        width, height = gray.shape
        for i in range(0, int(width/2)):
            if np.unique(gray[i, :]).shape[0] > 1:
                xL = i
                break
        for i in range(width - 1, int(width/2), -1):
            if np.unique(gray[i, :]).shape[0] > 1:
                xR = i
                break
        for i in range(0, int(height/2)):
            if np.unique(gray[:, i]).shape[0] > 1:
                yL = i
                break
        for i in range(height - 1, int(height/2), -1):
            if np.unique(gray[:, i]).shape[0] > 1:
                yR = i
                break

        # Crop the image using the bounding box
        cropped_image = img[xL:xR, yL:yR, :]

        # Save the cropped image locally
        cv.imwrite(img_filename, cropped_image)

if __name__ == "__main__":
    print("Executando os códigos das questões em sequência.")
    print("\tExecutando código da questão 1...")
    questao1()
    print("\tExecutando código da questão 2...")
    questao2()
    print("\tExecutando código da questão 3...")
    questao3()
    print("\tExecutando código da questão 4...")
    questao4()
    print("Finalizando cortando as imagens")
    removeImgBorders()
    print("Finalizado!")