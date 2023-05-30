"""
funcao questao() implementa os codigos de cada questao
main() é chamada na execução e executa os codigos de todas as questoes
"""
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Arquivos para as questões
img1Filename = "content/keble_a.jpg"
img2Filename = "content/keble_b.jpg"
img3Filename = "content/keble_c.jpg"

# ----------------------------------------------------------------
# Questão
def findSiftKeypoints(imgG):
    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(imgG, None)
    return keypoints, descriptors

def drawKeyPoints(img, keypoints):
    image_with_keypoints = cv.drawKeypoints(img, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return image_with_keypoints

def imshow(ax, img):
    ax.imshow(img[:, :, np.array([2, 1, 0])], vmin=0, vmax=255)

def findSiftMatches(kp1, kp2, desc1, desc2):
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    return good_matches

def drawMatches(img1, img2, kp1, kp2, matches):
    matching_result = cv.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matching_result

def questao():
    img1 = cv.imread(img1Filename)
    img1G = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    kp1, desc1 = findSiftKeypoints(img1G)
    img1Draw = drawKeyPoints(img1, kp1)

    # Questão 2
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)    
    imshow(ax, img1Draw)
    ax.set_title(f"Keypoints do SIFT com tamanho e orientação")
    plt.savefig("output/img1.png", dpi=300)
    plt.show()

    # Questão 3
    kp_plot = kp1[0]
    for kp_i in kp1:
        if kp_i.size > kp_plot.size:
            kp_plot = kp_i

    attributes = {
        "pt": kp_plot.pt,
        "size": kp_plot.size,
        "angle": kp_plot.angle,
        "response": kp_plot.response,
        "octave": kp_plot.octave
    }
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)    
    img2Draw = drawKeyPoints(img1, [kp_plot])
    imshow(ax, img2Draw)

    keys = list(attributes.keys())
    values = list(attributes.values())
    ax.text(attributes["pt"][0], attributes["pt"][1] + 100, '\n'.join([f'{k}: {v}' for k, v in zip(keys, values)]),
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=12,
            bbox=dict(facecolor='red', alpha=0.5))
    
    ax.set_title(f"Olhando um Keypoint específico do SIFT")
    plt.savefig("output/img2.png", dpi=300)
    plt.show()

    # Questão 4
    img2 = cv.imread(img2Filename)
    img2G = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    kp2, desc2 = findSiftKeypoints(img2G)
    matches = findSiftMatches(kp1, kp2, desc1, desc2)
    img3Draw = drawMatches(img1, img2, kp1, kp2, matches)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)    
    imshow(ax, img3Draw)
    ax.set_title(f"Correspondências usando FLANN com KNN nos Keypoints SIFT entre duas imagens")
    plt.savefig("output/img3.png", dpi=300)
    plt.show()

    # Questão 5
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homography, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    dist_from_border = 100
    square_points = np.float32([[dist_from_border, dist_from_border], 
                                [img1G.shape[1] - dist_from_border, dist_from_border], 
                                [img1G.shape[1] - dist_from_border, img1G.shape[0] - dist_from_border],
                                [dist_from_border, img1G.shape[0] - dist_from_border]]).reshape(-1, 1, 2)
    projected_points = cv.perspectiveTransform(square_points, homography)
    img4DrawL = cv.polylines(img1.copy(), [np.int32(square_points)], True, (0, 255, 0), 2)
    img4DrawR = cv.polylines(img2.copy(), [np.int32(projected_points)], True, (0, 255, 0), 2)
    img4Draw = np.concatenate([img4DrawL, img4DrawR], axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)    
    imshow(ax, img4Draw)
    ax.set_title(f"Projeção do retângulo da imagem 1 na imagem 2 usando a homografia do RANSAC")
    plt.savefig("output/img4.png", dpi=300)
    plt.show()

    # Questão 6
    homography_inverse = np.linalg.inv(homography)
    img5Draw = cv.warpPerspective(img2, homography_inverse, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    img5Draw[0:img1.shape[0], 0:img1.shape[1]] = img1

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)    
    imshow(ax, img5Draw)
    ax.set_title(f"Costura das duas imagens usando a homografia inversa da imagem 2 para a imagem 1")
    plt.savefig("output/img5.png", dpi=300)
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
    print("\tExecutando códigos da questões...")
    questao()
    print("Finalizando cortando as imagens")
    removeImgBorders()
    print("Finalizado!")