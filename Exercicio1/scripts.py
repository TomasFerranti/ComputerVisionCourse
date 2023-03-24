"""
funcoes questao1(), questao2(), questao3() implementam os codigos de cada questao
main() é chamada na execução e executa os codigos de todas as questoes
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Arquivo para questão 1 e 2
img1_filename = 'content/1200px-Palazzo_Farnese_Fassade.jpg'

# Arquivos para questão 3
img2_filename = 'content/gol-vasco-flamengo 2023.jpg'
img3_filename = 'content/campo-futebol.png'

# Parâmetro de raio dos pontos quadrados
r = 5

# Função para plotar e salvar imagens
img_count = 0
def imshow(img):
    global img_count
    fig = plt.figure(frameon=False)
    h, w = img.shape[:-1]
    #fig.set_size_inches(w,h)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    img_count = img_count + 1
    fname = f'figs/img{img_count}.png'
    ax.imshow(img[:, :, np.array([2, 1, 0])])#, aspect='auto')
    fig.savefig(fname, dpi=300)
    plt.show()

# ----------------------------------------------------------------
# Questão 1
def my_rotation(img, angulo, centro):
    
    # nessa primeira parte, vamos definir a transformação que leva a posicao dos pixels da imagem original
    # para a posicao dos pixels do imagem rotacionada.
    
    # a primeira matriz de translação muda a origem das coordenadas do canto da imagem para o centro da imagem
    matriz_translacao = np.array([[1, 0, -centro[0]],
                                  [0, 1, -centro[1]],
                                  [0, 0, 1         ]])
    # a matriz de rotacao aplica a rotacao em torno da origem
    matriz_rotacao = np.array([[np.cos(angulo), -np.sin(angulo), 0],
                               [np.sin(angulo), np.cos(angulo) , 0],
                               [0             , 0              , 1]])
    # a composicao coloca todas as matrizes em uma só: aplica a translacao (muda a origem), rotaciona, volta para a origem anterior
    matriz_composicao = np.linalg.inv(matriz_translacao) @ matriz_rotacao @ matriz_translacao
    
    # criar imagem rotacionada em preto, com mesmas dimensões da original
    height, width = img.shape[:2]
    rotated_image = np.zeros((height, width, 3), np.uint8)
    # o próximo passo é percorrer cada pixel da nova imagem e verificar qual é o pixel correspondente na imagem original
    m_comp_inv = np.linalg.inv(matriz_composicao)
    for linr in range(height):
        for colr in range(width):
            pos_rot = np.array([linr, colr, 1]).reshape(-1, 1)
            pos_orig = m_comp_inv @ pos_rot
            pos_orig = pos_orig / pos_orig[2]
            lin = round(pos_orig[0, 0])
            col = round(pos_orig[1, 0])
            if (lin >= 0 and lin < height) and (col >= 0 and col < width):
                #opa, é um pixel pertencente à imagem original...
                rotated_image[linr, colr] = img[lin, col]
                
    return rotated_image

def questao1():
    # Ler imagem e dimensoes
    img = cv.imread(img1_filename)
    height, width = img.shape[:2]

    # Centro da imagem e angulo em radianos
    centro = (height/2, width/2)
    angulo = np.pi/6

    # Rotação
    my_rotated_image = my_rotation(img, angulo, centro)

    # Display
    imshow(my_rotated_image)
    
# ----------------------------------------------------------------
# Questão 2
def transformacao_projetiva(img, matriz):
    # criar imagem rotacionada em preto, com mesmas dimensões da original
    height, width = img.shape[:2]
    transformed_image = np.zeros((height,width,3), np.uint8)
    # o próximo passo é percorrer cada pixel da nova imagem e verificar qual é o pixel correspondente na imagem original
    m_comp_inv = np.linalg.inv(matriz)
    for linr in range(height):
        for colr in range(width):
            pos_rot = np.array([linr, colr, 1]).reshape(-1, 1)
            pos_orig = m_comp_inv @ pos_rot
            pos_orig = pos_orig / pos_orig[2]
            lin = round(pos_orig[0,0])
            col = round(pos_orig[1,0])
            if (lin >= 0 and lin < height) and (col >= 0 and col < width):
                #opa, é um pixel pertencente à imagem original...
                transformed_image[linr, colr] = img[lin, col]

    return transformed_image

def questao2():
    img = cv.imread(img1_filename)
    
    # Matriz de projeção da direção X para o ponto de fuga (2000, 0)
    T = np.array([[1, 0     , 0],
                  [0, 1     , 0],
                  [0, 1/2000, 1]])
    img = transformacao_projetiva(img, T)

    imshow(img)
    
# ----------------------------------------------------------------
# Questão 3
def encontrar_transformacao_projetiva(fonte, destino):
    # Matrizes a preencher do sistema linear H * h = b
    H = np.zeros((8, 8))
    b = np.zeros(8).reshape(-1, 1)

    for i in range(0, 4):
        x = fonte[i][0]
        y = fonte[i][1]
        u = destino[i][0]
        v = destino[i][1]
        H[2*i    , :] = np.array([x, y, 1, 0, 0, 0, -u*x, -u*y])
        H[2*i + 1, :] = np.array([0, 0, 0, x, y, 1, -v*x, -v*y])
        b[2*i    ] = u
        b[2*i + 1] = v
    
    h = np.linalg.solve(H, b).ravel().tolist()

    # Preencher matriz projetiva com os valores de h
    matrix = np.zeros((3, 3))
    matrix[0, :] = np.array([h[0], h[1], h[2]])
    matrix[1, :] = np.array([h[3], h[4], h[5]])
    matrix[2, :] = np.array([h[6], h[7], 1   ])
    return matrix

def questao3():
    # Pontos na foto do jogo
    img_jogo = cv.imread(img2_filename)
    fonte = [np.array([145, 132]),
             np.array([145, 615]),
             np.array([340, 2  ]),
             np.array([340, 205])]
    
    # Pontos na foto do campo
    img_campo = cv.imread(img3_filename)
    destino = [np.array([157, 27 ]),
               np.array([157, 192]),
               np.array([450, 27 ]),
               np.array([450, 81 ])]
    
    # Desenhar pontos acompanhados de imagens
    for img, pontos in ((img_jogo, fonte), (img_campo, destino)):
        img_draw = img.copy()
        for ponto in pontos:
            x, y = ponto
            img_draw[x-r:x+r, y-r:y+r, :] = 0
        imshow(img_draw)

    # Pontos referentes às posições dos jogadores na imagem do jogo
    pontos_desenhar_jogo = [(np.array([265, 195]), "r"),
                            (np.array([260, 240]), "g"),
                            (np.array([256, 104]), "b")]
    
    # Pontos referentes às posições dos jogadores na imagem do campo
    P = encontrar_transformacao_projetiva(fonte, destino)
    add_hom = lambda x : np.array([x[0], x[1], 1])
    rem_hom = lambda x : (x / x[2])[:-1]
    transformacao = lambda x : rem_hom(P @ add_hom(x))
    pontos_desenhar_campo = [(transformacao(x), c) for x, c in pontos_desenhar_jogo]
    
    # Desenhar ambas imagens com seus pontos e cores
    dic_cor = {"r":[255, 0, 0], "g":[0, 255, 0], "b":[0, 0, 255]}
    for img, pontos_cor in ((img_jogo, pontos_desenhar_jogo), (img_campo, pontos_desenhar_campo)):
        img_draw = img.copy()
        for ponto, cor in pontos_cor:
            x, y = ponto
            x, y = int(x), int(y)
            for idx, value in enumerate(dic_cor[cor]):
                img_draw[x-r:x+r, y-r:y+r, idx] = value
        imshow(img_draw)

if __name__ == "__main__":
    print("Executando os códigos das questões em sequência.")
    print("\nRealizando transformação da questão 1...")
    questao1()
    print("\nRealizando transformação da questão 2...")
    questao2()
    print("\nRealizando transformação da questão 3...")
    questao3()