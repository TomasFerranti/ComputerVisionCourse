#%%
import cv2
import numpy as np
#%%
#abrir o vídeo de input para a "camera lenta" artificial
video_path = '..\imagens\\onca.mp4'
# 8 frames inseridos a cada par de frames consecutivos do vídeo original
fator = 8 

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#%%
#  criar os vídeos de output
# interpolacao por repeticao
outrep_width = width
outrep_height = height
# interpolacao linear
outlin_width = width
outlin_height = height
# interpolacao por fluxo optico
outopt_width = width
outopt_height = height
# video com os 3 métodos combinados lado a lado
outcomb_width = 3*width 
outcomb_height = height

# Criar objeto VideoWriter para salvar os vídeos
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
outrep_path = '..\imagens\out_rep.mp4'
outlin_path = '..\imagens\out_lin.mp4'
outopt_path = '..\imagens\out_opt.mp4'
outcomb_path = '..\imagens\out_comb.mp4'

outrep = cv2.VideoWriter(outrep_path, fourcc, fps, (outrep_width, outrep_height))
outlin = cv2.VideoWriter(outlin_path, fourcc, fps, (outlin_width, outlin_height))
outopt = cv2.VideoWriter(outopt_path, fourcc, fps, (outopt_width, outopt_height))
outcomb = cv2.VideoWriter(outcomb_path, fourcc, fps, (outcomb_width, outcomb_height))

#%%
# Função auxiliar para combinar os frames dos vídeos em um único frame
def combinar_frames(frames):
    # Define as dimensões do novo frame
    height = frames[0][0].shape[0]
    width = frames[0][0].shape[1]
    channels = frames[0][0].shape[2]
    combined_frame = np.zeros((height, width * len(frames), channels), dtype=np.uint8)

    # Combina os frames dos vídeos
    for i, frame in enumerate(frames):
        combined_frame[:, i * width : (i + 1) * width, :] = frame[0]

    return combined_frame

#%%
# mapa das coordenadas x e y (video original) - para o método por fluxo ótico
coord_x, coord_y = np.meshgrid(np.arange(width), np.arange(height))

cont_frames = 0; total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT); bloco = int(total_frames/10)

#rebobinar o vídeo original para o início
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, prev_frame = cap.read()
while cap.isOpened():
    cont_frames += 1

    #imprimir o progresso do processamento
    if cont_frames % bloco== 0:
        print('Processando: ', int(cont_frames/bloco)*10, '%')
    ret, frame = cap.read()

    if not ret:
        break
    # sequencia comeca com o frame anterior (prev_frame)
    frame_repeat = cv2.resize(prev_frame, (outrep_width, outrep_height))
    frame_linear = cv2.resize(prev_frame, (outlin_width, outlin_height))
    frame_optflow = cv2.resize(prev_frame, (outopt_width, outopt_height))

    frame_combinado = combinar_frames([[frame_repeat], [frame_linear], [frame_optflow]])
    frame_combinado = cv2.resize(frame_combinado, (outcomb_width, outcomb_height))
    # escreve cada frame no video de saida correspondente
    outrep.write(frame_repeat)
    outlin.write(frame_linear)
    outopt.write(frame_optflow)
    outcomb.write(frame_combinado)
    
    # Efetuar o fluxo ótico
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calcular o fluxo ótico.  Use o método de Farneback já implementado no OpenCV
    # os parâmetros do método são:
    # prev_gray: frame anterior em escala de cinza
    # gray: frame atual em escala de cinza
    # None: sem máscara
    # 0.5: pirâmide de escala, fator de escala
    # 3: número de níveis da pirâmide
    # 15: tamanho da janela de vizinhança
    # 3: número de iterações do algoritmo
    # 5: tamanho da janela de média para suavização
    # 1.2: desvio padrão do filtro gaussiano
    # 0: flags
    
    flow = # insira a função para calcular o fluxo ótico#

    # inserir frames intermediários
    for i in range(1, fator):
        # Interpolação por repetição
        frame_repeat = prev_frame

        # Interpolação linear
        # a função addWeighted() do OpenCV pode ser usada para combinar dois frames
        # por interpolação linear.  Para combinar frames é melhor usar a função, que evita
        # que você se preocupe com o tipo numérico e problemas de arredondamento
        frame_linear = cv2.addWeighted(prev_frame, (fator - i) / fator,frame, i / fator, 0)

        # Interpolação por fluxo ótico
        # modificando as coordenadas x e y de acordo com o fluxo ótico
        # o mapa map_x contém as coordenadas x de cada pixel do frame atual
        # ele deve ser calculado a partir do mapa de coordenadas x original (coord_x)
        # e do fluxo ótico (flow)
        # use uma interpolação linear, com peso (i/fator), para incluir a fração do fluxo ótico que deve ser considerada
        # não é preciso usar a função cv2.addWeighted() aqui!  São apenas matrizes de números reais.
        map_x = coord_x #complete aqui# 
        map_y = coord_y #complete aqui#
        frame_optflow = cv2.remap(prev_frame, map_x.astype(np.float32),\
                        map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)
        # finalmente, faça a interpolação linear entre o próximo frame e o
        # frame transformado com fluxo ótico.
        # quais os pesos você deve usar?
        frame_optflow = cv2.addWeighted(frame_optflow, peso_fluxo_otico, frame, peso_proximo_frame, 0)

        # Combina os frames dos vídeos (repetição, linear e fluxo ótico)
        frame_combinado = combinar_frames([[frame_repeat], [frame_linear], [frame_optflow]])
        #frame_combinado = cv2.resize(frame_combinado, (outcomb_width, outcomb_height))
        # escreve cada frame no video de saida correspondente
        outrep.write(frame_repeat)
        outlin.write(frame_linear)
        outopt.write(frame_optflow)
        outcomb.write(frame_combinado)

    prev_frame = frame

cap.release()
outrep.release()
outlin.release()
outopt.release()
outcomb.release()
