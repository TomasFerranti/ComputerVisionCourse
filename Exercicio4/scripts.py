"""
funcoes questao1(), questao2(), questao3(), questao4() implementam os codigos de cada questao
main() é chamada na execução e executa os codigos de todas as questoes
"""
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile


# Arquivo para questão 1 e 2
audio1_filename = "content/StarWars60.wav"
audio2_filename = "content/Coruja.WAV"

# Arquivos para questão 3 e 4
img1_filename = "content/frutas.jpg"

# ----------------------------------------------------------------
# Questão 1
def questao1():
    # Carregar o arquivo de áudio
    sample_rate, audio = wavfile.read(audio1_filename) 
    
    # Duração em segundos desejada
    duration = 10

    # Número de amostrar equivalente a essa duracao
    num_samples = int(duration * sample_rate)

    # Reduzir dimensão do áudio para a duração
    shortened_audio = audio[:num_samples]

    # Calcular a transformada de Fourier
    fft_audio = np.abs(np.fft.fft(shortened_audio))

    # Pegar o eixo de frequência
    freq_axis = np.fft.fftfreq(shortened_audio.shape[-1], d=1/sample_rate)

    # Plotar o espectro do áudio
    plt.plot(freq_axis[:len(freq_axis)//2], fft_audio[:len(fft_audio)//2])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Absolute fourier coefficients of initial 10 seconds\nFrequency > 0")
    plt.savefig("output/img1.png", dpi=300)
    plt.show()

# ----------------------------------------------------------------
# Questão 2
def questao2():
    # Carregar o arquivo de áudio
    sample_rate, audio = wavfile.read(audio1_filename)

    # Calcular a transformada de Fourier
    fft_data = np.fft.fft(audio)

    # Pegar o eixo de frequência
    freq_axis = np.fft.fftfreq(len(audio), d=1/sample_rate)

    # Encontrar os indices das 100*p% frequencias mais baixas
    p = 0.2
    num_freq = len(freq_axis)
    idx = int(num_freq * p / 2)

    # Colocar os coeficientes das frequencias mais altas em 0
    fft_data[idx+1:-idx] = 0

    # Computar a inversa para salvar o audio filtrado
    filtered_audio = np.real(np.fft.ifft(fft_data))

    # Salvar os dados do áudio filtrado em um arquivo .wav
    wavfile.write("output/audio2.wav", sample_rate, filtered_audio.astype(np.int16))
    
# ----------------------------------------------------------------
# Questão 3
def convolveFFT(audio, impulse_response):
    # Normalizar
    audio = audio / np.max(np.abs(audio))
    impulse_response = impulse_response / np.max(np.abs(impulse_response))
    
    # Tamanho é a potencia de 2
    fft_size = 2**int(np.ceil(np.log2(len(audio) + len(impulse_response))))

    # Parte real da transformada de fourier
    audio_fft = np.fft.rfft(audio, fft_size)
    impulse_response_fft = np.fft.rfft(impulse_response, fft_size)
    
    # Convoluir os resultados
    convolved_fft = audio_fft * impulse_response_fft
    
    # Retornar de fourier
    convolved_audio = np.fft.irfft(convolved_fft, fft_size)[:len(audio)]

    # Normalizar
    convolved_audio = convolved_audio / np.max(np.abs(convolved_audio))
    return convolved_audio

def applyEcho(sample_rate, audio, second, attenuation):
    # Impulso do eco
    delay_samples_echo = int(sample_rate * second)
    impulse_response_echo = np.zeros(len(audio) + delay_samples_echo)
    impulse_response_echo[delay_samples_echo] = 1

    # Convolução
    convolved_audio_echo = attenuation * convolveFFT(audio, impulse_response_echo)
    
    return convolved_audio_echo

def questao3():
    # Carregar o arquivo de áudio
    sample_rate, audio = wavfile.read(audio2_filename)

    # Normalizar
    audio = audio / np.max(np.abs(audio))
    
    # Caracteristicas de ambos impulsos
    attenuation = 0.8

    # Audio do eco
    convolved_audio_echo = audio + applyEcho(sample_rate, audio, 0.5, attenuation)
    
    # Audio do reverb
    n_copies = 10
    dt = 0.1
    delay_times = [dt + n * dt for n in range(n_copies)]
    reverbs = []
    for idx, delay in enumerate(delay_times):
        reverbs.append(applyEcho(sample_rate, audio, delay, attenuation**(idx + 1)))
    convolved_audio_reverb = audio.copy()
    for reverb in reverbs:
        convolved_audio_reverb += reverb

    # Salvar os dados do áudio filtrado em um arquivo .wav
    convolved_audio_echo = convolved_audio_echo / np.max(np.abs(convolved_audio_echo))
    convolved_audio_reverb = convolved_audio_reverb / np.max(np.abs(convolved_audio_reverb))
    convolved_audio_echo = (2**15) * convolved_audio_echo
    convolved_audio_reverb = (2**15) * convolved_audio_reverb

    wavfile.write("output/audio3a.wav", sample_rate, convolved_audio_echo.astype(np.int16))
    wavfile.write("output/audio3b.wav", sample_rate, convolved_audio_reverb.astype(np.int16))

# ----------------------------------------------------------------
# Questão 4
def keepFrequencies(img, p, high=False):
    imgChannels = []
    # Realizar o corte para cada canal RGB
    for channel in range(3):
        imgC = img[:, :, channel]

        # Aplicar transformada de Fourier bidimensional
        f = np.fft.fft2(imgC)
        fshift = np.fft.fftshift(f)

        # Criar uma máscara para preservar apenas uma porcentagem p das frequencias
        rows, cols = imgC.shape
        crow, ccol = rows // 2, cols // 2
        
        if high:
            mask = np.ones((rows, cols), np.uint8)
            mask[int(crow - crow*p):int(crow + crow*p), int(ccol - ccol*p):int(ccol + ccol*p)] = 0
        else:
            mask = np.zeros((rows, cols), np.uint8)
            mask[int(crow - crow*p):int(crow + crow*p), int(ccol - ccol*p):int(ccol + ccol*p)] = 1
        
        # Aplicar a máscara e inverter a transformada de Fourier
        fshift = fshift * mask
        f_ishift = np.fft.ifftshift(fshift)
        imgC = np.fft.ifft2(f_ishift)
        imgC = np.abs(imgC)

        imgChannels.append(imgC)
    
    # Concatenar os cortes para cada canal RGB para uma só imagem de novo
    imgFinal = np.concatenate([imgC.reshape(imgC.shape[0], imgC.shape[1], 1) for imgC in imgChannels], axis=2)
    return imgFinal

def questao4():
    img = cv.imread(img1_filename)

    imgLow_p_10 = keepFrequencies(img, 0.1, False)
    imgLow_p_80 = keepFrequencies(img, 0.8, False)
    imgHigh = keepFrequencies(img, 0.1, True)

    cv.imwrite("output/img4baixas10.jpg", imgLow_p_10)    
    cv.imwrite("output/img4baixas80.jpg", imgLow_p_80)    
    cv.imwrite("output/img4altas.jpg", imgHigh)    
    
# ----------------------------------------------------------------
# Enquadrar imagens
def removeImgBorders():
    img_filenames = os.listdir("output/")
    for img_filename in img_filenames:
        if ".png" not in img_filename and ".jpg" not in img_filename:
            continue
        # Caminho para o diretório
        img_filename = "output/" + img_filename

        # Carregar imagem
        img = cv.imread(img_filename)

        # Converter para escala de cinza
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Achar os indexes dos primeros pixels nao brancos de cada direção
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
    print("\tExecutando código da questão 3...")
    questao3()
    print("\tExecutando código da questão 4...")
    questao4()
    print("Finalizando cortando as imagens")
    removeImgBorders()
    print("Finalizado!")