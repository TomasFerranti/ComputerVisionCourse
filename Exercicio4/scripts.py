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
    return convolved_audio

def questao3():
    # Carregar o arquivo de áudio
    sample_rate, audio = wavfile.read(audio1_filename)

    # Caracteristicas de ambos impulsos
    attenuation = 0.8

    # Impulso do eco
    echo_seconds = 0.5
    delay_samples_echo = int(sample_rate * echo_seconds)

    impulse_response_echo = np.zeros(len(audio) + delay_samples_echo)
    impulse_response_echo[0] = 1
    impulse_response_echo[delay_samples_echo] = attenuation

    # Impulso do reverb
    delay_times = [0.03, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    delay_times = [10*time for time in delay_times]

    delay_samples_reverb = int(sample_rate * delay_times[-1])

    impulse_response_reverb = np.zeros(len(audio) + delay_samples_reverb)
    impulse_response_reverb[0] = 1
    for idx, delay in enumerate(delay_times):
        idx = int(sample_rate * delay)
        impulse_response_reverb[idx:idx+len(audio)] += attenuation ** idx

    # Perform FFT convolution
    convolved_audio_echo = convolveFFT(audio, impulse_response_echo)
    convolved_audio_reverb = convolveFFT(audio, impulse_response_reverb)
    
    # Salvar os dados do áudio filtrado em um arquivo .wav
    wavfile.write("output/audio3a.wav", sample_rate, convolved_audio_echo.astype(np.int16))
    wavfile.write("output/audio3b.wav", sample_rate, convolved_audio_reverb.astype(np.int16))

# ----------------------------------------------------------------
# Questão 4
def questao4():
    img = cv.imread(img1_filename)
    imgChannelsLow = []
    imgChannelsHigh = []
    pLowKeep = 0.1
    pHighKeep = 0.1
    # Realizar o corte para cada canal RGB
    for channel in range(3):
        imgC = img[:, :, channel]

        # Aplicar transformada de Fourier bidimensional
        f = np.fft.fft2(imgC)
        fshift = np.fft.fftshift(f)

        # Criar uma máscara para preservar apenas uma porcentagem p das frequencias
        rows, cols = imgC.shape
        crow, ccol = rows // 2, cols // 2

        maskLow = np.zeros((rows, cols), np.uint8)
        maskLow[int(crow - crow*pLowKeep):int(crow + crow*pLowKeep), int(ccol - ccol*pLowKeep):int(ccol + ccol*pLowKeep)] = 1

        maskHigh = np.ones((rows, cols), np.uint8)
        maskHigh[int(crow - crow*pHighKeep):int(crow + crow*pHighKeep), int(ccol - ccol*pHighKeep):int(ccol + ccol*pHighKeep)] = 0

        # Aplicar a máscara e inverter a transformada de Fourier
        fshiftLow = fshift * maskLow
        f_ishiftLow = np.fft.ifftshift(fshiftLow)
        imgCLow = np.fft.ifft2(f_ishiftLow)
        imgCLow = np.abs(imgCLow)

        fshiftHigh = fshift * maskHigh
        f_ishiftHigh = np.fft.ifftshift(fshiftHigh)
        imgCHigh = np.fft.ifft2(f_ishiftHigh)
        imgCHigh = np.abs(imgCHigh)

        imgChannelsLow.append(imgCLow)
        imgChannelsHigh.append(imgCHigh)
    
    # Concatenar os cortes para cada canal RGB para uma só imagem de novo
    imgLow = np.concatenate([img.reshape(img.shape[0], img.shape[1], 1) for img in imgChannelsLow], axis=2)
    imgHigh = np.concatenate([img.reshape(img.shape[0], img.shape[1], 1) for img in imgChannelsHigh], axis=2)
    cv.imwrite("output/img4baixas.jpg", imgLow)    
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