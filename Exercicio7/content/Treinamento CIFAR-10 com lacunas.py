#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

#%%
# você poderia carregar a base de dados CIFAR-10 desta forma:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Define as classes do CIFAR-10
classes = ['Avião', 'Carro', 'Pássaro', 'Gato', 'Cervo', 'Cachorro', 'Sapo', 'Cavalo', 'Navio', 'Caminhão']

#%%
# Índice da imagem
indice_imagem_escohlida = # complete aqui#

# Obtém a imagem e o rótulo correspondente
image = x_train[indice_imagem_escohlida]
label = # complete aqui



# Exibe a imagem e o rótulo
plt.imshow(image)
plt.axis('off')
plt.title('Classe: ' + classes[label[0]])
plt.show()
#%%
# Normalize pixel values between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Converte os rótulos para vetores (função indicadora)
# Esta conversão é necessária para que a rede neural possa
# comparar a saída da rede com o rótulo da imagem
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = # complete aqui #
#%% Create CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
# complete aqui #
# complete aqui #
#    ....   #
model.add(Flatten())
model.add(Dense(512, activation='relu'))
#output layer
model.add(Dense(10, #complete aqui#))

# Gerar o modelo (compile)
model.compile(optimizer='adam', loss=#complete aqui#, metrics=['accuracy'])

# Treinar o modelo
model.fit(x_train, y_train, batch_size=#complete#, epochs=#complete#, validation_data=(x_test, y_test))

# Avaliar o modelo com o conjunto de teste
loss, accuracy = model.evaluate(#complete aqui#, #complete aqui#)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# %%
im_nova = cv2.imread('..\Imagens\SUA IMAGEM')
im_normalizado = im_nova / 255.0
nova_entrada = np.expand_dims(im_normalizado, axis=0)
previsoes = model.predict(#complete aqui#)
predicted_class = np.argmax(#compolete aqui#)

print("previsoes: ", previsoes)
print(f"Predicted class: {predicted_class}")


# plot the image
plt.imshow(im_nova)
plt.show()
print(classes[predicted_class])
# %% visualizar os filtros da primeira camada
pesos_primeira_camada = #complete aqui#
#use a função get_weights() para obter os pesos da primeira camada

# Normalização dos pesos para valores entre 0 e 255
pesos_normalizados = (pesos_primeira_camada - np.min(pesos_primeira_camada)) / (np.max(pesos_primeira_camada) - np.min(pesos_primeira_camada))
pesos_normalizados *= 255

num_linhas = 4
num_colunas = 8
# Cria a figura e os subplots
fig, axs = plt.subplots(num_linhas, num_colunas, figsize=(32, 8))

# Percorre as imagens e exibe em cada subplot
for i in range(32):
    ax = axs[i // num_colunas, i % num_colunas]  # obtém o subplot correto
    filtro = pesos_normalizados[:, :, :, i]
    filtro_img = np.reshape(filtro, (3, 3, 3))
    # imagem em tons de cinza
    filtro_pb = cv2.cvtColor(filtro_img, cv2.COLOR_BGR2GRAY)
    filtro_pb = filtro_pb.astype(np.uint8)
    ax.imshow(filtro_pb, cmap='gray')  # exibe a imagem
    ax.axis('off')  # remove os eixos

# Ajusta o espaçamento entre os subplots
plt.tight_layout()

# Exibe a figura
plt.show()