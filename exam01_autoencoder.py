# 압축을 하는 과정 -> 인코더 디코더 => 코덱

import matplotlib.pyplot as plt
import numpy as np
from keras.models import *
from keras.layers import *
from keras.datasets import mnist

input_img = Input(shape=(784,))
encoded = Dense(32, activation='relu') # Dense 레이어만 존재 -> input_img로 입력
encoded = encoded(input_img) # 입력을 받은 덴스레이어
decoded = Dense(784, activation='sigmoid') #왜 sigmoid를 쓰는가? 0~1로 정규화 min_max정규화
decoded = decoded(encoded)
autoencoder = Model(input_img, decoded) # (입력, 출력) 입력~출력이 범위
autoencoder.summary()

encoder = Model(input_img, encoded)
encoder.summary()

encoder_input = Input(shape=(32,))
decoder_layer = autoencoder.layers[-1] #모델을 슬라이싱 할 수 있다
decoder = Model(encoder_input, decoder_layer(encoder_input)) #인코더 입력 ~ 디코더 레이어까지 입출력
decoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# 라벨을 쓰지않는다 -> 비지도 학습
(x_train, _), (x_test, _) = mnist.load_data() # y_train가 필요없다 -> 라벨을 쓰지 않는다  -> 우리는 입력만 알고 싶어한다(비지도 학습, 자기주도학습)
x_train = x_train / 255
x_test = x_train / 255

flatted_x_train = x_train.reshape(-1, 784)
flatted_x_test = x_test.reshape(-1, 784)

fit_hist = autoencoder.fit(flatted_x_train, flatted_x_train, epochs=50,
            batch_size=256, validation_data=(flatted_x_test, flatted_x_test))

encoded_img = encoder.predict(x_test[:10].reshape(-1, 784))
decoded_img = decoder.predict(encoded_img)

n = 10
plt.gray()
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i]) #입력이미지 출력
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 10, i + 1 + n)
    plt.imshow(x_test[i].reshape(28, 28)) # 디코더 후 입력이미지 출력
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

plt.plot(fit_hist.history['loss'])
plt.plot(fit_hist.history['val_loss'])

plt.show()