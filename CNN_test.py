import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
import os
import cv2
# 학습 데이터 준비 (예시)
IMG_SIZE = 100
DATADIR = "./dataset/fire_dataset/"
CATEGORIES = ["fire", "no_fire"]
X_train = []
y_train = []
for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    class_num = CATEGORIES.index(category)  # 0 for fire, 1 for no_fire
    
    for img in os.listdir(path):
        try:
            # 이미지 로드 및 크기 변경
            img_array = cv2.imread(os.path.join(path, img))
            resized_img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            
            # 이미지 및 라벨을 데이터 목록에 추가
            X_train.append(resized_img_array)
            y_train.append(class_num)
        except Exception as e:
            print(f"Error: {e}")

# 배열을 numpy 배열로 변환하고, 이미지 데이터를 0-1 사이의 값으로 정규화
X_train = np.array(X_train).astype('float32') / 255.0
y_train = np.array(y_train)
# 모델 정의
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# 학습
model.fit(X_train, y_train, epochs=30, validation_split=0.1)

model.save('model.h5')
