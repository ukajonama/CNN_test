import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator

# 데이터 세트를 로드합니다.
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

train_generator = train_datagen.flow_from_directory('./dataset/fire_dataset/',
                                                   target_size=(224, 224),
                                                   batch_size=32,
                                                   class_mode='binary')
test_generator =train_datagen.flow_from_directory('./dataset/fire_dataset/',
                                                   target_size=(224, 224),
                                                   batch_size=32,
                                                   class_mode='binary')
# CNN을 만듭니다.
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# 모델을 컴파일합니다.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델을 학습합니다.
model.fit(train_generator, epochs=10)

# 모델을 저장합니다.

evaluation = model.evaluate(test_generator)
print('Test Accuracy:', evaluation[1])
model.save('model.h5')