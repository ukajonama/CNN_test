import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
# 저장된 모델 로드
model = tf.keras.models.load_model('model.h5')

# 테스트 데이터셋을 로드합니다.
# test_datagen = ImageDataGenerator(rescale=1./255)  # 전처리 스케일링만 적용합니다.
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_generator =train_datagen.flow_from_directory('./dataset/fire_dataset/',
                                                   target_size=(224, 224),
                                                   batch_size=32,
                                                   class_mode='binary')

# 모델 평가
evaluation = model.evaluate(test_generator)
print('Test Accuracy:', evaluation[1])