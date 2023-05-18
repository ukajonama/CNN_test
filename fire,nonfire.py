import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# 저장된 모델 로드
model_path = 'model.h5'
model_path = model_path.encode('utf-8').decode('cp949')  # 파일 경로 인코딩 수정

model = tf.keras.models.load_model(model_path)

# 테스트 데이터셋을 로드합니다.
test_datagen = ImageDataGenerator(rescale=1./255)  # 이미지 스케일링만 적용합니다.

test_generator = test_datagen.flow_from_directory('./dataset/test/',
                                                   target_size=(224, 224),
                                                   batch_size=32,
                                                   class_mode='binary')

# 모델 평가
evaluation = model.evaluate(test_generator)
print('Test Accuracy:', evaluation[1])

