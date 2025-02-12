# 25_AI_Practice
2025 AI 실습자료 업로드

# Image 다중분류 Sample Code
필요한 라이브러리 설치:
pip install tensorflow numpy matplotlib
```
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
```
# 1. 데이터 전처리
```
train_datagen = ImageDataGenerator(
    rescale=1./255,  # 이미지 픽셀값을 [0, 1] 범위로 정규화
    rotation_range=20,  # 회전 범위
    width_shift_range=0.2,  # 수평 이동 범위
    height_shift_range=0.2,  # 수직 이동 범위
    shear_range=0.2,  # 기울이기
    zoom_range=0.2,  # 확대/축소
    horizontal_flip=True,  # 수평 뒤집기
    fill_mode='nearest'  # 비어있는 공간 채우기
)

test_datagen = ImageDataGenerator(rescale=1./255)
```

# 경로를 설정해주세요. (train_dir, test_dir은 학습 및 테스트 이미지 폴더 경로)
```
train_dir = 'train_data'  # 학습 데이터가 있는 폴더
test_dir = 'test_data'  # 테스트 데이터가 있는 폴더

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # MobileNetV2의 입력 사이즈 (224x224)
    batch_size=32,
    class_mode='categorical',  # 다중 클래스 분류
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)
```

# 2. 모델 구축
```
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# MobileNetV2 모델의 마지막 레이어를 제외하고 추가 학습
for layer in base_model.layers:
    layer.trainable = False  # pre-trained weights를 사용할 때는 대부분의 레이어를 고정시킴

# 새로운 classifier 부분 추가
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dense(5, activation='softmax')  # 5개의 클래스로 분류
])
```

# 3. 모델 컴파일
```
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
```

# 4. 모델 학습
```
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_generator,
    epochs=20,  # 필요에 따라 epoch 수를 조정하세요
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    callbacks=[early_stop]
)
```

# 5. 모델 평가 (테스트 데이터에서 정확도 확인)
```
test_loss, test_accuracy = model.evaluate(test_generator)
```

# 정확도 출력
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

설명:
데이터 전처리 (ImageDataGenerator): 이미지를 실시간으로 증강하고 정규화하여 모델이 더 잘 학습할 수 있도록 돕습니다.

MobileNetV2 모델: ImageNet에서 미리 학습된 MobileNetV2 모델을 불러옵니다. include_top=False로 최상위 분류 레이어는 제외합니다.

분류 레이어 추가: 모델의 출력 부분에 Dense 레이어를 추가하여 5개의 클래스를 분류할 수 있도록 합니다.

컴파일 및 학습: Adam 옵티마이저와 categorical_crossentropy 손실 함수를 사용하여 모델을 컴파일합니다. EarlyStopping을 사용해 과적합을 방지하고 최상의 모델을 선택합니다.

모델 평가: 테스트 데이터를 사용하여 모델의 정확도를 평가하고 출력합니다.

결과:
테스트 데이터에서 모델이 분류한 정확도를 Test Accuracy로 출력하게 됩니다.
