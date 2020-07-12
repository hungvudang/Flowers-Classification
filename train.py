# Thêm thư viện
import os
import random

import numpy as np
from tqdm import tqdm
from imutils import paths
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.layers import Input
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.models import Model
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder

# Lấy các đường dẫn đến ảnh.
image_path = list(paths.list_images('dataset'))

# Đổi vị trí ngẫu nhiên các đường dẫn ảnh
random.shuffle(image_path)

# Đường dẫn ảnh sẽ là dataset/tên_loài_hoa/tên_ảnh ví dụ dataset/Bluebell/image_0241.jpg nên p.split(os.path.sep)[-2]
# sẽ lấy ra được tên loài hoa
labels = [p.split(os.path.sep)[-2] for p in image_path]
# labels = os.listdir('dataset')

# Chuyển tên các loài hoa thành số
le = LabelEncoder()
labels = le.fit_transform(labels)

# One-hot encoding
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Load ảnh và resize về đúng kích thước mà VGG 16 cần là (224,224)
print('Load image...')
list_image = []
for j in tqdm(range(len(image_path))):
    image = load_img(image_path[j], target_size=(224, 224))
    image = img_to_array(image)

    image = np.expand_dims(image, 0)
    image = imagenet_utils.preprocess_input(image)

    list_image.append(image)

list_image = np.vstack(list_image)

print('Build model...')
# Load model VGG 16 của ImageNet dataset, include_top=False để bỏ phần Fully connected layer ở cuối.
baseModel = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Xây thêm các layer
# Lấy output của ConvNet trong VGG16
fcHead = baseModel.output

# Flatten trước khi dùng FCs
fcHead = Flatten(name='flatten')(fcHead)

# Thêm FC
fcHead = Dense(256, activation='relu')(fcHead)
fcHead = Dropout(0.5)(fcHead)

# Output layer với softmax activation
fcHead = Dense(17, activation='softmax')(fcHead)

# Xây dựng model bằng việc nối ConvNet của VGG16 và fcHead
model = Model(inputs=baseModel.input, outputs=fcHead)

print(model.summary())

# Chia traing set, test set tỉ lệ 80-20
X_train, X_test, y_train, y_test = train_test_split(list_image, labels, test_size=0.2, random_state=42)

# augmentation cho training data
aug_train = ImageDataGenerator(rescale=1. / 255, rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
                               shear_range=0.2,
                               zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
# augementation cho test
aug_test = ImageDataGenerator(rescale=1. / 255)

# freeze VGG model
for layer in baseModel.layers:
    layer.trainable = False

opt = RMSprop(0.001)
model.compile(opt, 'categorical_crossentropy', ['accuracy'])
numOfEpoch = 50
print('Training...')
H = model.fit(aug_train.flow(X_train, y_train, batch_size=32),
              steps_per_epoch=len(X_train) // 32,
              validation_data=(aug_test.flow(X_test, y_test, batch_size=32)),
              validation_steps=len(X_test) // 32,
              epochs=numOfEpoch)

print('Fine Tuning model...')
# unfreeze some last CNN layer:
for layer in baseModel.layers[15:]:
    layer.trainable = True

numOfEpoch = 100
opt = SGD(0.001)
model.compile(opt, 'categorical_crossentropy', ['accuracy'])
H = model.fit(aug_train.flow(X_train, y_train, batch_size=32),
              steps_per_epoch=len(X_train) // 32,
              validation_data=(aug_test.flow(X_test, y_test, batch_size=32)),
              validation_steps=len(X_test) // 32,
              epochs=numOfEpoch)

print('Save model...')
# Lưu model
model.save('model.h5')
print('Done !')