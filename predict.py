import os

import numpy as np
import cv2 as cv
from imutils import paths
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.applications import imagenet_utils



image_path = list(paths.list_images('testset'))

# Load model
model = load_model('model.h5')
print(model.summary())

# Load image
list_image = []
for j in range(len(image_path)):
    img = load_img(image_path[j], target_size=(224, 224))
    img = img_to_array(img)

    img = np.expand_dims(img, 0)
    img = imagenet_utils.preprocess_input(img)

    list_image.append(img)

list_image = np.vstack(list_image)


preds = model.predict(list_image)

# Hiện thị kết quả:
labels = os.listdir('dataset')
colors = np.random.randint(50, 250, size=(len(image_path),3), dtype=np.uint8)

j = 0
for p in image_path:
    img = cv.imread(p)
    b,g,r = colors[j]
    label = labels[np.argmax(preds[j])]
    w_text, h_text = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
    cv.rectangle(img, (0,0), (w_text + 2, h_text + 2), (255, 255, 255), -1)
    cv.putText(img, label, (0, h_text), cv.FONT_HERSHEY_SIMPLEX, 2,(int(b),int(g),int(r)) , 3)
    


    cv.namedWindow('Output '+str(j), 0)
    cv.imshow('Output '+str(j), img)
    j += 1

cv.waitKey(0)
cv.destroyAllWindows()