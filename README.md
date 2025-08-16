# Brain-Tumor-Classification

This project detects brain tumors from MRI images using **Convolutional Neural Networks (CNN)** with **Transfer Learning (VGG16)**. The model achieves **~96% accuracy** on a small dataset.

## Dataset
The dataset contains two folders:
- **yes/** → images with tumor (155 images)
- **no/** → images without tumor (98 images)

## Workflow
1. **EDA** → Load images, resize to 224×224, normalize, encode labels.  
2. **Model** → Use VGG16 (pretrained on ImageNet), freeze base layers, add custom Dense layers.  
3. **Training** → Augmentation with ImageDataGenerator, train for 10 epochs, batch size 8.  
4. **Evaluation** → Classification report, confusion matrix, accuracy ~96.15%.  
5. **Visualization** → Training vs validation accuracy & loss plotted with matplotlib.

## Sample Code
```python
from imutils import paths
import cv2, os, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Dense, AveragePooling2D, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Load dataset
path = "./brain_tumor_dataset"
image_paths = list(paths.list_images(path))
images, labels = [], []
for p in image_paths:
    label = p.split(os.path.sep)[-2]
    img = cv2.resize(cv2.imread(p), (224,224))
    images.append(img); labels.append(label)

images = np.array(images)/255.0
labels = to_categorical(LabelBinarizer().fit_transform(labels))
train_X, test_X, train_Y, test_Y = train_test_split(images, labels, test_size=0.1, stratify=labels)

# Build model with VGG16
base = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224,224,3)))
for layer in base.layers: layer.trainable = False
x = Flatten()(AveragePooling2D(pool_size=(4,4))(base.output))
x = Dropout(0.5)(Dense(64, activation="relu")(x))
output = Dense(2, activation="softmax")(x)
model = Model(inputs=base.input, outputs=output)
model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

