"""
Brain Tumor Detection using CNN and Transfer Learning (VGG16)

Step 1. Perform Exploratory Data Analysis (EDA)
------------------------------------------------
The dataset contains 2 folders: "no" and "yes" with 98 and 155 images each.
We load the images, resize them to 224×224, normalize them, 
and convert labels to categorical format.
"""

from imutils import paths 
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load the images directories
path = "./Desktop/DataFlair/brain_tumor_dataset"
image_paths = list(paths.list_images(path))

images = []
labels = []

for image_path in image_paths:
    label = image_path.split(os.path.sep)[-2]  # folder name is label
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    images.append(image)
    labels.append(label)

# Plot a sample image
def plot_image(image):
    plt.imshow(image)
    plt.axis("off")

plot_image(images[0])

# Normalize and prepare labels
images = np.array(images) / 255.0
labels = np.array(labels)

label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)
labels = to_categorical(labels)

# Split dataset
(train_X, test_X, train_Y, test_Y) = train_test_split(
    images, labels, test_size=0.10, random_state=42, stratify=labels
)

"""
Step 2. Build a CNN Model
--------------------------
We use Transfer Learning with VGG16 as the base model. 
We freeze its layers and add new fully connected layers suitable for binary classification.
We also use ImageDataGenerator for data augmentation.
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, AveragePooling2D, Dropout, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Data augmentation
train_generator = ImageDataGenerator(fill_mode='nearest', rotation_range=15)

# Base model
base_model = VGG16(weights="imagenet", input_tensor=Input(shape=(224, 224, 3)), include_top=False)
base_input = base_model.input
base_output = base_model.output

# Add custom layers
base_output = AveragePooling2D(pool_size=(4, 4))(base_output)
base_output = Flatten(name="flatten")(base_output)
base_output = Dense(64, activation="relu")(base_output)
base_output = Dropout(0.5)(base_output)
base_output = Dense(2, activation="softmax")(base_output)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Build model
model = Model(inputs=base_input, outputs=base_output)
model.compile(optimizer=Adam(learning_rate=1e-3), 
              metrics=["accuracy"], 
              loss="binary_crossentropy")

# Show model summary
model.summary()

"""
Step 3. Train and Evaluate the Model
-------------------------------------
We set batch size, epochs, and train the model.
After training, we evaluate using classification report and confusion matrix.
"""

from sklearn.metrics import classification_report, confusion_matrix

batch_size = 8
train_steps = len(train_X) // batch_size
validation_steps = len(test_X) // batch_size
epochs = 10

# Train model
history = model.fit(
    train_generator.flow(train_X, train_Y, batch_size=batch_size),
    steps_per_epoch=train_steps,
    validation_data=(test_X, test_Y),
    validation_steps=validation_steps,
    epochs=epochs
)

# Evaluate
predictions = model.predict(test_X, batch_size=batch_size)
predictions = np.argmax(predictions, axis=1)
actuals = np.argmax(test_Y, axis=1)

print(classification_report(actuals, predictions, target_names=label_binarizer.classes_))

cm = confusion_matrix(actuals, predictions)
print("Confusion Matrix:\n", cm)

# Calculate accuracy manually
total = sum(sum(cm))
accuracy = (cm[0, 0] + cm[1, 1]) / total
print("Accuracy: {:.4f}".format(accuracy))

"""
Step 4. Plot Training Metrics
------------------------------
We plot training loss, validation loss, training accuracy, 
and validation accuracy across epochs.
"""

N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Brain Dataset")
plt.xlabel("Epoch")
plt.ylabel("Loss / Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.jpg")
plt.show()
