import os
import numpy as np
import tensorflow as tf
from keras.api.models import Model
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.api.optimizers import Adam
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from keras.api.applications import VGG16
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from PIL import Image

# Parameters
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 32
NUM_CLASSES = 3  # Set the correct number of classes (adjusted from 10 to 3)
WEBP_DIR = 'E:/Development_data/Application development/Image comparison project/product_classifier/dataset_sample'  # Dataset path
JPG_DIR = 'E:/Development_data/Application development/Image comparison project/product_classifier/dataset_sample_converted'  # Converted .jpg directory

# Function to convert .webp image to .jpg and save
def convert_webp_to_jpg(img_path, save_path):
    img = Image.open(img_path)
    img = img.convert("RGB")  # Convert .webp to RGB before saving as .jpg
    img.save(save_path, "JPEG")

# Function to convert all .webp images in the dataset to .jpg
def convert_all_webp_to_jpg(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.webp'):
                # Get the class folder structure
                class_folder = os.path.basename(root)
                target_class_dir = os.path.join(target_dir, class_folder)

                # Create the target class directory if it doesn't exist
                if not os.path.exists(target_class_dir):
                    os.makedirs(target_class_dir)

                # Full paths
                img_path = os.path.join(root, file)
                jpg_img_path = os.path.join(target_class_dir, file.replace(".webp", ".jpg"))

                # Convert .webp to .jpg and save
                if not os.path.exists(jpg_img_path):
                    convert_webp_to_jpg(img_path, jpg_img_path)

# Convert all .webp images to .jpg
convert_all_webp_to_jpg(WEBP_DIR, JPG_DIR)

# Custom ImageDataGenerator with enhanced augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,             # Reduce rotation range to prevent extreme transformations
    width_shift_range=0.2,         # Slightly reduce width shift range
    height_shift_range=0.2,        # Slightly reduce height shift range
    shear_range=0.2,               # Slightly reduce shear for more perspective diversity
    zoom_range=0.3,                # Keep zoom range for more variety
    horizontal_flip=True,         # Flip horizontally
    vertical_flip=False,           # Avoid vertical flip for certain data
    fill_mode='nearest',
    brightness_range=[0.7, 1.3],  # Adjust brightness
    channel_shift_range=30.0,      # Adjust color channels
)

# Load the dataset using flow_from_directory with the new .jpg dataset
train_generator = train_datagen.flow_from_directory(
    JPG_DIR,  # Use the converted .jpg directory
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',  # Ensure labels are one-hot encoded
    shuffle=True
)

# Debugging: Print information about the dataset
print(f"Classes found: {train_generator.class_indices}")
print(f"Total samples: {train_generator.samples}")
print(f"Batch size: {train_generator.batch_size}")
print(f"Steps per epoch: {train_generator.samples // BATCH_SIZE}")

# Transfer learning model with VGG16 pre-trained on ImageNet
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Fine-tuning: Unfreeze the last few layers
for layer in base_model.layers[-10:]:  # Unfreeze last 10 layers for fine-tuning
    layer.trainable = True

# Add new layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Use Global Average Pooling
x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)  # L2 regularization
x = Dropout(0.5)(x)  # Dropout to prevent overfitting
predictions = Dense(NUM_CLASSES, activation='softmax')(x)  # Output layer

# Final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model with a lower learning rate
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Compute class weights to handle class imbalance (if needed)
y_train = train_generator.classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Early stopping callback to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Reduce learning rate when the validation loss plateaus
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Train the model
model.fit(
    train_generator,
    epochs=50,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    verbose=2,
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weights_dict
)

# Save the trained model
model.save('product_classifier_transfer_learning_improved.h5')
