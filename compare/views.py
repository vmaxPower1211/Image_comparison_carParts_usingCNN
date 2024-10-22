import numpy as np
import tensorflow as tf
from django.shortcuts import render
from django.http import JsonResponse
from PIL import Image
from io import BytesIO
from keras.api.preprocessing.image import img_to_array, load_img
from keras.api.models import load_model
# Constants
IMG_HEIGHT = 256
IMG_WIDTH = 256
NUM_CLASSES = 3  # Update to the correct number of classes
MODEL_PATH = 'E:\Development_data\Application development\Image comparison project\product_classifier\product_classifier_transfer_learning_improved.h5'  # Path to your trained model

# Load the model
model = load_model(MODEL_PATH)

def preprocess_image(image_file):
    """Process the uploaded image file."""
    # Open the image and convert to a format suitable for the model
    img = Image.open(BytesIO(image_file.read()))
    img = img.convert("RGB")
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def compare_images(request):
    if request.method == 'POST':
        # Get the uploaded image files
        image_file_1 = request.FILES.get('upload_image_1')
        image_file_2 = request.FILES.get('upload_image_2')

        if not image_file_1 or not image_file_2:
            return JsonResponse({'result': 'error', 'message': 'Both images are required!'})

        # Preprocess images
        img_array_1 = preprocess_image(image_file_1)
        img_array_2 = preprocess_image(image_file_2)

        # Get predictions for both images
        prediction_1 = model.predict(img_array_1)
        prediction_2 = model.predict(img_array_2)

        # Compare the predictions (you can improve this part based on your comparison logic)
        prediction_1_class = np.argmax(prediction_1, axis=1)[0]
        prediction_2_class = np.argmax(prediction_2, axis=1)[0]

        if prediction_1_class == prediction_2_class:
            result='identical'
            print(result)
            return JsonResponse({'result': result, 'message': 'The images are identical!'})
            
        else:
            result='different'
            print(result)
            return JsonResponse({'result': result, 'message': 'The images are different!'})

    return render(request, 'index.html')

