from django.shortcuts import render
from .forms import ImageUploadForm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, UnidentifiedImageError
import numpy as np
import base64

model = VGG16(weights='imagenet')

def get_image_base64(img_file):
    img_file.seek(0)
    img_data = base64.b64encode(img_file.read()).decode('utf-8')
    return f'data:image/jpeg;base64,{img_data}'

def predict(request):
    prediction = None
    img_data = None

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']
            # Convert to BytesIO and reset file pointer
            img_file.seek(0)
            img_data = get_image_base64(img_file)
            img_file.seek(0)

            try:
                # Load the image
                img = Image.open(img_file)

                # Resize and preprocess the image
                img = img.resize((224, 224))
                img_array = img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                # Make predictions
                preds = model.predict(img_array)
                top_pred = decode_predictions(preds, top=1)[0][0]
                prediction = top_pred[1]  # Assuming you want the class name

            except UnidentifiedImageError:
                prediction = "画像ファイルを識別できませんでした。形式を確認してください。"
            except Exception as e:
                prediction = f"予期しないエラーが発生しました: {str(e)}"

        return render(request, 'home.html', {'form': form, 'prediction': prediction, 'img_data': img_data})

    form = ImageUploadForm()
    return render(request, 'home.html', {'form': form})

