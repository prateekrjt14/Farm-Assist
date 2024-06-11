from keras.models import load_model
import pickle
import os
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
from groq import Groq
import base64

app = Flask(__name__)

# Initialize Groq client
client = Groq(api_key="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")

# Crop recommendation model
with open('CropPredict.pkl', 'rb') as f:
    model = pickle.load(f)

# Disease prediction model
model2 = load_model('DiseasePrediction.h5')

# Class indices
with open('class_indices.pkl', 'rb') as f:
    class_indices = pickle.load(f)

# Mapping of prediction labels to online image URLs
crop_images = {
    'rice': 'https://www.indianhealthyrecipes.com/wp-content/uploads/2023/07/basmati-rice-recipe.jpg',
    'maize': 'https://www.keshrinandan.com/wp-content/uploads/2015/08/ke_maize_nutrition.jpg',
    'chickpea': 'https://upload.wikimedia.org/wikipedia/commons/e/ea/Sa-whitegreen-chickpea.jpg',
    'kidneybeans': 'https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Kidney_beans.jpg/2560px-Kidney_beans.jpg',
    'pigeonpeas': 'https://www.thedailymeal.com/img/gallery/pigeon-peas-are-a-truly-unique-topper-for-sprucing-up-your-salad/intro-1693253591.jpg',
    'mothbeans': 'https://img2.exportersindia.com/product_images/bc-full/dir_103/3069234/moth-beans-1184105.jpg',
    'mungbean': 'https://upload.wikimedia.org/wikipedia/commons/8/86/Mung_beans_%28Vigna_radiata%29.jpg',
    'blackgram': 'https://2.bp.blogspot.com/-ACOHUAygDZ8/W9XRZdhYIRI/AAAAAAAAFYI/acMeNKD-mD8E4L_kEFPOqBI4EZ4z791hQCLcBGAs/s600/Black%2BGram.jpg',
    'lentil': 'https://cdn.britannica.com/14/157214-050-3A82D9CD/kinds-lentils.jpg',
    'pomegranate': 'https://www.health.com/thmb/uwTh7XB2O92hRMRupm6Hmkq1Jpw=/2125x0/filters:no_upscale():max_bytes(150000):strip_icc()/Pomegranate-fc75404736b446c5aa26194e1d2b1d0a.jpg',
    'banana': 'https://www.usatoday.com/gcdn/-mm-/ac688eec997d2fce10372bf71657297ff863814d/c=171-0-1195-768/local/-/media/2022/01/25/USATODAY/usatsports/gettyimages-174959827.jpg',
    'mango': 'https://www.adityabirlacapital.com/healthinsurance/active-together/wp-content/uploads/2023/05/Is-Mango-Good-For-Diabetes-1200x1200.jpg',
    'grapes': 'https://www.thespruceeats.com/thmb/l1_lV7wgpqRArWBwpG3jzHih_e8=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/what-are-grapes-5193263-hero-01-80564d77b6534aa8bfc34f378556e513.jpg',
    'watermelon': 'https://static.toiimg.com/thumb/resizemode-4,width-1280,height-720,msid-109165284/109165284.jpg',
    'muskmelon': 'https://blog-images-1.pharmeasy.in/blog/production/wp-content/uploads/2021/05/18150019/shutterstock_1376235665-1.jpg',
    'apple': 'https://media-cldnry.s-nbcnews.com/image/upload/t_social_share_1200x630_center,f_auto,q_auto:best/rockcms/2022-09/apples-mc-220921-e7070f.jpg',
    'orange': 'https://www.allrecipes.com/thmb/y_uvjwXWAuD6T0RxaS19jFvZyFU=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/GettyImages-1205638014-2000-d0fbf9170f2d43eeb046f56eec65319c.jpg',
    'papaya': 'https://media.post.rvohealth.io/wp-content/uploads/2020/09/papaya-benefits-732x549-thumbnail.jpg',
    'coconut': 'https://i.unu.edu/media/ourworld.unu.edu-en/article/4538/Videobriefs-19.jpg',
    'cotton': 'https://cdn.britannica.com/18/156618-050-39339EA2/cotton-harvesting.jpg',
    'jute': 'https://truejute.com/wp-content/uploads/2021/12/JUTE-27-11-19.jpg',
    'coffee': 'https://upload.wikimedia.org/wikipedia/commons/c/c5/Roasted_coffee_beans.jpg',
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/crop_predict')
def crop_predict():
    return render_template('cropPredict.html')

@app.route('/disease_predict')
def disease_predict():
    return render_template('result.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    features = [float(i) for i in data]
    prediction = model.predict([features])[0]
    image_url = crop_images.get(prediction, 'https://static.vecteezy.com/system/resources/previews/006/208/684/non_2x/search-no-result-concept-illustration-flat-design-eps10-modern-graphic-element-for-landing-page-empty-state-ui-infographic-icon-vector.jpg')  # Fallback to a default image
    return jsonify({'prediction': prediction, 'image': image_url})

# Define function to preprocess image
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# Define prediction function
def predict_image(img_array, model2, class_indices):
    prediction = model2.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_label = list(class_indices.keys())[predicted_class_index]
    return predicted_label

# Function to get disease information using Groq API
def get_disease_info(disease_name):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"Provide detailed information about {disease_name} in the following format:\n\n" \
                               "Causes:\n- Cause 1\n- Cause 2\n\n" \
                               "Precautions:\n- Precaution 1\n- Precaution 2\n\n" \
                               "Treatments:\n- Treatment 1\n- Treatment 2"
                }
            ],
            model="llama3-8b-8192",
        )
        info = chat_completion.choices[0].message.content
        return info
    except Exception as e:
        return f"Could not retrieve information: {e}"

# Modify the Flask route to include the image URL in the response
@app.route('/diseasePredict', methods=['POST'])
def disease_Predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        img_path = "temp.jpg"  # Save the file temporarily
        file.save(img_path)

        img_array = preprocess_image(img_path)
        predicted_label = predict_image(img_array, model2, class_indices)

        os.remove(img_path)  # Remove the temporary file

        # Get detailed information about the disease
        disease_info = get_disease_info(predicted_label)

        # Construct the image URL to be sent in the response
        image_url = f"/uploaded_images/{file.filename}"  # Assuming you serve uploaded images from a directory named "uploaded_images"

        return jsonify({'prediction': predicted_label, 'info': disease_info, 'image_url': image_url})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
