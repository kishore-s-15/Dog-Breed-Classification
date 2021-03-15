# Importing the required libraries
import os
import base64
import numpy as np

# For Ignoring Tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = FastAPI()

# Loading the saved model
model = load_model("./model")

# Class Labels
class_list = ["Beagle", "Chihuahua", "Doberman", "french_bulldog", "golden_retriever",
              "malamute", "pug", "saint_bernard", "scottish_deerhound", "tibetian_mastiff"]

# Image Pydantic Model for validation
class Image(BaseModel):
    base64_encoded_image: str

def decode_predictions(B64_string: bytes) -> str:
    """
    Function which returns the class label 
    predicted by the model

    params:
    B64_string: bytes

    returns:
    class_label: str
    """

    IMG_URL = "test.jpg"

    # Saving the encoded string as an image file for prediction
    with open(IMG_URL, "wb") as img_file:
        img_file.write(base64.b64decode(B64_string))

    image = load_img(IMG_URL, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    y_pred = model.predict(image)

    idx = np.argmax(np.squeeze(y_pred))

    return class_list[idx]

@app.post('/inference/')
def inference(image: Image) -> str:
    """
    Route takes Base64 encoded string as parameter
    and returns the predicted class label
    """

    return decode_predictions(image.base64_encoded_image)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)