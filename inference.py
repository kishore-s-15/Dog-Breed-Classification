# Importing the required libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import base64
import numpy as np

from keras.models import load_model
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array

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
    class_list = ["Beagle", "Chihuahua", "Doberman", "french_bulldog", "golden_retriever",
                  "malamute", "pug", "saint_bernard", "scottish_deerhound", "tibetian_mastiff"]

    model = load_model("./model")

    with open(IMG_URL, "wb") as img_file:
        img_file.write(base64.b64decode(B64_string))

    image = load_img(IMG_URL, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    y_pred = model.predict(image)

    idx = np.argmax(np.squeeze(y_pred))

    return class_list[idx]

if __name__ == "__main__":
    with open("image.jpg", "rb") as img_file:
        b64_string = base64.b64encode(img_file.read())

    print("Base 64 Encoded String : ")
    print(b64_string)

    string = base64.b64decode(b64_string)

    breed = decode_predictions(b64_string)

    print(f"The dog in the image is a {breed}")