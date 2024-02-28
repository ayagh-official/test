from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2

app = FastAPI()

# Allow any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("./VGG.h5")

# Class names for the different skin diseases
CLASS_NAMES = ['no','yes']

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    # Resize the image to (150,150)
    image = tf.image.resize(image, (224, 224))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    imaget = read_file_as_image(await file.read())
    
    predictions = model.predict(np.expand_dims(imaget, axis=0))
    predicted_class_index = np.argmax(predictions[0])
    
    return CLASS_NAMES[predicted_class_index]


    # # Replace this with the actual path to the image file
    # file_path = "./image(1).jpg"
    # img = Image.open(file_path)
    # opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # img = cv2.resize(opencvImage, (150, 150))
    # img = img.reshape(1, 150, 150, 3)
    # p = model.predict(img)
    # p = np.argmax(p, axis=1)[0]

    # return {"prediction": p}

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
