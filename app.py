#deep learning libraries




#web frameworks
from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
import uvicorn
import aiohttp
import asyncio
import numpy as np
from skimage import transform
from tensorflow.keras.models import load_model

import io
import os
import sys
import base64 
from PIL import Image, ImageOps

async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

app = Starlette()

#importing model
MODEL_PATH = 'model.h5'
model = load_model(MODEL_PATH)

@app.route("/upload", methods = ["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)

@app.route("/classify-url", methods = ["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes)

def predict_image_from_bytes(bytes):
    #load byte data into a stream
    img_file = io.BytesIO(bytes)
    #encoding the image in base64 to serve in HTML
    im = Image.open(img_file)
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(im, size, Image.ANTIALIAS)
    #im.save("img.jpg")
    
    #img_uri = base64.b64encode(open("img.jpg", 'rb').read()).decode('utf-8')
    
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    #image = Image.open("img.jpg")

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    result = model.predict(data)
    result=np.argmax(result)
    
    # process an array and print result
    if result == 0:
        prediction = "Green_bin"
    elif result == 1:
        prediction = "Blue_bin"
    elif result == 2:
        prediction = "Black_bin"
    elif result == 3:
        prediction = "Trash_bin"
    else:
        prediction = "error"

    print(prediction)

    return JSONResponse({'Prediction' : prediction })
        
@app.route("/")
def form(request):
        return HTMLResponse(
            """
            <h1> WasteAPI </h1>
            <p> What bin should your item go into?? </p>
            <form action="/upload" method = "post" enctype = "multipart/form-data">
                <u> Select picture to upload: </u> <br> <p>
                1. <input type="file" name="file"><br><p>
                2. <input type="submit" value="Upload">
            </form>
            <br>
            <br>
            <u> Submit picture URL </u>
            <form action = "/classify-url" method="get">
                1. <input type="url" name="url" size="60"><br><p>
                2. <input type="submit" value="Upload">
            </form>
            """)
        
@app.route("/form")
def redirect_to_homepage(request):
        return RedirectResponse("/")
        
if __name__ == "__main__":
    if "serve" in sys.argv:
        port = int(os.environ.get("PORT", 8008)) 
        uvicorn.run(app, host = "0.0.0.0", port = port)
