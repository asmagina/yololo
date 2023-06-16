import yolov5
import torch
from PIL import Image
import base64
import io
import numpy as np
import transformers
from transformers import pipeline
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from potassium import Potassium, Request, Response


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"

app = Potassium("my_app")

@app.init
def init():   
    device = 0 if torch.cuda.is_available() else -1
    transformers.utils.move_cache()

    # load model
    model = yolov5.load('yolov5s.pt', device='cpu')

    # set model parameters
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1000  # maximum number of detections per image

    # # image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
    # model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")
    # model = pipeline("object-detection", model=model, device=device)

    # model.conf = 0.25  # NMS confidence threshold
    # model.iou = 0.45  # NMS IoU threshold
    # model.agnostic = False  # NMS class-agnostic
    # model.multi_label = False  # NMS multiple labels per box
    # model.max_det = 1000  # maximum number of detections per image
    context = {
        "model": model
    }

    return context    

@app.handler()
def handler(context: dict, request: Request) -> Response:
    model = context.get("model")

   # get the base64 encoded string
    im_b64 = request.json.get('image')

    # convert it into bytes  
    img_bytes = base64.b64decode(im_b64.encode('utf-8'))

    # convert bytes data to PIL Image object
    img = Image.open(io.BytesIO(img_bytes))
    # inputs = image_processor(images=img, return_tensors="pt")
    outputs = model(img)

    return str(outputs.__len__())
