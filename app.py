import yolov5
import torch
from PIL import Image
import base64
import io
import numpy as np
from transformers import pipeline
from transformers import AutoImageProcessor, AutoModelForObjectDetection


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global image_processor
    
    device = 0 if torch.cuda.is_available() else -1
    model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")
    model = pipeline("object-detection", model=model, image_processor=image_processor, device=device)
    image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")

    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1000  # maximum number of detections per image

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global image_processor

   # get the base64 encoded string
    im_b64 = model_inputs.json['image']

    # convert it into bytes  
    img_bytes = base64.b64decode(im_b64.encode('utf-8'))

    # convert bytes data to PIL Image object
    img = Image.open(io.BytesIO(img_bytes))
    # inputs = image_processor(images=img, return_tensors="pt")
    outputs = model(img)

    return str(outputs.__len__())
