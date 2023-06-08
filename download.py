# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model


from transformers import pipeline
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import torch

def download_model():
    device = 0 if torch.cuda.is_available() else -1
    image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
    model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")
    model = pipeline("object-detection", model=model, image_processor=image_processor, device=device)


if __name__ == "__main__":
    download_model()