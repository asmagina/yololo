# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model


import yolov5

def download_model():
    yolov5.load('yolov5s.pt', device='0')


if __name__ == "__main__":
    download_model()