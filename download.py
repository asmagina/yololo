# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model


import yolov5

def download_model():
    model = yolov5.load('fcakyon/yolov5s-v7.0', device='0')


if __name__ == "__main__":
    download_model()