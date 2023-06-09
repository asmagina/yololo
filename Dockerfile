# Must use a Cuda version 11+
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0


# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Add your model weight files 
# (in this case we have a python script)
ADD yolov5s.pt . 
ADD download.py .
RUN python3 download.py

ADD app.py .


EXPOSE 8000

CMD python3 -u app.py
