# Face Recognition Based on Arcface ONNX Model 
---
Face Recognition on Arcface ONNX Model. The Face Recognition model used in this repository is pretrained Arcface Resnet100.  Tested with MXNet Runtime to perform the model processing. Face detection performed with MTCNN. Recognition process involve embedding data of input face image and compare with the other.
---
## Environment and Installation
This repository only tested in Linux OS and Google Colab
1. Use python environment or anaconda(conda) with cudatoolkit and cudnn support (need to access *.so library)
2. Proposed python version => 3.6
3. Install requirements with shell command
   ```
   sh install-req.sh
   ```
## How to RUN
1. Download mtcnn dependencies with model-downloader.py. This is also downloading the pretrained Arcface ResNet100 model.
   ```
   python model-downloader.py
   ```
2. Run the program with arguments
   ```
   python recognize.py -1 [first image] -2 [second image]
   ```
3. The result are printed on terminal as embedding distance between 2 face images and the similarity
   ```
   Distance : 0.413434
   Similarity : 0.786454
   ```