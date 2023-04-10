# American Sign Language (ASL)
# Overview
American Sign Language (ASL), is a natural language that serves as the predominant sign language of Deaf communities in the United States of America and most of Anglophone Canada. ASL is a complete and organized visual language that is expressed by employing both manual and nonmanual features.

# Dataset
The ASL Classifier uses the [ASL Alphabet Dataset](https://www.kaggle.com/datasets/sachinmlwala/asl-dataset) uploaded on Kaggle and can be found in [Releases](https://github.com/sachin11226/ASL/releases/tag/Model) also . The dataset consists of 10,000+ images of ASL signs, with 26 classes (letters A-Z approx 400 images per class ).

# Model
The ASL Classifier uses a hand tracking module from [Mediapipe](https://mediapipe.dev/) to localize the hand and convolutional neural network (CNN) to classify ASL signs. The model architecture consists of two convolutional layers followed by two fully connected layers. The final layer is a softmax layer that outputs the probability distribution over the 26 classes.

# Usage
To run the Code on you machine.
1.  Download model from [Releases](https://github.com/sachin11226/ASL/releases/tag/Model) and Clone the repository:
```bash
git clone https://github.com/sachin11226/ASL 
```
2.  Install the required packages and Run main.py.
```bash
pip install opencv,mediapipe,tensorflow
```
# Video demo
https://user-images.githubusercontent.com/78334981/227418019-7545bcf6-1ee2-4461-baa1-e07ba68deb9e.mp4

