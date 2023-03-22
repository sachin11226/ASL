# importing libararies 
from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np
#cuda giving error so I disabled cuda
#you can delete below two line if you want to use cuda
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

# reading video input from camera
cap = cv2.VideoCapture(0)


label =  ['N', 'R', 'B', 'I', 'F', 'H', 'E', 'U', 'M', 'X', 'K', 'Q', 'Y',
                'S', 'G', 'A', 'O', 'T', 'V', 'Z', 'C', 'P', 'L', 'W', 'D', 'J'] 

#At the back end of cvzone is using mediapipe to to implement hand detection module 
detector = HandDetector(maxHands=1)
offset = 15
imgsize = 300
#loading trained model 
model = tf.keras.models.load_model('my_model')

#below code is used to save images when creating data set 

# counter=0
# letter=input("enter the capture letter")
# folder='Data/'+letter.capitalize()
# print(folder)
x_lim=400
def predict(img):
    '''
    Take image as and return the index of max probable class
    '''
    # Normalizaton and changing the colour  of image 
    img=(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))/255
    
    # expanding dim because model.predict take a 4d tensor as input
    img = tf.expand_dims(img,0)
   
    # prediction the most probable class 
    interval=model.predict(img)
    pred = np.argmax(interval)
    
    return pred,interval

while True:
    success, img = cap.read()
    # img=cv2.flip(img,1)
    hands, img = detector.findHands(img) # Return the coordinate of hand and  Keypoints 

    imgWhite = np.ones((imgsize, imgsize, 3), np.uint8)*255 
    try:
        if hands:
            
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgcrop = img[y-offset:y+h+offset, x-offset:x+10+w+offset]
            aspectRatio = h/w
            '''If the height is greater than width than setting height to 300px and
                centring the image about the widht and vice versa '''
            if aspectRatio > 1:
                k = imgsize/h
                wcal = int(k*w)
                imgresize = cv2.resize(imgcrop, (wcal, imgsize))
                wgap = int((300-wcal)/2)
                imgWhite[:, wgap:wcal+wgap] = imgresize
                pred,interval=predict(imgWhite)
                pred= label[pred]
                cv2.putText(img,pred,(100,100), cv2.FONT_HERSHEY_SIMPLEX,2 ,(0,255,0),2)

            if aspectRatio < 1:
                k = imgsize/w
                hcal = int(k*h)
                imgresize = cv2.resize(imgcrop, (imgsize, hcal))
                hgap: int = int((300-hcal)/2)
                imgWhite[hgap:hcal+hgap, :] = imgresize
                pred,interval=predict(imgWhite)
                pred= label[pred]
                cv2.putText(img=img,text=pred,org=(100,100),fontFace= cv2.FONT_HERSHEY_SIMPLEX,fontScale=5,color=(0,255,0),thickness= 2)
            
            cv2.imshow('imgwhite',imgWhite)
    except:
        pass
    cv2.imshow('img',img[:,:x_lim,:])

    # if cv2.waitKey(2) == ord('s'):
    #     counter+=1
    #     cv2.imwrite(f'{folder}\\{letter}{counter}.jpg',imgWhite)
    #     print(counter)
    

    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
