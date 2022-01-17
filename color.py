# import cv2
# import numpy as np

# def create_blank(width, height, rgb_color=(0, 0, 0)):
#     """Create new image(numpy array) filled with certain color in RGB"""
#     # Create black blank image
#     image = np.zeros((height, width, 3), np.uint8)

#     # Since OpenCV uses BGR, convert the color first
#     color = tuple(reversed(rgb_color))
#     # Fill image with color
#     image[:] = color

#     return image

# # Create new blank 300x300 red image
# width, height = 80, 80

# red = (255, 0, 0)
# blue = (0, 0, 255)
# green=(0, 255, 0)
# redim = create_blank(width, height, rgb_color=red)
# blueim = create_blank(width, height, rgb_color=blue)
# green = create_blank(width, height, rgb_color=green)
# im_h = cv2.hconcat([redim, blueim,green])
  
# # show the output image
# cv2.imshow('man_image.jpeg', im_h)
# # cv2.imshow("im",blueim)
# cv2.waitKey(0)
# cv2.imwrite('red.jpg', red)




import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional
import cv2
import numpy as np

def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

# Create new blank 300x300 red image
width, height = 80, 80
# Text preprocessing function

str_punc = string.punctuation.replace(',', '').replace("'",'')

def clean(text):
    global str_punc
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.lower()
    return text   

lfile = open("labelEncoder.pickle",'rb')
le = pickle.load(lfile)
lfile.close()

tkfile = open("tokenizer.pickle",'rb')
tokenizer = pickle.load(tkfile)
tkfile.close()

json_file = open('cnn_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("cnn_weight.h5")
print("Loaded model from disk")

sentences = [
            "He's over the moon about being accepted to the university",
            "Your point on this certain matter made me outrageous, how can you say so? This is insane.",
            "I can't do it, I'm not ready to lose anything, just leave me alone",
            "Merlin's beard harry, you can cast the Patronus charm! I'm amazed!"
            ]
# for sentence in sentences:
# print(sentence)
sentence = clean("i feel so cold a href http irish")
sentence = tokenizer.texts_to_sequences([sentence])
sentence = pad_sequences(sentence, maxlen=256, truncating='pre')
print(model.predict(sentence))
proba = model.predict(sentence)
probanger= proba[0][0]*100
probfear= proba[0][1]*100
probjoy= proba[0][2]*100
problove= proba[0][3]*100
probsad= proba[0][4]*100

print("anger",probanger)
print("fear",probfear)
print("sad",probsad)
print("joy",probjoy)
print("love",problove)
angrycolor=[(255, 0, 0),(255, 51, 51),(255, 102, 102),(255, 173, 153)]
fearcolor=[(255, 119, 51),(255, 136, 77),(255, 153, 102),(255, 230, 204)]
joycolor=[(51, 204, 51),(92, 214, 92),(133, 224, 133),(214, 245, 214)]
lovecolor=[(255, 26, 140),(255, 102, 179),(255, 153, 204),(255, 204, 230)]
sadcolor=[(153, 102, 51),(204, 153, 102),(217, 179, 140),(236, 217, 198)]
# for i in proba[0]:
    
#     print(i*100)
result = le.inverse_transform(np.argmax(model.predict(sentence), axis=-1))[0]
proba =  np.max(model.predict(sentence))
print(f"{result} : {proba}\n\n")


if(probanger>80):
    angim=create_blank(width, height, rgb_color=angrycolor[0])
elif(probanger>1):
    angim=create_blank(width, height, rgb_color=angrycolor[1])
elif(probanger>0.01):
    angim=create_blank(width, height, rgb_color=angrycolor[2])
else:
    angim=create_blank(width, height, rgb_color=angrycolor[3])



if(probjoy>80):
    joyim=create_blank(width, height, rgb_color=joycolor[0])
elif(probjoy>1):
    joyim=create_blank(width, height, rgb_color=joycolor[1])
elif(probjoy>0.01):
    joyim=create_blank(width, height, rgb_color=joycolor[2])
else:
    joyim=create_blank(width, height, rgb_color=joycolor[3])


if(probfear>80):
    fearim=create_blank(width, height, rgb_color=fearcolor[0])
elif(probfear>1):
    fearim=create_blank(width, height, rgb_color=fearcolor[1])
elif(probfear>0.01):
    fearim=create_blank(width, height, rgb_color=fearcolor[2])
else:
    fearim=create_blank(width, height, rgb_color=fearcolor[3])

if(problove>80):
    loveim=create_blank(width, height, rgb_color=lovecolor[0])
elif(problove>1):
    loveim=create_blank(width, height, rgb_color=lovecolor[1])
elif(problove>0.01):
    loveim=create_blank(width, height, rgb_color=lovecolor[2])
else:
    loveim=create_blank(width, height, rgb_color=lovecolor[3])

if(probsad>80):
    sadim=create_blank(width, height, rgb_color=sadcolor[0])
elif(probsad>1):
    sadim=create_blank(width, height, rgb_color=sadcolor[1])
elif(probsad>0.01):
    sadim=create_blank(width, height, rgb_color=sadcolor[2])
else:
    sadim=create_blank(width, height, rgb_color=sadcolor[3])

im_h = cv2.hconcat([angim, fearim,sadim,loveim,joyim])
  
# show the output image
cv2.imshow('man_image.jpeg', im_h)
# cv2.imshow("im",blueim)
cv2.waitKey(0)