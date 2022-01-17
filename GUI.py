

from tkinter import *
import time

#Import scikit-learn metrics module for accuracy calculation
import pickle
from PIL import Image, ImageTk  

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

def pp(a):
    global mylist
def predict(val):
    print(val)
    sentence = clean(val)
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
    
    
    
    
def userHome():
    global root, mylist,shrslt
    root = Tk()
    root.geometry("1200x700+0+0")
    root.title("Home Page")

    image = Image.open("colorbg.jpg")
    image = image.resize((1200, 700), Image.ANTIALIAS) 
    pic = ImageTk.PhotoImage(image)
    lbl_reg=Label(root,image=pic,anchor=CENTER)
    lbl_reg.place(x=0,y=0)
  
    #-----------------INFO TOP------------
    lblinfo = Label(root, font=( 'aria' ,20, 'bold' ),text="Color Palette generation",fg="white",bg="#000955",bd=10,anchor='w')
    lblinfo.place(x=100,y=50)
    lblinfo2 = Label(root, font=( 'aria' ,10 ),text="Generating color palette based on text emotions  ",fg="white",bg="#000955",anchor='w')
    lblinfo2.place(x=120,y=130)
    lblinfo3 = Label(root, font=( 'aria' ,20 ),text="Provide input",fg="#000955",anchor='w')
    lblinfo3.place(x=800,y=180)
    E1 = Entry(root,width=30,font="veranda 20")
    E1.place(x=650,y=260)
     
    btntrn=Button(root,padx=16,pady=8, bd=6 ,fg="white",font=('ariel' ,16,'bold'),width=10, text="Generate", bg="red",command=lambda:predict(E1.get()))
    btntrn.place(x=780, y=340)

      


    # btnhlp=Button(root,padx=16,pady=8, bd=6 ,fg="white",font=('ariel' ,10,'bold'),width=7, text="Help?", bg="blue",command=lambda:predict(E1.get()))
    # btnhlp.place(x=50, y=450)
    # rslt = Label(root, font=( 'aria' ,20, ),text="RESULT :",fg="black",bg="white",anchor=W)
    # rslt.place(x=640,y=480)
    # shrslt = Label(root, font=( 'aria' ,20, ),text="",fg="blue",bg="white",anchor=W)
    # shrslt.place(x=730,y=480)

    def qexit():
        root.destroy()
     

    root.mainloop()


userHome()