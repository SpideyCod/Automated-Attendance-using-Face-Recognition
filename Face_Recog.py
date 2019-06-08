import os
import cv2
import numpy as np
import os
import csv
import pandas as pd
from PIL import Image
from pymongo import database


def retrieve_input():
    global date
    inputValue=textBox.get("1.0","end-1c")
    date = str(inputValue)


#################################################################
###############-------Training the Model---------################
#################################################################

def training():
    getImagesWithID(path)

    


    Base_Dir=os.path.dirname(os.path.abspath(__file__))
    image_dir=os.path.join(Base_Dir,"dataSet")
#recognizer=cv2.createLBPHFaceRecognizer();
#recognizer=cv2.face.createLBPHFaceRecognizer();
recognizer=cv2.face.LBPHFaceRecognizer_create();
#recognizer=EigenFaceRecognizer_create();
path='dataSet'


def getImagesWithID(path):
        import os
        import cv2
        import numpy as np
        from PIL import Image


    
        imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
        print(imagePaths)
        faces=[]
        IDs=[]
        for imagePath in imagePaths:
            faceImg=Image.open(imagePath).convert("L");
            faceNp=np.array(faceImg,'uint8')
            ID=int(os.path.split(imagePath)[-1].split('.')[1])
            faces.append(faceNp)
            IDs.append(ID)
            cv2.imshow("training",faceNp)
            cv2.waitKey(10)
        return IDs,faces
Ids,faces=getImagesWithID(path)
recognizer.train(faces,np.array(Ids))
recognizer.write('trainingData.yml')
cv2.destroyAllWindows()
recognizer
        
#################################################################        
##############---------Face Detection-----------#################
#################################################################


def detector():
    names = pd.read_csv('Names.csv')
    
    names = pd.DataFrame(names)
    names.set_index(['Id'], inplace  =True)
    print(names)

    import cv2

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    cam=cv2.VideoCapture(0)
    rec = cv2.face.LBPHFaceRecognizer_create()
#rec=cv2.createLBPHFaceRecognizer();
#rec=cv2.face.createLBPHFaceRecognizer();
#rec=cv2.face.LBPHFaceRecognizer_create();
#rec.load("recognizer/trainingData.yml")
    rec.read("trainingData.yml")
#print(res)
    id2=""
    while True:
        ret,img=cam.read()
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
             roi_gray = gray[y:y+h, x:x+w]
             roi_color = img[y:y+h, x:x+w]
             cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
             id,conf=rec.predict(roi_gray)

             

             df =  pd.read_csv('Attendance.csv')
             df.set_index(['Id'], inplace = True)
             
             for i in names.index:
                 if i ==id:
                     id2 = names.Names[id]
                     df.loc[id,date] = 'Present'
             
            
             os.path.isfile('Attendance.csv')
             df.to_csv('Attendance.csv')
                
         
         #cv2.putText(img, (x,y+h),str(id),font,255,2,cv2.LINE_AA)
         #cv2.putText(img,id2,(x,y+h),  gray, 4,(255,255,255),2,cv2.LINE_AA)
             cv2.putText(img, id2, (x, h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255)) 
         #cv2.putText(cv2.fromarray(img),str(id),(x,y+h),gray,255)
        cv2.imshow('img',img)
        k=cv2.waitKey(30)&0xff
        if k==27:
            break
    cam.release()
    cv2.destroyAllWindows()
    

from tkinter import *

win = Tk()
win.title("Attendance")
frame = Frame(win, height=15, width=200)
frame.pack()


textBox=Text(win, height=2, width=10)
textBox.pack(side = TOP)
button_date=Button(win, height=1, width=10, fg = 'blue', text="Enter Date", 
                    command=lambda: retrieve_input())

button_date.pack(fill = 'x', side = TOP)

button_recog = Button(win, text = "Recognize", fg = 'red', command = detector)
button_recog.pack(fill = 'both')




win.mainloop()

        
        



