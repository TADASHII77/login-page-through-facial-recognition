
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from tkinter import *
from PIL import Image, ImageTk
data_path = 'C:/Users/yamim/AppData/Local/Programs/Python/Python311/TrainingImage/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

Training_Data, Labels = [], []

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Training_Data), np.asarray(Labels))

#print("Dataset Model Training Complete!!!!!")

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def main():
    win = Tk()
    win.title("Login page")
    win.geometry("900x600")
    win.config(bg="#1f618d")  # Changed background color
    

    # Open the image
    load = Image.open('atm.jpg')

    # Define the new dimensions
    new_width = 500
    new_height = 500

    # Resize the image
    load = load.resize((new_width, new_height))

    render = ImageTk.PhotoImage(load)
    img = Label(win, image=render, bg="#1f618d")  # Added background color
    img.pack(side=RIGHT)

    f1 = Frame(win, width=400, height=300, bg="#f5f5f5", borderwidth=2, relief="groove", border=10, bd=0, padx=30, pady=30)
    f1.pack(side=LEFT, padx=50, pady=100)

    lb = Label(f1, text="Login Page", font=('italic', 24, 'bold'), fg="#1f618d", bg="#f5f5f5")  # Changed text color
    lb.grid(row=0, column=0, pady=(0, 20), padx=5, columnspan=2)

    # Create faded text for username and password
    username_entry = Entry(f1, font=('italic', 15, 'normal'), fg="#bdbdbd", insertbackground="#bdbdbd", bd=2, relief="ridge")
    username_entry.insert(0, "Username")
    username_entry.grid(row=1, column=0, columnspan=2, sticky=W, padx=5, pady=10, ipadx=10, ipady=5)

    password_entry = Entry(f1, show="*", font=('italic', 15, 'normal'), fg="#bdbdbd", insertbackground="#bdbdbd", bd=2, relief="ridge")
    password_entry.insert(0, "Password")
    password_entry.grid(row=2, column=0, columnspan=2, sticky=W, padx=5, pady=10, ipadx=10, ipady=5)

    forget_password_label = Label(f1, text="Forget Password?", font=('italic', 12, 'normal'), fg="#1f618d", bg="#f5f5f5", cursor="hand2")
    forget_password_label.grid(row=3, column=0, columnspan=2, pady=5)
    forget_password_label.bind("<Button-1>", lambda e: print("Forget Password Clicked"))  # Add functionality here

    # Create rounded buttons
    login_button = Button(f1, text="Log In", font=('italic', 15, 'bold'), bg="#ffa600", fg="white", bd=0, relief="ridge", padx=20, pady=5)
    login_button.grid(row=4, column=0, columnspan=2, pady=20, padx=10, ipadx=10, sticky='ew')
    win.mainloop()
def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return img,[]

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))

    return img,roi

cap = cv2.VideoCapture(0)
while True:

    ret, frame = cap.read()

    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        if result[1] < 500:
            confidence = int(100*(1-(result[1])/300))



        if confidence > 82:
            cv2.putText(image, "Rohit", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Face Cropper', image)
            cap.release()
            cv2.destroyAllWindows()
            main()
            break
            

        else:
            cv2.putText(image, "Unknown", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)


    except:
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', image)
        pass

    if cv2.waitKey(1)==13:
        break


cap.release()
cv2.destroyAllWindows()
