from __future__ import print_function
from tkinter import *
import cv2
import numpy as np
import cv2 as cv
from pytesseract import pytesseract


root = Tk()
label =Label(root, width=300, height=110)
label.place(x=0,y=475)

#Camara
cap= cv2.VideoCapture(1)

#inicio de variable texto GLOBAL para la lectura
global text_codigo
text_codigo= " "

#codigo que extrae el numero de lote
def codLOte(text):
   global text_codigo
   text_codigo= " "
   n = 0
   for y in text:
      n += 1
      k = len(text)
      if y=="F" or y =="f" and n<=k:
      
       text_codigo += text[n-1:(n+10)]
       n += 1

   print("Texto original")
   print(text)
   print ("El codigo es")
   print ( text_codigo)
   r2.delete('1.0', END)    
   r2.insert(END, text_codigo)
   return text_codigo

def show_frames():
   cv2image= cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2RGB)
   rot0 = cv2.rotate(cv2image, cv2.ROTATE_90_CLOCKWISE)
   ImgDerecha = cv2.rotate(rot0, cv2.ROTATE_90_CLOCKWISE)

   ImgDerecha = ImgDerecha[185:290,150:350] #Esta linea recorta la imagen

   img = Image.fromarray(ImgDerecha)
   imgtk = ImageTk.PhotoImage(image = img)
   label.imgtk = imgtk
   label.configure(image=imgtk)
   label.after(20, show_frames)


def reorient_photo(image):
   gray_0 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   gray = cv2.bitwise_not(gray_0)
   im_blurred = cv2.GaussianBlur(gray, (0, 0), 3)
   gray = cv2.addWeighted(gray_0, 0.5, im_blurred, 1.2, 0)


   thresh = cv2.threshold(gray, 0, 255,
      cv2.THRESH_BINARY | cv2.THRESH_OTSU)[-1]

   coords = np.column_stack(np.where(thresh > 0))
   angle = cv2.minAreaRect(coords)[-1]

   if angle < -45:
      angle = -(90 + angle)
      
   else:
      angle = -angle

   (h, w) = image.shape[:2]
   center = (w // 2, h // 2)
   M = cv2.getRotationMatrix2D(center, angle, 1.0)
   rotated = cv2.warpAffine(image, M, (w, h),
      flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# boton scanear
def photo_filter (image):

   new_image = np.zeros(image.shape, image.dtype)

   alpha = 1.0 
   beta = 0    

   print(' Lectura completa del lector ')
   print('-------------------------')
   try:
      alpha = float(2) #gamma
      beta = int(40) #Brillo
   except ValueError:
      print('Error, not a number')
      
   for y in range(image.shape[0]):
      for x in range(image.shape[1]):
         for c in range(image.shape[2]):
               new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
            
   image = new_image

   gray = cv.cvtColor(new_image, cv.COLOR_BGR2GRAY)
   text = pytesseract.image_to_string(gray)
   print(text) 
   print('-------------------------')

   return text


def takePhoto():
   ret, frame = cap.read()                                                                                  
   if not ret:
      print('failed to grab frame')

   img_name = f'opencv_frame_.jpg'

   rot0 = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
   realFream = cv2.rotate(rot0, cv2.ROTATE_90_CLOCKWISE)
   cv2.imwrite(img_name, realFream)
   img1 = cv2.imread(img_name, 1)
   img1 = img1[185:310,0:700] #recorta la imagen
   cv2.imwrite(img_name, img1)
   reorient_photo(img1)

   codLOte(photo_filter (img1))


def obtener_codigo_lote():
   text_codigo = r2.get(1.0, "end-1c")
   return text_codigo


botonScanear = Button (root, text="Escanear", bg='#EF5350',font=('arial', '20'), width= 15, height=2 , fg='white', command=takePhoto)
botonScanear.place(x=580,y=415)

botonCargar = Button (root, text="Cargar", bg='#EF5350',font=('arial', '20'), width= 15, height=2 , fg='white')
botonCargar.place(x=580,y=570)

root.mainloop()