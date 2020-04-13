# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 20:14:22 2018

@author: Rishabh
"""
import numpy as np
import pandas as pd
import cv2
import RPi.GPIO as GPIO
import time
import pygame
#from pygame.locals import *
import pygame.camera
from sklearn.svm import SVC     #"Support Vector Classifier"

pygame.init()
pygame.camera.init()
   
LED1 =36;LED2=38;LED3=40;
i=0;j=0;
GPIO.setmode(GPIO.BOARD)
GPIO.setup(LED1, GPIO.OUT)
GPIO.setup(LED2, GPIO.OUT)
GPIO.setup(LED3, GPIO.OUT)

GPIO.setup(11,GPIO.OUT) #big disk
GPIO.setup(13,GPIO.OUT) #tilt
GPIO.setup(7,GPIO.OUT) #small disk
pwm2=GPIO.PWM(11,100)
pwm3=GPIO.PWM(13,100)
pwm1=GPIO.PWM(7,100)

GPIO.output(LED1,True)
time.sleep(2)
GPIO.output(LED1,False)

GPIO.output(LED2,True)
time.sleep(2)
GPIO.output(LED2,False)

GPIO.output(LED3,True)
time.sleep(2)
GPIO.output(LED3,False)

pwm1.start(5)
pwm2.start(5)
pwm3.start(5)
pwm1.ChangeDutyCycle(0)
pwm2.ChangeDutyCycle(0)
pwm3.ChangeDutyCycle(0)


'''---------ML CODE-------'''
x = pd.read_csv("output.csv")
a= np.array(x)
y = a[:,12]
x = a[:,:12]

#x.shape
#print (x)
#print (y)

clf = SVC(kernel='linear')
clf.fit(x, y)
'''======trained the data====='''



'''-----------LED CODE-----------'''   
#GPIO.output(LED1,True)
GPIO.output(LED2,True)
#GPIO.output(LED3,True)
pwm1.ChangeDutyCycle(0)
pwm2.ChangeDutyCycle(0)
pwm3.ChangeDutyCycle(0)
time.sleep(2)
GPIO.output(LED2,False)
GPIO.output(LED1,True)
GPIO.output(LED3,True)
    

i=0;j=0;
meanh=0.0;means=0.0;meanv=0.0;meanr=0.0;meang=0.0;meanb=0.0;
stdevh=0.0;stdevs=0.0;stdevv=0.0;stdevr=0.0;stdevg=0.0;stdevb=0.0;
while (j<3):
    pwm1.ChangeDutyCycle(0)
    angle = j*120
    duty0=(float)(angle)/10+2.5
    pwm1.ChangeDutyCycle(duty0)
        
    time.sleep(5) #wait 5 sec before capture
    GPIO.output(LED2,False)
    time.sleep(1)
       
    GPIO.output(LED2,True)
    GPIO.output(LED1,True)
    GPIO.output(LED3,True)
        
    '''----------CAMERA CAPTURE------------'''
    cam = pygame.camera.Camera("/dev/video0",(400,300))
    cam.start()
    image= cam.get_image()
    pygame.image.save(image,r'/home/pi/Desktop/MiniProject/testimage1.jpg')
    cam.stop()
    time.sleep(2)
    '''#IMP--> Change path name and file names'''
    GPIO.output(LED1,False)
    GPIO.output(LED2,False)
    GPIO.output(LED3,False)
    img = cv2.imread("101.jpg")
    crop_img = img[80:300, 70:220]
    #cv2.imshow("testimage.jpg", crop_img)
    cv2.imwrite('testimage.jpg',crop_img)
        

        
    '''------------TEST CLONING----------'''
    imgx= cv2.imread('testimage.jpg')
    hsv_img= cv2.cvtColor(imgx, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv_img)
        
    cv2.imwrite('splane.png',s)
    img= cv2.imread('splane.png')
    ret,thresh2=cv2.threshold(img,55, 255,cv2.THRESH_BINARY_INV)
        
    kernel = np.ones((5,5),np.uint8)
        
    closing=cv2.morphologyEx(thresh2,cv2.MORPH_CLOSE, kernel)
    mask_inv=cv2.bitwise_not(closing)
    cv2.imwrite('closed_image.png',closing)
        
    mask_inv=cv2.bitwise_and(mask_inv,imgx)
    cv2.imwrite('wo_back.jpg',mask_inv)
        
        
    '''----------------COLOR VECTOR---------------'''
    img1= cv2.imread('wo_back.jpg')
    r1, g1, b1= cv2.split(img1)
        
    hsv_img1= cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    h1,s1,v1= cv2.split(hsv_img1)
        
    tmeanh,_,_,_=cv2.mean(h1)
    tmeans,_,_,_=cv2.mean(s1)
    tmeanv,_,_,_=cv2.mean(v1)
    tmeanr,_,_,_=cv2.mean(r1)
    tmeang,_,_,_=cv2.mean(g1)
    tmeanb,_,_,_=cv2.mean(b1)
        
    tstdevh=np.std(h1) 
    tstdevs=np.std(s1) 
    tstdevv=np.std(v1) 
    tstdevr=np.std(r1) 
    tstdevg=np.std(g1) 
    tstdevb=np.std(b1)
    meanh= (float)(meanh);
    means= (float)(means);
    meanv= (float)(meanv);
    meanr= (float)(meanr);
    meang= (float)(meang);
    meanb= (float)(meanb);


    stdevh= (float)(stdevh);
    stdevs= (float)(stdevs);
    stdevv= (float)(stdevv);
    stdevr= (float)(stdevr);
    stdevg= (float)(stdevg);
    stdevb= (float)(stdevb);


    tmeanh= (float)(format(tmeanh, '.4f'))
    tmeans= (float)(format(tmeans, '.4f'))
    tmeanv= (float)(format(tmeanv, '.4f'))
    tmeanr= (float)(format(tmeanr, '.4f'))
    tmeang= (float)(format(tmeang, '.4f'))
    tmeanb= (float)(format(tmeanb, '.4f'))

    tstdevh= (float)(format(tstdevh, '.4f'))
    tstdevs= (float)(format(tstdevs, '.4f'))
    tstdevv= (float)(format(tstdevv, '.4f'))
    tstdevr= (float)(format(tstdevr, '.4f'))
    tstdevg= (float)(format(tstdevg, '.4f'))
    tstdevb= (float)(format(tstdevb, '.4f'))
        
    meanh= meanh+tmeanh
    means= means+tmeans
    meanv= meanv+tmeanv
    meanr= meanr+tmeanr
    meang= meang+tmeang
    meanb= meanb+tmeanb

    stdevh= stdevh+tstdevh
    stdevs= stdevs+tstdevs
    stdevv= stdevv+tstdevv
    stdevr= stdevr+tstdevr
    stdevg= stdevg+tstdevg
    stdevb= stdevb+tstdevb

    j=j+1


pwm1.ChangeDutyCycle(0)

stdevh=(float)(stdevh/3)
stdevs=(float)(stdevs/3)
stdevv=(float)(stdevv/3)
stdevr=(float)(stdevr/3)
stdevg=(float)(stdevg/3)
stdevb=(float)(stdevb/3)
meanh= (float)(meanh/3)
means= (float)(means/3)
meanv= (float)(meanv/3)
meanr= (float)(meanr/3)
meang= (float)(meang/3)
meanb= (float)(meanb/3)

#print (meanh, means, meanv, meanr, meang, meanb, stdevh, stdevs, stdevv, stdevr, stdevg, stdevb )
    
'''--------------CLASSIFIER-------------'''
ans = clf.predict([[meanh,means,meanv,meanr,meang,meanb,stdevh,stdevs,stdevv,stdevr,stdevg,stdevb]])
print ("Classified as:")
print (ans)
if(ans==1):
    GPIO.output(LED1,True)

elif(ans==2):
    GPIO.output(LED2,True)

elif(ans==3):
    GPIO.output(LED3,True)

'''-------MOTOR with TUB---------'''

angle= (ans-1)*120
duty=(float)(angle)/10+2.5
pwm2.ChangeDutyCycle(duty)
time.sleep(4)
    

duty0=(float)(70)/10+2.5
pwm3.ChangeDutyCycle(duty0)
time.sleep(4)
duty1=(float)(0)/10+2.5
pwm3.ChangeDutyCycle(duty1)
time.sleep(4)
'''=====continue WHILE LOOP======'''
GPIO.setwarnings(False)
        



