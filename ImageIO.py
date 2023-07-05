import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

#Changes the directory from curr to new
def ChangeDirectory(curr,new):
    path=os.path.realpath(__file__)
    dir1=os.path.dirname(path)
    dir2=dir1.replace(curr,new)
    os.chdir(dir2)

#Reads the image as either grayscale or colour  
def readImage(file,scale):
    if(scale=="gray"): #greyscale
        img=cv2.imread(file,0)
    elif(scale=="color"):       #normal
        img=cv2.imread(file)
    assert img is not None, file+" file could not be read, check with os.path.exists()"
    return img

#Saves the image in location 'file' as either grayscale or color
def saveImage(file,image,scale):
    if(scale=="gray"):    #greyscale
        plt.imsave(file,image,cmap="gray")
    elif(scale=="color"):   #normal
        plt.imsave(file,image)
        
#Plots the image in either grayscale or color        
def PlotOne(image,scale):
    if(scale=="gray"): 
        plt.imshow(image,cmap="gray")
    elif(scale=="color"):       #normal
        plt.imshow(image)
        
#PLots images in a matrix        
def Plot(images,names,scale,rows,cols,length=20,breadth=20):
    plt.figure(figsize = (length,breadth))
    for i,im in enumerate(images):
        plt.subplot(rows,cols,(i+1))
        plt.title(names[i])
        PlotOne(im,scale[i])
        plt.axis('off')
    plt.show()
    
#Showing the images in a new window    
def Show(images,names):
    for i,image in enumerate(images):
        cv2.namedWindow(names[i], cv2.WINDOW_NORMAL)
        cv2.imshow(names[i],image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

   