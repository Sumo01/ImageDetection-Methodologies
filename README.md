# ImageDetection-Methodologies

This repo has all the image detection methodologies that are commonly used when performing analysis on image datasets and while performing computer vision tasks. It mainly employs the open-cv library in python. Explanation for each file in this repo is given below:

## Methodologies:
These are implementations of common methodologies along with examples for simple understanding. 
1. <a href='CannyEdgeDetection.ipynb'> Canny Edge Detection</a>
2. <a href='DilationAndErosion.ipynb'>Dilation and Erosion</a>
3. <a href='FourierTransform.ipynb'>Fourier Transform</a>
4. <a href='HistogramEqualization.ipynb'>Histogram Equalization</a>
5. <a href='HoughTransform.ipynb'>Hough Transform</a>

## Modules (in utils):
These files have all the codes for different types of image operations and feature extraction.

### <a href='utils/ImageIO.pyImageIO.py'>ImageIO</a>
This file has all the input output operations that can be performed on the images.

1. ```ChangeDirectory(curr,new)``` - Changes the directory that the os is in from curr to new
2. ```readImage(file,scale)``` - gets the image from storage path file in the scale required. 'scale' can be "gray" for grayscale or "color" for bgr
3. ```saveImage(file,image,scale)```- saves the image 'image' at location 'file' in the color 'scale'. 'scale' can be "gray" for grayscale or "color" for bgr.
4. ```PlotOne(image,scale)``` - plots one image 'image' in the color 'scale'. 'scale' can be "gray" for grayscale or "color" for bgr.
5. ```Plot(images, names, scale, rows, cols, length, breadth)``` - Plots multiple images. Employs matplotlib. Here,
    - ```images```: list of all the images to be plotted
    - ```names```: list of headings for each image
    - ```scale```: list of the colorscale for each image ("gray" or "color")
    - ```rows```: number of rows (default =10)
    - ```cols```: number of cols (default = 5)
    - ```length```: length of each image (default=20)
    - ```breadth```: breadth of each image(default=20)
6. ```Show(images,names)```: Creates a new window with the images. Shows each image in list 'images' in a different window with the 'names' having the list of their titles.

### <a href='utils/ImageProcessingMethodologies.py'> Methods </a>
This file contains all the image processing methods that can be used to transform an image and extract only the relevant features from it.

#### Data Augumentation Methods
These methods are used to make small changes to the images so as to create a larger dataset for better model performance
1. ```Flip(image, orientation)```- Flips the image 'image' either horizontally or vertically (depends on orientation ('horiz' or 'vert'))
2. ```RandRotate(image)```- rotates the 'image' by a randomly generated angle
3. ```RandCrop```- Crops the image by a random size
4. ```ColorJitter(image)```- Modifies the color of the 'image' by changing the HSV values.
5. ```GaussianNoise(image)```- Introduces gaussian noise into the 'image'
6. ```Rescale(image)```- rescales the 'image' to a different size
7. ```Translation(image)```- Shifts the 'image' in a random direction by a random amount
8. ```ElasticTransformations(image)```- Distorts the 'image' using random displacement fields
9. ```Cutout(image)```- randomly masks our rectangular regions in the 'image'
10. ```Mixup(image1, image2)```- combines two images by blending their pixels
11. ```Cutmix(image1,image2)```- randomly replaces a portion of 'image1' with a portiaion of 'image2'

#### Image Transformation Methods
1. ```Convert(img,curr,new)```- converts the image from current filter to new filter. converts from ('bgr','hsv','rgb') to ('bgr','hsv','gray','rgb')
2. ```Dilate(image, shape, kernel)```- dilation function
    - ```image```: image to be dilated
    - ```shape```: shape to be used for dilation (default=cv2.MORPH_ELLIPSE)
    - ```kernel```: iterations of dilation (default=10)
3. ```Erode(image, shape, kernel)```- erosion function
    - ```image```: image to be eroded
    - ```shape```: shape to be used for erosion (default=cv2.MORPH_ELLIPSE)
    - ```kernel```: iterations of erosion (default=10)
4. ```CannyEdgeDetection(img,threshold)```: Edge detection method. 'threshold' is a list with 2 values - the lower threshold and the higher threshold. Default is [100,300]
5. ```HistogramEqualization(image)```: performs histogram equalization on the image. (Converts to greyscale as well)
6. ```Binary(image)```: performs binary thresholding on the image
7. ```BinInverse(img)```: performs inverse binary thresholding on the image 'img'
8. ```Resize(img, resizepercent)```: resizes or rescales an image by a percentage 'resizepercent'. Default is 0.5
