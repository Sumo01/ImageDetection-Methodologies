# ImageDetection-Methodologies

This repo has all the image detection methodologies that are commonly used when performing analysis on image datasets and while performing computer vision tasks. It mainly employs the open-cv library in python. Explanation for each file in this repo is given below:

## Methodologies:
These are implementations of common methodologies along with examples for simple understanding. 
1. <a href='CannyEdgeDetection.ipynb'> >Canny Edge Detection</a>
2. DilationandErosion.ipynb
3. FourierTransform.ipynb
4. HistogramEqualization.ipynb
5. HoughTransform.ipynb

## Modules:
These files have all the codes for the different methodologies for easy importing

### ImageIO.py
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

