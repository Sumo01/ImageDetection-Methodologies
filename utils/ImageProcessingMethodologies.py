import cv2
import numpy as np

#### Data Augumentation Methodologies ####
# The functions given below are methods that can be used to increase the size of a dataset by applying transformations to existing images.

#Function to change the orientation of an image
def Flip(image, orientation):
    if(orientation=="horiz"):
        flipped_image = cv2.flip(image, 1)
    else:
        flipped_image = cv2.flip(image, 0)
    return flipped_image


#Function to rotate an image by a random degree
def RandRotate(image):
    rows, cols, _ = image.shape
    rotation_angle = np.random.randint(-30, 30)
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    return rotated_image


#Function to crop an image by a random size
def RandCrop(image):
    rows, cols, _ = image.shape
    crop_size = np.random.randint(100, min(rows, cols) // 2)
    x = np.random.randint(0, cols - crop_size)
    y = np.random.randint(0, rows - crop_size)
    cropped_image = image[y:y+crop_size, x:x+crop_size]
    return cropped_image


# Function to modify the color of an image by applying random brightness, contrast, saturation, or hue changes 
def ColorJitter(image):
    brightness = np.random.randint(-50, 50)
    contrast = np.random.uniform(0.5, 1.5)
    saturation = np.random.uniform(0.5, 1.5)
    hue = np.random.randint(-10, 10)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 2] = cv2.add(cv2.multiply(hsv_image[:, :, 2], contrast), brightness)
    hsv_image[:, :, 1] = cv2.multiply(hsv_image[:, :, 1], saturation)
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue) % 180
    jittered_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return jittered_image


# Function to introduce Gaussian Noise into the image
def GaussianNoise(image):
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image


# Function to rescale an image to different sizes or randomly resize it.
def Rescale(image):
    scale_factor = np.random.uniform(0.7, 1.3)
    resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
    return resized_image


# Function to shift an image in different random directions
def Translation(image):
    rows, cols, _ = image.shape
    x_translation = np.random.randint(-50, 50)
    y_translation = np.random.randint(-50, 50)
    translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
    translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
    return translated_image


# Function to distorts images using random displacement fields
def ElasticTransformations(image):
    rows, cols, _ = image.shape
    alpha = np.random.randint(30, 70)
    sigma = np.random.uniform(6, 9)
    random_state = np.random.RandomState(None)
    dx = cv2.GaussianBlur((random_state.rand(rows, cols) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((random_state.rand(rows, cols) * 2 - 1), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    distorted_indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    
    # Resize distorted indices
    resized_indices = cv2.resize(distorted_indices[1], (cols, rows)).astype(np.float32), \
                      cv2.resize(distorted_indices[0], (cols, rows)).astype(np.float32)

    elastic_transformed_image = cv2.remap(image, resized_indices[0], resized_indices[1],
                                          interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return elastic_transformed_image


# Function to randomly mask out rectangular regions from an image. 
def Cutout(image):
    rows, cols, _ = image.shape
    cutout_image = image.copy()
    cutout_size = np.random.randint(10, 30)
    x = np.random.randint(0, cols - cutout_size)
    y = np.random.randint(0, rows - cutout_size)
    cutout_image[y:y+cutout_size, x:x+cutout_size] = 0
    return cutout_image


# Function to combine pairs of images by blending their pixels and corresponding labels, 
def Mixup(image1,image2):
    alpha = np.random.uniform(0.3, 0.7)
    mixed_image = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)
    return mixed_image


# Function to randomly replace a portion of one image with a portion from another image
def Cutmix(image1,image2):
    rows, cols, _ = image1.shape
    x = np.random.randint(0, cols)
    y = np.random.randint(0, rows)
    crop_size = np.random.randint(100, min(rows, cols) // 2)
    image2[y:y+crop_size, x:x+crop_size] = image1[y:y+crop_size, x:x+crop_size]
    cutmix_image = cv2.resize(image2, (cols, rows))
    return cutmix_image


#############################################################################################
################################ Image Transformation Methods ###############################

#Function to convert an image from current filter to new filter 
#Returns new image
#Can add more conversions if req.
def Convert(img,curr,new):
    if(curr=="bgr"):
        if(new=="gray"):
            new_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        elif(new=="rgb"):
            new_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        elif(new=="hsv"):
            new_img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    elif(curr=="rgb"):
        if(new=="gray"):
            new_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        elif(new=="bgr"):
            new_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        elif(new=="hsv"):
            new_img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    elif(curr=="hsv"):
        if(new=="gray"):
            new_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        elif(new=="rgb"):
            new_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        elif(new=="bgr"):
            new_img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    return new_img

def Dilate(image,shape=cv2.MORPH_ELLIPSE,kernel=10):
    #Dilates the objects based on the shape - usually only black and white images are used
    element = cv2.getStructuringElement(shape, (2 * kernel + 1, 2 * kernel + 1),(kernel, kernel))
    dilated_image = cv2.dilate(image, element)
    return dilated_image

def Erode(image,shape=cv2.MORPH_ELLIPSE,kernel=10):
    #Erodes the image - usually only black and white images used
    element = cv2.getStructuringElement(shape, (2 * kernel + 1, 2 * kernel + 1),(kernel, kernel))
    eroded_image = cv2.erode(image, element)
    return eroded_image

def FloodFill(img):
    #Flood filling finds the external edges and fills in huge objects
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.fillPoly(img,pts=contours,color=(255,255,255))
    return img

def HoleFill(img):
    #Filling in small holes 
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.fillPoly(img,pts=contours,color=(255,255,255))
    return img

def CannyEdgeDetection(img,threshold=[100,300]):
    #Usually the img will be the result of histogram equalization and binary thresholding
    edges=cv2.Canny(img,threshold[0],threshold[1])
    return edges

def HistogramEqualization(image):
    #Image can be used without any further processing
    #Converting to gryscale
    img=Convert(image, 'bgr', 'gray')
    
    #Finding the histogram and bins of the flattened image
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    
    #Getting the cumulative distributive function
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    
    #Converting to image form
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    equ = cv2.equalizeHist(img)
    return equ

def Binary(img):
    #Binary thresholding - usually done for images whcih are histogram equalized
    ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    return thresh

def BinInverse(img):
    #Inverse Binary Thresholding - images are usually first histogram equalized
    ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    return thresh

def findContours(img):
    #Finding the contours of the image
    ret, thresh = cv2.threshold(img, 1, 255, 0)
    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def drawContours(img,contours):
    #Drawing the contours on the image
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    return img

def drawContoursNew(img,contours):
    #Drawing the contours on a new blank image
    contour_image = np.zeros_like(img)
    # Draw contours on the blank image
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    return contour_image

def contourRange(img):
    #Finding contours whose radii are in a certain range. Also finds the centers of the contours.
    contours=findContours(img)
    centers = []
    radii = []
    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        
        # Check if the radius is within the desired range
        if 80 <= radius <= 200:
            center = (int(x), int(y))
            radius = int(radius)
            print('Contour: centre {},{}, radius {}'.format(x, y, radius))
            img=drawContours(img,[cnt])
            centers.append(center)
            radii.append(radius)
    # Save the image with the drawn contours
    return centers,radii,img

#Method to resize images
def Resize(img, resizepercent=0.5):
    return cv2.resize(img,None,fx=resizepercent, fy= resizepercent)
