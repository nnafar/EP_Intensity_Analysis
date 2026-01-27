# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 17:29:55 2025

@author: nnafar
"""

# importing packages
import numpy as np
import matplotlib.pyplot as plt
import cv2
from cv2.ximgproc import guidedFilter
import os 
import glob
from natsort import natsorted
import scipy.optimize as sc


# Defining frame rate
fps = 19.15 # Frame rate of the Andor Camera (fastest setting for time series to see anything really)

# Select path for the data folder 
path = 'C:/Users/matya/Documents/Thesis code/Intensity Project/Data/W3_blank/W3*.tif'

# Grabbing paths to all files in the given directory 
files = glob.glob(path) # this just grabbed all of the file paths and sorted them as if computer was sorting it (1, 10, 100, 1000, 1001 .... )

# Sorting file paths by order of ascending integres (X1, X2, X3, ... , X1999)
sorted_files=natsorted(files) # this sorts files the same way that a human would


# Initialize variables for freehand drawing
drawing = False
points = []

# Declaring color globally for convenience 
global color
color = 255,255,255

# Mouse callback function, DONT TOUCH THIS, IF IT WORKS IT WORKS
def draw_freehand(event, x, y, flags, param):
    
    global drawing, points, img_copy, mask
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points = [(x, y)]
        cv2.circle(img_copy, (x, y), 1, color, -1)
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x, y))
            cv2.line(img_copy, points[-2], points[-1], color, 2)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        points.append((x, y))
        cv2.line(img_copy, points[-1], points[0], color, 2)
        create_mask(points, img.shape[:2])
        points = []

# Function to create and display the mask
def create_mask(pts, shape):
    global mask
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.array([pts], dtype=np.int32)
    cv2.fillPoly(mask, pts, 1)
    cv2.imshow("Mask", mask * 255)  # Display the mask for visualization
    
###########################
# Creating storage arrays #
###########################

# Creating an empty list to which the video frames will be subsequently appended
im_stack0=[]

# Appending images from the directory to te im_stack0 list 
for i in range(2000):
    img_load = cv2.imread(sorted_files[i],cv2.IMREAD_ANYDEPTH) # this is the part of it that takes the fucking longest
    im_stack0.append(img_load)
    
# Converting the list into numpy array
im_stack = np.asarray(im_stack0)


# Load the last image of the video for drawing of the mask 
image_path = sorted_files[-1]

#
img = cv2.imread(image_path,cv2.IMREAD_UNCHANGED) # Important to include the cv2.IMREAD_UNCHANGED flag

# Check if the image was loaded successfully
if img is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")

# Normalize the image to the 8-bit range for display
normalized_image = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

# Convert the normalized image to 8-bit
normalized_image = np.uint8(normalized_image)

# Apply a colormap (e.g., COLORMAP_JET)
colored_image = cv2.applyColorMap(normalized_image, cv2.COLORMAP_JET)

######### DO NOT TOUCH ########
img_copy = colored_image.copy()
mask = None
######### DO NOT TOUCH ########


# Create a window and set the mouse callback
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', draw_freehand)

# Display the image and wait for user interaction
while True:
    cv2.imshow('Image', img_copy)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press 'ESC' to exit
        break
cv2.destroyAllWindows()

# Function to apply guided filter
def apply_guided_filter(img, mask, radius=2, eps=1e-6):
    guided_filter = cv2.ximgproc.guidedFilter(guide=img, src=mask, radius=radius, eps=eps) # this is actually very smart filtering thingy, it uses the original image to enhance the mask
    return guided_filter

# Apply the guided filter to the mask
mask_guided = apply_guided_filter(img, mask.astype(np.float32))
cv2.imshow("Guided Mask", mask_guided)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Multiplication of the image stack by the mask of region of interest
ROI_stack = np.multiply(im_stack,mask_guided)

# Summation of intensities along axes 1 and 2 (y,x) while z dimension remains
intensity_total_stack = np.sum(ROI_stack,axis=(1,2)) 

 # The mean intensity is obtained through division of total intensity by the total amount of non_zero pixels in the guided mask 
intensity_mean_stack = intensity_total_stack/np.count_nonzero(mask_guided)
# Jump identification sensitivity parameter
s = 3

# Identification of the fluorescence intensity increase time point
for i in range(len(intensity_mean_stack)):
    if intensity_mean_stack[i]>np.average(intensity_mean_stack[i-10:i])+s: # i indexes the first frame containing intensity increase
        background = np.array([intensity_mean_stack[i-1]]) # background frame
        b = i-1 # index of te background frame
        break

# Subtracting background intensity from the image stack
intensity_mean_curve=intensity_mean_stack[b:]-background
t = np.linspace(0,len(intensity_mean_curve)/fps,len(intensity_mean_curve),endpoint=True) # Creates time array matching the length of the intensity curve

# Saving of the time stamp when experiment begins
exp_start = t[b]

# Definition of the model for non-linear data fit 
def dyn_model(t,A,b):
    return A*(1-np.exp(-t/b))

# Manual declaration of the endpoint of data that is going to be fit to the model
time_stamp = 750

# Initial guess
initial_guess = [1000,10]

# Curve fitting 
params,covariance = sc.curve_fit(dyn_model,t[:time_stamp],intensity_mean_curve[:time_stamp],p0=initial_guess)

# Saving of extracted parameters into individual variables
A = params[0] # Intensity fitted amplitude I_max
tau = params[1] # time constant of pore resealing

# Plotting of data in comparison with a cruve produced by the model fit 
fig,ax = plt.subplots(nrows=1,ncols=1)
ax.plot(t[:time_stamp],dyn_model(t[:time_stamp],A,tau))
ax.plot(t[:time_stamp],intensity_mean_curve[:time_stamp])

plt.show()
