# from the following tutorial: https: // realpython.com/python-opencv-color-spaces/

from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np

flags = [i for i in dir(cv2) if i.startswith('COLOR_')]


len(flags)  #shows how many flags are available

# 
'''
The first characters after COLOR_ indicate the origin color space,
and the characters after the 2 are the target color space. 
This flag represents a conversion from BGR (Blue, Green, Red) to RGB.
As you can see, the two color spaces are very similar,
with only the first and last channels swapped.

'''
flags[40]
# output: 'COLOR_BGR2HLS' 

nemo = cv2.imread(
    '/Users/hefler/Library/Mobile Documents/com~apple~CloudDocs/programming/ComputerVision/bla2.5577e4ec1f8e.jpg')

plt.imshow(nemo)
plt.show()

'''
the output image is not as the original, that is because OpenCV
by default reads images as BRG formart. use the .cvtColor(image, flag)
to set the format to RGB

'''

nemo = cv2.cvtColor(nemo, cv2.COLOR_BGR2RGB)

plt.imshow(nemo)
plt.show()

# Visualing Nemo in RGB Colour Space

# import these Matplotlib libraries:
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

'''
place each pixel in its location based 
on its components and colour it by its 
colour using OpenCV.split() method

'''
r, g, b = cv2.split(nemo)

# this sets up the plot graph
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection='3d')

'''

Now that you have set up the plot, you need to set up the pixel colors. 
In order to color each pixel according to its true color, there’s a bit 
of reshaping and normalization required. It looks messy, but essentially 
you need the colors corresponding to every pixel in the image to be 
flattened into a list and normalized, so that they can be passed to the 
facecolors parameter of Matplotlib scatter().

'''

'''
Normalizing just means condensing the range of colors from 0-255 to 0-1
as required for the facecolors parameter. Lastly, facecolors wants a list,
not an NumPy array:
'''
pixel_colors = nemo.reshape((np.shape(nemo)[0]*np.shape(nemo)[1], 3))
norm = colors.Normalize(vmin=-1., vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()

# Building the scatter plot:
axis.scatter(r.flatten(), g.flatten(), b.flatten(), \
    facecolors=pixel_colors, marker='.')

axis.set_xlabel('Red')
axis.set_ylabel('Green')
axis.set_zlabel('Blue')


plt.show()
