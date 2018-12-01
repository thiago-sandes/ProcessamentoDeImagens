import matplotlib 
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def imread(file):
    img = mpimg.imread(file)
    return np.asarray(img, np.uint8)

def nchannels(img):
	if len(img.shape) == 2:
		return 1
	return len(img.shape)

def size(img):
		x, y = img.shape[0], img.shape[1]
		return [y, x]

def rgb2gray(img):
	x, y = size(img)
	ans = np.zeros((y, x), np.uint8)
	i = -1
	for col in img:
		i+=1
		for j in range(len(col)):
			p = col[j]
			ans[i][j] = (p[0]*0.299 + p[1]*0.587 + p[2]*0.114)
			
	return ans

def imreadgray(file):
	img = imread(file)
	if nchannels(img) == 1:
		return img
	return rgb2gray(img)

def imshow(img):
	if nchannels(img) == 1:
		plt.imshow(img, cmap='gray')
	else:
		plt.imshow(img)
	plt.show()

def thresh(img, limiar):
	imgOut = []
	if nchannels(img) == 1:
		for col in img:
			imgOut.append([255 if p >= limiar else 0 for p in col])
	else:
		for col in img:
			line = []
			for p in col:
				if p[0] >= limiar[0]:
					a0 = 255
				else:
					a0 = 0
				if p[1] >= limiar[1]:
					a1 = 255
				else:
					a1 = 0
				if p[2] >= limiar[2]:
					a2 = 255
				else:
					a2 = 0
				line.append([a0, a1, a2,])
			imgOut.append(line)
	return np.asarray(imgOut, np.uint8)

def negative(img):
	return 255 - img;

def contrast(img, r, m):
	return r * (img - m) + m
	
def hist(img):
	if nchannels(img) == 1:
		h = [0 for _ in range(256)]
		for col in img:
			for p in col:
				h[p]+=1
		return h
	else:
		h = [[0 for _ in range(256)] for _ in range(3)]
		for col in img:
			for p in col:
				h[0][p[0]]+=1
				h[1][p[1]]+=1
				h[2][p[2]]+=1
		return h

def showhist(h, Bin = 1):
	if len(h) != 3:
		x = [sum(h[i : i+Bin])/Bin*1. for i in range(0, 256, Bin)]
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_xlim(0, len(x))	
		ax.stem([i for i in range(len(x))], x)
		plt.show()
	else:
		x = []
		x.append([sum(h[0][i : i+Bin])/Bin*1. for i in range(0, 256, Bin)])
		x.append([sum(h[1][i : i+Bin])/Bin*1. for i in range(0, 256, Bin)])
		x.append([sum(h[2][i : i+Bin])/Bin*1. for i in range(0, 256, Bin)])
		
		fig = plt.figure()
		ax1 = fig.add_subplot(111, projection = '3d')

		xpos = [i for i in range(len(x[0]))] + [i for i in range(len(x[0]))] + [i for i in range(len(x[0]))]
		ypos = [1 for _ in range(len(x[0]))] + [2 for _ in range(len(x[0]))] + [3 for _ in range(len(x[0]))]
		zpos = [0 for i in range(len(x[0])*3)]

		dx = np.ones(len(x[0])*3)
		dy = np.ones(len(x[0])*3)
		dz = x[0]+x[1]+x[2]

		ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color = 'red')
		plt.show()
	
######### testes ############
img1 = imread('in1.jpg')
print (type(img1))

'''
# Q2 printing image
plt.imshow(img1)
#plt.show()

# Q3 pegando numero de canais
nCh = nchannels(img1)
print (nCh)

# Q4 
print (size(img1))

# Q5
img1Gray = rgb2gray(img1)
'''
# Q6
img2 = imreadgray('in1.jpg')

# Q7 imshow()
#imshow(img1)
#imshow(img2)

# Q8 threshold
#imgT = thresh(img2, 200)
#imgT = thresh(img1, [100, 250, 250])
#imshow(imgT)

# Q9 
#imshow(negative(img1))

# Q10
#imshow(contrast(img2, 4, 20))

# Q11
h = hist(img1)

# Q12 e Q13
#showhist(h)
#showhist(h, 2)

# Q14


