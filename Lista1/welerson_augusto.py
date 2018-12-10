########################################
#
# Nome: WELERSON AUGUSTO LINO DE JESUS MELO
# Matricula: 201600017230
# E-mail: welerson.a.melo@gmail.com
#
# Nome: THIAGO JOSE SANDES MELO
# Matricula: 201600092557	
# E-mail: thiago_sandes@outlook.com
#
########################################


'''
Lista 1 para a disciplina Processamento de Imagens . . .
Periodo 2018-2
'''


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
		plt.imshow(img, cmap='gray', vmin = 0, vmax = 255)
	else:
		plt.imshow(img, vmin = 0, vmax = 255)
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
	out = r * (img - m) + m
	out = [[255 if x > 255 else x for x in arr] for arr in out]
	out = [[0 if x < 0 else x for x in arr] for arr in out]
	
	return np.asarray(out, np.uint8)
	
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

def histeq(img):
	#Imagem de entrada
	#imshow(img)
	
	###### REFAZER, ESTA SATURANDO A IMAGEM, VER O PQ DISSO
	
	data = img.copy().flatten()
	hist, bins = np.histogram(data, 256, density=True)
	cdf = hist.cumsum()
	cdf = 255*cdf/cdf[-1]
	img_eq = np.interp(data, bins[:-1], cdf)
	imgOut = np.asarray(img_eq.reshape(img.shape), np.uint8)
	#Imagem de saida
 	#imshow(imgOut)
	return imgOut
	
def convolve(img, kernel):
	# para imagens grays por enquanto, adaptar	e fazer map para os valores ficarem no range [0,255]
	Sx, Sy = size(img)
	
	a = len(kernel)
	b = len(kernel[0])
	a2 = int(a/2)
	b2 = int(b/2)
	outAux = np.zeros((Sy, Sx), np.float64)
	
	for i in range(Sy):
		for j in range(Sx):
			g = 0.0
			for s in range(a):
				for t in range(b):
					x = j+t-a2
					y = i+s-b2
					x = min(max(x, 0), Sx-1)
					y = min(max(y, 0), Sy-1)
					g += (kernel[s][t] * img[y][x])
			outAux[i][j] = g
	
	#Se for simplesmente saturar valores fora do range [0,255]
	#outAux = [[255 if x > 255 else x for x in arr] for arr in outAux]
	#outAux = [[0 if x < 0 else x for x in arr] for arr in outAux]
	#return np.asarray(outAux, np.uint8)
	
	#Se for fazer remapeamento para 0,255
	outAux = np.interp(outAux, (outAux.min(), outAux.max()), (0, 255))
	return outAux

def maskBlur():
	return [[1/16,2/16,1/16],[2/16,4/16,2/16],[1/16,2/16,1/16]]

def blur(img):
	kernel = maskBlur()
	return convolve(img, kernel)

def seSquare3():
	return np.asarray([[1,1,1],[1,1,1],[1,1,1]], np.uint8)

def seCross3():
	return np.asarray([[0,1,0],[1,1,1],[0,1,0]], np.uint8)

def erode(img, structure):
	img.setflags(write=1)
	out = img
	rows, cols, channels = img.shape
	rowSruct, colStruct = structure.shape
	auxArrayB = np.zeros(structure.shape)
	auxArrayG = np.zeros(structure.shape)
	auxArrayR= np.zeros(structure.shape)
	b, g, r    = out[:, :, 0], out[:, :, 1], out[:, :, 2]
	
	## Iterate over each cell
	for row in range(rows):
		for col in range(cols):
			try:
				auxArrayB[0][0] = b[row - 1 ][col - 1]
			except:
				auxArrayB[0][0] = auxArrayB[0][0] = b[np.clip((row - 1), row, rows)][np.clip((col - 1), col, cols)]
			try:
				auxArrayB[0][1] = b[row - 1 ][col]
			except:
				auxArrayB[0][1] = b[np.clip((row - 1), row, rows)][np.clip((col), col, cols)]
			try:
				auxArrayB[0][2] = b[row - 1 ][col + 1]
			except:
				auxArrayB[0][2] = b[np.clip((row - 1), row, rows)][np.clip((col + 1), col, cols)]
			try:
				auxArrayB[1][0] = b[row][col - 1]
			except:
				auxArrayB[1][0] = b[np.clip((row), row, rows)][np.clip((col - 1), col, cols)]
			try:
				auxArrayB[1][1] = b[row][col]
			except:
				auxArrayB[1][1] = b[np.clip((row), row, rows)][np.clip((col), col, cols)]
			try:
				auxArrayB[1][2] = b[row][col + 1]
			except:
				auxArrayB[1][2] = b[np.clip((row), row, rows)][np.clip((col + 1), col, cols)]
			try:
				auxArrayB[2][0] = b[row + 1][col - 1]
			except:
				auxArrayB[2][0] = b[np.clip((row + 1), row, rows)][np.clip((col - 1), col, cols)]
			try:
				auxArrayB[2][1] = b[row + 1][col]
			except:
				auxArrayB[2][1] = b[np.clip((row + 1), row, rows)][np.clip((col), col, cols)]
			try:
				auxArrayB[2][2] = b[row + 1][col + 1]
			except:
				auxArrayB[2][2] = b[np.clip((row + 1), row, rows)][np.clip((col + 1), col, cols)]

			try:
				auxArrayG[0][0] = g[row - 1 ][col - 1]
			except:
				auxArrayG[0][0] = g[np.clip((row - 1), row, rows)][np.clip((col - 1), col, cols)]
			try:
				auxArrayG[0][1] = g[row - 1 ][col]
			except:
				auxArrayG[0][1] = g[np.clip((row - 1), row, rows)][np.clip((col), col, cols)]
			try:
				auxArrayG[0][2] = g[row - 1 ][col + 1]
			except:
				auxArrayG[0][2] = g[np.clip((row - 1), row, rows)][np.clip((col + 1), col, cols)]
			try:
				auxArrayG[1][0] = g[row][col - 1]
			except:
				auxArrayG[1][0] = g[np.clip((row), row, rows)][np.clip((col - 1), 0, cols)]
			try:
				auxArrayG[1][1] = g[row][col]
			except:
				auxArrayG[1][1] = g[np.clip((row), row, rows)][np.clip((col), col, cols)]
			try:
				auxArrayG[1][2] = g[row][col + 1]
			except:
				auxArrayG[1][2] = g[np.clip((row), row, rows)][np.clip((col + 1), col, cols)]
			try:
				auxArrayG[2][0] = g[row + 1][col - 1]
			except:
				auxArrayG[2][0] = g[np.clip((row + 1), row, rows)][np.clip((col - 1), col, cols)]
			try:
				auxArrayG[2][1] = g[row + 1][col]
			except:
				auxArrayG[2][1] = g[np.clip((row + 1), row, rows)][np.clip((col), col, cols)]
			try:
				auxArrayG[2][2] = g[row + 1][col + 1]
			except:
				auxArrayG[2][2] = g[np.clip((row + 1), row, rows)][np.clip((col + 1), col, cols)]

			try:
				auxArrayR[0][0] = r[row - 1][col - 1]
			except:
				auxArrayR[0][0] = r[np.clip((row - 1), row, rows)][np.clip((col - 1), col, cols)]
			try:
				auxArrayR[0][1] = r[row - 1][col]
			except:
				auxArrayR[0][1] = r[np.clip((row - 1), row, rows)][np.clip((col), col, cols)]
			try:
				auxArrayR[0][2] = r[row - 1][col + 1]
			except:
				auxArrayR[0][2] = r[np.clip((row - 1), row, rows)][np.clip((col + 1), col, cols)]
			try:
				auxArrayR[1][0] = r[row][col - 1]
			except:
				auxArrayR[1][0] = r[np.clip((row), row, rows)][np.clip((col - 1), col, cols)]
			try:
				auxArrayR[1][1] = r[row][col]
			except:
				auxArrayR[1][1] = r[np.clip((row), row, rows)][np.clip((col), col, cols)]
			try:
				auxArrayR[1][2] = r[row][col + 1]
			except:
				auxArrayR[1][2] = r[np.clip((row), row, rows)][np.clip((col + 1), col, cols)]
			try:
				auxArrayR[2][0] = r[row + 1][col - 1]
			except:
				auxArrayR[2][0] = r[np.clip((row + 1), row, rows)][np.clip((col - 1), col, cols)]
			try:
				auxArrayR[2][1] = r[row + 1][col]
			except:
				auxArrayR[2][1] = r[np.clip((row + 1), row, rows)][np.clip((col), col, cols)]
			try:
				auxArrayR[2][2] = r[row + 1][col + 1] 
			except:
				auxArrayR[2][2] = r[np.clip((row + 1), row, rows)][np.clip((col + 1), col, cols)]

			listaAuxB = []
			listaAuxG = []
			listaAuxR = []
			for rowSt in range(rowSruct):
				for colSt in range(colStruct):
						if(structure[rowSt][colSt] == 1):
							listaAuxB.append(auxArrayB[rowSt][colSt])
							listaAuxG.append(auxArrayB[rowSt][colSt])
							listaAuxR.append(auxArrayR[rowSt][colSt])
							

			b[row][col] = min(listaAuxB)
			g[row][col] = min(listaAuxG)
			r[row][col] = min(listaAuxR)

	erosion = np.dstack((b,g,r))
	erosion = erosion.astype(np.uint8)
	return erosion

######### testes ############
#img1 = imread('in1.jpg')
#print (type(img1))

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
#img2 = imreadgray('in1.jpg')

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
#h = hist(img1)

# Q12 e Q13
#showhist(h)
#showhist(h, 2)

# Q14
#img1 = imread('in1.jpg')
#histeq(img1)

# Q15
# http://aishack.in/tutorials/image-convolution-examples/
#img = imread('cinza.jpg')
#kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]) # Filtro Edge Detection 
#convolve(img, kernel)

# Q16
#maskBlur()


#LEMBRAR: TESTAR SE ALGUMA FUNCAO ESTA MODIFICANDO A IMG DE ENTRADA
# LEMBRAR: ver se em alguma funcao os valores estao passando de 255 ou 0 e saturar
