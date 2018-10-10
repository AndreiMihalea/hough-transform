import numpy as np
from numba import stencil, jit
from PIL import Image
import sys
import os
import time
import scipy.misc
from scipy.ndimage import filters
import cv2
from math import *

@jit()
def grayscale(a):
	return np.dot(a[...,:3], [0.299, 0.587, 0.114])

@jit()
def gaussian_blur(a):
	ashape = a.shape
	res = np.copy(a)
	for i in range(2,ashape[0]-2):
		for j in range(2,ashape[1]-2):
			res[i,j] = (a[i-2,j-2] * 0.003  + a[i-1,j-2] * 0.0133 + a[i,j-2] * 0.0219 + a[i+1,j-2] * 0.0133 + a[i+2,j-2] * 0.0030 +
						a[i-2,j-1] * 0.0133 + a[i-1,j-1] * 0.0596 + a[i,j-1] * 0.0983 + a[i+1,j-1] * 0.0596 + a[i+2,j-1] * 0.0133 +
						a[i-2,j+0] * 0.0219 + a[i-1,j+0] * 0.0983 + a[i,j+0] * 0.1621 + a[i+1,j+0] * 0.0983 + a[i+2,j+0] * 0.0219 +
						a[i-2,j+1] * 0.0133 + a[i-1,j+1] * 0.0596 + a[i,j+1] * 0.0983 + a[i+1,j+1] * 0.0596 + a[i+2,j+1] * 0.0133 +
						a[i-2,j+2] * 0.003  + a[i-1,j+2] * 0.0133 + a[i,j+2] * 0.0219 + a[i+1,j+2] * 0.0133 + a[i+2,j+2] * 0.0030)
	return res

@jit()
def step4(i, j, T1, T2):
	dirs = [-1, 0, 1]
	for k in dirs:
		ok = True
		for l in dirs:
			if k != 0 or l != 0:
				if T2[i + k][j + l] == 1:
					ok = False
					break
		if ok == False:
			break
		for l in dirs:
			if k != 0 or l != 0:
				if T2[i + k][j + l] == 0 and T1[i + k][j + l] == 1:
					T2[i + k][j + l] = 1
					step4(i + k, j + l, T1, T2)

@jit()
def complete(T2):
    width, height = T2.shape
    cT2 = np.copy(T2)
    for x in range(1, width - 1):
        for y in range(1, height - 1):
        	if cT2[x][y] == 0:
        		if cT2[x][y - 1] and cT2[x][y + 1]:
        			T2[x][y] = 1

@jit()
def vote(T2, A, radii):
	(width, height) = T2.shape
	for x in range(width):
		for y in range(height):
			if T2[x][y]:
				for r in range(len(radii)):
					for t in range(0, 360):
						a = x - radii[r] * cos(t * np.pi / 180)
						b = y - radii[r] * sin(t * np.pi / 180)
						if a >= 0 and a < width and b >= 0 and b < height:
							A[r][round(a)][round(b)] += 1

@jit()
def area(x, y, img):
	pass

@jit()
def flood(targetc, replacementc, x, y, img):
	width, height, _ = img.shape
	if x < 0 or x >= width or y < 0 or y >= height:
		return
	if img[x][y][0] == img[x][y][1] == img[x][y][2] == 255:
		img[x][y] = replacementc
	if not (img[x][y][0] == targetc[0] and img[x][y][1] == targetc[1] and img[x][y][2] == targetc[2]):
		return
	img[x][y] = replacementc
	flood(targetc, replacementc, x - 1, y, img)
	flood(targetc, replacementc, x + 1, y, img)
	flood(targetc, replacementc, x, y - 1, img)
	flood(targetc, replacementc, x, y + 1, img)

def main(*args):
	input_file = "brad.png"

	img = scipy.misc.imread(input_file, flatten = True)
	img = img.astype('int32')

	width = img.shape[0]
	height = img.shape[1]

	filtered = gaussian_blur(img)
	scipy.misc.imsave('filtered.png', filtered)

	sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32)
	sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32)

	gx = np.empty((width, height), dtype = np.int32)
	gy = np.empty((width, height), dtype = np.int32)

	gx = filters.convolve(filtered, sx)
	gy = filters.convolve(filtered, sy)

	Gm = np.hypot(gx, gy)
	Gd = np.arctan2(gy, gx)
	Gd = np.rad2deg(Gd) % 180

	scipy.misc.imsave('magnitudes.png', Gm)

	for x in range(width):
	    for y in range(height):
	    	if 0 <= Gd[x][y] < 22.5 or 157.5 <= Gd[x][y] < 180:
	    		Gd[x][y] = 0
	    	elif 22.5 <= Gd[x][y] < 67.5:
	    		Gd[x][y] = 45
	    	elif 67.5 <= Gd[x][y] < 112.5:
	    		Gd[x][y] = 90
	    	elif 112.5 <= Gd[x][y] < 157.5:
	    		Gd[x][y] = 135

	Gdc = np.empty([width, height, 3])
	for x in range(width):
	    for y in range(height):
	    	if Gd[x][y] == 0:
	    		Gdc[x][y] = np.array([255., 0., 0.]) * Gm[x][y] / 255.
	    	if Gd[x][y] == 45:
	    		Gdc[x][y] = np.array([0., 255., 0.]) * Gm[x][y] / 255.
	    	if Gd[x][y] == 90:
	    		Gdc[x][y] = np.array([0., 0., 255.]) * Gm[x][y] / 255.
	    	if Gd[x][y] == 135:
	    		Gdc[x][y] = np.array([255., 0., 255.]) * Gm[x][y] / 255.
	    	if Gm[x][y] < 2:
	    		Gdc[x][y] = [0, 0, 0]

	scipy.misc.imsave('directions.png', Gdc)

	nm = np.zeros([width, height], dtype = np.int32)

	for x in range(1, width - 1):
		for y in range(1, height - 1):
			if Gd[x][y]==0:
				if (Gm[x][y] >= Gm[x][y+1]) and (Gm[x][y] >= Gm[x][y-1]):
					nm[x][y] = Gm[x][y]
			elif Gd[x][y] == 45:
				if (Gm[x][y] >= Gm[x-1][y+1]) and (Gm[x][y] >= Gm[x+1][y-1]):
					nm[x][y] = Gm[x][y]
			elif Gd[x][y] == 90:
				if (Gm[x][y] >= Gm[x+1][y]) and (Gm[x][y] >= Gm[x-1][y]):
					nm[x][y] = Gm[x][y]
			elif Gd[x][y] == 135:
				if (Gm[x][y] >= Gm[x+1][y+1]) and (Gm[x][y] >= Gm[x-1][y-1]):
					nm[x][y] = Gm[x][y]

	scipy.misc.imsave('nmsuppress.png', nm)

	P1 = 1
	P2 = 3

	T1 = np.zeros([width, height], np.int32)
	T2 = np.zeros([width, height], np.int32)

	for i in range(width):
		for j in range(height):
			if nm[i][j] > P1:
				T1[i][j] = 1
			if nm[i][j] > P2:
				T2[i][j] = 1

	T1 = T1 - T2

	for i in range(1, width - 1):
		for j in range(1, height - 1):
			if T2[i][j]:
				T2[i][j] = 1
				step4(i, j, T1, T2)

	#complete(T2)

	scipy.misc.imsave('canny.png', T2)

	radii = [i for i in range(20, 60)]
	A = np.zeros([len(radii), width, height], dtype = np.int32)

	vote(T2, A, radii)

	circles = []

	for r in range(len(radii)):
		A_max = np.amax(A[r])
		if A_max > 170:
			for x in range(width):
				for y in range(height):
					if A[r][x][y] < 180:
						A[r][x][y] = 0
			for x in range(width):
				for y in range(height):
					if A[r][x][y] == A_max:
						circles.append((radii[r] + 1, x, y))

	hough = final = np.empty([width, height, 3], dtype = np.int32)

	for x in range(width):
		for y in range(height):
			if T2[x][y] == 1:
				hough[x][y] = [255, 255, 255]
				for c in circles:
					r, cx, cy = c[0], c[1], c[2]
					dist = (x - cx) * (x - cx) + (y - cy) * (y - cy)
					if dist >= r * r - 625 and dist <= r * r + 625:
						hough[x][y] = [255, 0, 0]

	scipy.misc.imsave('hough.png', hough)

	final = np.copy(hough)
		
	for c in circles:
		for x in range(c[1] - c[0], c[1] + 1):
			for y in range(c[2] - c[0], c[2] + 1):
				'''
				if (x - c[1]) * (x - c[1]) + (y - c[2]) * (y - c[2]) <= c[0] * c[0]:
					x2 = 2 * c[1] - x
					y2 = 2 * c[2] - y
					final[x][y] = [255, 0, 0]
					final[x][y2] = [255, 0, 0]
					final[x2][y] = [255, 0, 0]
					final[x2][y2] = [255, 0, 0]
				'''
				flood([0, 0, 0], [255, 0, 0], c[1], c[2], final)

	img = scipy.misc.imread(input_file)
	img = img.astype('int32')

	nr = 0
	flood([0, 0, 0], [img[515][292][0], img[515][292][1], img[515][292][2]], 515, 292, final)

	scipy.misc.imsave('final.png', final)

	'''
	img = cv2.imread('filtered.png',0)
	edges = cv2.Canny(img,0.1,1)
	scipy.misc.imsave('edges.png', edges)
	'''

if __name__ == "__main__":
    main(*sys.argv[1:])
