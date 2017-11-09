import math, numpy as np, scipy.misc, pdb

channels = 4 ## r, g, b, alpha
dim = 1024
radius = dim/2

img = np.zeros((dim, dim, channels))

for x in range(dim):
	for y in range(dim):
		x_ = float(x - radius) / radius
		y_ = float(y - radius) / radius
		# print x_, y_
		z_squared = 1 - (x_**2 + y_**2)
		if z_squared >= 0:
			alpha = 1
			r, g = x_, y_
			b = math.sqrt(z_squared)
			# r, g, b = 1, 1, 1
			# print math.sqrt(r**2 + g**2 + b**2)
		else:
			alpha = 0
			r, g, b = 0, 0, 0
		img[x][y] = (r,g,b,alpha)

# pdb.set_trace()
img[:,:,:channels-1] = img[:,:,:channels-1]/2. + .5
print img.max(0).max(0)
print img.min(0).min(0)
scipy.misc.imsave('normals_vector.png', img)

for ind in range(channels-1):
	copy = img.copy()
	for mask in range(channels-1):
		if mask != ind:
			copy[:,:,mask] = 0
	scipy.misc.imsave('mask_' + str(ind) + '.png', copy)

# pdb.set_trace()
norm = (img.copy() - .5)*2
summed = np.power(norm[:,:,:channels-1],2).sum(channels-2)
rep = np.tile(summed[:,:,np.newaxis], (1,1,channels-1))
norm[:,:,:channels-1] = rep
# pdb.set_trace()
scipy.misc.imsave('norm.png', norm)
