import scipy as sp
import numpy as np

#Pauli matrices
x = np.matrix([[0,1],[1,0]])
y = np.matrix([[0, -1j], [1j,0]])
z = np.matrix([[1,0], [0,-1]])

#Spin up and down
Up = np.matrix([[1],[0]])
Down = np.matrix([[0], [1]])

#Spin operators
sx = 1./2*x
sy = 1./2*y
sz = 1./2*z

#Identity matrix
I = np.identity(2)

#Tensor products
#Sz
sz1 = np.tensordot(sz, I, axes=0)
S_1z = np.tensordot(sz1, I, axes=0)

sz2 = np.tensordot(I, sz, axes=0)
S_2z = np.tensordot(sz2, I, axes=0)

sz3 = np.tensordot(I, I, axes=0)
S_3z = np.tensordot(sz3, sz, axes=0)

#Sx
sx1 = np.tensordot(sx, I, axes=0)
S_1x = np.tensordot(sx1, I, axes=0)

sx2 = np.tensordot(I, sx, axes=0)
S_2x = np.tensordot(sx2, I, axes=0)

sx3 = np.tensordot(I, I, axes=0)
S_3x = np.tensordot(sx3, sx, axes=0)

#Sy
sy1 = np.tensordot(sy, I, axes=0)
S_1y = np.tensordot(sy1, I, axes=0)

sy2 = np.tensordot(I, sy, axes=0)
S_2y = np.tensordot(sy2, I, axes=0)

sy3 = np.tensordot(I, I, axes=0)
S_3y = np.tensordot(sy3, sy, axes=0)

"""
def main(particles):
    l = [[0] for i in range(0, 2**particles)]
    print(l)
    #for i in range(0, particles):
    
main(3)
"""

#Defining basis for 3 particles
def main(particles):
	a = [[0] for i in range(0,1<<particles)]

	for i in range(0, 1, particles):
		arr = []
		for j in range(0, particles):
			if(i & (1<<j)):
				arr.append(Up)
			else:
				arr.append(Down)
		arr.reverse()
		a[i] = arr
	a.reverse()
	print(a)

main(3)

#Hamiltonian
"""
H = S_1x*S_2x + S_1y*S_2y + S_1z*S_2z + S_2x*S_3x + S_2y*S_3y + S_2z*S_3z

print H

for j in range(
"""





























