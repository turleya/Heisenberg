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
S_1z = np.tensordot(np.tensordot(sz, I, axes=0), I, axes=0)

S_2z = np.tensordot(np.tensordot(I, sz, axes=0), I, axes=0)

S_3z = np.tensordot(np.tensordot(I, I, axes=0), sz, axes=0)

"""
for i in range(0, particles):
	Test = np.tensordot(np.identity(particles-1), sz, axes=0)
"""

#Sx
S_1x = np.tensordot(np.tensordot(sx, I, axes=0), I, axes=0)

S_2x = np.tensordot(np.tensordot(I, sx, axes=0), I, axes=0)

S_3x = np.tensordot(np.tensordot(I, I, axes=0), sx, axes=0)

#Sy
S_1y = np.tensordot(np.tensordot(sy, I, axes=0), I, axes=0)

S_2y = np.tensordot(np.tensordot(I, sy, axes=0), I, axes=0)

S_3y = np.tensordot(np.tensordot(I, I, axes=0), sy, axes=0)


###Defining basis for 3 particles
#Returns all possible combinations of spin up and down for given number of particles

def particleInit(particles):
    a = [[0] for i in range(0,1<<particles)]
    for i in range(0, 1<<particles):
        arr = []
        for j in range(0, particles):
            if(i & (1<<j)):
                arr.append(Up)
            else:
                arr.append(Down)	         
            a[i] = arr
    return a
 
def main(particles):
    parts = particleInit(particles)
    for part in parts:
        part.reverse()
    
    parts.reverse()    
    #print(parts)
    
main(2)

#Hamiltonian
"""
H = S_1x*S_2x + S_1y*S_2y + S_1z*S_2z + S_2x*S_3x + S_2y*S_3y + S_2z*S_3z

print H

for j in range(
"""

A = S_1z*S_2z




B = S_2z*S_3z

print B
























