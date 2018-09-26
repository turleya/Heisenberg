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

splus = np.matrix([[0,1],[0,0]])
sminus = np.matrix([[0,0],[1,0]])


#Identity matrix
I = np.identity(2)
ZeroMat = np.matrix([[0, 0], [0, 0]])

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

def createMatrix(sign, index, particles):
    if sign == "plus":
        sign = splus
    else:
        sign = sminus
    l = []
    mat = np.array(l, ndmin=particles)
    for i in range(0, particles):
        l.append(ZeroMat)
        np.append(mat, l)
    print(mat)
    
    
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
 
def normaliseMatrices(list):
    mats = []
    for element in list:
        l = []
        for x in range(0, len(element)):
            if element[x][0] == [1]:
                l.append([1])
                l.append([0])
            else:
                l.append([0])
                l.append([1])
        mat = np.matrix(l)
        mats.append(mat)
    return mats
    
#Defining basis for 3 particles
def main(particles):
    parts = particleInit(particles)
    for part in parts:
        part.reverse()
    parts.reverse()    
    
main(2)
print("-------")
main(3)
#print(splus)
#print(sminus)

createMatrix("plus", 1, 2)


'''
def _init_(particles,J_xy,J_zz,counts):
     #length of matrix
    self.J_xy = J_xy #strength of flip-flop term
    self.J_zz = J_zz #strength of Ising interaction
    self.counts = counts #number of times the code will sweep through states
    parts = particleInit(particles)   
    
    
    #Forming matrix containing each part of basis    
    self.B = np.vstack(part in parts.reshape((3,1<<particles)))
    print self.B
    
    
    l = 0.0
    E = 0.0
    E_sq = 0.0
    
    for l in range(self.counts):
        l += 1
        
        for i in range(0,particles):
            for j in range(0,1<<particles):
                s = self.B[i,j]
                #trying to define all operators to act on basis B
                Z_i*Up = (1./2)*Up     
                Z_i*Down = (1./2)*Down
                Plus_i*Up = 0
                Plus_i*Down = (1./2)*Up
                Minus_i*Up = (1./2)*Down
                Minus_i*Down = 0
                
                #Hamiltonian                 
                H = J_zz*Z_i*Z_(i+1)*self.B + J_xy*Plus_i*Minus_(i+1)*self.B + J_xy*Minus_i*Plus_(i+1)*self.B
                #How to turn this into the Hamiltonian matrix?
                
        return H
        print H
   '''             
"""
#Hamiltonian

H = S_1x*S_2x + S_1y*S_2y + S_1z*S_2z + S_2x*S_3x + S_2y*S_3y + S_2z*S_3z

print H

for j in range(
"""