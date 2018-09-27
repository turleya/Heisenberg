import numpy as np

#Pauli matrices
x = np.array([[0,1],[1,0]])
y = np.array([[0, -1j], [1j,0]])
z = np.array([[1,0], [0,-1]])

#Spin up and down
Up = np.array([[1],[0]])
Down = np.array([[0], [1]])

#Spin operators
sx = 1./2*x
sy = 1./2*y
sz = 1./2*z

splus = np.array([[0,1],[0,0]])
sminus = np.array([[0,0],[1,0]])


#Identity matrix
I = np.identity(2)
ZeroMat = np.array([[0, 0], [0, 0]])

"""
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
"""


#Defining basis for 3 particles    
def particleInit(particles):
    a = [[0] for i in range(0,1<<particles)]
    for i in range(0, 1<<particles):
        arr = []
        for j in range(0, particles):
            if(i & (1<<j)):
                arr.append(1)
            else:
                arr.append(0)	         
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
        mat = np.array(l)
        mats.append(mat)
    return mats

    
#Puts spins in order: up up up first, down down down last for ease
def main(particles):
    parts = particleInit(particles)
    for part in parts:
        part.reverse()
    parts.reverse() 
    #print(parts)
   
#prints all posssible combination of spins for x number of particles  
main(2)
print("-------")


#Spin operators
#Sz_i operators
def spinoperatorsz(particles):
    for i in range(0, particles):
	W = np.identity(2*particles)
	W_i = W
	W_i[2*i:2+2*i, 2*i:2+2*i] = sz
	#print(W_i)

spinoperatorsz(2)

#Splus_i operators
def spinoperatorsplus(particles):
    for i in range(0, particles):
	W = np.identity(2*particles)
	W_i = W
	W_i[2*i:2+2*i, 2*i:2+2*i] = splus
	#print(W_i)

spinoperatorsplus(2)

#Sminus_i operators
def spinoperatorsminus(particles):

    for i in range(0, particles):
	W = np.identity(2*particles)
	W_i = W
	W_i[2*i:2+2*i, 2*i:2+2*i] = sminus
	#print(W_i)

spinoperatorsminus(2)



#def spinoperator(inp,spin,outp):





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

