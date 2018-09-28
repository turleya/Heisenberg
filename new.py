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


Iden = np.identity(8)
print Iden

#Creating basis
def particleInit(particles):
    for i in range(0,1<<particles):
	B = np.arange(1<<particles)
	B = B.reshape((1<<particles,1))
	C = np.zeros_like(B)
	C_i = C
	C_i[1-i,0] = 1
	#print(C_i)

particleInit(3)

#Creating spin operator matrices
def spinoperatorsz(particles):
    for i in range(0, particles):
	P = np.tensordot(I, I, axes=0)
	P_i = P
	P_i = np.tensordot(I, P, axes=0)
	#print(P_i)

spinoperatorsz(1)
	 
	
	
	

