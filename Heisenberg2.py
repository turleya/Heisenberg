import numpy as np
from numpy import linalg as la

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
I = np.identity(2)

#Defining basis of all possible combinations of spins for n particles
def particleInit(particles): # not necessary
    for i in range(0,1<<particles):
	B = np.arange(1<<particles)
	B = B.reshape((1<<particles,1))
	C = np.zeros_like(B)
	C_i = C
	C_i[1-i,0] = 1
	#print(C_i)

particleInit(3)

#Spin operators
#Sz Operator
def spinoperatorz(particles,index): 
    if particles == index: 
        P_i = sz
    else:
        P_i = I
    for i in range(1, particles): 
        if (particles-i) == index: 
            P_i = np.bmat([[sz[0,0]*P_i, sz[0,1]*P_i], [sz[1,0]*P_i, sz[1,1]*P_i]]) 
        else:
            P_i = np.bmat([[I[0,0]*P_i, I[0,1]*P_i], [I[1,0]*P_i, I[1,1]*P_i]]) 
    return(P_i)

spinoperatorz(3,1) 
  
#Splus operator
def spinoperatorplus(particles,index): 
    if particles == index: 
        P_i = splus
    else:
        P_i = I
    for i in range(1, particles): 
        if (particles-i) == index:  
            P_i = np.bmat([[splus[0,0]*P_i, splus[0,1]*P_i], [splus[1,0]*P_i, splus[1,1]*P_i]]) 
        else:
            P_i = np.bmat([[I[0,0]*P_i, I[0,1]*P_i], [I[1,0]*P_i, I[1,1]*P_i]]) 
    return(P_i)

spinoperatorplus(3,3) 

#Sminus operator
def spinoperatorminus(particles,index): 
    if particles == index: 
        P_i = sminus
    else:
        P_i = I
    for i in range(1, particles): 
        if (particles-i) == index:  
            P_i = np.bmat([[sminus[0,0]*P_i, sminus[0,1]*P_i], [sminus[1,0]*P_i, sminus[1,1]*P_i]]) 
        else:
            P_i = np.bmat([[I[0,0]*P_i, I[0,1]*P_i], [I[1,0]*P_i, I[1,1]*P_i]]) 
    return(P_i)

spinoperatorminus(3,3) 


J_ij = 1.0
H_i = 0.5*J_ij*(spinoperatorplus(3,1)*spinoperatorminus(3,2) + spinoperatorplus(3,2)*spinoperatorminus(3,1)) + J_ij*(spinoperatorz(3,1)*spinoperatorz(3,2)) + 0.5*J_ij*(spinoperatorplus(3,3)*spinoperatorminus(3,2) + spinoperatorplus(3,2)*spinoperatorminus(3,3)) + J_ij*(spinoperatorz(3,3)*spinoperatorz(3,2))+ 0.5*J_ij*(spinoperatorplus(3,1)*spinoperatorminus(3,3) + spinoperatorplus(3,3)*spinoperatorminus(3,1)) + J_ij*(spinoperatorz(3,1)*spinoperatorz(3,3))
print(H_i)

w, v = la.eig(H_i)
print w
print v

"""
#Heisenberg Hamiltonian 
def Hamiltonian(particles):
    H = np.zeros([2**particles,2**particles])
    J_ij = 1.0 #anistropy parameter
    B_i = 0.0 #local magnetic field
    for i in range(particles-1):	
	H_i = 0.5*J_ij*(spinoperatorplus(particles,i+1)*spinoperatorminus(particles,i+2) + spinoperatorplus(particles,i+2)*spinoperatorminus(particles,i+1)) + J_ij*(spinoperatorz(particles,i+1)*spinoperatorz(particles,i+2)) + B_i*spinoperatorz(particles, i+1) 
	H = H + H_i
    print(H)

Ham = Hamiltonian(3)

#Computing eigenvalues of Hamiltonian
def eigen(particles):
    w, v = la.eig(Ham)
    for i in range(0, 2**particles):
	w_i = w[i] 
	v_i = v[i]
	print(round(w_i,2)) #eigenvalues
	print(np.around(v_i, decimals=2)) #eigenvectors

eigen(3)   


 + 0.5*J_ij*(spinoperatorplus(particles,1)*spinoperatorminus(particles,particles) + spinoperatorplus(particles,particles)*spinoperatorminus(particles,1)) + J_ij*(spinoperatorz(particles,1)*spinoperatorz(particles,particles))

+ 0.5*J_ij*(spinoperatorplus(3,1)*spinoperatorminus(3,3) + spinoperatorplus(3,3)*spinoperatorminus(3,1)) + J_ij*(spinoperatorz(3,1)*spinoperatorz(3,3))
"""










