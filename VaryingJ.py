import numpy as np
from numpy import linalg as la

#Spin operators
sx = np.array([[0,1./2],[1./2,0]])
sy = np.array([[0, 1./2j], [-1./2j,0]]) #?????
sz = np.array([[1./2,0], [0,-1./2]])

splus = np.array([[0,1],[0,0]])
sminus = np.array([[0,0],[1,0]])

#Spin up and down
Up = np.array([[1],[0]])
Down = np.array([[0], [1]])

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

#Sx operator
def spinoperatorx(particles,index): 
    if particles == index: 
        P_i = sx
    else:
        P_i = I
    for i in range(1, particles): 
        if (particles-i) == index:  
            P_i = np.bmat([[sx[0,0]*P_i, sx[0,1]*P_i], [sx[1,0]*P_i, sx[1,1]*P_i]]) 
        else:
            P_i = np.bmat([[I[0,0]*P_i, I[0,1]*P_i], [I[1,0]*P_i, I[1,1]*P_i]]) 
    return(P_i)

#Sy operator
def spinoperatory(particles,index): 
    if particles == index: 
        P_i = sy
    else:
        P_i = I
    for i in range(1, particles): 
        if (particles-i) == index:  
            P_i = np.bmat([[sy[0,0]*P_i, sy[0,1]*P_i], [sy[1,0]*P_i, sy[1,1]*P_i]]) 
        else:
            P_i = np.bmat([[I[0,0]*P_i, I[0,1]*P_i], [I[1,0]*P_i, I[1,1]*P_i]]) 
    return(P_i)
"""
#Creating random magnetic fields
np.random.seed(0)
randomint = np.around(np.random.uniform(low=-0.5, high=0.5, size=3), decimals=3)
print(randomint)
"""

#Heisenberg Hamiltonian 
def Hamiltonian(particles):
    np.random.seed(0)
    J_ij = np.around(np.random.uniform(low=0.0, high=2.0, size=(particles-1)), decimals=5)
    B_i = np.around(np.random.uniform(low=-0.2, high=0.2, size=particles), decimals=5)
    print(J_ij)
    print(B_i)

    #Chain
    H = np.zeros([2**particles,2**particles])

    #Loop
    #H = 0.5*J_ij*(spinoperatorplus(particles,1)*spinoperatorminus(particles,particles) + spinoperatorplus(particles,particles)*spinoperatorminus(particles,1)) + J_ij*(spinoperatorz(particles,1)*spinoperatorz(particles,particles))

    #Particles not interacting
    #H = -(0.5*J_ij*(spinoperatorplus(particles,2)*spinoperatorminus(particles,3) + spinoperatorplus(particles,3)*spinoperatorminus(particles,2)) + J_ij*(spinoperatorz(particles,2)*spinoperatorz(particles,3)))

    for i in range(particles-1):
	H_i = 0.5*J_ij[i]*(spinoperatorplus(particles,i+1)*spinoperatorminus(particles,i+2) + spinoperatorplus(particles,i+2)*spinoperatorminus(particles,i+1)) + J_ij[i]*(spinoperatorz(particles,i+1)*spinoperatorz(particles,i+2))
	H = H + H_i
    for j in range(particles):	
	H_j = B_i[j]*spinoperatorz(particles, j+1)
	H = H + H_j
    return(H)

Ham = Hamiltonian(3)
print(np.around(Ham, decimals=3))


#Eigenvalue and eigenvector calculation
w, v = la.eig(Ham) #Cant use eigh function since matrix becomes non symmetric once random magnetic fields are applied
E = np.around(w,decimals=4) #Full array of all eigenvalues

print(E)
V = np.around(v,decimals=4)
print(V) #Incorrect eigenvectors

#Computing row eigenvectors of Hamiltonian
def eigenvectorsr(particles,index):
    for i in range(0, 2**particles):
	l = v[0:(2**particles),i].reshape(1,2**particles)
	if i == (index-1):
	    return(l)
ER = np.around(eigenvectorsr(3,3), decimals=4)
#print(ER)

#Computing column eigenvectors of Hamiltonian
def eigenvectorsc(particles,index):
    for i in range(0, 2**particles):
	v_i = v[0:(2**particles),i] 
	if i == (index-1):
	    return(v_i)
#D = eigenvectorsc(2,1)
#print(D)


def spinoperator(particles,index):
    so = [spinoperatorx(particles,index), spinoperatory(particles,index), spinoperatorz(particles,index)]
    return(so)
#SO = spinoperator(2,1)
#print(SO)

#Computing expectation values for Sx, Sy and Sz seperately
def expecval(particles):
    for i in range(particles):
	SO = spinoperator(particles, i+1)
	if i == 0:
	    print("PARTICLE , 1")
	else:
	    print("PARTICLE" ,(i+1),)
	for j in range(2**particles):
	    if j == 0:
		print("Expectation value of spin vector for eigenvector , 1")
	    else:
		print("Expectation value of spin vector for eigenvector " ,(j+1),)
	    for k in range(0,3):
        	Ket_i = SO[k]*eigenvectorsc(particles,j+1)
        	Bra_i = eigenvectorsr(particles,j+1)*Ket_i
		
    		print(Bra_i)
expecval(3)

"""
#Computing expectation values for 
def expecvalZ(particles):
    for i in range(particles):
     if i == 0:
	    print("PARTICLE , 1")
     else:
	    print("PARTICLE" ,(i+1),)
     for j in range(2**particles):
         if j == 0:
		print("Eigenvector , 1")
         else:
		print("Eigenvector " ,(j+1),)
         Ket_i = spinoperatorz(particles, i+1)*eigenvectorsc(particles,j+1)
         Bra_i = eigenvectorsr(particles,j+1)*Ket_i
         print(Bra_i)
    
Si = expecvalZ(2)
#print(Si)
"""

