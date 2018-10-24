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


#Design Matrix
np.random.seed(0)
def design_matrix(samples):
    Res = np.zeros([samples,6])
    for i in range(samples):
        #np.random.seed(0)
        Rand = np.random.uniform(low=-1.0, high=1.0, size=6)
	Rand_i = Rand
	Res[i,0:6] = Rand_i
        
    return(Res)

#design_matrix = design_matrix(5)
#print(design_matrix[0])	


#Creating J_ij for Hamiltonian
def energies(samples,particles):
    J = np.zeros((particles,particles))
    Energy = np.zeros([samples,16])
    for k in range(samples):
        exchange = design_matrix[k]
        for i in range(particles-1):
            for j in range(particles-2):
    	        J[i][i+1] = exchange[i]
    	        J[j][j+2] = exchange[j+3]
        #print(J[0][1])
        J[0][particles-1] = exchange[5]

	#Heisenberg Hamiltonian   
        H = np.zeros([2**particles,2**particles])
        for j in range(particles):
	    i=0
	    while i<j:
	        if J[i][j] != 0:
		    #print i+1,j+1
		    H += J[i][j]*(0.5*(spinoperatorplus(particles,i+1)*spinoperatorminus(particles,j+1) + spinoperatorplus(particles,j+1)*spinoperatorminus(particles,i+1)) + spinoperatorz(particles,i+1)*spinoperatorz(particles,j+1))
	        i += 1
	
        #print(H)
	w,v = la.eigh(H)
	E = np.around(w,decimals=8) #Full array of all eigenvalues
	E_i = E
	Energy[k,0:16] = E_i
    return(Energy)

#Energies = energies(5,4)
#print(Energies)




#Supervised Machine Learning (Regression)
samples = 200

design_matrix = design_matrix(samples) #each row contains a feature vector
features = np.array(['J_12', 'J_23', 'J_34', 'J_13', 'J_24', 'J_14'])
Energies = energies(samples,4) #Target vectors

print("Number of Features: %i\nFeatures:"% design_matrix.shape[1])
print(features)
print("First feature vector:\n", design_matrix[0])
print("\nNumber of Examples: %i" %design_matrix.shape[0])



#Exploritory Data Analysis (EDA)

#import matplotlib.pyplot as plt
import seaborn as sns #advanced graphing library









