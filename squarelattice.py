import numpy as np
from numpy import linalg as la
#from tempfile import TemporaryFile



#Spin operators
sx = np.array([[0,1./2],[1./2,0]])
sy = np.array([[0, 1./2j], [-1./2j,0]]) 
sz = np.array([[1./2,0], [0,-1./2]])

splus = np.array([[0,1],[0,0]])
sminus = np.array([[0,0],[1,0]])

#Spin up and down
Up = np.array([[1],[0]])
Down = np.array([[0], [1]])

I = np.identity(2)

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

### Parameters ###
samples = 10000
dimension = 3
### Parameters ###

#Design Matrix
np.random.seed(0)
def design_matrix(samples,dimension):
    Res = np.zeros([samples,(4*(dimension**2)-(6*dimension)+2)])
    for i in range(samples):
        #np.random.seed(0)
        Rand = np.random.uniform(low=-1.0, high=1.0, size=(4*(dimension**2)-(6*dimension)+2))
        Rand_i = Rand
        Res[i,0:(4*(dimension**2)-(6*dimension)+2)] = Rand_i
        
    return(Res)

design_matrix = design_matrix(samples,dimension)
print(design_matrix)	

#dm = np.save(sldata, design_matrix)


#Creating J_ij for Hamiltonian
def energies(samples,dimension):
    J = np.zeros((dimension**2,dimension**2))
    Energy = np.zeros([samples,2**(dimension**2)])
    Ground_state = np.zeros(samples)
    for k in range(samples):
        exchange = design_matrix[k]

        #horizontal exchange interactions
        l=0
        for i in range(dimension):
            for j in range(dimension-1):
                J[j+dimension*i][j+1+dimension*i] = exchange[l]
                #print(l)
                #print(exchange[l])
                l += 1

        #vertical exchange interactions
        n=0
        for i in range(dimension-1):
            for j in range(dimension):
                J[j+dimension*i][j+(dimension*i)+dimension] = exchange[(dimension**2)-dimension + n]
                #print(n)
                n += 1

        #exchange interactions across diagonals of each individual square lattice
        m=0
        for i in range(dimension-1):
            for j in range(dimension-1):
                J[j+dimension*i][j+(dimension*i)+dimension+1] = exchange[2*((dimension**2)-dimension) + m]
                J[j+(dimension*i)+1][j+(dimension*i)+dimension] = exchange[2*((dimension**2)-dimension) + (dimension-1)**2 +m]
                #print(m)
                m += 1

	#Heisenberg Hamiltonian   
        H = np.zeros([2**(dimension**2),2**(dimension**2)])
        for j in range(dimension**2):
            i=0
            while i<j:
                if J[i][j] != 0:
                    #print(i+1,j+1)
                    H += J[i][j]*(0.5*(spinoperatorplus((dimension**2),i+1)*spinoperatorminus((dimension**2),j+1) + spinoperatorplus((dimension**2),j+1)*spinoperatorminus((dimension**2),i+1)) + spinoperatorz((dimension**2),i+1)*spinoperatorz((dimension**2),j+1))
                i += 1
	
        #print(H)
        w,v = la.eigh(H)
        E = np.around(w,decimals=8) #Full array of all eigenvalues
        E_i = E
        Energy[k,0:(2**(dimension**2))] = E_i
        Ground_state[k] = Energy[k,0]
    return(Ground_state)
    #return(Energy)

Energies = energies(samples,dimension)
print(Energies)

data = np.savez('sldata', design_matrix, Energies)



