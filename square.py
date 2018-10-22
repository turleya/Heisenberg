import numpy as np
from numpy import linalg as la

for l in range(5):
    Run = 0

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

    #Creating random J
    #np.random.seed(5)
    Rand = np.around(np.random.uniform(low=-1.0, high=1.0, size=6), decimals=5)
    print("Exchange Interactions J_ij")
    print(Rand)


    strength = 1 #overall multiplicative factor of interation strength
    particles = 4
    J = np.zeros((particles,particles))
    #J = 1.0*(np.ones((particles,particles)))
    #print(J)

    for k in range(particles + 2):
        for i in range(particles-1):
	    for j in range(particles-2):
    	        J[i][i+1] = Rand[i]
    	        J[j][j+2] = Rand[j+3]
    J[0][particles-1] = Rand[5]

    J= J*strength
    #print(J)



    #Heisenberg Hamiltonian 
    def Hamiltonian(particles):
        B_i = 0.0  
        H = np.zeros([2**particles,2**particles])

        for j in range(particles):
	    i=0
	    while i<j:
	        if J[i][j] != 0:
		    #print i+1,j+1
		    H += J[i][j]*(0.5*(spinoperatorplus(particles,i+1)*spinoperatorminus(particles,j+1) + spinoperatorplus(particles,j+1)*spinoperatorminus(particles,i+1)) + spinoperatorz(particles,i+1)*spinoperatorz(particles,j+1))
	        i += 1


        for k in range(particles):	
	    H_k = B_i*spinoperatorz(particles, k+1)
	    H = H + H_k
        return(H)

    Ham = Hamiltonian(4)
    #print(Ham)


    #Eigenvalue and eigenvector calculation
    w, v = la.eigh(Ham) #Cant use eigh function since matrix becomes non symmetric once    random magnetic fields are applied
    E = np.around(w,decimals=4) #Full array of all eigenvalues

    print("Energies E_i")
    print(E)
    V = np.around(v,decimals=4)
    #print(V) #Incorrect eigenvectors
    print("--------------")

    Run = Run + l



