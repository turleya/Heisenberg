import numpy as np
from numpy import linalg as la
import matplotlib


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


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}
matplotlib.rc('font', **font)

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
def designmatrix(samples):
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
    Ground_state = np.zeros(samples)
    for k in range(samples):
        exchange = designmatrix[k]
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
        Energy[k,0:16] = E_i #All Energies
        Ground_state[k] = Energy[k,0] #Ground state Energies
    #return(Energy)
    return(Ground_state)

#Energies = energies(5,4)
#print(Energies)


#Supervised Machine Learning (Regression)
samples = 2000

designmatrix = designmatrix(samples) #each row contains a feature vector
features = np.array(['J_12', 'J_23', 'J_34', 'J_13', 'J_24', 'J_14'])
Energies = energies(samples,4) #Target vectors

print("Number of Features: %i\nFeatures:"% designmatrix.shape[1])
print(features)
print("First feature vector:\n", designmatrix[0])
print("\nNumber of Examples: %i" %designmatrix.shape[0])


#Exploritory Data Analysis (EDA)

import matplotlib.pyplot as plt
import seaborn as sns #advanced graphing library

plt.figure()
plt.title('Histogram of Targets')
sns.distplot(Energies, bins=20)
plt.xlabel("Ground State Energies")

plt.figure()
plt.title('Histogram of Targets')
plt.hist(Energies, bins=20)
plt.xlabel("Ground State Energies")
plt.ylabel("Number of Samples")

#Compute covariance between the standardized features
# -1 <= Cov(x,y) <= 1
#Covariance is a measure of the joint availability of two random variables
plt.figure()
plt.title('Covariance Matrix')
X_std = (designmatrix - np.mean(designmatrix, axis=0))/np.std(designmatrix, axis=0)
cov = np.dot(X_std.T, X_std)/float(designmatrix.shape[0])
sns.heatmap(cov)

#plt.show()

#Train/Validation/Test sets
import sklearn
from sklearn.model_selection import train_test_split

designmatrix_tv, designmatrix_test, Energies_tv, Energies_test = train_test_split(designmatrix, Energies, test_size=0.2)

designmatrix_train, designmatrix_val, Energies_train, Energies_val = train_test_split(designmatrix_tv, Energies_tv, test_size=0.2)

print("\nTraining Set:\nNumber of Examples: %i\nNumber of Features: %i" % designmatrix_train.shape)
print("\nValidation Set:\nNumber of Examples: %i\nNumber of Features: %i" % designmatrix_val.shape)
print("\nTest Set:\nNumber of Examples: %i\nNumber of Features: %i" % designmatrix_test.shape)


#Standardization
designmatrix_mu = np.mean(designmatrix_train, axis=0)
designmatrix_std = np.std(designmatrix_train, axis=0)

designmatrix_train_std = (designmatrix_train - designmatrix_mu) / designmatrix_std
designmatrix_val_std = (designmatrix_val - designmatrix_mu) / designmatrix_std
designmatrix_test_std = (designmatrix_test - designmatrix_mu) / designmatrix_std

Energies_mu = np.mean(Energies_train, axis=0)
Energies_std = np.std(Energies_train, axis=0)

Energies_train_std = (Energies_train - Energies_mu) / Energies_std
Energies_val_std = (Energies_val - Energies_mu) / Energies_std
Energies_test_std = (Energies_test - Energies_mu) / Energies_std

plt.subplot(121)
plt.hist(Energies_train, bins=10)
plt.xlabel('Energies (before)')
plt.subplot(122)
plt.hist(Energies_train_std, bins=10)
plt.xlabel('Energies (after)')
#plt.show()

#Support Vector Regression
from sklearn.svm import SVR 
from sklearn.metrics import mean_squared_error, mean_absolute_error

train_mse = []
val_mse = []

Cs = np.logspace(-3,5,20)
print(Cs)

for C in Cs:
    model = SVR()
    model.C = C
    model.fit(designmatrix_train_std, Energies_train_std)
    Energies_train_pred = Energies_mu + model.predict(designmatrix_train_std)*Energies_std
    Energies_val_pred = Energies_mu + model.predict(designmatrix_val_std)*Energies_std

    train_mse.append(mean_squared_error(y_true=Energies_train, y_pred=Energies_train_pred))
    val_mse.append(mean_squared_error(y_true=Energies_val, y_pred=Energies_val_pred))

plt.figure()
plt.plot(Cs, train_mse, label='Training set MSE')
plt.plot(Cs, val_mse, label='Validation set MSE')
plt.ylabel('Mean Squared Error')
plt.xlabel('C')
plt.xscale('log')
plt.legend()

#model with lowest validation MSE
best = np.argmin(val_mse)
plt.axvline(x=Cs[best], linestyle='dashed', color='black')

#retrain with optimal C
model = SVR()
model.C = Cs[best]
model.fit(designmatrix_train_std, Energies_train_std)

print("Lowest Validation MSE: %f" % np.min(val_mse))
#plt.show()

#Generalization Error
Energies_test_pred = Energies_mu + model.predict(designmatrix_test_std)*Energies_std
print("Generalization MSE: %.4f" % (mean_squared_error(y_true=Energies_test, y_pred=Energies_test_pred)))
print("Generalization MAE: %.4f" % (mean_absolute_error(y_true=Energies_test, y_pred=Energies_test_pred)))

plt.figure()
plt.scatter(Energies_test, Energies_test_pred)
plt.xlabel('True Energies')
plt.ylabel('Predicted Energies')
plt.plot([-2,0], [-2,0], linestyle='dashed', color='k')
plt.grid()
plt.show()






