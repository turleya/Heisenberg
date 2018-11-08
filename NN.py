import numpy as np
from numpy import linalg as la
import matplotlib.pylab as plt


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


#Neural Network (NN) - Regression
samples = 10000

designmatrix = designmatrix(samples) #each row contains a feature vector
features = np.array(['J_12', 'J_23', 'J_34', 'J_13', 'J_24', 'J_14'])
Energies = energies(samples,4) #Target vectors

from keras.models import Sequential, load_model
from keras.layers import Dense

model = Sequential() #defining model for Heisenberg Hamiltonian
model.add(Dense(input_dim=6, units=256, activation='relu'))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=1))

## Compile ##
model.compile(optimizer='adam', loss='mse')
print(model.summary())

#Train/Validation/Test sets
import sklearn
from sklearn.model_selection import train_test_split
import seaborn as sns #advanced graphing library

#Histogram of Targets
#plt.figure()
plt.title('Histogram of targets')
sns.distplot(Energies, bins=20)
plt.xlabel("Ground State Energies")

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


from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
import h5py

callback_list = [EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=10, verbose=1), ModelCheckpoint(filepath='NN.h5', monitor='val_loss', verbose=1, save_best_only=True)] #creates a HDF5 file 'my_model.h5'
n_epochs = 100

model_history = model.fit(x=designmatrix_train, y=Energies_train_std, validation_data=(designmatrix_val, Energies_val_std), 
                batch_size=32, verbose=1, 
                epochs=n_epochs, callbacks=callback_list, shuffle=True)

print("Epochs Taken:", len(model_history.history['loss']))


#Generalization Error
#Predictions
model_best = load_model('NN.h5')
Energies_pred_train = model_best.predict(designmatrix_train)*Energies_std + Energies_mu
Energies_pred_val = model_best.predict(designmatrix_val)*Energies_std + Energies_mu

plt.figure()
plt.scatter(Energies_val, Energies_pred_val)
plt.xlabel('Energies_true')
plt.ylabel('Energies_pred')
plt.plot([-2,0], [-2,0], linestyle='dashed', color='k')
plt.grid()

#Accuracy
plt.figure()
plt.plot(range(1, len(model_history.history['loss'])+1), model_history.history['loss'], label='Train')
plt.plot(range(1, len(model_history.history['val_loss'])+1), model_history.history['val_loss'], label='Val')
plt.legend()
plt.xlabel('Number of Epochs')
plt.ylabel('Mean Squared Error')

plt.show()


