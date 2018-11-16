import numpy as np
import matplotlib.pylab as plt
import imp
import pandas as pd
import seaborn as sns #advanced graphing library
import random

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

seed = 1010
np.random.seed(seed)
random.seed(seed)

Heis_data =np.load('sldata.npz') #each row contains a feature vector
designmatrix = Heis_data['designmatrix']
Energies = Heis_data['GS_Energies'] #Target vectors

model = Sequential() #defining model for Heisenberg Hamiltonian
model.add(Dense(input_dim=20, units=256, activation='relu'))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=1))


MSE_val = []
MSE_train = []
dm = []
#for i in range(200, 10000, 200):
for i in [10000]:
    designmatrix_tv = []
    designmatrix_test = []
    Energies_tv = []
    Energies_test = []

    #Train/Validation/Test sets
    designmatrix_tv, designmatrix_test, Energies_tv, Energies_test = train_test_split(designmatrix[0:i], Energies[0:i], test_size=0.2)
    #print('------')
    #print(designmatrix_tv)


    designmatrix_train, designmatrix_val, Energies_train, Energies_val = train_test_split(designmatrix_tv, Energies_tv, test_size=0.2)

    #Standardisation
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

    
    ## Compile ##
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())

    callback_list = [EarlyStopping(monitor='val_loss', min_delta=1e-7, patience=10, verbose=1), ModelCheckpoint(filepath='lattice.h5', monitor='val_loss', verbose=1, save_best_only=True)] #creates a HDF5 file 'my_model.h5'
    n_epochs = 100

    #Model fit
    model_history = model.fit(x=designmatrix_train, y=Energies_train_std, validation_data=(designmatrix_val, Energies_val_std), 
                batch_size=32, verbose=1, 
                epochs=n_epochs, callbacks=callback_list, shuffle=True)

    #Generalization Error
    #Predictions
    model_best = load_model('lattice.h5')
    Energies_pred_train = model_best.predict(designmatrix_train)*Energies_std + Energies_mu
    Energies_pred_val = model_best.predict(designmatrix_val)*Energies_std + Energies_mu

    final_val = len(model_history.history['val_loss'])
    val_loss = (model_history.history['val_loss'])[final_val-1]

    final_train = len(model_history.history['loss'])
    train_loss = (model_history.history['loss'])[final_train -1]
    
    MSE_val.append(val_loss)
    MSE_train.append(train_loss)
    dm.append(len(designmatrix_train))
    print(dm)


print(MSE_train)
print(MSE_val)

#Learning Curve
plt.figure()
plt.plot(dm, MSE_train, label='Train')
plt.plot(dm, MSE_val, label='Val')
plt.legend()
plt.xlabel('Number of Samples in Training set')
plt.ylabel('Mean Squared Error')
plt.show()
