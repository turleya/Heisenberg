import numpy as np
import matplotlib.pylab as plt
import h5py

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

import sklearn
from sklearn.model_selection import train_test_split

Heis_data =np.load('sldata.npz') #each row contains a feature vector
designmatrix = Heis_data['designmatrix']
Energies = Heis_data['GS_Energies'] #Target vectors

#Train/Validation/Test sets
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

## Defining Model ##
model = Sequential() #defining model for Heisenberg Hamiltonian
model.add(Dense(input_dim=20, units=256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=1))

## Compile ##
model.compile(optimizer='adam', loss='mse')
print(model.summary())

callback_list = [EarlyStopping(monitor='val_loss', min_delta=1e-7, patience=10, verbose=1), ModelCheckpoint(filepath='dim3_lattice.h5', monitor='val_loss', verbose=1, save_best_only=True)] #creates a HDF5 file 'my_model.h5'
n_epochs = 100

model_history = model.fit(x=designmatrix_train, y=Energies_train_std, validation_data=(designmatrix_val, Energies_val_std), 
                batch_size=32, verbose=1, 
                epochs=n_epochs, callbacks=callback_list, shuffle=True)

print("Epochs Taken:", len(model_history.history['loss']))

#Generalization Error
#Predictions
model_best = load_model('dim3_lattice.h5')
Energies_pred_train = model_best.predict(designmatrix_train)*Energies_std + Energies_mu
Energies_pred_val = model_best.predict(designmatrix_val)*Energies_std + Energies_mu

#plt.plot.print_out(y_true=Energies_train, y_pred=Energies_pred_train, setname='Train')
#plt.plot.print_out(y_true=Energies_val, y_pred=Energies_pred_val, setname='Val')
#plt.plot.plot_history(model_history)
plt.scatter(Energies_val, Energies_pred_val)
plt.xlabel('Energies_true')
plt.ylabel('Energies_pred')
plt.plot([-4,-1], [-4,-1], linestyle='dashed', color='k')
plt.grid()

#Accuracy
plt.figure()
plt.plot(range(1, len(model_history.history['loss'])+1), model_history.history['loss'], label='Train')
plt.plot(range(1, len(model_history.history['val_loss'])+1), model_history.history['val_loss'], label='Val')
plt.legend()
plt.xlabel('Number of Epochs')
plt.ylabel('Mean Squared Error')

plt.show()

