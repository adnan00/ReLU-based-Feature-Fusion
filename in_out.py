
# first neural network with keras tutorial
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow as tf
import os

# split into input (X) and output (y) variables
X_in = np.load('X_ucm.npy',allow_pickle=True)
print(len(X_in))
X_out = np.load('X_no_ucm.npy',allow_pickle=True)
print(len(X_out))

y_ucm = np.load('Y_ucm.npy',allow_pickle=True)
y_no_ucm = np.load('Y_no_ucm.npy',allow_pickle=True)
y_in = np.ones(len(X_in))
y_out = np.zeros(len(X_out))

X = np.concatenate((X_in, X_out), axis=0)
y = np.concatenate((y_in, y_out), axis=0)
print(X.shape)
print(y.shape)
#exit()
checkpoint_path = "model/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_best_only=True)
# define the keras model
model = Sequential()
model.add(Dense(512, input_dim=1872, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset

#model = create_model()

# Save the weights using the `checkpoint_path` format


# Train the model with the new callback


model.fit(X, y, epochs=1, batch_size=32,callbacks=[cp_callback],validation_split=0.2)


#model.save_weights(checkpoint_path.format(epoch=0))
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

y_pred = model.predict(X_in)
print(y_pred)
y_pred = y_pred.flatten()
y_pred[y_pred>=0.5]=1
y_pred[y_pred<0.5]=0
print(np.sum(y_pred==y_in)/len(y_in))
import pickle as pk
pca = pk.load(open("pca_aerial_no_ucm.pkl", "rb"))
lda = pk.load(open("lda_aerial_no_ucm.pkl", "rb"))
clf = pk.load(open("svm_aerial_no_ucm.pkl", "rb"))
X_out = lda.transform(pca.transform(X_out))
print(clf.score(X_out,y_no_ucm))

pca = pk.load(open("pca_aerial_ucm.pkl", "rb"))
lda = pk.load(open("lda_aerial_ucm.pkl", "rb"))
clf = pk.load(open("svm_aerial_ucm.pkl", "rb"))
X_in = lda.transform(pca.transform(X_in))
print(clf.score(X_in,y_ucm))