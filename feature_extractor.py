from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="directory of folder where each image in in a subfolder of its class")
parser.add_argument("--block", help="choose blocks")
#parser.add_argument("--npca", help="number of componenets of PCA")
args = parser.parse_args()
if args.dir:
    dir = args.dir
else:
    print("please enter a directory")
    exit(0)
if args.block:
    blocks = args.block.split("_")
else:
    print("please enter the block 3_6_13 if you want to choose 3,6 and 13 block from net")
    exit(0)
#if args.npca:
#    num_pca = args.npca
#else:
#    print("please enter the number of PCA components")
#    exit(0)


def get_output(model,name):
    x = model.get_layer('block_'+str(name)+'_depthwise_BN').output  # for block3 use 'block_3_depthwise_BN', for block 4 use 'block_4_depthwise_BN' etc
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    return x

model = MobileNetV2(weights = 'imagenet',include_top=False,input_shape=(224,224,3))

layers=[]
for name in blocks:
    layer_output = get_output(model,name)
    layers.append(layer_output)

'''
alternatively the following block can be used
x = model.get_layer('block_3_depthwise_BN').output # for block3 use 'block_3_depthwise_BN', for block 4 use 'block_4_depthwise_BN' etc
x = tf.keras.layers.GlobalAveragePooling2D()(x)

y = model.get_layer('block_6_depthwise_BN').output
y = tf.keras.layers.GlobalAveragePooling2D()(y)

z = model.get_layer('block_13_depthwise_BN').output
z = tf.keras.layers.GlobalAveragePooling2D()(z)

w = model.get_layer('block_16_depthwise_BN').output   #remove it whether to try with block 3,6,13
w = tf.keras.layers.GlobalAveragePooling2D()(w)       #remove it whether to try with block 3,6,13
'''
combined = tf.keras.layers.concatenate(layers,axis=-1)
#combined = tf.keras.layers.concatenate([x,y,z,w],axis=-1)  # change this to [x,y,z] if you want to remove w
shared_model = Model(inputs=model.input, outputs=combined)
shared_model.save('aerial_model.h5')
print(shared_model.summary())

datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)


'''
images should be saved in the folder in the following format:
-AID
....Airport
...........airport_1.jpg
...........airport_2.jpg
....Bareland
...........bareland_1.jpg
...........bareland_2.jpg

'''
#dir = "add dir name here"


batch_size=1
genX = datagen.flow_from_directory(
        dir,
        target_size=(224,224),
        shuffle=False,
        class_mode='categorical',
        batch_size=batch_size)

def generator_two_img(batch_size):
    while True:
            X1i = genX.next()
            yield X1i[0], X1i[1]

nb_samples = len(genX.filenames)
X = shared_model.predict_generator(generator_two_img(batch_size=1),verbose=1,steps=nb_samples)
y = genX.labels



print(X.shape)
print(y.shape)


np.save('X.npy',X)
np.save('Y.npy',y)

