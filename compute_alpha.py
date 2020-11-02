from tensorflow.keras.preprocessing import image
import numpy as np
import os
import glob
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import pandas as pd
from tensorflow.keras.models import Model



def split_string(str):
    arr = str.split("_")
    block_name = arr[0]+arr[1]
    block_type = arr[2]
    return block_name,block_type
images = []
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="directory of folder where each image in in a subfolder of its class")
args = parser.parse_args()
if args.dir:
    dir = args.dir
else:
    print("please enter a directory")
    exit(0)



sub_dir = [x[0] for x in os.walk(dir)]
n_classes = len(sub_dir)-1

#print(sub_dir)

prev_y=[]
model = MobileNetV2(weights='imagenet', include_top=True, input_shape=(224, 224, 3))

layer_name=[]
for i in range(len(model.layers)):
    if "relu" in model.layers[i].name and 'block' in model.layers[i].name:
        name = model.layers[i].name
        layer_name.append(name)
np.save('test_layers_final.npy',layer_name)
layer_name=np.load('test_layers_final.npy')
print(layer_name)
arr=[]
alpha=1
forbidden=['Conv1_relu','expanded_conv_depthwise_relu','out_relu']
count=0
prev_value=0
##block
for layer in layer_name:
    if layer in forbidden:
        continue

    model_test = Model(inputs=model.input, outputs=model.get_layer(layer).output)
    zero_percentage=0
    for i in range(1,len(sub_dir)):

        filenames = [img for img in glob.glob(sub_dir[i]+"/*.jpg")]

        for filename in filenames[0:5]:

            name = filename.split("\\")[-1]
            img = image.load_img(filename, target_size=(224, 224))
            img = np.expand_dims(img, axis=0)
            img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
            #img = np.ones((1, 224, 224, 3)) #for white image
            y = model_test.predict(img)
            params = model_test.output.shape[1]*model_test.output.shape[2]*model_test.output.shape[3]
            zero_percentage = zero_percentage + (1 - (np.count_nonzero(y) / params))


    #print(layer + ": "+str(zero_percentage/(len(sub_dir)*5)))
    curr_value = zero_percentage/(n_classes*5)
    if count%2==1:
        alpha = prev_value/curr_value
        print(layer.split("_")[0]+"_"+layer.split("_")[1]+"_alpha: "+str(alpha))
    count = count+1
    block_name,block_type = split_string(layer)
    arr.append([block_name,block_type,"%0.4f" % (curr_value)])
    prev_value= curr_value
df2 = pd.DataFrame(np.array(arr),columns=['block_name', 'block_type', 'zero_percentage'])
df2.to_csv('zero_percentage_test.csv')
print("csv file saved")

