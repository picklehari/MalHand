import pandas
import numpy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os



data = pandas.read_csv("CSVs/Handwritten_V2_train.csv",header=None)
data_2 = pandas.read_csv("CSVs/Handwritten_V2_test.csv",header=None)
data_3 = pandas.read_csv("CSVs/Handwritten_V2_valid.csv",header=None)

data = data.transpose()
file_names = list(data.iloc[0,:])
filepath = "Data/"
data = data.drop(index=0)
data = data.transpose()
data = data.to_numpy()
x = 0

train_path = filepath + "train/"
test_path = filepath + "test/"
valid_path = filepath + "valid/"

for i in set(file_names):
    os.makedirs(train_path + str(i),exist_ok=True)
    os.makedirs(test_path + str(i),exist_ok=True)
    os.makedirs(valid_path + str(i),exist_ok=True)

for rows in data:
    img = rows
    img = numpy.reshape(img,(32,32))
    img = img.transpose()
    
    name = train_path +str(file_names[x]) +"/" + str(x) + ".png"
    x = x+1
    plt.imsave(name,img,cmap=cm.gray)

data_2 = data_2.transpose()
file_names = list(data_2.iloc[0,:])
data_2 = data_2.drop(index=0)
data_2 = data_2.transpose()
data_2 = data_2.to_numpy()
y = 0
for rows in data_2:
    img = rows
    img = numpy.reshape(img,(32,32))
    img = img.transpose()
    
    name = test_path +str(file_names[y]) +"/" + str(x) + ".png"
    x = x+1
    y = y+1
    plt.imsave(name,img,cmap=cm.gray)

data_3 = data_3.transpose()
file_names = list(data_3.iloc[0,:])
data_3 = data_3.drop(index=0)
data_3 = data_3.transpose()
data_3 = data_3.to_numpy()
y = 0
for rows in data_3:
    img = rows
    img = numpy.reshape(img,(32,32))
    img = img.transpose()
    
    name = valid_path +str(file_names[y]) +"/" + str(x) + ".png"
    x = x+1
    y = y+1
    plt.imsave(name,img,cmap=cm.gray)

