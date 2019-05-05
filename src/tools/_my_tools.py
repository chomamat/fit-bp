from collections import defaultdict
import csv
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def showImg(name,*img,folder=None):
    if len(img) == 0:
        return
    else:
        res = img[0]
        for i in img[1:]:
            res = np.concatenate((res,i),axis=1)
            
    
    win = cv.namedWindow(name,cv.WINDOW_NORMAL)
    cv.resizeWindow(name, 1700,900)
    cv.imshow(name,res)
    cv.waitKey(0)
    cv.destroyAllWindows()
    if folder is not None:
        # res = (res * 255).astype('int')
        cv.imwrite(folder+name+".png",res)

def showImgGC(name,*img,folder=None,size=(30,30)):
    if len(img) == 0:
        return
    else:
        res = img[0]
        for i in img[1:]:
            res = np.concatenate((res,i),axis=1)
    
    res = res.astype('float32')
    fig, ax = plt.subplots(figsize=size)
    ax.grid(False)
    ax.imshow(res.squeeze(), cmap='binary_r')
    if folder is not None:
        res = (res * 255).astype('int')
        cv.imwrite(folder+name+".png",res)

def compare(i,X,y,res,folder=None, channels_last=True):
    if channels_last is True:
        showImgGC(str(i).zfill(2),X[i,:,:,0],y[i,:,:,0],res[i,:,:,0],X[i,:,:,1],folder=folder)
    else:
        showImgGC(str(i).zfill(2),X[i,0,:,:],y[i,0,:,:],res[i,0,:,:],X[i,1,:,:],folder=folder)

# =================================================================================================

# files expects a list of filenames
def load(files, typeF=None, channels_last=False):
    out = []

    for f in files:
        print("Loading",f)
        x = np.load(f)

        if typeF is not None:
            x = x.astype(typeF)
            x /= 255

        if channels_last is True:
            x = x.swapaxes(1,3)
            x = x.swapaxes(1,2)

        out.append(x)

    return out

def loadData(folder, train=False, val=False, test=False, typeF=None, channels_last=False):
    names = []

    if train is True:
        names.append(folder+"X_train.npy")
        names.append(folder+"y_train.npy")

    if val is True:
        names.append(folder+"X_val.npy")
        names.append(folder+"y_val.npy")

    if test is True:        
        names.append(folder+"X_test.npy")
        names.append(folder+"y_test.npy")

    return load(names, typeF=typeF, channels_last=channels_last)

# Concatenate the datasets to create one dataset with dimensions [:,1,I,J], where (I,J) are image dimensions
def loadDataByOne(folder, train=False, val=False, test=False, typeF=None, channels_last=False):
    out = []

    if train is True:
        X_train = loadData(folder, train=True, typeF=typeF, channels_last=channels_last)[0]
        X_train = X_train.reshape((-1,1,X_train.shape[2],X_train.shape[3]))
        out.append(X_train)

    if val is True:
        X_val = loadData(folder, val=True, typeF=typeF, channels_last=channels_last)[0]
        X_val = X_val.reshape((-1,1,X_val.shape[2],X_val.shape[3]))
        out.append(X_val)

    if test is True:
        X_test = loadData(folder, test=True, typeF=typeF, channels_last=channels_last)[0]
        X_test = X_test.reshape((-1,1,X_test.shape[2],X_test.shape[3]))
        out.append(X_test)

    return out

# =================================================================================================

def loadDataFloat(folder):
    print("This is function is deprecated, replace it please with loadData().")
    X_train, y_train, X_test, y_test = loadData(folder)
    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')
    X_test = X_test.astype('float32')
    y_test = y_test.astype('float32')
    X_train /= 255
    y_train /= 255
    X_test /= 255
    y_test /= 255

    return X_train, y_train, X_test, y_test

def postProcess(X, channels, bound=1):
    c = channels - 1
    if bound != 1:
        tmp = (X / bound) * c * 2
        res = ( (np.floor(tmp) + np.ceil(tmp)) // 3 ) / c
        res = (res * bound).astype(int)     
    else:
        tmp = X * c * 2
        res = ( (np.floor(tmp) + np.ceil(tmp)) // 3 ) / c
    return res

def splitX(X):
    X_1 = np.squeeze(X[:,0,:,:])
    X_2 = np.squeeze(X[:,1,:,:])
    return X_1, X_2

def toCSV(file, d):
    with open(file, 'w') as f:
        w = csv.writer(f)
        w.writerow(d.keys())
        w.writerows(zip(*d.values()))

def fromCSV(file):
    d = defaultdict(list)
    for record in csv.DictReader(open(file)):
        for key, val in record.items():
            d[key].append(float(val))

    return dict(d)

def plotHistory(history, save=None, size=(10,10)) :
    plt.figure(figsize=size)
    plt.grid(True)

    for key, val in history.items():
        assert type(val) == list
        plt.plot(range(len(val)),val,label=key)
    plt.legend(loc='upper right')#, prop={'size': 24})

    if save is not None:
        plt.savefig(save)
    else:
        plt.show()