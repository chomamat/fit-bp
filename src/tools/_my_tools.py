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

def showImgGC(name,*img,folder=None):
    if len(img) == 0:
        return
    else:
        res = img[0]
        for i in img[1:]:
            res = np.concatenate((res,i),axis=1)
            
    fig, ax = plt.subplots(figsize=(30,30))
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

def loadData(folder, typeF=None, channels_last=False):
    X_train = np.load(folder+"X_train.npy")
    y_train = np.load(folder+"y_train.npy")
    X_test = np.load(folder+"X_test.npy")
    y_test = np.load(folder+"y_test.npy")

    if typeF is not None:
        X_train = X_train.astype(typeF)
        y_train = y_train.astype(typeF)
        X_test = X_test.astype(typeF)
        y_test = y_test.astype(typeF)
        X_train /= 255
        y_train /= 255
        X_test /= 255
        y_test /= 255

    if channels_last is True:
        X_train = X_train.swapaxes(1,3)
        X_train = X_train.swapaxes(1,2)
        X_test = X_test.swapaxes(1,3)
        X_test = X_test.swapaxes(1,2)

        y_train = np.expand_dims(y_train,3)
        y_test = np.expand_dims(y_test,3)
    else:
        y_train = np.expand_dims(y_train, 1)
        y_test = np.expand_dims(y_test, 1)

    return X_train, y_train, X_test, y_test

def loadDataFloat(folder):
    print("This is function is deprecated, replace it please.")
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

def plotHistory(history) :
    for key, val in history.items():
        assert type(val) == list
        plt.plot(range(len(val)),val,label=key)
    plt.show()