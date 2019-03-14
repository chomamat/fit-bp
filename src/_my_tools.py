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
        res = (res * 255).astype('int')
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

def loadData(folder):
    X_train = np.load(folder+"X_train.npy")
    y_train = np.load(folder+"y_train.npy")
    X_test = np.load(folder+"X_test.npy")
    y_test = np.load(folder+"y_test.npy")

    return X_train, y_train, X_test, y_test

def loadDataFloat(folder):
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

def splitX(X):
    X_1 = np.squeeze(X[:,0,:,:])
    X_2 = np.squeeze(X[:,1,:,:])
    return X_1, X_2