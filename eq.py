import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import scipy, scipy.misc, scipy.signal
import cv2
import sys

def conj_transpose(mat):
    x = mat.conj()
    return x.T


def setlowerEps(arr, eps):
    arr[arr == 0] = eps
    return arr

def conj_transpose(mat):
    x = mat.conj()
    return x.T


def setlowerEps(arr, eps):
    arr[arr == 0] = eps
    return arr


def transpose(mat):
    return mat.T


def rms(a, b):
    asqr = a ** 2
    bsqr = b ** 2
    sumofsquares = asqr + bsqr
    rms = np.sqrt(sumofsquares).astype(np.uint32)
    return rms


def getunpadded(a, h, w, padding):
    return a[padding : h + padding, padding : w + padding]


def build_is_hist(img):
    
    height, width, channels = len(img), len(img[0]), len(img[0][0])
    temp = [[[0.0 for i in range(channels)] for j in range(width + 4)] for k in range(height + 4)]
    Img = np.array(temp)
    for i in range(channels):
        Img[:, :, i] = np.pad(img[:, :, i], (2, 2), 'edge')
    
    hsv = cv2.cvtColor(np.array(Img, dtype=np.uint8), cv2.COLOR_RGB2HSV)

    for i in range(len(hsv)):
        for j in range(len(hsv[0])):
            hsv[i][j][0] *= 255
            hsv[i][j][1] *= 255
            for k in range(len(hsv[0][0])):
                if hsv[i][j][k] > 255:
                    hsv[i][j][k] = 255
                if hsv[i][j][k] < 0:
                    hsv[i][j][k] = 0


    hsv = hsv.astype(np.uint8).astype(np.float64)
    H = hsv[:,:,0]
    S = hsv[:,:,1]
    I = hsv[:,:,2]
    
    kernel = [
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0]
    ]
    
    fh = np.array(kernel)
    fv = conj_transpose(fh)

    kernel = np.rot90(fh)
    kernel = np.rot90(kernel)
    dIh = cv2.filter2D(I, -1, kernel)
    dSh = cv2.filter2D(S, -1, kernel)

    kernel = np.rot90(fv)
    kernel = np.rot90(kernel)
    dIv = cv2.filter2D(I, -1, kernel)
    dSv = cv2.filter2D(S, -1, kernel)

    setlowerEps(dIh, 0.00001)
    setlowerEps(dIv, 0.00001)
    setlowerEps(dSh, 0.00001)
    setlowerEps(dSv, 0.00001)

    di = getunpadded(rms(dIh, dIv), height, width, 2)
    ds = getunpadded(rms(dSh, dSv), height, width, 2)
    
    h = getunpadded(H, height, width, 2)
    s = getunpadded(S, height, width, 2)
    i = getunpadded(I, height, width, 2).astype(np.uint8)

    Rho = [[0 for i in range(width + 4)] for j in range(height + 4)]
    Rho = np.array(Rho, dtype = np.float64)

    for p in range(2, height + 2):
        for q in range(2, width + 2):
            tmpi = transpose(I[p-2:p+3,q-2:q+3])
            tmpi.flatten()
            tmps = transpose(S[p-2:p+3,q-2:q+3])
            tmps.flatten()
            corre = np.corrcoef(tmpi, tmps)
            Rho[p,q] = corre[0,1]
    
    rho = np.abs(Rho[2 : height + 2, 2 : width + 2])
    rho[np.isnan(rho)] = 0
    rd = (rho * ds).astype(np.uint32)

    Hist_I = np.array([[0 for i in range(1)] for j in range(256)])
    Hist_S = np.array([[0 for i in range(1)] for j in range(256)])
    
    for n in range(0,255):
        temp = np.array([[0 for i in range(len(di[0]))] for j in range(len(di))])
        temp[i==n] = di[i==n]
        Hist_I[n+1] = np.sum(transpose(temp).flatten())
        temp = np.array([[0 for i in range(len(di[0]))] for j in range(len(di))])
        temp[i==n] = rd[i==n]
        Hist_S[n+1] = np.sum(transpose(temp).flatten())

    return Hist_I, Hist_S


def dhe(img, alpha=0.5):

    hist_i, hist_s = build_is_hist(img)
    hist_c = alpha*hist_s + (1-alpha)*hist_i
    hist_sum = np.sum(hist_c)
    hist_cum = hist_c.cumsum(axis=0)
    
    hsv = matplotlib.colors.rgb_to_hsv(img)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    i = hsv[:,:,2].astype(np.uint8)
    
    c = hist_cum / hist_sum
    s_r = (c * 255)
    i_s = np.zeros(i.shape)
    for n in range(0,255):
        i_s[i==n] = s_r[n+1]/255.0
    i_s[i==255] = 1
    hsi_o = np.stack((h,s,i_s), axis=2)
    result = matplotlib.colors.hsv_to_rgb(hsi_o)
    
    result = result * 255
    result[result>255] = 255
    result[result<0] = 0
    return result.astype(np.uint8)


def he(outImg):

    for i in range(len(outImg)):
        for j in range(len(outImg[0])):
            for k in range(len(outImg[0][0])):
                if(outImg[i][j][k] > 255):
                    outImg[i][j][k] = 255
                if(outImg[i][j][k] < 0):
                    outImg[i][j][k] = 0
    outImg = outImg.astype(np.uint8)
    return outImg


def histogram_rgb(img):
    height, width, channels = img.shape
    outImg = np.zeros((height, width, channels))

    for channel in range(channels): 
        outImg[:, :, channel] = ex.equalize_hist(img[:, :, channel]) * 255
    return outImg


def conj_transpose(mat):
    x = mat.conj()
    return x.T

def transpose(mat):
    return mat.T


def image_normalise(img):
    img = img.astype('float64')
    img = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX)
    return img

    
def TextureWeightHorizontal(fin, SIGMA, sharpness):
    a = conj_transpose(np.diff(fin, n=1, axis=1)), conj_transpose(fin[:,0]) - conj_transpose(fin[:,-1])
    dt0_h = conj_transpose(np.vstack(a))
    kernel = np.ones((1, SIGMA))
    gauker_h = cv2.filter2D(dt0_h, -1, kernel)
    t1 = np.abs(gauker_h)
    t2 = np.abs(dt0_h)
    W_h = t1 * t2 + sharpness
    return  1 / W_h


def TextureWeightVertical(fin, SIGMA, sharpness):
    dt0_v = np.vstack((np.diff(fin, n=1, axis=0), fin[0,:] - fin[-1,:]))
    kernel = np.ones((SIGMA, 1))
    gauker_v = cv2.filter2D(dt0_v, -1, kernel)
    t1 = np.abs(gauker_v)
    t2 = np.abs(dt0_v)
    W_v = t1 * t2 + sharpness
    return 1 / W_v


def compute(IN, wx, wy, LAMBDA):
    rows = len(IN)
    cols = len(IN[0])
    k = rows * cols

    dx =  -LAMBDA * transpose(wx).flatten()
    dxa = -LAMBDA * transpose(np.roll(wx, 1, axis = 1)).flatten()
    dxd1 = -LAMBDA * transpose(np.concatenate((wx[:, -1][:,None], np.zeros((rows,cols - 1))), axis=1)).flatten()
    wx[:,-1] = 0
    dxd2 = -LAMBDA * transpose(wx).flatten()
    Ax = scipy.sparse.spdiags(np.concatenate((dxd1[:,None], dxd2[:,None]), axis=1).T, np.array([-k+rows,-rows]), k, k)

    dy =  -LAMBDA * transpose(wy).flatten()
    dya = -LAMBDA * transpose(np.roll(wy, 1, axis = 0)).flatten()
    dyd1 = -LAMBDA * transpose(np.concatenate((wy[-1,:][None,:], np.zeros((rows-1,cols))), axis=0)).flatten()
    wy[-1,:] = 0
    dyd2 = -LAMBDA * transpose(wy).flatten()
    
    Ay = scipy.sparse.spdiags(np.concatenate((dyd1[None,:], dyd2[None,:]), axis=0), np.array([-rows+1,-1]), k, k)
    a = dx + dy + dxa + dya
    D = 1 - a
    A_sum = Ax + Ay 
    A = (A_sum + conj_transpose(A_sum) + scipy.sparse.spdiags(D, 0, k, k)).T
    
    tin = IN[:,:]
    tout = scipy.sparse.linalg.spsolve(A, transpose(tin).flatten())
    OUT = np.reshape(tout, (rows, cols), order='F')
    
    return OUT


def tsmooth(norm_img, LAMBDA=0.01, SIGMA=3.0, sharpness=0.001):
    
    I = norm_img
    x = np.copy(I)
    S = compute(I, TextureWeightHorizontal(x, SIGMA, sharpness),  TextureWeightVertical(x, SIGMA, sharpness), LAMBDA)
    return S

def function(x, a, b):
    t1 = (1 - x**a)
    t2 = t1 * b
    return np.exp(t2)

def applyK(I, k, a=-0.3293, b=1.1258):
    gamma = np.power(k, a)
    beta = function(k, a, b)
    J = np.power(I, gamma)
    J = J * beta
    # J = (I**gamma)*beta
    return J

def entropy(X):
    tmp = (X * 255).astype(np.uint8)

    for i in range(len(tmp)):
        for j in range(len(tmp[0])):
            if (tmp[i][j] > 255):
                tmp[i][j] = 255
            if (tmp[i][j] < 0):
                tmp[i][j] = 0
            
    pk = np.array(np.unique(tmp, return_counts=True)[1])
    sum1 = np.sum(pk, axis=0)
    pk = 1.0 * pk / sum1
    S = -np.sum(pk * np.log2(pk), axis=0)
    return S

def maxEntropyEnhance(I, isBad, a=-0.3293, b=1.1258):
    
    tmp = cv2.resize(I, (50,50), interpolation=cv2.INTER_AREA)

    for i in range(len(tmp)):
        for j in range(len(tmp[0])):
            for k in range(len(tmp[0][0])):
                tmp[i][j][k] = max(tmp[i][j][k], 0)

    tmp = tmp.real

    height, width, channels = tmp.shape
    if (height == 3):    
        tmp = image_normalise(tmp)
        tmp = (tmp[:,:,0]*tmp[:,:,1]*tmp[:,:,2])
        tmp = tmp**(1/3)
    
    Y = tmp
    
    isBad = isBad * 1
    isBad = scipy.misc.imresize(isBad, (50,50), interp='bicubic', mode='F')

    for i in range(len(isBad)):
        for j in range(len(isBad[0])):    
            if(isBad[i][j] < 0.5):
                isBad[i][j] = 0
            if(isBad[i][j] >= 0.5):
                isBad[i][j] = 1
          
    Y = Y[isBad == 1]
    
    if Y.size == 0:
       J = I
       return J
    
    var = applyK(Y, k)
    
    f = lambda k: -entropy(var)

    opt_k = scipy.optimize.fminbound(f, 1, 7)
    
    J = applyK(I, opt_k, a, b) - 0.01
    
    return J
    
def compute_weight(t_our, I, MU, J):
    t_a = t_our.shape[0]
    t_b = t_our.shape[1]
    i_a = I.shape[2]

    t = np.zeros((t_a, t_b, i_a))
    for i in range(i_a):
        t[:,:,i] = t_our
    W = np.power(t, MU)

    I2 = I * W
    J2 = J * (1 - W)
    result = (I2 + J2) * 255

    for i in range(len(result)):
        for j in range(len(result[0])):
            for k in range(len(result[0][0])):
                result[i][j][k] = min(result[i][j][k], 255)
                result[i][j][k] = max(result[i][j][k], 0)

    return result.astype(np.uint8)    


def enhance(norm_img):
    MU = 0.5
    LAMBDA = 0.5
    SIGMA = 5
    
    I = norm_img

    maxBGR = np.max(I, axis=2)
    resized = cv2.resize(maxBGR, (0, 0), fx = 0.5, fy = 0.5)
    
    height, width = len(maxBGR), len(maxBGR[0])
    l = (width, height)

    smoothened = tsmooth(resized, LAMBDA, SIGMA)

    t_our = cv2.resize(smoothened, l, interpolation=cv2.INTER_AREA)
    isBad = t_our < 0.5
    J = maxEntropyEnhance(I, isBad)
    result = compute_weight(t_our, I, MU, J)
    return result