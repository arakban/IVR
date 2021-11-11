import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy.io


def extractprops(img, fig3, fig4, fig5, fig6, fig7):

    img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # convert color image to grey image
    ret, binimg = cv2.threshold(img_grey, 72, 255, cv2.THRESH_BINARY_INV)
    # do thersholding, pixels of background are 0, pixels of object are 255

    vec = getproperties(binimg)
    return vec


def getproperties(img):
    height = img.shape[0]
    width = img.shape[1]

    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    area = hist[255]
    # print(area)

    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(img,contours,-1,(255,0,255),10)
    # cv2.imshow('img_contours', img)
    # cv2.waitKey(1000)
    perim = len(contours[0])

    compactness = perim*perim/(4*np.math.pi*area)

    # get scale-normalized complex central moments
    c11 = complexmoment(img, 1, 1) / (area**2)
    c20 = complexmoment(img, 2, 0) / (area**2)
    c30 = complexmoment(img, 3, 0) / (area**2.5)
    c21 = complexmoment(img, 2, 1) / (area**2.5)
    c12 = complexmoment(img, 1, 2) / (area**2.5)

    # get invariants, scaled to [-1,1] range
    ci1 = c11.real
    ci2 = (1000*c21*c12).real
    tmp = c20*c12*c12
    ci3 = 10000*tmp.real
    ci4 = 10000*tmp.imag
    tmp = c30*c12*c12*c12
    ci5 = 1000000*tmp.real
    ci6 = 1000000*tmp.imag

    vec = np.array([compactness, ci1, ci2]).squeeze()
    return vec


def complexmoment(img, u, v):
    # gets a given complex central moment value
    r_list = []
    c_list = []
    momlist = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] > 0:
                r_list.append(i)
                c_list.append(j)
    rbar, cbar = sum(r_list) / len(r_list), sum(c_list) / len(c_list)
    for i in range(len(r_list)):
        c1 = np.complex(r_list[i] - rbar, c_list[i] - cbar)
        c2 = np.complex(r_list[i] - rbar, cbar - c_list[i])
        momlist.append(c1**u * c2**v)
    muv = sum(momlist)
    return muv


def classify(v, N, Means, Invcors, Dim, Aprioris):
    # classifies a test feature vector v into one of N classes given the class means (Means) and inverse of covariance matrices (Invcors) and aprori probabilities (Aprioris)
    evals = np.zeros(N)
    IC = np.zeros((Dim, Dim))
    for i in range(N):
        IC = Invcors[i]
        evals[i] = float(multivariate(v, Means[i], IC, Aprioris[0][i]))
        bestclass = np.where(evals == np.max(evals))
    return bestclass


def multivariate(Vec, Mean, Invcor, apriori):
    # computes the probability density of a test feature vector Vec and a test class with mean vector Mean and inverse covariance matrix Invcor assuming a multi-variate gaussian distribution. It also weights by the class's a priori probability. You need to normalize by p(x)=sum(p(x|c)*p(c)) over all classes c to turn these numbers into true probabilities p(c|x).
    diff = Vec-Mean
    dist = diff@Invcor@diff.T
    n = Vec.shape[0]
    wgt = 1/pow(np.linalg.det(np.linalg.inv(Invcor)), 0.5)
    prob = apriori * (1 / (2*np.math.pi)**(n/2)) * wgt * np.math.exp(-0.5*dist)
    #print(f"multivariate prob: {prob}")
    return prob


img = cv2.imread('f1.jpg', 1)
vec = extractprops(img, 0, 0, 0, 0, 0)
print(vec)


module_data = scipy.io.loadmat("blocks.mat")  # 读取mat文件
Invcors = module_data["Invcors"]
Aprioris = module_data["Aprioris"]
maxclasses = module_data["maxclasses"][0][0]
Means = module_data["Means"]


# imgstem = input("Test image file stem ...(filestem)\n?")
imgstem = 'f'
imgnum = 0

while imgnum < 12:
    imgnum = imgnum + 1
    current_img = cv2.imread(imgstem+str(imgnum)+'.jpg', 1)
    vec = extractprops(current_img, 0, 0, 0, 0, 0)
    bestclass = classify(vec, maxclasses, Means, Invcors, 3, Aprioris)
    print(f"bestclass of image '{imgstem}{imgnum}.jpg' is: {bestclass[0]}")
    next_num = imgnum+1
    

