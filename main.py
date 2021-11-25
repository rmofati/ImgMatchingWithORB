# Resolução do Exercício Aula 8 - ORB e Matching
# Aluno: Rafael Mofati Campos

import numpy as np
import cv2
#from matplotlib import pyplot as plt

# Desenvolvimento do exercício 1
# Ao aumentor o valor do fast_threshold, a quantidade de pontos detectados diminui
for fastThreshold in range(10, 110, 10):
    orb = cv2.ORB_create(patchSize = 21, fastThreshold = fastThreshold, nlevels = 8, scaleFactor = 1.2)

    img_mulher = cv2.imread('img/rostoMulher1.jpg', 0)

    kp, des = orb.detectAndCompute(img_mulher, None)

    img_mulher = cv2.drawKeypoints(img_mulher,kp,None)

    scale = 2
    width = int(img_mulher.shape[1] * scale)
    height = int(img_mulher.shape[0] * scale)
    dim = (width, height)

    image_mulher_resized = cv2.resize(img_mulher, dim, interpolation = cv2.INTER_AREA)

    cv2.imshow("Imagem", image_mulher_resized)
    print("FastThreshold:", fastThreshold, " | Número de pontos e descritores:", len(kp))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Desenvolvimento do exercício 2
img_ref = cv2.imread("img/imgRef.jpg", 0)
img1 = cv2.imread("img/img1.jpg", 0)
img2 = cv2.imread("img/img2.jpg", 0)
img3 = cv2.imread("img/img3.jpg", 0)
img4 = cv2.imread("img/img4.jpg", 0)

imgs = [img1, img2, img3, img4]

orb = cv2.ORB_create(patchSize = 21, fastThreshold = 60, nlevels = 5, scaleFactor = 1.2)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
kp1, des1 = orb.detectAndCompute(img_ref, None)

for img in imgs:
    kp2, des2 = orb.detectAndCompute(img, None)
    matches = bf.match(des1,des2)
    imgMatch = cv2.drawMatches(img_ref,kp1,img,kp2,matches,None)

    cv2.imshow("imgMatch", imgMatch)
    cv2.waitKey(0)
    cv2.destroyAllWindows()