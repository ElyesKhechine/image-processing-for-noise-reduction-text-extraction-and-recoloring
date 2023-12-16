
				Mini-Projet Traitement d'images
		   		 Realisee par: Elyes KHECHINE
###################################### PARTIE I #####################################
"""
Pour la premiere partie du projet, puisque il s'agit d'une operation d'elimination de bruit d'une image par un filtrage, il y a principalement 3 methodes a utiliser:
- Filtrage Spacial non lineaire: filtre median ou filtre moyenneur
- Filtrage dans le domaine frequentiel par la DFT (Direct Fourier Transform)
- Transformations morphologiques: Dilatation ou Fermeture(dilatation->erosion)
Pour ce cas, j'ai choisi la methode du filtrage dans le domaine frequentiel vu que ce type de bruit (traits horizontales noirs) presente un aspect frequentiel. Apres comparison avec la methode de fermeture des transf. morphologiques, je trouve que le resultat du filtre DFT est plus net.

"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
path = r'C:\Users\Elyes\Downloads\TI\TP\MiniProjet\liftingbodybruite.png'
img = cv2.imread(path,0)

#appliquer la DFT
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum =20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)  # center

# Declaration d'un mask LPF, le centre du cercle est 1, le reste sont des zeros
rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)

mask = np.zeros((rows, cols, 2), np.uint8)
r = 70
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
mask[mask_area] = 1

# Appliquer le masque et l'inverse DFT
fshift = dft_shift * mask

fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
cv2.imwrite(r'C:\Users\Elyes\Downloads\TI\TP\MiniProjet\liftingbodyFFTinverse.png',img_back)

plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Image d'entree'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Apres FFT'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(fshift_mask_mag, cmap='gray')
plt.title('FFT + Mask'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(img_back, cmap='gray')
plt.title('Apres FFT Inverse'), plt.xticks([]), plt.yticks([])
plt.savefig(r'C:\Users\Elyes\Downloads\TI\TP\MiniProjet\comparison.png', bbox_inches='tight')
plt.show()

cv2.waitKey(0);
cv2.destroyAllWindows();
cv2.waitKey(1)
###################################### PARTIE II #####################################
"""
Pour la partie II, le but c'est de filtrer une image afin d'eliminer toute sorte de bruit. Pour cela, on a plusieurs methodes (comme j'ai mentionnee dans la premiere partie)
J'ai utiliser un filtre non-lineaire mediane de taille 5 car il est effective il est assez efficace pour réduire un certain type de bruit (comme bruit sel et poivre), avec considérablement moins de flou sur les bords par rapport aux autres filtres linéaires de même taille
Apres filtrer l'image, on procede comme suit:
- Cadrer l'image a une region d'interet (ROI) comportant le texte souhaite
- Transformer l'image colore en image en niveau de gris
- Binairisation et seuillage de l'image (avec un seuil de 125, apres verification avec l'histogramme de l'image en niveau de gris) afin d'obtenir les caracteres en blanc sur un fond noir
- Detection des contours par le detecteur de Canny
On sauvegrade chaque resultat dans un fichier.

"""
#importation des bibliotheques
import numpy as np
import cv2

#chargement de l'image bruitee
path= r'C:\Users\Elyes\Downloads\TI\TP\MiniProjet\cartebruitee.png'
img = cv2.imread(path)

#application du filtre mediane sur l'image bruitee
img_filtre = cv2.medianBlur(img, 5)
cv2.waitKey(0);
cv2.imwrite(r'C:\Users\Elyes\Downloads\TI\TP\MiniProjet\cartefiltree.png',img_filtre)

#Cadrer l'image a une region d'interet (ROI) comportant le texte souhaite
ROI = img_filtre[220:395,175:990] 
cv2.imshow('ROI',ROI)
cv2.waitKey(0);
cv2.imwrite(r'C:\Users\Elyes\Downloads\TI\TP\MiniProjet\carteROI.png',ROI) 

#Transformer l'image colore en image en niveau de gris
ROI_gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
cv2.imshow('ROI_gray',ROI_gray)
cv2.waitKey(0)
cv2.imwrite(r'C:\Users\Elyes\Downloads\TI\TP\MiniProjet\carteROIgray.png',ROI_gray)

#Seuillage et binairisation de l'image
ret, ROI_thresh = cv2.threshold(ROI_gray, 125, 255, cv2.THRESH_BINARY)
cv2.imshow('ROI_text',ROI_thresh)
cv2.waitKey(0)
cv2.imwrite(r'C:\Users\Elyes\Downloads\TI\TP\MiniProjet\carteROItext.png',ROI_thresh)

#application du 'Canny Edge Detector'
img_canny = cv2.Canny(image=ROI_thresh, threshold1=100, threshold2=200)
cv2.imshow('img_canny',img_canny)
cv2.waitKey(0)
cv2.imwrite(r'C:\Users\Elyes\Downloads\TI\TP\MiniProjet\carteROIcanny.png',img_canny)

#Determination des contours retournees par le detecteur de Canny
contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #trouver les contours

cv2.waitKey(0);
cv2.destroyAllWindows();
cv2.waitKey(1)
###################################### PARTIE III #####################################
"""
L'algorithme de la partie 3 consiste a un changement de couleurs d'une image d'un drapeau d'Allemange en un drapeau de Lithuanie.
Pour cela, on procede par 3 etapes principales:
- Declaration des intervalles d'interets pour chaque couleur du drapeau initial
- Designation du mask de chaque zone de couleur en utilisant les intervalles precedament definis.
- Changement du drapeau par affectation des couleurs desirees (du drapeau de Lithuanie)

"""
#importation des bibliotheques

import numpy as np
import cv2

#chargement de l'image
path= r'C:\Users\Elyes\Downloads\TI\TP\MiniProjet\DrapeauAllemagne.png'
img = cv2.imread(path)
#affichage de l'image initiale
cv2.imshow('DrapeauAllemagne',img) 
cv2.waitKey(0)
#declaration  des intervalles d'interets pour chaque couleur
red_lo=np.array([0,0,180])
red_hi=np.array([80,80,255])

yellow_lo = np.array([0, 180,180])
yellow_hi = np.array([80, 255, 255])

black_lo = np.array([0,0,0])
black_hi = np.array([40,40,40])

#designation du masque de chaque couleur
mask_red=cv2.inRange(img,red_lo,red_hi)
mask_yellow = cv2.inRange(img, yellow_lo, yellow_hi)
mask_black = cv2.inRange(img, black_lo, black_hi)

#Changement du drapeau par affectation des couleurs desirees
img[mask_red>0]=(68,106,0)
img[mask_yellow>0]=(45,39,193)
img[mask_black>0]=(19,185,253)
0,
#affichage de l'image resultante
cv2.imshow("DrapeauLithuania",img) #afficher l'image modifiée (drapeau de Lithuanie)
cv2.imwrite("DrapeauLithuania.png",img) #enregistrer l'image obtenue

cv2.waitKey(0);
cv2.destroyAllWindows();
cv2.waitKey(1) 