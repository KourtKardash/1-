import argparse
import cv2
import numpy as np
from scipy.ndimage import convolve, shift


'''The derivative of the residual'''
def grad_nev(x, ker, u) :
    return 2 * (convolve(convolve(x, ker), ker[::-1, ::-1]) - convolve(u, ker[::-1, ::-1]))

'''The derivative of the functional'''
def grad_TV(x):
    dx = np.sign(shift(x, (0, 1)) - x)
    dy = np.sign(shift(x, (1, 0)) - x)
    dx = shift(dx, (0, -1)) - dx
    dy = shift(dy, (-1, 0)) - dy
    return dx + dy

def deblur_algorithm(u, ker, alpha, myu, beta_k):
    iterations = 100
    x = u.copy()
    v = np.zeros(u.shape)
    
    '''The Nesterov accelerated gradient method'''
    for i in range(iterations):
        grad = grad_nev(x + myu * v, ker, u) + alpha * grad_TV(x + myu * v)
        beta = beta_k ** (float(i) / iterations)
        v = myu * v - beta * grad
        x += v
    x = np.clip(x, a_min=0, a_max=255)
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_image')
    parser.add_argument('kernel')
    parser.add_argument('output_image')
    parser.add_argument('noise_level')
    args = parser.parse_args()

'''
img = np.array(cv2.imread("reference.bmp", cv2.IMREAD_GRAYSCALE)).astype('float')
kernel = np.array(cv2.imread("kernel2.bmp", cv2.IMREAD_GRAYSCALE)).astype('float')
kernel /= np.sum(kernel)
res = convolve(img, kernel)
cv2.imwrite("blurred_2.bmp", res.astype('int'))
'''
img_blurred = np.array(cv2.imread(args.input_image, cv2.IMREAD_GRAYSCALE)).astype('float')
kernel = np.array(cv2.imread(args.kernel, cv2.IMREAD_GRAYSCALE)).astype('float')
kernel /= np.sum(kernel)
noise_level = float(args.noise_level)

a = 0
b = 0
myu = 0
beta_k = 0

if noise_level == 0 :
    a = 0.56263
    b = 0.017567
    myu = 0.941267
    beta_k = 0.103674
elif noise_level > 0 and noise_level <= 1:
    a = 0.4144
    b = 0.0227
    myu = 0.8975
    beta_k = 0.26214 
elif noise_level > 1 and noise_level < 6:
    a = 0.1552
    b = 0.5975
    myu = 0.7468
    beta_k = 0.29754
else:
    a = 0.56263
    b = 0.017567
    myu = 0.941267
    beta_k = 0.103674

img_deblurred = deblur_algorithm(img_blurred, kernel, noise_level * a + b, myu, beta_k)
cv2.imwrite(args.output_image, img_deblurred.astype('int'))

#a, b, myu, beta_k
