
try:
    from ulab import numpy as np
    #print('maximum number of dimensions: ', ulab.__version__)
except:
    import numpy as np
import random
import math
import time
import board
import neopixel
from gaussian import posterior_1d, sample_multivariante_normal_1d, norm_samples

# Finite number of points
X = np.arange( 0.1, 1.1, 0.02 )*10

# Noise free training data
X_train = np.array([ 0.1, 0.3, 0.4,0.7 , 1])*10
Y_train = np.sqrt(X_train)
# Compute mean and covariance of the posterior distribution
mu_sqrt, cov_sqrt = posterior_1d(X, X_train, Y_train, l=2.25, sigma_f=.5, noise=0.1)

Y_train = np.sin(X_train)
mu_sin, cov_sin = posterior_1d(X, X_train, Y_train, l=2.25, sigma_f=.5, noise=0.1)

mean_diff = np.mean( np.diff(X_train) )
min_diff = np.min( np.diff(X_train) )
max_diff = np.max( np.diff(X_train) )
std_diff = np.std( np.diff(X_train) )
print(min_diff)
print(mean_diff," +/- ",std_diff) # ???
print(max_diff)

# get samples for r,g,b -> three variables --> three samples
samples1 = sample_multivariante_normal_1d( mu_sqrt, cov_sqrt, epsilon=1e-5 )
samples2 = sample_multivariante_normal_1d( mu_sin, cov_sin, epsilon=1e-5 )
samples3 = sample_multivariante_normal_1d( mu_sin, cov_sin, epsilon=1e-5 )
samples1 = norm_samples(samples1)*256
samples2 = norm_samples(samples2)*256
samples3 = norm_samples(samples3)*256

# initialize neopixel
pixels = neopixel.NeoPixel(board.NEOPIXEL, 1)
pixels[0] = (0,200,0)

n = len(samples1)
i = 1
p = 1
while True:
    r = int(samples1[i]*.99)
    g = int(samples2[i]*.1)
    b = int(samples3[i]*.2)
    print(i,r,g,b)
    pixels[0] = ( r, g ,b )
    if i == n-1 or i == 0:
        p = p*-1
    i += p*1
    print(i,p)
    time.sleep(.1)
