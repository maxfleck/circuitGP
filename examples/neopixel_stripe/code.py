try:
    from ulab import numpy as np
    #print('maximum number of dimensions: ', ulab.__version__)
except:
    import numpy as np
import time
import board
import neopixel
import random
from gaussian import posterior_1d, sample_multivariante_normal_1d, norm_samples

number_of_pixels = 50
color = [250,50,250]
pixels = neopixel.NeoPixel(board.D5, number_of_pixels)    # Feather wiring!


x = np.linspace(0,2*np.pi,5)
y = np.sin(x)

x = x - np.min(x)
x = x/np.max(x)
y = y - np.min(y)
y = y/np.max(y)
y = y*0.6
y = y+0.4
X = np.arange( 0., 2*np.pi, 2*np.pi/10 )
color_no = len(X)
mu, cov = posterior_1d(X, x, y, l=0.25, sigma_f=0.3, noise=0.08)

colors = []
for _ in range(number_of_pixels):
    dummy = []
    for i in range(3):
        Y = sample_multivariante_normal_1d( mu, cov, epsilon=1e-5 )
        Y[ Y > 1.0 ] = 1.0
        #Y[ Y < 0.4 ] = 0.4
        Y *= color[i]
        dummy.append( np.array(Y) )
    #colors.append( np.dot( np.array([Y]).transpose() , np.array([color]) ))
    dummy = np.array(dummy)
    colors.append( dummy.transpose())
print("Done")

dt = 0.15
while True:
    #print(  len(colors), len(colors[0]), number_of_pixels )
    for c in range(color_no):
    	time_start = time.monotonic()
    	for i in range(number_of_pixels):
            #print(  len(colors), len(colors[0]), number_of_pixels )
            #print(i,c)
            #print( colors[i][c] )
            pixels[i] = tuple(colors[i][c])
            #time.sleep(0.001)
    	zzz = dt - time.monotonic() + time_start
    	if zzz > 0.:
            #print("sleep",zzz)
    	    time.sleep(zzz)
            #pass
        #else:
        #    print("no sleep ", zzz)
        #    pass