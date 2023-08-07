## circuitGP

# Gaussian Processes for circuitpython and micopython

Should run on boards with a rp2040 or nRF52840 chip using circuitpython (therefore micropython should work too). 

It allows the usage of simple 2D gaussian processes for parameter-free applications and mainly relys on ulab. If your microcontroller is able to handle the process depends on your data and the parametrization of your gaussian process. In general, it is recommended to avoid borderline cases. Therefore, the available memory of the chip should be respected and normalised and thinned out data sets should be used.

Examples and working .py-scripts can be found in the examples folder (tested on adafruit feather nRF52840 bluefruit sense and adafruit feather rp2040).

**TO DO**

- add/implement additional kernel functions
- set up runner to auto compile code


**GOALS**

- bring something you might call creativity to small devices through samples from possibility spaces
- allows complex control functions that do not follow a fixed functional form and thus data-driven wrappers for different types of sensors
- ...which can adjust themselves on the basis of current data
