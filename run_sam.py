#use python3

import numpy as np
import shutil, sys, os, math
import mesa_reader as mr

msun = 1.989e33
rsun = 6.955e10
mjup = 1.898e30
rjup = 7.149e9
mearth = 5.97e27
sigma=5.67e-5
au = 1.496e13
L_sun = 3.839e33
expect_r = 0.1396907404744788

z = 0.02                    # metallicity of both planet and star
y = 0.273                   # helium fraction of both planet and (initial) star
Teq = 1447
total_extraheat = 5.84e26
P_surf = 1.2e6
kappa_v = 4e-3

LOGS_name = 'LOGS_SAM'

rp = 0 
record = 0
radius_initial = 2.4955500000e+10 #4.4955500000e+10 #1.3e10 for m=0.69
Z_all_HELM = 0.06 # parameter to change e.o.s 
extraheat = total_extraheat/mass_initial/mjup
T_HEAT = []
T_radius = []

TEM_array = []
RAD_array = []
GRADA_array = []
GD_array = []
MW_array = []

cnt=0 #Sam
ratio_arr=[]
loop=True
while loop:
    cnt+=1
    print('extraheat',total_extraheat)


    print(abs(expect_r-rp)/expect_r)
    ratio_arr.append((expect_r-rp)/expect_r)
	if abs(expect_r-rp)/expect_r <= 0.01:
        loop = False
    else:
        Factor = 15 # if iteration not converge, choose a smaller Factor 
		extraheat *= (1+Factor*(expect_r-rp)/expect_r)#6e+27/mass_initial/mjup
		total_extraheat *= (1+Factor*(expect_r-rp)/expect_r)
    
    
