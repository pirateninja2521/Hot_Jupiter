#!/usr/bin/env python
## run MESA by this file

            #For  HD 209458b: 
            #Mp=0.69Mj=1.3097097e30,, Rp=1.359Rj=0.13656591, F_incident=9.93e8 cgs units Teq = (9.93e8/(4*5.671e-5))^0.25=1447
            #Mz = 57.9 M_p^0.61=46.1717,Zp=46.1717*0.00314635457/0.69=0.21053991202

            #For  WASP-19b:    
            #Mp=1.133Mj, Rp=1.386Rj, F_incident=41.33e8 cgs units Teq = (41.33e8/(4*5.67e-5))^0.25=2066
            #Mz = 57.9 M_p^0.61=63.1098,Zp=63.1098*0.00314635457/1.133=0.17525683094

            #FOr  WASP-103b:  Mp=1.488Mj, Rp=1.528Rj, F_incident=89.34e8 cgs units Teq = (89.34e8/(4*5.67e-5))^0.25=2505
            #For  HAT-P-33b:    Mp=0.76Mj, Rp=1.686Rj, F_incident=22.78e8 cgs units Teq = (22.78e8/(4*5.67e-5))^0.25=1780
            #For  XO-1b:           Mp=0.918Mj, Rp=1.206Rj, F_incident=4e8 cgs units Teq = (4e8/(4*5.67e-5))^0.25=1152
             # e.g., might use T_eq = Teff_star*(R_star/(2*Distance_to_star))^(1/2)


import math
import numpy as np
import os
import shutil
import pdksubsprograms as my
import sys
import mesa_reader as mr

###################################################################################
## some constants
###################################################################################
msun = 1.989e33
rsun = 6.955e10
mjup = 1.898e30
rjup = 7.149e9
mearth = 5.97e27
sigma=5.67e-5
au = 1.496e13
L_sun = 3.839e33

###################################################################################
## parameter and control 
###################################################################################

# flags to skip steps
do_create_planet = False ## not used, default false
# to generate new planet
do_relax_irrad = True   # set star mass & relax irradiation & core setting
do_relax_z = True		#relax_metalicity
do_evolve_planet = True
do_evolve_planet2 = False

# parameter to change
#expect_r = 0.13656591 # expected radius of planet (sun unit)
expect_r = 0.1396907404744788 # expected radius of planet (sun unit), HD 209458 b
#expect_r = 0.1424660531991373 # WASP 19b
#expect_r = 0.12396396836808052 # XO-1b


z = 0.02                    # metallicity of both planet and star
y = 0.273                   # helium fraction of both planet and (initial) star

# core parameter
putincore = '.False.' #.false.
core_mass = 0.0001326 #0.0001326 in sun units
core_density = 5.50

# eq(49) parameter
#### HD 209458 b
Teq = 1447 #1447                  # equilibrium temperature in do_evolve_planet
Teq2 = 1447                 # equilibrium temperature in do_evolve_planet2 (may be higher or lower than Teq)
mass_initial = 0.69          #1.133#1.488  #jupiter
#### WASP 19b
#Teq = 2066                  # equilibrium temperature in do_evolve_planet
#Teq2 = 2066                 # equilibrium temperature in do_evolve_planet2 (may be higher or lower than Teq) 
#mass_initial = 1.133          #WASP 19b  #jupiter 
### XO-1b
#Teq = 1152                  # equilibrium temperature in do_evolve_planet
#Teq2 = 1152                 # equilibrium temperature in do_evolve_planet2 (may be higher or lower than Teq) 
#mass_initial = 0.918          #WASP 19b  #jupiter 



total_extraheat = 5.84e26 #3.8418676605428115e+26
#total_extraheat = 2.84e26 #3.8418676605428115e+26
#P_surf = 3.5e5
#kappa_v = 3e-2
#total_extraheat = 7.15e26 #3.8418676605428115e+26
#total_extraheat = 8.0e27
P_surf = 1.2e6
#P_surf = 1.7e6
#P_surf = 6.0e5
kappa_v = 4e-3
#LOGS_name = 'LOGS_XO1b'
#LOGS_name = 'LOGS_HD209458'
LOGS_name = 'LOGS_WASP_19b'
stop_loop =  False

###################################################################################
## main
###################################################################################
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

### two choice: 
### use while loop to do multiple run to reach given radius
### use for loop to do one run 

cnt=0 #Sam
ratio_arr=[]
while abs(expect_r-rp)/expect_r > 0.01:
	cnt+=1 #Sam
	print(abs(expect_r-rp)/expect_r)
	ratio_arr.append((expect_r-rp)/expect_r)
	###########################################################################
	# make rp converge to expect_r
	##########################################################################
	if rp != 0:
		Factor = 15 # if iteration not converge, choose a smaller Factor 
		extraheat = extraheat*(1+Factor*(expect_r-rp)/expect_r)#6e+27/mass_initial/mjup
		total_extraheat = total_extraheat*(1+Factor*(expect_r-rp)/expect_r)
	print('extraheat',total_extraheat)
	#########################################################################
	# back up all inlist and run_MESA.py scripts
	#########################################################################
	inlist1 = 'LOGS/opt_inlist_created'
	inlist2 = 'LOGS/opt_inlist_rlx_irrad'
	inlist3 = 'LOGS/opt_inlist_rlx_z'
	inlist6 = 'LOGS/opt_inlist_irrad_pdk'
	inlist7 = 'LOGS/opt_inlist_irrad2_pdk'
	if os.path.exists(inlist1):
		os.remove(inlist1)
	if os.path.exists(inlist2):
		os.remove(inlist2)
	if os.path.exists(inlist3):
		os.remove(inlist3)
	if os.path.exists(inlist6):
		os.remove(inlist6)
	if os.path.exists(inlist7):
		os.remove(inlist7)
	#shutil.copyfile('run_MESA_20190718_PDK.py','LOGS/run_MESA_backup.py')

	createmodel = 'created.mod'
	rlx_irradmodel = 'rlx_irrad.mod'
	irrad1model = 'evolve.mod'
	rlx_metal = 'rlx_z0.mod'

	##########################################################################
	### remove existing model to make sure new run correctly builded
	##########################################################################
	if os.path.exists(createmodel) and do_create_planet:
		os.remove(createmodel)
	if os.path.exists(rlx_irradmodel) and do_relax_irrad:
		os.remove(rlx_irradmodel)
	if os.path.exists(rlx_metal) and do_relax_z:
		os.remove(rlx_metal) 
	if os.path.exists(irrad1model) and do_evolve_planet:
		os.remove(irrad1model) 

	if do_create_planet: #False
		run_time = my.create_planet(mass_initial,radius_initial,y,z,inlist1,createmodel)
	
	if do_relax_irrad:
		saved_model = createmodel
		save_model = rlx_irradmodel
		run_time = my.relax_irrad_planet(inlist2,saved_model,save_model,mass_initial,radius_initial,putincore,core_mass,core_density,y,z)
		print("end")

	if do_relax_z:
		saved_model = rlx_irradmodel
		save_model = rlx_metal
		new_z = 0.02
		run_time = my.relax_z_planet(Teq,inlist3,saved_model,save_model,new_z,extraheat,Z_all_HELM,P_surf,kappa_v)
		### mutistep to reach given metallicity
		for jj in range(0): 
			if os.path.exists(rlx_metal):
				new_z = new_z + 0.01
				saved_model = rlx_metal
				save_model = 'temp'+rlx_metal
				run_time = my.relax_z_planet(Teq,inlist3,saved_model,save_model,new_z,extraheat,Z_all_HELM,P_surf,kappa_v)
				os.remove(saved_model)
				if os.path.exists(save_model):
					os.rename(save_model,rlx_metal) ## to rename the model to iterate
			else:
				break
		print("end_2")

	if do_evolve_planet:
		saved_model = rlx_metal
		save_model = irrad1model
		runtime = my.irrad_planet(Teq,inlist6,saved_model,save_model,extraheat,Z_all_HELM,P_surf,kappa_v)
		print("end_3")

	if do_evolve_planet2: #False
		saved_model = irrad1model
		runtime = my.irrad_planet2(Teq2,inlist7,saved_model,extraheat,Z_all_HELM,P_surf,kappa_v)

	###############################################################################
	## MESAreader
	###############################################################################

	m = mr.MesaData()
	g = mr.MesaLogDir()
	p = g.profile_data()
	optical_p = p.data('tau')
	grav_p    = p.data('grav')
	pressure_p= p.data('pressure')
	opacity_p = p.data('opacity')
	radius_p = p.data('radius')
	logT_p   = p.data('logT')
	grada_p  = p.data('grada')
	cp_p = p.data('cp')
	eta_p = p.data('eta')
	mu_p = p.data('mu')
	'''
	for taking tau = 2/3 as boundary case
	o = 0
	optical_p[0]
	while optical_p[o] < 2./3:
		o += 1

	star_age = m.data('star_age')
	radius_h = m.data('radius')
	print(total_extraheat,radius_p[o])
	print(o)
	rp = radius_p[o]
	record += 1

	T_HEAT.append(total_extraheat)
	T_radius.append(rp) 
	
	TEM_array.append(logT_p[o:])
	RAD_array.append(radius_p[o:])
	GRADA_array.append(grada_p[o:])
	GD_array.append( eta_p[o:])
	MW_array.append( mu_p[o:] )
	'''
	# The first element of array is the outer boundary of planet
	print(total_extraheat,radius_p[0])
	print(optical_p[0],'surface tau')
	print(kappa_v/opacity_p[0],'kap_v/kap_th')
	print(np.power(10,logT_p[0]),'surface temp')
	kk = 0
	op_v = kappa_v*pressure_p/grav_p
	rp = radius_p[0]
	record = record + 1
	T_HEAT.append(total_extraheat)
	T_radius.append(rp) 

	TEM_array.append(logT_p)
	RAD_array.append(radius_p)
	GRADA_array.append(grada_p)
	GD_array.append( eta_p)
	MW_array.append( mu_p)

	print (T_HEAT, T_radius,'radius_here')
	#print(CNTR_T,'center T here')
	####################################################################
	if (stop_loop):
		print('no loop iteration')
		break
#end while

print('while count: ', cnt)
print(ratio_arr)

##########################################################################
### save LOGS file
##########################################################################

ans = input('Successful? (1/0)\n')
while (1 == ans):
	if os.path.exists(LOGS_name):
		print('directory ' + LOGS_name + ' has existed. \n')
		ans2 = input('replace it (1/0)?\n')
		if 	(1 == ans2):
			shutil.rmtree(LOGS_name)
			shutil.copytree('LOGS',LOGS_name)
			break
		else:
			LOGS_name = input('input new LOGS name (please input format like \'LOGS_PDK\')\n')
			shutil.copytree('LOGS',LOGS_name)
			break
	else:
		shutil.copytree('LOGS',LOGS_name)
		break

'''
import matplotlib.pyplot as plt
plt.figure()
for j in range(5):
	plt.plot(RAD_array[j],TEM_array[j],label=str(T_HEAT[j]))
plt.xlabel('radius')
plt.ylabel('logT')
plt.title('M=1.133Mj,Z=0.04, Teq = 2066')
plt.legend()
plt.figure()
for j in range(5):
	plt.plot(RAD_array[j],GRADA_array[j],label=str(T_HEAT[j]))
plt.xlabel('radius')
plt.ylabel('grada')
plt.title('M=1.133Mj,Z=0.04, Teq = 2066')
plt.legend()

plt.figure()
for j in range(5):
	plt.plot(RAD_array[j],GD_array[j],label=str(T_HEAT[j]))
plt.xlabel('radius')
plt.ylabel('eta')
plt.title('M=1.133Mj,Z=0.04, Teq = 2066')
plt.legend()

plt.figure()
for j in range(5):
	plt.plot(RAD_array[j],MW_array[j],label=str(T_HEAT[j]))
plt.xlabel('radius')
plt.ylabel('mu')
plt.title('M=1.133Mj,Z=0.04, Teq = 2066')
plt.legend()

plt.show()
'''
