#!/usr/bin/env python

import numpy as np
import os
import shutil
import time
import random
import sys

msun = 1.9892e33
rsun = 6.9598e10
mjup = 1.8986e30
rjup = 6.9911e9
mearth = 5.97e27
sigma=5.67e-5
au = 1.496e13

##
def relax_irrad_planet(inlist2,prevoius_model,savingmodel,minitial,rinitial,putincore,core_mass,core_density,y,z):
	start_time = time.time()
	print("relax irradiation planet \n---------------------------------------------------------------")
	for pp in range(5):
		print("\n---------------------------------------------------------------")
	f = open('inlist2_rlx_irrad', 'r')
	g = f.read()
	f.close()
	print('initial mass = ',minitial,'jupiter mass')
	g = g.replace("<<initial_radius>>", str(rinitial))
	g = g.replace("<<initial_mass>>", str(minitial*mjup))

	g = g.replace("<<putincore>>", str(putincore))
	g = g.replace("<<core_mass>>", str(core_mass))
	g = g.replace("<<core_density>>", str(core_density))

	g = g.replace("<<z>>",str(z))
	g = g.replace("<<y>>",str(y))
	g = g.replace("<<smwtfname>>", '"' + savingmodel + '"')
	g = g.replace('<<rdprvsname>>','"' + prevoius_model + '"')
	h = open(inlist2, 'w')
	h.write(g)
	h.close()
	shutil.copyfile(inlist2,"inlist") # rewrite inlist with inlist2 
	os.system('./star')
	run_time = time.time() - start_time
	print("run time for create_planets in sec=",run_time)
	return run_time

##

def relax_z_planet(Teq,inlist2,prevoius_model,savingmodel,new_z,extraheat,z_helm,P_surf,kappa_v):
	start_time = time.time()
	print("relax metallicity \n---------------------------------------------------------------")
	for pp in range(2):
		print("\n---------------------------------------------------------------")
	f = open('inlist3_rlx_z', 'r')
	g = f.read()
	f.close()
	g = g.replace("<<extraheat>>",str(extraheat))
	g = g.replace("<<smwtfname>>", '"' + savingmodel + '"')
	g = g.replace('<<rdprvsname>>','"' + prevoius_model + '"')
	g = g.replace('<<new_z>>',str(new_z))
	g = g.replace("<<Teq>>", str(Teq) ) 
	g = g.replace("<<z_HELM>>", str(z_helm) ) 
	g = g.replace("<<P_surf>>", str(P_surf) ) 
	g = g.replace("<<kap_v>>", str(kappa_v) ) 
	h = open(inlist2, 'w')
	h.write(g)
	h.close()
	shutil.copyfile(inlist2,"inlist") # rewrite inlist with inlist2
	os.system('./star')
	run_time = time.time() - start_time
	print("run time for create_planets in sec=",run_time)
	return run_time

##

def irrad_planet(Teq,inlist6,prevoius_model,savingmodel ,extraheat,z_helm,P_surf,kappa_v):
	start_time = time.time()
	#kappa_v=1/float(irrad_col)
	print("evolve planet\n------------------------------------------------------------------------------")
	for pp in range(2):
		print("\n---------------------------------------------------------------")
	f = open('inlist4_irrad', 'r')
	g = f.read()
	f.close()
	g = g.replace("<<extraheat>>",str(extraheat))
	g = g.replace("<<rdprvsname>>",'"' + prevoius_model  + '"')
	g = g.replace("<<smwtfname>>", '"' + savingmodel + '"')
	g = g.replace("<<Teq>>", str(Teq) ) 
	g = g.replace("<<z_HELM>>", str(z_helm) ) 
	g = g.replace("<<P_surf>>", str(P_surf) ) 
	g = g.replace("<<kap_v>>", str(kappa_v) ) 
	#g = g.replace("<<loadfile>>",'"' + removemod + '"')
	#g = g.replace("<<smwtfname>>", '"' + evolvemod2 + '"')
	#g = g.replace("<<irrad_col>>", str(irrad_col) )
	#g = g.replace("<<flux_dayside>>", str(flux_dayside) )
	#g = g.replace("<<maxage>>",str(maxage))
	#g = g.replace("<<initial_age>>",str(initialage))
		
	#    g = g.replace("<<knob>>", str(knob) ) 
	#g = g.replace("<<kappa_v>>", str(kappa_v) )
	#    g = g.replace("<<pl_param>>", str(random.uniform(0,2)) )
	#    g= g.replace("<<orbital_distance>>", str(orb_sep) )
	#    g= g.replace("<<historyName>>", str(orb_sep) +"_"+str(Rmp) +"_"+str(enFrac) + "_" + str(maxEntropy) )
	h = open(inlist6, 'w')
	h.write(g)
	h.close()
	shutil.copyfile(inlist6,"inlist")
	os.system('./star')
	run_time = time.time() - start_time
	print("run time to evolve in sec=",run_time)
	return run_time
	
###############################
def irrad_planet2(Teq,inlist7,prevoius_model,extraheat,z_helm,P_surf,kappa_v):
	start_time = time.time()
	#kappa_v=1/float(irrad_col)
	print("evolve planet for second step\n------------------------------------------------------------------------------")
	for pp in range(2):
		print("\n---------------------------------------------------------------")
	f = open('inlist5_irrad2', 'r')
	g = f.read()
	f.close()
	g = g.replace("<<extraheat>>",str(extraheat))
	g = g.replace("<<rdprvsname>>",'"' + prevoius_model + '"')
	g = g.replace("<<Teq>>", str(Teq)) 
	g = g.replace("<<z_HELM>>", str(z_helm) ) 
	g = g.replace("<<P_surf>>", str(P_surf) ) 
	g = g.replace("<<kap_v>>", str(kappa_v) ) 

	h = open(inlist7, 'w')
	h.write(g)
	h.close()
	shutil.copyfile(inlist7,"inlist")
	try:
		os.system('./star')
	except:
		print('error')
		exit(-1)
	run_time = time.time() - start_time
	print("run time to evolve in sec=",run_time)
	return run_time

