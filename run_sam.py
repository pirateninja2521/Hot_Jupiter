#use python3

import numpy as np
import shutil, sys, os, math
import mesa_reader as mr
from param_HD209458b import *
#from param_WASP_19b import *

def init_inlist2():
	os.system("cp inlist2_rlx_irrad inlist2")
	f = open('inlist2', 'r')
	g = f.read()
	f.close()

	print('initial mass = ',mass_initial,'jupiter mass')
	g = g.replace("<<initial_radius>>", str(radius_initial))
	g = g.replace("<<initial_mass>>", str(mass_initial*mjup))
	g = g.replace("<<z>>",str(z))
	g = g.replace("<<y>>",str(y))
	g = g.replace("<<smwtfname>>", '"rlx_irrad.mod"')
	h = open('inlist2', 'w')
	h.write(g)
	h.close()

def init_inlist3():
	os.system("cp inlist3_rlx_z inlist3")
	f = open('inlist3', 'r')
	g = f.read()
	f.close()
	g = g.replace("<<extraheat>>",str(extraheat))
	g = g.replace('<<rdprvsname>>','"rlx_irrad.mod"')
	g = g.replace("<<smwtfname>>", '"evolve.mod"')
	g = g.replace('<<new_z>>',str(new_z))
	g = g.replace("<<Teq>>", str(Teq) ) 
	g = g.replace("<<z_HELM>>", str(Z_all_HELM) ) 
	g = g.replace("<<P_surf>>", str(P_surf) ) 
	g = g.replace("<<kap_v>>", str(kappa_v) ) 
	h = open('inlist3', 'w')
	h.write(g)
	h.close()

def init_inlist4():
	os.system("cp inlist4_irrad inlist4")
	f = open('inlist4', 'r')
	g = f.read()
	f.close()
	g = g.replace("<<extraheat>>",str(extraheat))
	g = g.replace("<<rdprvsname>>",'"evolve.mod"')
	g = g.replace("<<smwtfname>>", '"rlx_z0.mod"')
	g = g.replace("<<Teq>>", str(Teq) ) 
	g = g.replace("<<z_HELM>>", str(P_surf) ) 
	g = g.replace("<<P_surf>>", str(P_surf) ) 
	g = g.replace("<<kap_v>>", str(kappa_v) ) 
	h = open('inlist4', 'w')
	h.write(g)
	h.close()

def run_inlist3_only():
	init_inlist3()
	os.system("cp inlist3 inlist")
	os.system("./rn1")
	exit()

#run_inlist3_only()


cnt=0 #Sam
ratio_arr=[]
loop=True
while loop:
	#os.system('rm -f *.mod')
	os.system('rm -f LOGS/*')
	cnt+=1
	print('total extraheat',total_extraheat)

	init_inlist2()
	init_inlist3()
	init_inlist4()
	os.system('./rn_sam')

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

	print(T_HEAT, "total extraheat array")
	print(T_radius,'radius_here')
	print("expect radius: ", expect_r)
	print("actual radius: ", rp)
	print("ratio: ", (expect_r-rp)/expect_r, "\n\n\n")
	if abs(expect_r-rp)/expect_r <= 0.01:
		loop = False
	else:
		Factor = 15 # if iteration not converge, choose a smaller Factor 
		extraheat *= (1+Factor*(expect_r-rp)/expect_r)#6e+27/mass_initial/mjup
		total_extraheat *= (1+Factor*(expect_r-rp)/expect_r)
	break

##########################################################################
### save LOGS file
##########################################################################

#ans = input('Successful? (1/0)\n')
ans = '0'
while ('1' == ans):
	if os.path.exists(LOGS_name):
		print('directory ' + LOGS_name + ' has existed. \n')
		ans2 = input('replace it (1/0)?\n')
		if 	('1' == ans2):
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
