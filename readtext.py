import mesa_reader as mr
import numpy as np
import sys
from scipy import special
import scipy

#star age vs. radius
use_DIR_1 = True
## need to be modified manually
##########################################
#DIR_1 = 'LOGS_HD209458'
DIR_1 = 'LOGS_SAM'
#DIR_1 = 'LOGS_XO1b'
#DIR_1 = 'LOGS_CCC'
#DIR_1 = 'LOGS_HD_P=12d5_new_extend5_fixed'
#DIR_1 = 'LOGS_HDP=35d4_extend5_kap=3dm2_fixed'
kappa_v = float(eval(input('input kappa_v \n'))) #kappa_v = 4e-3
T_eq = float(eval(input('input Teq \n'))) #1447
P_up_lim = 1e-1
##########################################
#star age vs. radius
#m = mr.MesaData(DIR_1 + '/history.data')
if use_DIR_1:
	m = mr.MesaData(DIR_1 + '/history.data')
	g = mr.MesaLogDir(DIR_1)
else:
	m = mr.MesaData()
	g = mr.MesaLogDir()
star_age = m.data('star_age')
radius_h = m.data('radius')
final_radius = radius_h[len(radius_h)-1]
luminosity_h = m.data('luminosity')
Teff_h = m.data('effective_T')
mass_h = m.data('star_mass')
fp = open("writedata.txt", "a")        #open file and append data



print(('radius=',radius_h[-1]))
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig    


#initial age changed
j = 0
while (star_age[j]<1e6):
	j = j+1

#######################################################
## profile columns variables
p = g.profile_data()
pressure_p = p.data('pressure')
radius_p = p.data('radius')
brunt_N2 = p.data('brunt_N2')
#brunt_B = p.data('brunt_B')
mass_coor_p = p.data('mass')
logRho_p = p.data('logRho')
logT_p = p.data('logT')
T_p = p.data('temperature')
Gamma1_p = p.data('gamma1')
Gamma3_p = p.data('gamma3')
opacity_p = p.data('opacity')
optical_p = p.data('tau')
grav_p = p.data('grav')
#atom_M_p = p.data('abar')
mole_mass = p.data('mu')

entropy_p = p.data('entropy')
log_irrad_heat_p = p.data('log_irradiation_heat')
#luminosity = p.data('luminosity')
luminosity_rad_p = p.data('luminosity')*p.data('lum_rad_div_L')

dkap_dlnrho_p = p.data('dkap_dlnrho_face')
dkap_dlnT_p = p.data('dkap_dlnT_face')

chiRho_p = p.data('chiRho')
chiT_p   = p.data('chiT')

cv_p = p.data('cv')
cp_p = p.data('cp')

eta_p = p.data('eta')
print(eta_p)

## check Pgas ~= pressure
gas_approx = p.data('pgas_div_ptotal')
if min(gas_approx) < 0.999:
	print('not a good approxmation pressure ~= P_gas')

### can't apply 
# dlnR_p = p.data('dlnR') 
# dlnRho_p = p.data('dlnd')
# dlnT_p  = p.data('dlnT')

########################################################

#print(luminosity_p,'luminosity')
#print(Teff_h[-1],'Teff')
#print('Lumino',luminosity_p[0])
L_sun = 3.839e33
R_sun = 6.955e10
M_sun=1.989e33
sigma=5.67e-5
## calculate T_intrinsic
#print(np.power(luminosity_p[0]*L_sun/sigma/4/3.1416/(radius_p[0]*R_sun)**2,0.25) ,'Tint')
print((grav_p[0],'grav'))
print((kappa_v/opacity_p[0],'gamma'))

#print(dr_p)
internal_f_p = luminosity_rad_p*L_sun/(4*np.pi*(radius_p*R_sun)**2)
#########################################################
from scipy import special
E_2 = special.expn(2,optical_p*0.4)
###############################################################
#find the number which brunt_N2=0
index_intersec = []
for n in range (1,len(brunt_N2)):
	if(brunt_N2[n]*brunt_N2[n-1] < 0):
		index_intersec.append(n)
if len(index_intersec) != 1:
	print('Two or more intersection')
# find tau = 2/3
o = 0
while optical_p[o] < 2./3:
	o = o+1

#print(kappa)
#while ((pressure_p[m]- 2*gravity_p[m]/kappa[m]/3) < 0):
#	m = m + 1
#print (pressure_p[m])
#print(gravity_p,kappa)

#profile, pressure vs. radius

plt.figure(figsize=(10,10))
plt.subplot(2,2,4)

plt.plot(radius_p, np.log10(pressure_p))
plt.plot(radius_p[index_intersec[0]], np.log10(pressure_p)[index_intersec[0]],'bo')
plt.xlabel('radius')
plt.ylabel('log(pressure)')
#filename_b = filename_a + '_age=4.5d9'  #name with age
#plt.title(filename_b)
plt.title(DIR_1+'_profile')

#test = mr.MesaData('python_history/history_test__m=1_T=1260_ini_r=3.7_Z=0.0134_nocore_plmtchange.data')
#test = mr.MesaData('LOGS__aa/history.data')

#profile, density vs. radius

plt.subplot(2,2,3)
plt.plot(radius_p, logRho_p)
plt.plot(radius_p[index_intersec[0]], logRho_p[index_intersec[0]],'bo')
plt.xlabel('radius')
plt.ylabel('log(density)')

#profile, pressure vs. temperature
plt.subplot(2,2,2)
plt.plot(np.log10(pressure_p), logT_p)
plt.plot(np.log10(pressure_p)[index_intersec[0]], logT_p[index_intersec[0]],'bo')
plt.xlabel('log(pressure)')
plt.ylabel('log(T)')

#profile, radius vs. pressure
#plt.title(filename_a)
plt.subplot(2,2,1)
plt.plot(radius_p, logT_p)
plt.plot(radius_p[index_intersec[0]], logT_p[index_intersec[0]],'bo')
plt.xlabel('radius')
plt.ylabel('log(T)')


plt.figure()
plt.plot(radius_p, grav_p)
#plt.plot(radius_p[index_intersec[0]], np.log10(pressure_p)[index_intersec[0]],'bo')
plt.xlabel('radius')
plt.ylabel('grav')
plt.title(DIR_1+'_r vs grav')
#print('total mass',m.data('star_mass')[0])


#profile, radius vs. opacity
'''
plt.figure()
plt.plot(radius_p,brunt_B,'o')
#plt.plot(radius_p[index_intersec[0]],opacity_p[index_intersec[0]],'bo')
plt.xlabel('radius')
plt.ylabel('')
plt.title('aaa')
print(pressure_p[0])
T_atm = T_p[0]
plt.figure()
plt.plot(radius_p,pressure_p,'o')
#plt.plot(radius_p[index_intersec[0]],opacity_p[index_intersec[0]],'bo')
plt.xlabel('radius')
plt.ylabel('')
plt.title('aaa')

logP_atm = np.arange(3.7, 6.1, 0.1)

rho_atm = np.power(10,logP_atm)*2.7/(8.314*10**7)/T_atm
#print(logP_atm[-1],'rho_atm')
plt.plot(logP_atm,rho_atm*np.exp(-4e-3*np.power(10,logP_atm)/grav_p[0]),label='T='+str(round(T_atm,0))+'K as ideal gas')
plt.legend()
'''
#print(m.data('surface_optical_depth'))

#profile, radius vs. opacity log plot

#plt.xlim(star_age.min() * 6)
#path_b = 'python_profile/' + filename_b + '.jpg'
#plt.savefig(path_b)

#profile, radius vs. entropy


'''
plt.figure()
plt.plot(radius_p,entropy_p)
plt.plot(radius_p[index_intersec[0]],entropy_p[index_intersec[0]],'bo')
plt.xlabel('radius')
plt.ylabel('entropy')
plt.title('entropy vs. radius')

plt.figure()

plt.plot(radius_p,log_irrad_heat_p)
plt.plot(radius_p[index_intersec[0]],(log_irrad_heat_p)[index_intersec[0]],'bo',label='interface')
plt.xlabel('radius')
plt.ylabel('log irradiaiton')
plt.legend()

plt.figure()
plt.plot(radius_p,((optical_p*grav_p/pressure_p/opacity_p)))
plt.plot(radius_p[index_intersec[0]],((optical_p*grav_p/pressure_p/opacity_p))[index_intersec[0]],'bo',label='interface')
plt.xlabel('radius')
plt.ylabel('kappa_th')
plt.legend()

plt.figure()
plt.plot(radius_p,E_2)
plt.plot(radius_p[index_intersec[0]],E_2[index_intersec[0]],'bo',label='interface')
plt.xlabel('radius')
plt.ylabel('E_2')
plt.legend()
'''
#plt.close('all')
################################################################################################################3
## output txt file
       #open file and append data
f_prmt = open('txtfile/' + DIR_1 + ".txt", "w") 
Rho_p = np.power(10,logRho_p)

dRho_dp = np.ones(len(Rho_p))
dRho_dp[1:-1] = ( (Rho_p[2:]-Rho_p[1:-1])*(pressure_p[1:-1]-pressure_p[:-2])/(pressure_p[2:]-pressure_p[1:-1])+(Rho_p[1:-1]-Rho_p[:-2])*(pressure_p[2:]-pressure_p[1:-1])/(pressure_p[1:-1]-pressure_p[:-2]) )/(pressure_p[2:]-pressure_p[:-2])
dRho_dp[0] = (Rho_p[1]-Rho_p[0])/(pressure_p[1]-pressure_p[0])
dRho_dp[-1]= (Rho_p[-1] - Rho_p[-2])/((pressure_p[-1]-pressure_p[-2]))

lnT_p = np.log(T_p)

dr_dlnT = np.ones(len(lnT_p))
dr_dlnT[1:-1] = ( (radius_p[2:]-radius_p[1:-1])*(lnT_p[1:-1]-lnT_p[:-2])/(lnT_p[2:]-lnT_p[1:-1])+(radius_p[1:-1]-radius_p[:-2])*(lnT_p[2:]-lnT_p[1:-1])/(lnT_p[1:-1]-lnT_p[:-2]) )/(lnT_p[2:]-lnT_p[:-2])
dr_dlnT[0] = (radius_p[1]-radius_p[0])/(lnT_p[1]-lnT_p[0])
dr_dlnT[-1]= (radius_p[-1] - radius_p[-2])/((lnT_p[-1]-lnT_p[-2]))
#print((Rho_p[2:]-Rho_p[:-2]))
#print(len(dRho_dp),len(Rho_p))
#print(np.array([1.,2.,3.])/np.array([2,2,2]))


##########################3#######################################################################################
##  Build outside profile with isothermal assumpion 
##  use eq(49) in Guillot 2011

def TP_Guillot(pressure,gam,k_ir,T_int,T_eq,g):

    tau = k_ir * pressure / g
    term1 = 3./4* T_int**4 * (2./3 + tau)
    term2 = 2./3. + 2./(3*gam)*(1 + (0.5*tau*gam - 1)*np.exp(-tau*gam))
    term3 = 2.*gam/3 * (1-tau**2*0.5) * scipy.special.expn(2,tau*gam)

    T_4 = term1 + 3./4*T_eq**4 * ( term2 + term3)

    return  T_4**0.25

### some constant and set
#atom_M = 2.7#atom_M_p[0] ## atomic mass
atm_M = mole_mass[0]
atm_g = grav_p[0] ## using the surface gravity as constant

### Parameter using in eq(49)
k_ir = opacity_p[0]
gam = kappa_v/k_ir
T_int = np.power(luminosity_rad_p*L_sun/sigma/4/3.1416/(radius_p*R_sun)**2,0.25)
g = grav_p[0]


'''
M_p = 1.310034e+30
print(kappa_v*(sigma*1447**4/4+0*pressure_p[1:10]),'AAA')
print((mass_coor_p[2:11]-mass_coor_p[1:10])*M_sun/4/np.pi/(radius_p[1:10]*R_sun)**2)
print((luminosity_rad_p[1:10] - luminosity_rad_p[2:11])*L_sun,'LLL')
#print(mass_coor_p[2:11]-mass_coor_p[1:10])
plt.plot(pressure_p[1:10],(luminosity_rad_p[1:10] - luminosity_rad_p[2:11])*L_sun,'r')
#plt.plot(pressure_p[1:10],opacity_p[1:10]*(sigma*T_p[1:10]**4+0*pressure_p[1:10])*(mass_coor_p[2:11]-mass_coor_p[1:10])*M_p,'b')
plt.plot(pressure_p[1:10],kappa_v*(sigma*1447**4/4+0*pressure_p[1:10])*(mass_coor_p[2:11]-mass_coor_p[1:10])*M_sun,'m')
'''
################################
plt.figure()
plt.plot(np.log10(pressure_p),np.power(10,logRho_p)*np.exp(-kappa_v*pressure_p/grav_p),'r',label='MESA')
#plt.plot(radius_p[index_intersec[0]],opacity_p[index_intersec[0]],'bo')
plt.xlabel('pressure')
plt.ylabel('')
plt.title(DIR_1+'\n irradiation heating ')
print((pressure_p[0]))
T_atm = T_p[0]

logP_atm = np.arange(np.log10(P_up_lim), np.log10(pressure_p[0])+0.1, 0.1)

rho_atm = np.power(10,logP_atm)*2.7/(8.314*10**7)/T_atm
#print(logP_atm[-1],'rho_atm')
plt.plot(logP_atm,rho_atm*np.exp(-kappa_v*np.power(10,logP_atm)/grav_p[0]),label='atmosphere')
plt.legend()
#print(m.data('surface_optical_depth'))
#################

#atm_T = T_p[0] ## using surface temperature as atmospheric temperature
bot_atm_P = pressure_p[0] ## using surface pressure as bottom of atmospheric pressure
bot_atm_rad = radius_p[0] ##using surface radius as bottom of atmospheric radius
#c_gas = 8.314*10**7 ## ideal gas constant in cgs unit
c_gas  = pressure_p[0]*mole_mass[0]/np.power(10,logRho_p[0])/T_p[0] ## using MESA table ideal gas constant 
#######
atm_GN  	 = 3000 # atm grid point
#atm_pressure = bot_atm_P - np.power(10, np.linspace(0,np.log10(bot_atm_P-P_up_lim),atm_GN) )+1 ## about from 10^4~10^6
power_for_grid = 0.9 # use to redistribute pressure grid point
atm_pressure = np.power(np.linspace(np.power(bot_atm_P,power_for_grid),np.power(P_up_lim,0.8),atm_GN),1/power_for_grid)
atm_T        = TP_Guillot(atm_pressure,gam,k_ir,T_int[0],T_eq,g)
atm_rho      = atm_pressure*atm_M/(c_gas)/(atm_T) ## use ideal gas and const temperature law, molecular weight=2.7
#atm_radius   = bot_atm_rad + ((c_gas)*(atm_T)/atm_M/atm_g)*np.log(bot_atm_P/atm_pressure)/R_sun ## isothermal atmosphere radius in R_sun unit 
atm_radius   = np.linspace(0,0,atm_GN)
atm_radius[0]= bot_atm_rad
for j in range(1,len(atm_pressure)):
	atm_radius[j] = atm_radius[j-1] + c_gas/atm_M/atm_g*(atm_T[j]+atm_T[j-1])*(atm_pressure[j-1]-atm_pressure[j])/(atm_pressure[j]+atm_pressure[j-1])/R_sun
#atm_radius   = bot_atm_rad + ((c_gas)*(atm_T)/atm_M/atm_g)*np.log(bot_atm_P/atm_pressure)/R_sun ## isothermal atmosphere radius in R_sun unit
####### mass coordinate calculation
MGRT = atm_M*atm_g/c_gas/atm_T
RADI = (atm_radius-bot_atm_rad)*R_sun
atm_mass_coor= mass_coor_p[0]\
-4*3.14159*atm_rho[0]*np.exp(-MGRT*RADI)*(MGRT**2*RADI**2+2*MGRT*RADI+2)/MGRT**3/M_sun \
+4*3.14159*atm_rho[0]*np.exp(-MGRT*RADI[0])*(MGRT**2*RADI[0]**2+2*MGRT*RADI[0]+2)/MGRT**3/M_sun 
#### below is constant
#atm_dRho_dp  = atm_M/c_gas/atm_T 
#atm_N2       = atm_g*atm_g*atm_M/c_gas/atm_T*(1-1/Gamma1_p[0])  ## simplified by ideal gas law
###### new parameter
atm_dRho_dp = np.ones(len(atm_rho))
atm_dRho_dp[1:-1] = ( (atm_rho[2:]-atm_rho[1:-1])*(atm_pressure[1:-1]-atm_pressure[:-2])/(atm_pressure[2:]-atm_pressure[1:-1])+(atm_rho[1:-1]-atm_rho[:-2])*(atm_pressure[2:]-atm_pressure[1:-1])/(atm_pressure[1:-1]-atm_pressure[:-2]) )/(atm_pressure[2:]-atm_pressure[:-2])
atm_dRho_dp[0] = (atm_rho[1]-atm_rho[0])/(atm_pressure[1]-atm_pressure[0])
atm_dRho_dp[-1]= (atm_rho[-1] - atm_rho[-2])/((atm_pressure[-1]-atm_pressure[-2]))

atm_dT_dp = np.ones(len(atm_T))
atm_dT_dp[1:-1] = ( (atm_T[2:]-atm_T[1:-1])*(atm_pressure[1:-1]-atm_pressure[:-2])/(atm_pressure[2:]-atm_pressure[1:-1])+(atm_T[1:-1]-atm_T[:-2])*(atm_pressure[2:]-atm_pressure[1:-1])/(atm_pressure[1:-1]-atm_pressure[:-2]) )/(atm_pressure[2:]-atm_pressure[:-2])
atm_dT_dp[0] = (atm_T[1]-atm_T[0])/(atm_pressure[1]-atm_pressure[0])
atm_dT_dp[-1]= (atm_T[-1] - atm_T[-2])/((atm_pressure[-1]-atm_pressure[-2]))

atm_N2       = atm_g*atm_g*(atm_M/c_gas/atm_T*(1-1/Gamma1_p[0])-atm_rho*atm_dT_dp/atm_T)  ## simplified by ideal gas law

atm_gamma1   = Gamma1_p[0] #cp_p[0]/cv_p[0]
atm_gamma3   = Gamma3_p[0] #cp_p[0]/cv_p[0]
atm_kappaT = dkap_dlnT_p[0] # use the surface value of MESA output
atm_kappaRho = dkap_dlnrho_p[0] # use the surface value of MESA output
atm_dr_dlnT = 0 # it's actually 1/0
atm_dlnRho_dlnp = 1 # pressure is propotional to density
atm_dlnRho_dlnT = -1 # pressure is inversely propotional to density


#print(atm_pressure,pressure_p[0])
#print(atm_rho,np.power(10,logRho_p[0:5]))
#print (brunt_N2[0:5],atm_N2)
#print (mole_mass[0]*grav_p[0]/c_gas/T_p[0],(np.log10(pressure_p[0:5])-np.log10(pressure_p[1:6]))/(radius_p[0:5]-radius_p[1:6])/R_sun)
#print(pressure_p[0]*mole_mass[0]/np.power(10,logRho_p[0])/T_p[0],'aa')
plt.figure()
plt.plot(np.append(atm_radius,radius_p[0:5]),np.log10(np.append(atm_pressure,pressure_p[0:5])),'ro')
#plt.figure()
#plt.plot(np.log10(pressure_p),brunt_N2)

### plot P-T profile
temp = TP_Guillot(atm_pressure,gam,k_ir,T_int[0],T_eq,g)
plt.figure()
p_bar = atm_pressure/1.e6
plt.semilogy(temp,p_bar,label='atmosphere ($\gamma=$'+str(round(gam,4))+')')
plt.semilogy(T_p[1:60],pressure_p[1:60]/1e6,'r',label='MESA')
#plt.semilogy(T_int[0:300],pressure_p[0:300]/1.e6,label='$T_{int}$')
plt.xlabel('T (K)')
plt.ylabel('P (bar)')
plt.legend()
plt.gca().invert_yaxis()
plt.title(DIR_1 + '_$\kappa_{th}=$'+str(round(opacity_p[0],4)))

##############################
## Energy plot


us = 1/1.73
fH = 0.5
fK = 1./3
gamma = kappa_v/opacity_p[0]
tau   = optical_p[0]-opacity_p[0]*(bot_atm_P-atm_pressure)/atm_g

Hv0 = -us*sigma*1447**4/np.pi
H   = luminosity_rad_p[0]*L_sun/(4*np.pi*radius_p[0]*R_sun)**2
Jv = -Hv0/us*np.exp(-gamma*tau/us)
B  = H*(1/fH+tau/fK) - Hv0*(1/fH+us/gamma/fK+(gamma/us-us/gamma/fK)*np.exp(-gamma*tau/us))

Jth = (H-Hv0)/fH + H*tau/fK + -Hv0*us/gamma/fK*(1-np.exp(-tau/us))
Jth1= (H-Hv0)/fH + H*tau/fK
Jth2= -Hv0*us/gamma/fK*(1-np.exp(-tau/us))
print(tau)
print((Jv,H,Hv0,'AAAA'))
print((Jv,B,'HB'))
plt.figure()
plt.text(1e-5, 2e5, '$\kappa_{th}=$'+str(round(opacity_p[0],5)))
plt.text(1e-5, 1e5, '$\kappa_v=$'+str(round(kappa_v,3)))
plt.title(DIR_1+'\n atmosphere energy plot')
plt.loglog(atm_pressure/1e6,Jv,'r',label='$J_v$')
plt.loglog(atm_pressure/1e6,B,'b',label='B')
plt.loglog(atm_pressure/1e6,B-gamma*Jv,'m',label='$J_{th}$')
#plt.loglog(atm_pressure/1e6,Jth,'y',label='Jth')
plt.loglog(atm_pressure/1e6,Jth1,'y-.',label='Jth1')
plt.loglog(atm_pressure/1e6,Jth2,'y--',label='Jth2')

plt.legend()
plt.xlabel('P (bar)')

plt.figure()
plt.text(1e-3, 2e5, '$\kappa_{th}=$'+str(round(opacity_p[0],5)))
plt.text(1e-3, 1e5, '$\kappa_v=$'+str(round(kappa_v,3)))
plt.title(DIR_1+'\n atmosphere energy plot')
plt.loglog(tau,Jv,'r',label='$J_v$')
plt.loglog(tau,B,'b',label='B')
plt.loglog(tau,B-gamma*Jv,'m',label='$J_{th}$')
#plt.loglog(tau,Jth,'y',label='Jth')
plt.loglog(tau,Jth1,'y-.',label='Jth1')
plt.loglog(tau,Jth2,'y--',label='Jth2')
plt.legend()
plt.xlabel('optical depth')


##################################################################################################################
## output data to file
print(DIR_1, '\n', 'radius \n', file=f_prmt)
for j in range(len(radius_p)):
	#print >>f_prmt, '{:e}'.format(radius_p[j]) 
	print(radius_p[j], file=f_prmt)

print('\n', file=f_prmt)     
  
print('-------------------------------------------------------------------------\n', file=f_prmt)
print('mass_coordinate \n', file=f_prmt)
for j in range(len(mass_coor_p)):
	print(mass_coor_p[j], file=f_prmt) 
print('\n', file=f_prmt) 

print('-------------------------------------------------------------------------\n', file=f_prmt)
print('pressure \n', file=f_prmt)
for j in range(len(pressure_p)):
	print(pressure_p[j], file=f_prmt)
print('\n', file=f_prmt) 

print('-------------------------------------------------------------------------\n', file=f_prmt)
print('density \n', file=f_prmt)
for j in range(len(np.power(10,logRho_p))):
	print(np.power(10,logRho_p)[j], file=f_prmt) 
print('\n', file=f_prmt) 

print('-------------------------------------------------------------------------\n', file=f_prmt)
print('dRho/dp \n', file=f_prmt)
for j in range(len(dRho_dp)):
	print(dRho_dp[j], file=f_prmt)
print('\n', file=f_prmt) 
 
print('-------------------------------------------------------------------------\n', file=f_prmt)
print('N^2 \n', file=f_prmt)
for j in range(len(brunt_N2)):
	print(brunt_N2[j], file=f_prmt) 
print('\n', file=f_prmt) 

print('-------------------------------------------------------------------------\n', file=f_prmt)
print('Gamma1 \n', file=f_prmt)
for j in range(len(Gamma1_p)):
	print(Gamma1_p[j], file=f_prmt)
print('\n', file=f_prmt) 
  
print('-------------------------------------------------------------------------\n', file=f_prmt)
print('kappa_th = tau g/P \n', file=f_prmt)
for j in range(len(optical_p*grav_p/pressure_p)):
	print(((optical_p*grav_p/pressure_p))[j], file=f_prmt)
print('\n', file=f_prmt) 

print('-------------------------------------------------------------------------\n', file=f_prmt)
print('opacity', file=f_prmt)
for j in range(len(opacity_p)):
	print(opacity_p[j], file=f_prmt)
print('\n', file=f_prmt) 

print('-------------------------------------------------------------------------\n', file=f_prmt)
print('grav', file=f_prmt)
for j in range(len(grav_p)):
	print((grav_p)[j], file=f_prmt)
print('\n', file=f_prmt) 

print('-------------------------------------------------------------------------\n', file=f_prmt)
print('temperature', file=f_prmt)
for j in range(len(T_p)):
	print((T_p)[j], file=f_prmt)
print('\n', file=f_prmt) 

print('-------------------------------------------------------------------------\n', file=f_prmt)
print('optical depth', file=f_prmt)
for j in range(len(optical_p)):
	print((optical_p)[j], file=f_prmt)
print('\n', file=f_prmt) 

print('-------------------------------------------------------------------------\n', file=f_prmt)
print('E_2 \n', file=f_prmt)
for j in range(len(E_2)):
	print((E_2)[j], file=f_prmt)
print('\n', file=f_prmt) 

print('-------------------------------------------------------------------------\n', file=f_prmt)
print('eta = Rp*pressure/internal flux \n', file=f_prmt)
for j in range(len(internal_f_p)):
	print(((radius_p[0]*R_sun*pressure_p/internal_f_p))[j], file=f_prmt)
print('\n', file=f_prmt) 

print('-------------------------------------------------------------------------\n', file=f_prmt)
print('Gamma3 \n', file=f_prmt)
for j in range(len(Gamma3_p)):
	print(Gamma3_p[j], file=f_prmt)
print('\n', file=f_prmt) 

print('-------------------------------------------------------------------------\n', file=f_prmt)
print('kappa_rho \n', file=f_prmt)
for j in range(len(dkap_dlnrho_p/opacity_p)):
	print((dkap_dlnrho_p/opacity_p)[j], file=f_prmt)
print('\n', file=f_prmt) 

print('-------------------------------------------------------------------------\n', file=f_prmt)
print('kappa_T \n', file=f_prmt)
for j in range(len(dkap_dlnT_p/opacity_p)):
	print((dkap_dlnT_p/opacity_p)[j], file=f_prmt)
print('\n', file=f_prmt) 

print('-------------------------------------------------------------------------\n', file=f_prmt)
print('dR_dlnT_total \n', file=f_prmt)
for j in range(len(dr_dlnT*R_sun)):
	print((dr_dlnT*R_sun)[j], file=f_prmt)
print('\n', file=f_prmt) 

print((-16*5.67e-5*T_p[100]**4/3/opacity_p[100]/np.power(10,logRho_p[100])/(dr_dlnT[100]*R_sun),'F value'))

print('-------------------------------------------------------------------------\n', file=f_prmt)
print('dlnRho_dlnP_const_T \n', file=f_prmt)
for j in range(len(1/chiRho_p)):
	print((1/chiRho_p)[j], file=f_prmt)
print('\n', file=f_prmt) 

print('-------------------------------------------------------------------------\n', file=f_prmt)
print('dlnRho_dlnT_const_P \n', file=f_prmt)
for j in range(len(-chiT_p/chiRho_p)):
	print((-chiT_p/chiRho_p)[j], file=f_prmt)
print('\n', file=f_prmt) 

print('-------------------------------------------------------------------------\n', file=f_prmt)
print('radiation_flux \n', file=f_prmt)
for j in range(len(luminosity_rad_p*L_sun/4/3.1416/(radius_p*R_sun)**2)):
	print((luminosity_rad_p*L_sun/4/3.1416/(radius_p*R_sun)**2)[j], file=f_prmt)
print('\n', file=f_prmt)

print('=========================================================================\n', file=f_prmt)
print('Below is for atmosphere \n', file=f_prmt)
print('\n', file=f_prmt) 

print('-------------------------------------------------------------------------\n', file=f_prmt)
print('atm_radius \n', file=f_prmt)
for j in range(len(atm_radius)):
	print(atm_radius[j], file=f_prmt)
print('\n', file=f_prmt)

print('-------------------------------------------------------------------------\n', file=f_prmt)
print('atm_mass_coor \n', file=f_prmt)
for j in range(len(atm_mass_coor)):
	print(atm_mass_coor[j], file=f_prmt)
print('\n', file=f_prmt)

print('-------------------------------------------------------------------------\n', file=f_prmt)
print('atm_pressure \n', file=f_prmt)
for j in range(len(atm_pressure)):
	print(atm_pressure[j], file=f_prmt)
print('\n', file=f_prmt) 

print('-------------------------------------------------------------------------\n', file=f_prmt)
print('atm_rho \n', file=f_prmt)
for j in range(len(atm_rho)):
	print(atm_rho[j], file=f_prmt) 
print('\n', file=f_prmt) 

print('-------------------------------------------------------------------------\n', file=f_prmt)
print('atm_T \n', file=f_prmt)
for j in range(len(atm_T)):
	print(atm_T[j], file=f_prmt) 
print('\n', file=f_prmt) 

print('-------------------------------------------------------------------------\n', file=f_prmt)
print('atm_dRho/dp \n', file=f_prmt)
for j in range(len(atm_dRho_dp)):
	print(atm_dRho_dp[j], file=f_prmt) 
print('\n', file=f_prmt) 

print('-------------------------------------------------------------------------\n', file=f_prmt)
print('atm_N^2 \n', file=f_prmt)
for j in range(len(atm_N2)):
	print(atm_N2[j], file=f_prmt) 
print('\n', file=f_prmt)

##############################################################################################
### below is constant

print('-------------------------------------------------------------------------\n', file=f_prmt)
print('atm_opacity \n', file=f_prmt)
print(opacity_p[0], file=f_prmt)
print('\n', file=f_prmt) 

print('-------------------------------------------------------------------------\n', file=f_prmt)
print('atm_gamma1 \n', file=f_prmt)
print(atm_gamma1, file=f_prmt)
print('\n', file=f_prmt)

print('-------------------------------------------------------------------------\n', file=f_prmt)
print('atm_gamma3 \n', file=f_prmt)
print(atm_gamma3, file=f_prmt)
print('\n', file=f_prmt)

print('-------------------------------------------------------------------------\n', file=f_prmt)
print('atm_kappaT \n', file=f_prmt)
print(atm_kappaT, file=f_prmt)
print('\n', file=f_prmt)

print('-------------------------------------------------------------------------\n', file=f_prmt)
print('atm_kappaRho \n', file=f_prmt)
print(atm_kappaRho, file=f_prmt)
print('\n', file=f_prmt)

print('-------------------------------------------------------------------------\n', file=f_prmt)
print('atm_dr_dlnT \n', file=f_prmt)
print(atm_dr_dlnT, file=f_prmt)
print('\n', file=f_prmt)

print('-------------------------------------------------------------------------\n', file=f_prmt)
print('atm_dlnRho_dlnp \n', file=f_prmt)
print(atm_dlnRho_dlnp, file=f_prmt)
print('\n', file=f_prmt)

print('-------------------------------------------------------------------------\n', file=f_prmt)
print('atm_dlnRho_dlnT \n', file=f_prmt)
print(atm_dlnRho_dlnT, file=f_prmt)
print('\n', file=f_prmt)

'''
#### below is constant
atm_dRho_dp  = atm_M/c_gas/atm_T 
atm_N2       = atm_g*atm_g*atm_M/c_gas/atm_T*(1-1/Gamma1_p[0])  ## simplified by ideal gas law
atm_gamma1   = cp_p[0]/cv_p[0]
atm_gamma3   = cp_p[0]/cv_p[0]
atm_kappaT = dkap_dlnT_p[0] # use the surface value of MESA output
atm_kappaRho = dkap_dlnrho_p[0] # use the surface value of MESA output
atm_dr_dlnT = 0 # it's actually 1/0
atm_dlnRho_dlnp = 1 # pressure is propotional to density
atm_dlnRho_dlnT = -1 # pressure is inversely propotional to density

'''

print('-------------------------------------------------------------------------\n', file=f_prmt)
print('index of optical_depth = 2/3 ==> surface\n', file=f_prmt)

print(o, file=f_prmt)
print('\n', file=f_prmt)  
print('-------------------------------------------------------------------------\n', file=f_prmt)




#plt.interactive(True)

'''
p = g.profile_data()
pressure_p = p.data('pressure')
radius_p = p.data('radius')
brunt_N2 = p.data('brunt_N2')
mass_coor_p = p.data('mass')
logRho_p = p.data('logRho')
logT_p = p.data('logT')
m  
p  
rho  
d(rho)/dp  
N^2  
Gamma_1  Gamma_2
'''

fp.close()
f_prmt.close()
#plt.show()
####################################################################
# PT_Guillot
print((grav_p[0],'grav'))
print((kappa_v/opacity_p[0],'gamma'))

#(49) T. Guillot 2010
'''
def TP_Guillot(pressure,gam,k_ir,T_int,T_eq,g):

    tau = k_ir * pressure / g
    term1 = 3./4* T_int**4 * (2./3 + tau)
    term2 = 2./3. + 2./(3*gam)*(1 + (0.5*tau*gam - 1)*np.exp(-tau*gam))
    term3 = 2.*gam/3 * (1-tau**2*0.5) * scipy.special.expn(2,tau*gam)

    T_4 = term1 + 3./4*T_eq**4 * ( term2 + term3)

    return  T_4**0.25


log_pressure = np.arange(-3, 3, 0.1)
pressure = np.power(10,log_pressure)*1.e6
k_ir = opacity_p[0]

gam = 4e-3/k_ir
T_int = np.power(luminosity_rad_p*L_sun/sigma/4/3.1416/(radius_p*R_sun)**2,0.25)
T_eq = 1447.
g = grav_p[0]
temp = TP_Guillot(pressure,gam,k_ir,T_int[0],T_eq,g)
print(TP_Guillot(1.e4,gam,k_ir,T_int[0],T_eq,g))
p_bar = pressure/1.e6
# plot P-T profile
plt.figure()
plt.semilogy(temp,p_bar,label='$\gamma=$'+str(round(gam,4)))
plt.semilogy(T_int[0:300],pressure_p[0:300]/1.e6,label='$T_{int}$')
plt.xlabel('T (K)')
plt.ylabel('P (bar)')
plt.legend()
plt.gca().invert_yaxis()
plt.title(DIR_1 + '_$\kappa_{th}=$'+str(round(opacity_p[0],4)))
'''

# plot tau-T profile
'''
plt.figure()
tau = pressure * k_ir / g
plt.semilogy(temp,tau,label='$\gamma=$'+str(round(gam,4)))
plt.xlabel('T (K)')
plt.ylabel(r'$\tau_{th}$')
plt.legend()
plt.gca().invert_yaxis()
plt.title(DIR_1 + '_$kappa table$')

'''

plt.show()
#  plot P(and thus tau)-T profile

#modified by sam
tau = atm_pressure * k_ir / g
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.semilogy(temp,p_bar,label='$\gamma=5$')
ax2.semilogy(temp,tau,label='$\gamma=5$')
ax1.set_xlabel('T (K)')
ax1.set_ylabel('P (bar)')
ax2.set_ylabel(r'$\tau_{th}$')
#plt.xlabel('T (K)')
#plt.ylabel(r'$\tau_{th}$')
ax1.legend()
#plt.gca().invert_yaxis()
ax1.set_ylim(ax1.get_ylim()[::-1])
ax2.set_ylim(ax2.get_ylim()[::-1])
plt.title('$\kappa_{th}=0.01$')
plt.show()


mu = 2.35
R = 8.314e7

#modified by sam
rho = atm_pressure * mu / (R*temp)
tau_v = gam * tau
heat = rho*np.exp(-tau_v)
# plot stellar heating profile
fig, ax1 = plt.subplots()
#ax2 = ax1.twinx()
ax1.semilogy(heat,tau,label='$\gamma=5$')
#ax2.semilogy(heat,tau_v,label='$\kappa_v=0.004$')
#ax2.semilogy(heat2,tau2_v,'--',label='$\kappa_v=0.0004$')
ax1.set_xlabel(r'stellar heat: $\rho\ \exp(-\tau_v)$')
ax1.set_ylabel(r'$\tau_{th}$')
#ax2.set_ylabel(r'$\tau_v$')
ax1.legend()
#ax2.legend()
plt.gca().invert_yaxis()
plt.title('$\kappa_{th}=0.01$')
plt.show()
##############################################################################################
## adding atmosphere grid point





