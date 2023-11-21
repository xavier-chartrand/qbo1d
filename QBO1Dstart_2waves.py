#!/usr/bin/python -u
# Author : Xavier Chartrand
# Email  : xavier.chartrand@uqar.ca
#          xavier.chartrand@ens-lyon.fr

'''
Solving the dimensionless Plumb's model with two counterpropagating waves
'''

# Modules and custom utilities
from qbo1d_utils import *

## Parameter space
# ---------- #
# (Cp/Cb, Fp/Fb)
# Waves parameters and forcing
Lr   = 1                                    # attenuation length of ptb wave
Fref = 40                                   # fixed wave forcing of ref wave
nCr  = 101; minCr  = 0.01; maxCr  = 100     # Cr simulations; min; max
nFtr = 51;  minFtr = 0;    maxFtr = 1       # Ftr simulations; min; max
Ftr  = np.linspace(minFtr,maxFtr,nFtr)      # momentum flux
Cr   = np.logspace(np.log10(minCr),\
                   np.log10(maxCr),\
                   nCr,base=10)             # phase velocity

# Indices and values of parameters to vary between simulations
# 'fdiag': output frequency (every X steps specified)
# 'wdir':  directory to write and read data files (.csv and .nc)

# BEGIN STREAM EDITOR
wdir   = ''
fdiag  = 0
indCr  = 0
indFtr = 0
# END STREAM EDITOR, BEGIN MODEL CONFIGURATION

# Wave parameters of each forced components
simCr  = Cr[indCr]                              # phase velocity of ptb wave
simFtr = Ftr[indFtr]                            # forcing ratio of ptb wave
simKr  = Lr*simCr**(-2)                         # wavenumber of ptb wave
simF1  = Fref                                   # wave forcing of ref wave
simF2  = Fref*simFtr*simKr*simCr                # wave forcing of ptb wave

# ---------- #
# Grid and wave parameters
# Wave parameters arrays' layout (for 'c','k','F'):
# Diagnostics are implemented up to two counterpropagating waves.
# lines (2)   : number of waves, 1 or 2
# columns (2) : positive[0] and negative[1] forcing (see qbo_utils.py)
# Enter positive components only. Forcing is symmetrized during computation.
#* Note that the magnitude is in term of Reynolds: Re = nu*F0/(kc)
nz    = 200                                     # number of vertical points
H     = 3.5                                     # domain's height
c     = [1,simCr]                               # wave phase velocity
k     = [1,simKr]                               # wave wavenumber
F     = [simF1,simF2]                           # wave magnitude
hmuc  = 0.95                                    # u/c hmax(z) criterion
Fcond = 'noF_Uc'                                # forcing condition above Uc
L02   = Lr                                      # attenuation length of wave 2

# Scale timestep with forcing
dtf    = 1.E-4                                  # T0 timestep scaling
trspin = 4/5                                    # time ratio spin-up/run
nTmax  = 2500                                   # maximum T0s to simulate
nTspin = nTmax*trspin                           # save hot start up to spin-up
nT     = 500                                    # number of T0s to simulate
T01    = 1/simF1 if simF1 else 0                # characteristic time wave 1
T02    = Lr**2/simF2 if simF2 else 0            # characteristic time wave 2
T0     = min(T01,T02) if T01*T02\
         else T01 if T01 else T02 if T02 else 0 # minimum characteristic time
dt     = dtf*T0                                 # timestep/reversal condition

# Simulation parameters
params = {'Cr':{'Value':simCr,'Label':'Cr'},
          'Ftr':{'Value':simFtr,'Label':'Ftr'}} # identifiers

## Parameter dictionaries
# ---------- #
# Hot start options
# If no diagnostics, "itr" is at maximum the number required for the spin-up.
# Else, iterations can't exceed the maximum T0 ("nTmax") imposed.
# If no forcing, "itr = nTmax" to output zero values to plot some results.
# *Dont forget to create hotstarts directory to allow the model to run.
hsdir   = 'hotstarts/'                          # hot start file directory
hs      = hsInit(T0,nTspin,dtf,hsdir,params)    # hot start saving options
nTinit  = hs['T0opts']['nTinit']                # initial T0 of simulation
itrspin = int(np.ceil(max(nTspin-nTinit,0)/dtf))# iterations to spin-up
itrrun  = int(np.ceil(min(nTmax-nTinit,nT)/dtf))# iterations of asked run
itr     = (itrrun if T0 else nTmax) if fdiag\
           else itrspin if T0 else 0            # no diags above nTmax
nTrun   = round(itr*dt/T0) if T0 else nT        # number of T0s of the run

# Output fields
meanflow  = {'Save':True,'CSVfile':None}        # mean flow diags
refpaccel = {'Save':False,'CSVfile':None}       # d/dz u'w' diags (ref +)
refnaccel = {'Save':False,'CSVfile':None}       # d/dz u'w' diags (ref -)
ptbpaccel = {'Save':False,'CSVfile':None}       # d/dz u'w' diags (ptb +)
ptbnaccel = {'Save':False,'CSVfile':None}       # d/dz u'w' diags (ptb -)
diffaccel = {'Save':False,'CSVfile':None}       # implicit diffusion diags
wbaraccel = {'Save':False,'CSVfile':None}       # upwelling diags
fields    = {'MeanFlow':meanflow,
             'RefPosAccel':refpaccel,
             'RefNegAccel':refnaccel,
             'PtbPosAccel':ptbpaccel,
             'PtbNegAccel':ptbnaccel,
             'DiffAccel':diffaccel,
             'WbarAccel':wbaraccel}             # output fields

# Grid, wave parameters, initial conditions and diagnostics
grid  = gridInit(nz,H,dt,itr)                   # grid options
wF    = wfInit(c,k,F,hmuc,Fcond)                # wave forcing options
u0    = uInit(nz,hs)                            # initial velocity conditions
diags = diagInit(itr,itrspin,nTinit,nTrun,\
                 fdiag,fields,wdir,params)      # diagnostics saving options

## Runs
# ---------- #
# Print simulated parameters
print('''# ---------- #''')
print('SIMULATION PARAMETERS')
print('''Ref momflux   : %.2f'''%(Fref))
print('''Cp/Cb ratio   : %.2f'''%(simCr))
print('''Lp/Lb ratio   : %.2f'''%(Lr))
print('''Fp/Fb ratio   : %.2f'''%(simFtr))
print('\nDIAGNOSTICS')
print('Diagnotics step : %d'%(fdiag))
print('\nTIME OPTIONS')
print('T0 max          : %d'%(nTmax))
print('T0 spin-up      : %d'%(nTspin))
print('T0 init         : %d'%(nTinit))
print('T0 asked        : %d'%(nT))
print('T0 running      : %d'%(nTrun if T0 else 0))
print('''# ---------- #''')
runs(grid,u0,wF,diags,hs)                       # run simulation
convertToNC(grid,wF,diags,clear_csv=True)       # convert CSV to netCDF
