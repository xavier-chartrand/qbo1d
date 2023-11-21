#!/usr/bin/python -u
# Author : Xavier Chartrand
# Email  : xavier.chartrand@uqar.ca
#          xavier.chartrand@ens-lyon.fr

'''
Solving the dimensionless Plumb's model with one counterpropagating wave
'''

# Modules and custom utilities
from qbo1d_utils import *

## Parameter space
# ---------- #
# (Re)
# Waves parameters and forcing
nRe = 101; minRe = 0; maxRe = 100               # Re simulations; min; max
Re  = np.linspace(minRe,maxRe,nRe)              # wave Reynolds

# Indices and values of parameters to vary between simulations
# 'fdiag': output frequency index step
# 'wdir':  directory to write and read data files (.csv and .nc)

# BEGIN STREAM EDITOR
wdir  = ''
fdiag = 0
indRe = 0
# END STREAM EDITOR, BEGIN MODEL CONFIGURATION

# Wave parameters of each forced components
simRe = Re[indRe]                               # Reynolds

# ---------- #
# Grid and wave parameters
# Wave parameters arrays' layout (for 'c','k','F'):
# Diagnostics are implemented up to two counterpropagating waves.
# lines (2)   : number of waves, 1 or 2
# columns (2) : positive[0] and negative[1] forcing (see qbo_utils.py)
# Enter positive components only. Forcing is symmetrized during computation.
#* Note that the magnitude is in term of Reynolds: Re = nu*F0/(kc)
nz    = 100                                     # number of vertical points
H     = 3.5                                     # domain's height
c     = [1]                                     # wave phase velocity
k     = [1]                                     # wave wavenumber
F     = [simRe]                                 # wave magnitude
hmuc  = 0.95                                    # u/c hmax(z) criterion
Fcond = None                                    # forcing condition above Uc

# Adaptative timestep with the forcing
dtf    = 1.E-4                                  # T0 timestep scaling
trspin = 4/5                                    # time ratio spin-up/run
nTmax  = 2500                                   # maximum T0s to simulate
nTspin = nTmax*trspin                           # save hot start up to spin-up
nT     = 500                                    # number of T0s to simulate
T0     = k[0]**2/simRe if simRe else 0          # characteristic time
dt     = dtf*T0                                 # timestep/reversal condition

# Simulation parameters
params = {'Re':{'Value':simRe,'Label':'Re'}}    # identifiers

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
itrspin = int(np.ceil(max(nTspin-nTinit,0)/dtf))# iterations of spin-up
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
             'WbarAccel':wbaraccel}             # fields to diagnose

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
print('''Reynolds Re   : %.2f'''%(simRe))
print('''h/H ratio     : %.2f'''%(k[0]/H))
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

# END
