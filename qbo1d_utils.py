#!/usr/bin/python -u
# Author : Xavier Chartrand
# Email  : xavier.chartrand@uqar.ca
#          xavier.chartrand@ens-lyon.fr

# Modules
import numpy as np
import xarray as xr
import scipy
import os
import time
import csv
import matplotlib.pyplot as plt
from scipy.constants import pi
from scipy.sparse.linalg import spsolve
from dask.dataframe import read_csv

# Shell commands in python
def sh(script):
    os.system("bash -c '%s'" %script)

## Functions
# ---------- #
# Save to CSV file
def saveToCSV(fname,var):
    with open(fname,'a') as f:
        csv.writer(f).writerow(var)

# Load a variable associated to a CSV file
def loadCSV(field,clear_csv=False):
    # Use dask with chunks of 10MB to read file, not to overload memory
    df  = read_csv(field['CSVfile'],blocksize=10e6,header=None,delimiter=',')
    var = np.transpose(np.array(df,dtype='f8'))
    sh('rm %s'%(field['CSVfile'])) if clear_csv else None
    return var

# Convert all datas stored into CSVs of current directory to a netCDF file
def convertToNC(grid,wF,diags,clear_csv=False):
    # Loading CSV datas
    height   = grid['Height']['z']
    dimw     = np.prod(wF['c'].shape)
    dimt     = max(len(diags['Index']),0)
    fdiag    = diags['IndexFreq']
    time     = np.cumsum([grid['Time']['dt']*fdiag]*dimt)
    nulldiag = np.array([np.nan]*grid['Height']['nz'])

    # Convert to netCDF
    if fdiag:
        print('# ---------- #')
        print('Warning: saving datas before spin-up completed')\
        if diags['SaveState'] else None
        print('\nSaving diagnostics %s ...'%(diags['NCfile']))
        Dim1    = ['Height','Time']
        DNull   = ['Height']
        Crd1    = {'Height':height,'Time':time}
        CNull   = {'Height':height}
        uout    = xr.DataArray(\
                  loadCSV(diags['Fields']['MeanFlow'],clear_csv),\
                  dims=Dim1,coords=Crd1,attrs={'Units':'m/s'})\
                  if diags['Fields']['MeanFlow']['Save']\
                  else xr.DataArray(nulldiag,dims=DNull,coords=CNull)
        fzrpout = xr.DataArray(\
                  loadCSV(diags['Fields']['RefPosAccel'],clear_csv),\
                  dims=Dim1,coords=Crd1,attrs={'Units':'m/ss'})\
                  if diags['Fields']['RefPosAccel']['Save']\
                  else xr.DataArray(nulldiag,dims=DNull,coords=CNull)
        fzrnout = xr.DataArray(\
                  loadCSV(diags['Fields']['RefNegAccel'],clear_csv),\
                  dims=Dim1,coords=Crd1,attrs={'Units':'m/ss'})\
                  if diags['Fields']['RefNegAccel']['Save']\
                  else xr.DataArray(nulldiag,dims=DNull,coords=CNull)
        fzppout = xr.DataArray(\
                  loadCSV(diags['Fields']['PtbPosAccel'],clear_csv),\
                  dims=Dim1,coords=Crd1,attrs={'Units':'m/ss'})\
                  if diags['Fields']['PtbPosAccel']['Save']\
                  else xr.DataArray(nulldiag,dims=DNull,coords=CNull)
        fzpnout = xr.DataArray(\
                  loadCSV(diags['Fields']['PtbNegAccel'],clear_csv),\
                  dims=Dim1,coords=Crd1,attrs={'Units':'m/ss'})\
                  if diags['Fields']['PtbNegAccel']['Save']\
                  else xr.DataArray(nulldiag,dims=DNull,coords=CNull)
        nuout   = xr.DataArray(\
                  loadCSV(diags['Fields']['DiffAccel'],clear_csv),\
                  dims=Dim1,coords=Crd1,attrs={'Units':'m/ss'})\
                  if diags['Fields']['DiffAccel']['Save']\
                  else xr.DataArray(nulldiag,dims=DNull,coords=CNull)
        wbout   = xr.DataArray(\
                  loadCSV(diags['Fields']['WbarAccel'],clear_csv),\
                  dims=Dim1,coords=Crd1,attrs={'Units':'m/ss'})\
                  if diags['Fields']['WbarAccel']['Save']\
                  else xr.DataArray(nulldiag,dims=DNull,coords=CNull)
        DSout   = xr.Dataset({'MeanFlow':uout,
                              'RefPosAccel':fzrpout,
                              'RefNegAccel':fzrnout,
                              'PtbPosAccel':fzppout,
                              'PtbNegAccel':fzpnout,
                              'DiffAccel':nuout,
                              'WbarAccel':wbout})

        DSout.to_netcdf(diags['NCfile'],compute=False,engine='scipy')
        print('Diagnostics %s saved.'%(diags['NCfile']))

# Grid initialization
def gridInit(nz,H,dt,itr,Wcond=False,wb=0):
    dz     = H/nz
    z      = np.cumsum(nz*[dz])
    time   = {'itr':itr,'dt':dt}
    height = {'H':H,'nz':nz,'dz':dz,'z':z}
    upwell = {'Wcond':Wcond,'wb':wb*np.exp(-z/H)}
    return {'Time':time,'Height':height,'Upwelling':upwell}

# Wave forcing initialization (list of counterpropagating waves)
def wfInit(c,k,F,hmuc,Fcond):
    nw = len(c)
    cw = np.array([[c[i],-c[i]] for i in range(nw)])
    kw = np.array([2*[k[i]] for i in range(nw)])
    Fw = np.array([2*[F[i]] for i in range(nw)])
    return {'c':cw,'k':kw,'F':Fw,'hmuc':hmuc,'Fcond':Fcond}

# Diagnostics saving options
def diagInit(itr,itrspin,nTi,nTr,fout,fields,wdir,params):
    iout  = range(itrspin,itr+1,fout) if fout and itrspin<itr\
            else range(0,itr+1,fout) if fout else []
    sflag = (True if itrspin>=itr else False) if fout else False
    ncgen = 'QBO1D'+''.join(['_%s%.4f'%(params[p]['Label'],params[p]['Value'])\
                             for p in params])
    ncfile = wdir + ncgen + '_T%d-%d.nc'%(nTi,nTr+nTi)
    for f in fields:
        csvgen  = str(f) + ''.join(['_%s%.4f'%(params[p]['Label'],\
                                    params[p]['Value']) for p in params])
        csvfile = wdir + csvgen + '_T%d-%d.csv'%(nTi,nTr+nTi)
        sh('cat /dev/null > %s'%(csvfile)) if fields[f]['Save'] and iout\
        else None
        fields[f]['Save']    = False if not iout else fields[f]['Save']
        fields[f]['CSVfile'] = csvfile

    return {'NCfile':ncfile,'SaveState':sflag,'Fields':fields,'Index':iout,\
            'IndexFreq':fout}

# Hot start saving options
def hsInit(T0,nTspin,dtf,hsdir,params):
    iout   = [int(i/dtf) for i in range(1,int(nTspin)+1)]
    hsgen  = 'HS'+''.join(['_%s%.4f'%(params[p]['Label'],params[p]['Value'])\
                           for p in params])
    hsflag = False
    hsfile = None
    nTinit = nTspin if not T0 else 0

    # Check for existing hot start files
    if os.listdir(hsdir):
        TauArray = []
        # Get filenames matching simulated parameters
        for filename in os.listdir(hsdir):
            if filename.startswith(hsgen):
                hsflag = True
                SS     = filename.split('_')
                Tau    = float(SS[-1].split('.nc')[0].split('Tau')[1])
                TauArray.append(Tau) if Tau not in TauArray else None
        if hsflag:
            # Sort hotstart times
            TauArray.sort()
            # Get netCDF name of lastest hot start and update initial time
            hsfile = hsgen + '_Tau%d.nc'%(TauArray[-1])
            nTinit = TauArray[-1]

    hsinfo = {'HSfile':hsfile,'HSgen':hsgen,'HSdir':hsdir}
    t0opts = {'T0':T0,'nTinit':nTinit,'nTspin':nTspin}
    return {'Start':hsflag,'Params':params,'Index':iout,\
            'HSinfo':hsinfo,'T0opts':t0opts}

# Initial velocity
def uInit(nz,hs,A=-1.E-1,profile='ptb'):
    # Load hot start
    if hs['Start']:
        DS   = xr.open_dataset(hs['HSinfo']['HSdir']+hs['HSinfo']['HSfile'],\
                               engine='scipy')
        u    = DS['U']
        um1  = DS['Um1']
        fz   = DS['Fz']
        fzm1 = DS['Fzm1']
        fzm2 = DS['Fzm2']
    # If no hot start, input a classical velocity profile
    # 'ptb': small easterly perturbation
    # 'jet': vertical jet
    else:
        zz   = np.arange(nz)[::-1]
        u    = -np.cos(5*pi*zz/2/nz)*np.exp(-(1-zz/nz)*4) if profile=='jet'\
                else A*np.sin(pi/2*zz/nz+pi/2) if profile=='ptb'\
                else np.zeros(nz)
        um1  = np.zeros(nz)
        fz   = np.zeros(nz)
        fzm1 = np.zeros(nz)
        fzm2 = np.zeros(nz)
    return {'U':u,'Um1':um1,'Fz':fz,'Fzm1':fzm1,'Fzm2':fzm2}

# Compute h(max) criterion
def UcCrit(u,c,hmuc=0.95):
    dimw = c.shape
    return np.array([[(u>=hmuc*c[i,j]).argmax()\
                      if c[i,j]>=0 else (u<=hmuc*c[i,j]).argmax()\
                      for j in range(dimw[1])] for i in range(dimw[0])])

# WKB approximation integrand
def gz(u,c):
    return (u/c - 1)**(-2)

# WKB approximation computation
def WKB(grid,wF,u):
    dz    = grid['Height']['dz']
    c     = wF['c']
    k     = wF['k']
    F     = wF['F']
    Fcond = wF['Fcond']
    dimw  = c.shape
    u     = np.hstack([0,u,u[-2]])              # u=0 bottom, du/dz=0 top

    # Computing waveforcing with a WKB approximation
    # The additionnal 1/4 factor I/4 is to take account "staggering"
    uw = [[F[i,j]/k[i,j]/c[i,j]*np.exp(1/k[i,j]/c[i,j]**2*(\
       -   dz/2*np.cumsum(gz(u[1:],c[i,j]) + gz(u[:-1],c[i,j]))\
       +   dz/4*(gz(u[1:],c[i,j]) + gz(u[:-1],c[i,j]))))\
           for j in range(dimw[1])]\
           for i in range(dimw[0])]

    # No wave dissipation above critical layers or other conditions
    if Fcond:
        udir = np.sign(u[1])
        iuc  = UcCrit(u,c)
    if Fcond=='noF_Uc':
        for j in range(dimw[1]):
            for i in range(dimw[0]):
                kk = iuc[i,j]
                if kk:
                    uw[i][j][kk:] = uw[i][j][kk]
    elif Fcond=='noF_Uc_dir':
        (j,jj) = (1,0) if udir<0 else (0,1)
        kkdir  = iuc[:,j]
        kkdir  = np.min(kkdir[np.nonzero(kkdir)]) if kkdir.any() else None
        for i in range(dimw[0]):
            kkind = iuc[i,jj]
            if kkdir:
                uw[i][j][kkdir:] = uw[i][j][kkdir]
            if kkind:
                uw[i][jj][kkind:] = uw[i][jj][kkind]

    # Return vertical derivative of the forcing
    return -np.diff(uw,axis=-1)/dz

# Upwelling computation
def Upwelling(grid,u):
    dz = grid['Height']['dz']
    wb = grid['Upwelling']['wb']
    u  = np.hstack([0,u,u[-2]])
    return -wb*(u[2:] - u[:-2])/2/dz

# Iterate computation in time
def runs(grid,u0,wF,diags,hs):
    # Include parameters
    itr   = grid['Time']['itr']                 # number of time iterations
    dt    = grid['Time']['dt']                  # time step
    nz    = grid['Height']['nz']                # number of vertical points
    dz    = grid['Height']['dz']                # vertical resolution
    z     = grid['Height']['z']                 # vertical dimension array
    Wcond = grid['Upwelling']['Wcond']          # upwelling condition
    rD    = dt/dz**2                            # Fourier number
    dimw  = wF['c'].shape                       # wave array dimension
    hmuc  = wF['hmuc']                          # u/c hmax(z) criterion
    ibd1  = abs(z-0.1*z[-1]).argmin()           # bd 1 out index
    ibd2  = abs(z-0.75*z[-1]).argmin()          # bd 2 out index

    # Initial fields and stencil values
    u    = u0['U']                              # initial velocity
    um1  = u0['Um1']                            # previous (t-1) velocity
    fz   = u0['Fz']                             # initial wave forcing
    fzm1 = u0['Fzm1']                           # previous (t-1) wave forcing
    fzm2 = u0['Fzm2']                           # previous (t-2) wave forcing

    # Hot start and time/T0 options
    hsdir  = hs['HSinfo']['HSdir']              # hot start file directory
    hsgen  = hs['HSinfo']['HSgen']              # hot start generic file name
    T0     = hs['T0opts']['T0']                 # characteristic time
    nTinit = hs['T0opts']['nTinit']             # initial T0 time of the run
    nTspin = hs['T0opts']['nTspin']             # maximum T0s to simulation

    ## Computation
    # Initialization of the implicit diffusion matrix
    main  = np.zeros(nz)                        # principal branch
    lower = np.zeros(nz-1)                      # inferior branch
    upper = np.zeros(nz-1)                      # superior branch
    b     = np.zeros(nz)                        # RHS of the AU = b system

    # Append values to branches
    main[:]  = 1 + 2*rD                         # values of principal branch
    lower[:] = -rD                              # values of lower branch
    upper[:] = -rD                              # values of upper branch

    # Top boundary condition (du/dz)[-1] = 0 => u[end+1] = u[end-1]
    lower[-1] = -2*rD                           # "ghost" u[end+1] = u[end-1]

    # Construct the tridiagonal implicit diffusion matrix
    A = scipy.sparse.diags(diagonals=[main,lower,upper],
                           offsets=[0,-1,1],shape=(nz,nz),
                           format='csr')

    # No simulations if 'dt' not defined (i.e. if forcing is null), but allow
    # saving of null values
    if not dt:
        nulldiag_fields    = np.zeros((itr,nz)) # save zeros fields
        diags['Index']     = np.arange(itr)     # save in netCDF later
        diags['SaveState'] = False              # do not show spin-up warning
        itr                = -1                 # perform no simulation

        # Write to csv
        for row in nulldiag_fields:
            saveToCSV(diags['Fields']['MeanFlow']['CSVfile'],row)\
            if diags['Fields']['MeanFlow']['Save'] else None
            saveToCSV(diags['Fields']['RefPosAccel']['CSVfile'],row)\
            if diags['Fields']['RefPosAccel']['Save'] else None
            saveToCSV(diags['Fields']['RefNegAccel']['CSVfile'],row)\
            if diags['Fields']['RefNegAccel']['Save'] else None
            saveToCSV(diags['Fields']['PtbPosAccel']['CSVfile'],row)\
            if diags['Fields']['PtbPosAccel']['Save'] else None
            saveToCSV(diags['Fields']['PtbNegAccel']['CSVfile'],row)\
            if diags['Fields']['PtbNegAccel']['Save'] else None
            saveToCSV(diags['Fields']['DiffAccel']['CSVfile'],row)\
            if diags['Fields']['DiffAccel']['Save'] else None
            saveToCSV(diags['Fields']['WbarAccel']['CSVfile'],row)\
            if diags['Fields']['WbarAccel']['Save'] else None
        print('No simulation, null wave forcing.')

    # Iterate in time with a 3rd order Adams-Bashfort scheme
    itr  = itr-1 if not itr else itr
    AB3c = [23/12,-16/12,5/12]                  # AB3 coefficients
    for i in range(itr+1):
        # Update variables
        um1  = u                                # U[t-1] = U[t]
        fzm2 = fzm1                             # F[t-2] = F[t-1]
        fzm1 = fz                               # F[t-1] = F[t]

        # h(max) criterion
        hm = UcCrit(u,wF['c'],hmuc=hmuc).flatten()

        # Compute Reynolds stresses and upwelling
        dzUW = WKB(grid,wF,um1)                 # d(u'w')/dz each components
        WdzU = Upwelling(grid,um1)\
               if Wcond else 0                  # wbar d(ubar)/dz

        # Sum forcings
        fz = sum(sum(dzUW)) + WdzU

        # Resolve the linear system A u[t] = b(u[t-1]) (A: diffusion matrix)
        # RHS : b = u[t-1] + AB3(wave forcing)
        waccel = dt*(AB3c[0]*fz + AB3c[1]*fzm1 + AB3c[2]*fzm2)
        b      = um1 + waccel
        u      = spsolve(A,b)
        nudiff = np.array(u - um1 - waccel)/dt
        if abs(u).max()/abs(wF['c']).max()>1:
            print('Warning: max(u) velocity relatively exceeded max(c)')
            print('Umax: %.4f'%(abs(u).max()))
            print('Cmax: %.4f'%(abs(wF['c']).max()))

        # Write to csv at output frequencies to avoid overload memory usage
        if i in diags['Index']:
            saveToCSV(diags['Fields']['MeanFlow']['CSVfile'],u)\
            if diags['Fields']['MeanFlow']['Save'] else None
            saveToCSV(diags['Fields']['RefPosAccel']['CSVfile'],dzUW[0,0])\
            if diags['Fields']['RefPosAccel']['Save'] else None
            saveToCSV(diags['Fields']['RefNegAccel']['CSVfile'],dzUW[0,1])\
            if diags['Fields']['RefNegAccel']['Save'] else None
            saveToCSV(diags['Fields']['PtbPosAccel']['CSVfile'],dzUW[1,0])\
            if diags['Fields']['PtbPosAccel']['Save'] else None
            saveToCSV(diags['Fields']['PtbNegAccel']['CSVfile'],dzUW[1,1])\
            if diags['Fields']['PtbNegAccel']['Save'] else None
            saveToCSV(diags['Fields']['DiffAccel']['CSVfile'],nudiff)\
            if diags['Fields']['DiffAccel']['Save'] else None
            saveToCSV(diags['Fields']['WbarAccel']['CSVfile'],WdzU)\
            if diags['Fields']['WbarAccel']['Save'] else None

        # Saving progress to 'hotstart' files for posterior simulations
        # (won't need to rerun the spin-up everytime which is very long)
        nTiter = abs(np.array(hs['Index'])-i).argmin() + nTinit + 1
        if i in hs['Index'] and nTiter<=nTspin:
            # Remove previously saved hot starts
            sh('rm %s*'%(hsdir+hsgen)) if round(nTiter)>1 else None
            # Save progress
            Dim    = ['Height']
            Crd    = {'Height':z}
            hsu    = xr.DataArray(u,dims=Dim,coords=Crd,\
                                  attrs={'Units':'m/s'})
            hsum1  = xr.DataArray(um1,dims=Dim,coords=Crd,\
                                  attrs={'Units':'m/s'})
            hsfz   = xr.DataArray(fz,dims=Dim,coords=Crd,\
                                  attrs={'Units':'m/ss'})
            hsfzm1 = xr.DataArray(fzm1,dims=Dim,coords=Crd,\
                                  attrs={'Units':'m/ss'})
            hsfzm2 = xr.DataArray(fzm2,dims=Dim,coords=Crd,\
                                  attrs={'Units':'m/ss'})
            DShs   = xr.Dataset({'U':hsu,'Um1':hsum1,\
                                 'Fz':hsfz,'Fzm1':hsfzm1,'Fzm2':hsfzm2})
            ncname = hsgen + '_Tau%d.nc'%(nTiter)
            DShs.to_netcdf(hsdir+ncname,engine='scipy')
            print('\nHS T0 %.4f saved.\n'%(nTiter))

        # Show progress
        print('progress: %.4f %%'%(100*i/itr))\
        if i in range(0,itr,int(itr/100)) else None
    print('progress: %.4f %%'%(100)) if itr+1 else None

# END
