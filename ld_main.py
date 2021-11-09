#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:52:49 2020

@author: luke
"""


# =============================================================================
# SUMMARY
# =============================================================================


# This script generates detection and attribution results on LUMIP data

# New hydra version:
    # chop into cells (main script and functions) (done)
    # chop into subroutines on other scripts (done)
    # for unmasked, use all land-based ar6 regions per continent (too much uncertainty for luh2 selection) (done)
    # add option for uniform grids (per obs)
        # if statements before list comprehension to gather files
        # BETTER IDEA:
            # subroutine for all file extraction:
                # use flags as input; return things such as pi files, obs files and fp_files
                # use directories as input
                # currently, maps subroutine generates ar6_land maps to select grid cells
                    # but I no longer want to do that. 
                    # change maps subroutine to, based on grid_type, produce ar6_land masks either at obs resolution or as dict for mod resolutions
                    # take grid type as input to file allocation subroutine
                    # tres can be removed: no longer required
                    # based on need, can read in either ensmeans or individual realisations
                # no option exists for looking at area changes in the case of working at obs resolutions (map/*.nc files are all at mod resolution; will have to fix if I want to do this but not necessary now)
    # add option for obs type; needs to be added to subroutine functions
    # add d & a outputs per AR6 region, (latitudinally?)
    # put current (sep 30) fp_main and da_main and funcs scripts on backup branch on github
    # need solution for options in sr_mod_fp and sr_pi and sr_obs to:
        # run fp on one experiment; e.g. separate runs for historical and hist-nolu (for "sr_mod_fp")
        # can a function take fp_data_* objects and 
    # will always have 2 OF results for each obs type, but difference will be whether at model or obs grid
    # first establish working results for obs vs mod res, global vs continental vs ar6 results,
        # then establish historical vs hist-nolu single factor runs
    # continental map of detection results needs tweeks:
        # greenland shouldn't be part of eu; south american extent shouldn't include central america
        # mmm colors were wrong when all areas had valueof 3; seems that categories weren't established; double check
        # to avoid continental inclusion of regions not consisreed in the continents analysis via ar6, 
            # perhaps i shouldn't use continental shapefiles directly but rather merge shapefiles for ar6 regions (clipped by continents for land only)

#%%============================================================================
# import
# =============================================================================

import sys
import os
import numpy as np
import pickle as pk
import pandas as pd
import matplotlib.pyplot as plt
import copy as cp
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import matplotlib as mpl
import cartopy.feature as cfeature
import regionmask as rm
from random import shuffle
from matplotlib.lines import Line2D

#%%============================================================================
# path
#==============================================================================

curDIR='C:/Users/lgrant/Documents/repos/lakedepth'
os.chdir(curDIR)

# data input directories
ncDIR = os.path.join(curDIR, 'netcdf')
# sfDIR = os.path.join(curDIR, 'shapefiles')
txtDIR = os.path.join(curDIR, 'txtfiles')
outDIR = os.path.join(curDIR, 'figures')

# bring in functions
from ld_funcs import *

#%%============================================================================
# options - analysis
#==============================================================================

# adjust these flag settings for analysis choices only change '<< SELECT >>' lines

# << SELECT >>
flag_pickle=1     # 0: do not pickle objects
                  # 1: pickle objects after sections 'read' and 'analyze'

# << SELECT >>
flag_svplt=0      # 0: do not save plot
                  # 1: save plot in picDIR
        
letters = ['a', 'b', 'c',
           'd', 'e', 'f',
           'g', 'h', 'i',
           'j', 'k', 'l',
           'm', 'n', 'o',
           'p', 'q', 'r',
           's', 't', 'u',
           'v', 'w', 'x',
           'y', 'z']

#%%==============================================================================
# get data 
#==============================================================================

# lake depths
# os.chdir(sfDIR)
# ldeps = gp.read_file('glwd_2.shp')
# ldeps = ldeps.loc[ldeps['TYPE'] == 'Lake']
# ldeps
os.chdir(txtDIR)
ldeps = pd.read_csv('FreshWaterLakeDepthDataSet_v2_noheader.txt',
                    delim_whitespace=True,
                    header=0,
                    names=['Lat','Lon','Mean (m)','Max (m)','Surface_area (km**2)','International_name','National_name','Country'],
                    index_col=False,
                    encoding='iso-8859-1')
ldeps = ldeps[ldeps['Mean (m)'] != str(9999.0)]
# ldeps = ldeps[ldeps['Max (m)'] != 9999.0]


# isimip grid cell lake depths
os.chdir(ncDIR)
ds = xr.open_dataset('lakedepth.nc4')
da_depth = ds['LAKEDEPTH']

ds = xr.open_dataset('pctlake.nc4')
da_pct = ds['PCT_LAKE']
da_pct = xr.where(da_pct>0,1,0)

# testing some changes

da_depth = da_depth.where(da_pct == 1) # only where lake pixels exist
test_df = da_depth.to_dataframe() # convert to dataframe
test_df = test_df.dropna() # drop nans
test_df['GLDB'] = np.nan # empty row for gldb data
for gp in test_df.index:
    lat = gp[0]
    lon = gp[1]
    lat_bnd_0 = lat - 0.25
    lat_bnd_1 = lat + 0.25
    lon_bnd_0 = lon - 0.25
    lon_bnd_1 = lon + 0.25
    try:
        sample = ldeps.loc[(ldeps['Lat'] > lat_bnd_0)&\
                            (ldeps['Lat'] < lat_bnd_1)&\
                                    (ldeps['Lon'] > lon_bnd_0)&\
                                        (ldeps['Lon'] < lat_bnd_1)].mean()
        test_df.loc[(lat,lon)]['GLDB'] = sample
    except:
        pass
    
    

# add lat/lon bounds (+1 length from lat/lon grid points) to da_depth
# filter ldeps for non-missing depth data, lakes only, convert to meters, no 0 meters lakes
# check length of ldeps with index and len(idx)
# search for 
    # pandas dataframe to data array and vice versa
        # convert da_depth to df_depth with row for each lat/lon gridpoint combination from da_depth
        # have column for da_depth depth points
        # remove nan rows
        # loop lat/lon gridpoints:
            # loc selection from dataframe using bounds of lat/lon gridpoint:
                # statistics on loc selection -> pipe to same gridcell on empty arrays
                # *OR*
                # change coords to closest ISIMIP gridpointo
                # for every valid match, append to list for valid ISIMIP-to-ldeps gridpoints
            # *OR* Loop ldeps dataframe, per lake:
                # change coords to closest ISIMIP gridpoint
                # for every valid match, append to list for valid ISIMIP-to-ldeps gridpoints
        # After gridpoint assimilation, loop through list of valid ISIMIP-to-ldeps gridpoints
            # "loc" to select all sample lakes per valid point, compute statistics
            # assign value to empty data array with matching ISIMIP grid-scale




class Rectangle:
    pass


class Point:
    pass


def move_rect(rectangle, dx, dy):
    rectangle.corner.x = rectangle.corner.x + dx
    rectangle.corner.y = rectangle.corner.y + dy


box = Rectangle()
box.width = 100
box.height = 200

box.corner = Point()
box.corner.x = 0
box.corner.y = 0

move_rect(box, 10, 10)
   

#%%============================================================================

# mod ensembles
os.chdir(curDIR)
from da_sr_mod_ens import *
mod_ens,mod_ts_ens,nt = ensemble_subroutine(modDIR,
                                            maps,
                                            models,
                                            exps,
                                            var,
                                            lu_techn,
                                            measure,
                                            lulcc_type,
                                            y1,
                                            grid,
                                            freq,
                                            obs_types,
                                            continents,
                                            ns,
                                            fp_files,
                                            ar6_regs)
ts_pickler(curDIR,
           mod_ts_ens,
           grid,
           t_ext,
           obs_mod='model')

#%%============================================================================

# mod fingerprint (nx is dummy var not used in OLS OF)
os.chdir(curDIR)
from da_sr_mod_fp import *
fp,fp_continental,fp_ar6,nx = fingerprint_subroutine(obs_types,
                                                     grid,
                                                     ns,
                                                     nt,
                                                     mod_ens,
                                                     exps,
                                                     models,
                                                     ar6_regs,
                                                     continents,
                                                     continent_names,
                                                     exp_list)

#%%============================================================================

# pi data
os.chdir(curDIR)
from da_sr_pi import *
ctl_data,ctl_data_continental,ctl_data_ar6 = picontrol_subroutine(piDIR,
                                                                  pi_files,
                                                                  grid,
                                                                  models,
                                                                  obs_types,
                                                                  continents,
                                                                  continent_names,
                                                                  var,
                                                                  y1,
                                                                  freq,
                                                                  maps,
                                                                  ar6_regs,
                                                                  ns,
                                                                  nt)

#%%============================================================================

# obs data
os.chdir(curDIR)
from da_sr_obs import *
obs_data,obs_data_continental,obs_data_ar6,obs_ts = obs_subroutine(obsDIR,
                                                                   grid,
                                                                   obs_files,
                                                                   continents,
                                                                   continent_names,
                                                                   obs_types,
                                                                   models,
                                                                   y1,
                                                                   var,
                                                                   maps,
                                                                   ar6_regs,
                                                                   freq,
                                                                   nt,
                                                                   ns)

ts_pickler(curDIR,
           obs_ts,
           grid,
           t_ext,
           obs_mod='obs')

#%%============================================================================
# detection & attribution 
#==============================================================================

# optimal fingerprinting
os.chdir(curDIR)
from da_sr_of import *
var_sfs,\
var_ctlruns,\
proj,\
U,\
yc,\
Z1c,\
Z2c,\
Xc,\
Cf1,\
Ft,\
beta_hat,\
var_fin,\
models = of_subroutine(grid,
                       models,
                       nx,
                       analysis,
                       exp_list,
                       obs_types,
                       obs_data,
                       obs_data_continental,
                       obs_data_ar6,
                       fp,
                       fp_continental,
                       fp_ar6,
                       ctl_data,
                       ctl_data_continental,
                       ctl_data_ar6,
                       bs_reps,
                       ns,
                       nt,
                       reg,
                       cons_test,
                       formule_ic_tls,
                       trunc,
                       ci_bnds,
                       continents)
# save OF results
pickler(curDIR,
        var_fin,
        analysis,
        grid,
        t_ext,
        exp_list)
           
#%%============================================================================
# plotting scaling factors
#==============================================================================    

os.chdir(curDIR)    
if len(exp_list) == 2:
    
    pass

elif len(exp_list) == 1:
    
    start_exp = deepcopy(exp_list[0])
    if start_exp == 'historical':
        second_exp = 'hist-noLu'
    elif start_exp == 'hist-noLu':
        second_exp = 'historical'
    pkl_file = open('var_fin_1-factor_{}_{}-grid_{}_{}.pkl'.format(second_exp,grid,analysis,t_ext),'rb')
    var_fin_2 = pk.load(pkl_file)
    pkl_file.close()
    
    for obs in obs_types:
        for mod in models:
            var_fin[obs][mod][second_exp] = var_fin_2[obs][mod].pop(second_exp)
            
    exp_list = ['historical', 'hist-noLu']

if analysis == 'global':
    
    plot_scaling_global(models,
                        grid,
                        obs_types,
                        exp_list,
                        var_fin,
                        flag_svplt,
                        outDIR)

elif analysis == 'continental':
    
    plot_scaling_continental(models,
                             exps,
                             var_fin,
                             continents,
                             continent_names,
                             mod_ts_ens,
                             obs_ts,
                             flag_svplt,
                             outDIR,
                             lulcc_type,
                             t_ext,
                             freq,
                             measure,
                             var)
    
    plot_scaling_map_continental(sfDIR,
                                 obs_types,
                                 models,
                                 exp_list,
                                 continents,
                                 var_fin,
                                 grid,
                                 letters,
                                 outDIR)

elif analysis == 'ar6':
    
    plot_scaling_map_ar6(sfDIR,
                         obs_types,
                         models,
                         exp_list,
                         continents,
                         var_fin,
                         grid,
                         letters,
                         outDIR)              
    
    
         
    
    

# %%
