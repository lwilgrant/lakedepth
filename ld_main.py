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

# lake depths from GLDB
os.chdir(txtDIR)
ldeps = pd.read_csv('FreshWaterLakeDepthDataSet_v2_noheader.txt',
                    delim_whitespace=True,
                    header=0,
                    names=['Lat','Lon','Mean (m)','Max (m)','Surface_area (km**2)','International_name','National_name','Country'],
                    index_col=False,
                    encoding='iso-8859-1')
for col in ['Lat','Lon','Mean (m)','Max (m)','Surface_area (km**2)']:
    ldeps[col] = pd.to_numeric(ldeps[col],errors='coerce')
ldeps = ldeps[ldeps['Mean (m)'] != 9999.0]
ldeps = ldeps[ldeps['Max (m)'] != 9999.0]
ldeps['Mean (m)'] = ldeps['Mean (m)']/10
ldeps['Max (m)'] = ldeps['Max (m)']/10


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
test_df['GLDB_avg_mean_depth'] = np.nan # empty row for gldb data
test_df['GLDB_avg_max_depth'] = np.nan # empty row for gldb data
test_df['GLDB_med_mean_depth'] = np.nan # empty row for gldb data
test_df['GLDB_med_max_depth'] = np.nan # empty row for gldb data
test_df['GLDB_q1_mean_depth'] = np.nan # empty row for gldb data
test_df['GLDB_q3_mean_depth'] = np.nan # empty row for gldb data
test_df['GLDB_q1_max_depth'] = np.nan # empty row for gldb data
test_df['GLDB_q3_max_depth'] = np.nan # empty row for gldb data
test_df['number_lakes'] = np.nan
for gp in test_df.index:
    lat = gp[0]
    lon = gp[1]
    lat_bnd_0 = lat - 0.25
    lat_bnd_1 = lat + 0.25
    lon_bnd_0 = lon - 0.25
    lon_bnd_1 = lon + 0.25
    try:
        sample = ldeps.loc[(ldeps['Lat'] > lat_bnd_0)&(ldeps['Lat'] < lat_bnd_1)&\
                           (ldeps['Lon'] > lon_bnd_0)&(ldeps['Lon'] < lon_bnd_1)]
        sample_len = len(sample.index)
        test_df.loc[(lat,lon)]['GLDB_avg_mean_depth'] = sample['Mean (m)'].mean()
        test_df.loc[(lat,lon)]['GLDB_avg_max_depth'] = sample['Max (m)'].mean()
        test_df.loc[(lat,lon)]['GLDB_med_mean_depth'] = sample['Mean (m)'].median()
        test_df.loc[(lat,lon)]['GLDB_med_max_depth'] = sample['Max (m)'].median()
        test_df.loc[(lat,lon)]['GLDB_q1_mean_depth'] = sample['Mean (m)'].quantile(0.25)
        test_df.loc[(lat,lon)]['GLDB_q3_mean_depth'] = sample['Mean (m)'].quantile(0.75)
        test_df.loc[(lat,lon)]['GLDB_q1_max_depth'] = sample['Max (m)'].quantile(0.25)
        test_df.loc[(lat,lon)]['GLDB_q3_max_depth'] = sample['Max (m)'].quantile(0.75)
        test_df.loc[(lat,lon)]['number_lakes'] = sample_len
    except:
        pass
test_df = test_df.rename(columns={'LAKEDEPTH':'ISIMIP_gridcell_depth'}).dropna()
test_df.to_csv('dataframe.csv')
test_da = xr.Dataset.from_dataframe(test_df)
new_da = test_da.reindex_like(da_depth)
new_da['ISIMIP_gridcell_depth'].plot()
mean_bias = new_da['ISIMIP_gridcell_depth'] - new_da['GLDB_mean']
max_bias = new_da['ISIMIP_gridcell_depth'] - new_da['GLDB_max']


