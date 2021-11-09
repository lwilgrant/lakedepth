#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:52:49 2020

@author: luke
"""


# =============================================================================
# SUMMARY
# =============================================================================

# script for comparing lake depths between the ISIMIP field and GLDB source data

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
gldb = pd.read_csv('FreshWaterLakeDepthDataSet_v2_noheader.txt',
                    delim_whitespace=True,
                    header=0,
                    names=['Lat','Lon','Mean (m)','Max (m)','Surface_area (km**2)','International_name','National_name','Country'],
                    index_col=False,
                    encoding='iso-8859-1')

for col in ['Lat','Lon','Mean (m)','Max (m)','Surface_area (km**2)']:
    
    gldb[col] = pd.to_numeric(gldb[col],errors='coerce')
    
gldb.to_csv(txtDIR+'/'+'gldb_dataframe.csv')
gldb = gldb[gldb['Mean (m)'] != 9999.0]
gldb = gldb[gldb['Max (m)'] != 9999.0]
gldb['Mean (m)'] = gldb['Mean (m)']/10
gldb['Max (m)'] = gldb['Max (m)']/10

# isimip grid cell lake depths and percent coverage
os.chdir(ncDIR)
ds = xr.open_dataset('lakedepth.nc4')
da_depth = ds['LAKEDEPTH']
ds = xr.open_dataset('pctlake.nc4')
da_pct = ds['PCT_LAKE']
da_pct = xr.where(da_pct>0,1,0) # select lake grid cells with pct lake cover > 0
da_depth = da_depth.where(da_pct == 1) # only where lake pixels exist

# isimip + gldb in single dataframe
isimip = da_depth.to_dataframe() # convert to dataframe
isimip = isimip.dropna() # drop nans
isimip['GLDB_avg_mean_depth'] = np.nan # empty row for gldb data
isimip['GLDB_avg_max_depth'] = np.nan 
isimip['GLDB_med_mean_depth'] = np.nan 
isimip['GLDB_med_max_depth'] = np.nan 
isimip['GLDB_q1_mean_depth'] = np.nan 
isimip['GLDB_q3_mean_depth'] = np.nan 
isimip['GLDB_q1_max_depth'] = np.nan 
isimip['GLDB_q3_max_depth'] = np.nan 
isimip['number_lakes'] = np.nan

for gp in isimip.index:
    
    lat = gp[0]
    lon = gp[1]
    lat_bnd_0 = lat - 0.25
    lat_bnd_1 = lat + 0.25
    lon_bnd_0 = lon - 0.25
    lon_bnd_1 = lon + 0.25
    
    try:
        sample = gldb.loc[(gldb['Lat'] > lat_bnd_0)&(gldb['Lat'] < lat_bnd_1)&\
                           (gldb['Lon'] > lon_bnd_0)&(gldb['Lon'] < lon_bnd_1)]
        sample_len = len(sample.index)
        isimip.loc[(lat,lon)]['GLDB_avg_mean_depth'] = sample['Mean (m)'].mean()
        isimip.loc[(lat,lon)]['GLDB_avg_max_depth'] = sample['Max (m)'].mean()
        isimip.loc[(lat,lon)]['GLDB_med_mean_depth'] = sample['Mean (m)'].median()
        isimip.loc[(lat,lon)]['GLDB_med_max_depth'] = sample['Max (m)'].median()
        isimip.loc[(lat,lon)]['GLDB_q1_mean_depth'] = sample['Mean (m)'].quantile(0.25)
        isimip.loc[(lat,lon)]['GLDB_q3_mean_depth'] = sample['Mean (m)'].quantile(0.75)
        isimip.loc[(lat,lon)]['GLDB_q1_max_depth'] = sample['Max (m)'].quantile(0.25)
        isimip.loc[(lat,lon)]['GLDB_q3_max_depth'] = sample['Max (m)'].quantile(0.75)
        isimip.loc[(lat,lon)]['number_lakes'] = sample_len
        
    except:
        
        pass
    
isimip = isimip.rename(columns={'LAKEDEPTH':'ISIMIP_gridcell_depth'}).dropna()
isimip.to_csv(txtDIR+'/'+'dataframe.csv')
ds_depths = xr.Dataset.from_dataframe(isimip)
ds_depths = ds_depths.reindex_like(da_depth)


