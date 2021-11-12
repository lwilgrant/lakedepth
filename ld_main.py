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
        sample = gldb.loc[(gldb['Lat'] >= lat_bnd_0)&(gldb['Lat'] <= lat_bnd_1)&\
                           (gldb['Lon'] >= lon_bnd_0)&(gldb['Lon'] <= lon_bnd_1)]
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
ds_biases = ds_depths.copy(deep=True)

for var in ds_biases.keys():
    if 'GLDB' in var:
        ds_biases[var] = ds_biases['ISIMIP_gridcell_depth'] - ds_biases[var]
ds_biases = ds_biases.drop_vars(names=['ISIMIP_gridcell_depth','number_lakes'])
ds_depths['number_lakes'].plot()
lake_smps = ds_depths['number_lakes'].where(ds_depths['number_lakes']>10)
test = lake_smps.values.flatten()
test = test[~np.isnan(test)]
num_lake_smps = len(test)
print('num_lake_smps is {}'.format(num_lake_smps))

#%%============================================================================
# plot - median biases + num lakes
#==============================================================================

cmap_brbg = plt.cm.get_cmap('RdBu_r')
col_cbticlbl = '0'   # colorbar color of tick labels
col_cbtic = '0.5'   # colorbar color of ticks
col_cbedg = '0.9'   # colorbar color of edge
cb_ticlen = 3.5   # colorbar length of ticks
cb_ticwid = 0.4   # colorbar thickness of ticks
cb_edgthic = 0   # colorbar thickness of edges between colors
cblabel = 'corr'  # colorbar label
col_zero = 'gray'   # zero change color
sbplt_lw = 0.1   # linewidth on projection panels
cstlin_lw = 0.2   # linewidth foffr coastlines

title_font = 15
cbtitle_font = 14
tick_font = 14
legend_font=12

east = 180
west = -180
north = 80
south = -60
extent = [west,east,south,north]

# x=25
x=20
y=9.5
cstlin_lw = 0.2
t_left = 0.05
t_bottom = 0.05
t_right = 0.95
t_top = 0.95
t_rect = [t_left, t_bottom, t_right, t_top]


############################### colormaps ##################################

# identify colors for obs eof maps
cmap55 = cmap_brbg(0.01)
cmap50 = cmap_brbg(0.05)   #blue
cmap45 = cmap_brbg(0.1)
cmap40 = cmap_brbg(0.15)
cmap35 = cmap_brbg(0.2)
cmap30 = cmap_brbg(0.25)
cmap25 = cmap_brbg(0.3)
cmap20 = cmap_brbg(0.325)
cmap10 = cmap_brbg(0.4)
cmap5 = cmap_brbg(0.475)
cmap0 = col_zero
cmap_5 = cmap_brbg(0.525)
cmap_10 = cmap_brbg(0.6)
cmap_20 = cmap_brbg(0.625)
cmap_25 = cmap_brbg(0.7)
cmap_30 = cmap_brbg(0.75)
cmap_35 = cmap_brbg(0.8)
cmap_40 = cmap_brbg(0.85)
cmap_45 = cmap_brbg(0.9)
cmap_50 = cmap_brbg(0.95)  #red
cmap_55 = cmap_brbg(0.99)

colors_brbg = [cmap_45,
                cmap_35,
                cmap_30,
                cmap_25,
                cmap_10,
                cmap0,
                cmap10,
                cmap25,
                cmap30,
                cmap35,
                cmap45]

# declare list of colors for discrete colormap of colorbar
cmap_list_eof = mpl.colors.ListedColormap(colors_brbg,N=len(colors_brbg))
cmap_list_eof.set_over(cmap55)
cmap_list_eof.set_under(cmap_55)

q_samples = []
ds_biases = ds_biases.where(lake_smps > 0)
for met in ['avg','med']:        
    for dep in ['mean','max']:
        q_samples.append(np.abs(ds_biases['GLDB_{}_{}_depth'.format(met,dep)].quantile(0.95).item()))
        q_samples.append(np.abs(ds_biases['GLDB_{}_{}_depth'.format(met,dep)].quantile(0.05).item()))
    
# colorbar args
start = np.around(np.max(q_samples),decimals=4)
# start = 20
inc = start/5
values_eof = [-1*start,
              -1*start+inc,
              -1*start+inc*2,
              -1*start+inc*3,
              -1*start+inc*4,
              -0.001,
              0.001,
              start-inc*4,
              start-inc*3,
              start-inc*2,
              start-inc,
              start]

tick_locs_eof = [-1*start,
                 -1*start+inc,
                 -1*start+inc*2,
                 -1*start+inc*3,
                 -1*start+inc*4,
                 0,
                 start-inc*4,
                 start-inc*3,
                 start-inc*2,
                 start-inc,
                 start]

tick_labels_eof = [str(np.around(-1*start,decimals=1)),
                   str(np.around(-1*start+inc,decimals=1)),
                   str(np.around(-1*start+inc*2,decimals=1)),
                   str(np.around(-1*start+inc*3,decimals=1)),
                   str(np.around(-1*start+inc*4,decimals=1)),
                   str(0),
                   str(np.around(start-inc*4,decimals=1)),
                   str(np.around(start-inc*3,decimals=1)),
                   str(np.around(start-inc*2,decimals=1)),
                   str(np.around(start-inc,decimals=1)),
                   str(np.around(start,decimals=1))]

norm_eof = mpl.colors.BoundaryNorm(values_eof,cmap_list_eof.N)

f = plt.figure(figsize=(x,y))
gs1 = gridspec.GridSpec(2,2)
ax1 = f.add_subplot(gs1[0],projection=ccrs.PlateCarree())
ax2 = f.add_subplot(gs1[1],projection=ccrs.PlateCarree())
ax3 = f.add_subplot(gs1[2],projection=ccrs.PlateCarree())
ax4 = f.add_subplot(gs1[3],projection=ccrs.PlateCarree())
axes = [ax1,ax2,ax3,ax4]
cb_eof_x0 = 0.935
cb_eof_y0 = 0.0625
cb_eof_xlen = 0.015
cb_eof_ylen = 0.8725
cbax_eof = f.add_axes([cb_eof_x0, 
                       cb_eof_y0, 
                       cb_eof_xlen, 
                       cb_eof_ylen])
gs1.tight_layout(figure=f, 
                 rect=t_rect, 
                 h_pad=5, 
                 w_pad=5)

i = 0
for ax,met,dep in zip(axes,['avg','med']*2,['mean','mean','max','max']):
    
    ds_biases['GLDB_{}_{}_depth'.format(met,dep)].plot(ax=ax,
                                                       transform=ccrs.PlateCarree(),
                                                       cmap=cmap_list_eof,
                                                       cbar_ax=cbax_eof,
                                                       center=0,
                                                       norm=norm_eof,
                                                       add_labels=False)
    ax.coastlines(linewidth=cstlin_lw)
    ax.set_title(letters[i],
                 loc='left',
                 fontweight='bold',
                 fontsize=title_font)
    ax.set_extent(extent,
                  crs=ccrs.PlateCarree())
    if i == 0 or i == 1:
        ax.set_title('sample {}'.format(met),
                    loc='center',
                    fontweight='bold',
                    fontsize=title_font)
    if i == 0 or i == 2:
        if dep == 'mean':
            height = 0.4
        elif dep == 'max':
            height = 0.4
        ax.text(-0.05,
                height,
                '{} depth'.format(dep),
                fontsize=title_font,
                fontweight='bold',
                rotation='vertical',
                transform=ax.transAxes)

    i += 1
    
cb_eof = mpl.colorbar.ColorbarBase(ax=cbax_eof, 
                                   cmap=cmap_list_eof,
                                   norm=norm_eof,
                                   spacing='uniform',
                                   orientation='vertical',
                                   extend='both',
                                   ticks=tick_locs_eof,
                                   drawedges=False)
cb_eof.set_label('Depth bias (ISIMIP - GLDB)',
                 size=title_font)
cb_eof.ax.xaxis.set_label_position('bottom')
cb_eof.ax.tick_params(labelcolor=col_cbticlbl,
                      labelsize=tick_font,
                      color=col_cbtic,
                      length=cb_ticlen,
                      width=cb_ticwid,
                      direction='out'); 
cb_eof.ax.set_yticklabels(tick_labels_eof)
                        #   rotation=45)
cb_eof.outline.set_edgecolor(col_cbedg)
cb_eof.outline.set_linewidth(cb_edgthic)

f.savefig(outDIR+'/isimip_gldb_bias_numlakes.png',bbox_inches='tight',dpi=400)



#%%============================================================================
# plot - median/mean biases
#==============================================================================

cmap_brbg = plt.cm.get_cmap('RdBu_r')
col_cbticlbl = '0'   # colorbar color of tick labels
col_cbtic = '0.5'   # colorbar color of ticks
col_cbedg = '0.9'   # colorbar color of edge
cb_ticlen = 3.5   # colorbar length of ticks
cb_ticwid = 0.4   # colorbar thickness of ticks
cb_edgthic = 0   # colorbar thickness of edges between colors
cblabel = 'corr'  # colorbar label
col_zero = 'gray'   # zero change color
sbplt_lw = 0.1   # linewidth on projection panels
cstlin_lw = 0.2   # linewidth for coastlines

title_font = 15
cbtitle_font = 14
tick_font = 14
legend_font=12

# east = 180
# west = -180
# north = 80
# south = -60
# extent = [west,east,south,north]
east = 45
west = 0
north = 72.5
south = 50
extent = [west,east,south,north]

# x=25
x=20
y=9.5
cstlin_lw = 0.2
t_left = 0.05
t_bottom = 0.05
t_right = 0.95
t_top = 0.95
t_rect = [t_left, t_bottom, t_right, t_top]


############################### colormaps ##################################

# identify colors for obs eof maps
cmap55 = cmap_brbg(0.01)
cmap50 = cmap_brbg(0.05)   #blue
cmap45 = cmap_brbg(0.1)
cmap40 = cmap_brbg(0.15)
cmap35 = cmap_brbg(0.2)
cmap30 = cmap_brbg(0.25)
cmap25 = cmap_brbg(0.3)
cmap20 = cmap_brbg(0.325)
cmap10 = cmap_brbg(0.4)
cmap5 = cmap_brbg(0.475)
cmap0 = col_zero
cmap_5 = cmap_brbg(0.525)
cmap_10 = cmap_brbg(0.6)
cmap_20 = cmap_brbg(0.625)
cmap_25 = cmap_brbg(0.7)
cmap_30 = cmap_brbg(0.75)
cmap_35 = cmap_brbg(0.8)
cmap_40 = cmap_brbg(0.85)
cmap_45 = cmap_brbg(0.9)
cmap_50 = cmap_brbg(0.95)  #red
cmap_55 = cmap_brbg(0.99)

colors_brbg = [cmap_45,
                cmap_35,
                cmap_30,
                cmap_25,
                cmap_10,
                cmap0,
                cmap10,
                cmap25,
                cmap30,
                cmap35,
                cmap45]

# declare list of colors for discrete colormap of colorbar
cmap_list_eof = mpl.colors.ListedColormap(colors_brbg,N=len(colors_brbg))
cmap_list_eof.set_over(cmap55)
cmap_list_eof.set_under(cmap_55)

q_samples = []
ds_biases = ds_biases.where(lake_smps > 0)
for met in ['avg','med']:        
    for dep in ['mean','max']:
        q_samples.append(np.abs(ds_biases['GLDB_{}_{}_depth'.format(met,dep)].quantile(0.95).item()))
        q_samples.append(np.abs(ds_biases['GLDB_{}_{}_depth'.format(met,dep)].quantile(0.05).item()))
    
# colorbar args
# start = np.around(np.max(q_samples),decimals=4)
start = 10
# start = 20
inc = start/5
values_eof = [-1*start,
              -1*start+inc,
              -1*start+inc*2,
              -1*start+inc*3,
              -1*start+inc*4,
              -0.001,
              0.001,
              start-inc*4,
              start-inc*3,
              start-inc*2,
              start-inc,
              start]

tick_locs_eof = [-1*start,
                 -1*start+inc,
                 -1*start+inc*2,
                 -1*start+inc*3,
                 -1*start+inc*4,
                 0,
                 start-inc*4,
                 start-inc*3,
                 start-inc*2,
                 start-inc,
                 start]

tick_labels_eof = [str(np.around(-1*start,decimals=1)),
                   str(np.around(-1*start+inc,decimals=1)),
                   str(np.around(-1*start+inc*2,decimals=1)),
                   str(np.around(-1*start+inc*3,decimals=1)),
                   str(np.around(-1*start+inc*4,decimals=1)),
                   str(0),
                   str(np.around(start-inc*4,decimals=1)),
                   str(np.around(start-inc*3,decimals=1)),
                   str(np.around(start-inc*2,decimals=1)),
                   str(np.around(start-inc,decimals=1)),
                   str(np.around(start,decimals=1))]

norm_eof = mpl.colors.BoundaryNorm(values_eof,cmap_list_eof.N)

f = plt.figure(figsize=(x,y))
gs1 = gridspec.GridSpec(2,2)
ax1 = f.add_subplot(gs1[0],projection=ccrs.PlateCarree())
ax2 = f.add_subplot(gs1[1],projection=ccrs.PlateCarree())
ax3 = f.add_subplot(gs1[2],projection=ccrs.PlateCarree())
ax4 = f.add_subplot(gs1[3],projection=ccrs.PlateCarree())
axes = [ax1,ax2,ax3,ax4]
cb_eof_x0 = 0.935
cb_eof_y0 = 0.0625
cb_eof_xlen = 0.015
cb_eof_ylen = 0.8725
cbax_eof = f.add_axes([cb_eof_x0, 
                       cb_eof_y0, 
                       cb_eof_xlen, 
                       cb_eof_ylen])
gs1.tight_layout(figure=f, 
                 rect=t_rect, 
                 h_pad=5, 
                 w_pad=5)

i = 0
for ax,met,dep in zip(axes,['avg','med']*2,['mean','mean','max','max']):
    
    ds_biases['GLDB_{}_{}_depth'.format(met,dep)].plot(ax=ax,
                                                       transform=ccrs.PlateCarree(),
                                                       cmap=cmap_list_eof,
                                                       cbar_ax=cbax_eof,
                                                       center=0,
                                                       norm=norm_eof,
                                                       add_labels=False)
    ax.coastlines(linewidth=cstlin_lw)
    ax.set_title(letters[i],
                 loc='left',
                 fontweight='bold',
                 fontsize=title_font)
    ax.set_extent(extent,
                  crs=ccrs.PlateCarree())
    if i == 0 or i == 1:
        ax.set_title('sample {}'.format(met),
                    loc='center',
                    fontweight='bold',
                    fontsize=title_font)
    if i == 0 or i == 2:
        if dep == 'mean':
            height = 0.4
        elif dep == 'max':
            height = 0.4
        ax.text(-0.05,
                height,
                '{} depth'.format(dep),
                fontsize=title_font,
                fontweight='bold',
                rotation='vertical',
                transform=ax.transAxes)

    i += 1
    
cb_eof = mpl.colorbar.ColorbarBase(ax=cbax_eof, 
                                   cmap=cmap_list_eof,
                                   norm=norm_eof,
                                   spacing='uniform',
                                   orientation='vertical',
                                   extend='both',
                                   ticks=tick_locs_eof,
                                   drawedges=False)
cb_eof.set_label('Depth bias (ISIMIP - GLDB)',
                 size=title_font)
cb_eof.ax.xaxis.set_label_position('bottom')
cb_eof.ax.tick_params(labelcolor=col_cbticlbl,
                      labelsize=tick_font,
                      color=col_cbtic,
                      length=cb_ticlen,
                      width=cb_ticwid,
                      direction='out'); 
cb_eof.ax.set_yticklabels(tick_labels_eof)
                        #   rotation=45)
cb_eof.outline.set_edgecolor(col_cbedg)
cb_eof.outline.set_linewidth(cb_edgthic)

f.savefig(outDIR+'/isimip_gldb_bias_avg_median.png',bbox_inches='tight',dpi=400)


#%%============================================================================
# plot - q1/q3 biases
#==============================================================================

cmap_brbg = plt.cm.get_cmap('RdBu_r')
col_cbticlbl = '0'   # colorbar color of tick labels
col_cbtic = '0.5'   # colorbar color of ticks
col_cbedg = '0.9'   # colorbar color of edge
cb_ticlen = 3.5   # colorbar length of ticks
cb_ticwid = 0.4   # colorbar thickness of ticks
cb_edgthic = 0   # colorbar thickness of edges between colors
cblabel = 'corr'  # colorbar label
col_zero = 'gray'   # zero change color
sbplt_lw = 0.1   # linewidth on projection panels
cstlin_lw = 0.2   # linewidth for coastlines

title_font = 15
cbtitle_font = 14
tick_font = 14
legend_font=12

east = 45
west = 0
north = 72.5
south = 50
extent = [west,east,south,north]

# east = 180
# west = -180
# north = 80
# south = -60
# extent = [west,east,south,north]

x=20
y=9.5
cstlin_lw = 0.2
t_left = 0.05
t_bottom = 0.05
t_right = 0.95
t_top = 0.95
t_rect = [t_left, t_bottom, t_right, t_top]

############################### colormaps ##################################

# identify colors for obs eof maps
cmap55 = cmap_brbg(0.01)
cmap50 = cmap_brbg(0.05)   #blue
cmap45 = cmap_brbg(0.1)
cmap40 = cmap_brbg(0.15)
cmap35 = cmap_brbg(0.2)
cmap30 = cmap_brbg(0.25)
cmap25 = cmap_brbg(0.3)
cmap20 = cmap_brbg(0.325)
cmap10 = cmap_brbg(0.4)
cmap5 = cmap_brbg(0.475)
cmap0 = col_zero
cmap_5 = cmap_brbg(0.525)
cmap_10 = cmap_brbg(0.6)
cmap_20 = cmap_brbg(0.625)
cmap_25 = cmap_brbg(0.7)
cmap_30 = cmap_brbg(0.75)
cmap_35 = cmap_brbg(0.8)
cmap_40 = cmap_brbg(0.85)
cmap_45 = cmap_brbg(0.9)
cmap_50 = cmap_brbg(0.95)  #red
cmap_55 = cmap_brbg(0.99)

colors_brbg = [cmap_45,
                cmap_35,
                cmap_30,
                cmap_25,
                cmap_10,
                cmap0,
                cmap10,
                cmap25,
                cmap30,
                cmap35,
                cmap45]

# declare list of colors for discrete colormap of colorbar
cmap_list_eof = mpl.colors.ListedColormap(colors_brbg,N=len(colors_brbg))
cmap_list_eof.set_over(cmap55)
cmap_list_eof.set_under(cmap_55)

q_samples = []
ds_biases = ds_biases.where(lake_smps > 0)
# for met in ['avg','med']:        
#     for dep in ['mean','max']:
for met in ['q1','q3']:        
    for dep in ['mean','max']:
        q_samples.append(np.abs(ds_biases['GLDB_{}_{}_depth'.format(met,dep)].quantile(0.95).item()))
        q_samples.append(np.abs(ds_biases['GLDB_{}_{}_depth'.format(met,dep)].quantile(0.05).item()))
    
# colorbar args
# start = np.around(np.max(q_samples),decimals=4)
start = 10
inc = start/5
values_eof = [-1*start,
              -1*start+inc,
              -1*start+inc*2,
              -1*start+inc*3,
              -1*start+inc*4,
              -0.001,
              0.001,
              start-inc*4,
              start-inc*3,
              start-inc*2,
              start-inc,
              start]

tick_locs_eof = [-1*start,
                 -1*start+inc,
                 -1*start+inc*2,
                 -1*start+inc*3,
                 -1*start+inc*4,
                 0,
                 start-inc*4,
                 start-inc*3,
                 start-inc*2,
                 start-inc,
                 start]

tick_labels_eof = [str(np.around(-1*start,decimals=1)),
                   str(np.around(-1*start+inc,decimals=1)),
                   str(np.around(-1*start+inc*2,decimals=1)),
                   str(np.around(-1*start+inc*3,decimals=1)),
                   str(np.around(-1*start+inc*4,decimals=1)),
                   str(0),
                   str(np.around(start-inc*4,decimals=1)),
                   str(np.around(start-inc*3,decimals=1)),
                   str(np.around(start-inc*2,decimals=1)),
                   str(np.around(start-inc,decimals=1)),
                   str(np.around(start,decimals=1))]

norm_eof = mpl.colors.BoundaryNorm(values_eof,cmap_list_eof.N)

f = plt.figure(figsize=(x,y))
gs1 = gridspec.GridSpec(2,2)
ax1 = f.add_subplot(gs1[0],projection=ccrs.PlateCarree())
ax2 = f.add_subplot(gs1[1],projection=ccrs.PlateCarree())
ax3 = f.add_subplot(gs1[2],projection=ccrs.PlateCarree())
ax4 = f.add_subplot(gs1[3],projection=ccrs.PlateCarree())
axes = [ax1,ax2,ax3,ax4]
cb_eof_x0 = 0.935
cb_eof_y0 = 0.0625
cb_eof_xlen = 0.015
cb_eof_ylen = 0.8725
cbax_eof = f.add_axes([cb_eof_x0, 
                       cb_eof_y0, 
                       cb_eof_xlen, 
                       cb_eof_ylen])
gs1.tight_layout(figure=f, 
                 rect=t_rect, 
                 h_pad=5, 
                 w_pad=5)

i = 0
for ax,met,dep in zip(axes,['q1','q3']*2,['mean','mean','max','max']):
    
    ds_biases['GLDB_{}_{}_depth'.format(met,dep)].plot(ax=ax,
                                                       transform=ccrs.PlateCarree(),
                                                       cmap=cmap_list_eof,
                                                       cbar_ax=cbax_eof,
                                                       center=0,
                                                       norm=norm_eof,
                                                       add_labels=False)
    ax.coastlines(linewidth=cstlin_lw)
    ax.set_title(letters[i],
                 loc='left',
                 fontweight='bold',
                 fontsize=title_font)
    ax.set_extent(extent,
                  crs=ccrs.PlateCarree())
    if i == 0 or i == 1:
        ax.set_title('sample {}'.format(met),
                    loc='center',
                    fontweight='bold',
                    fontsize=title_font)
    if i == 0 or i == 2:
        if dep == 'mean':
            height = 0.4
        elif dep == 'max':
            height = 0.4
        ax.text(-0.05,
                height,
                '{} depth'.format(dep),
                fontsize=title_font,
                fontweight='bold',
                rotation='vertical',
                transform=ax.transAxes)

    i += 1
    
cb_eof = mpl.colorbar.ColorbarBase(ax=cbax_eof, 
                                   cmap=cmap_list_eof,
                                   norm=norm_eof,
                                   spacing='uniform',
                                   orientation='vertical',
                                   extend='both',
                                   ticks=tick_locs_eof,
                                   drawedges=False)
cb_eof.set_label('Depth bias (ISIMIP - GLDB)',
                 size=title_font)
cb_eof.ax.xaxis.set_label_position('bottom')
cb_eof.ax.tick_params(labelcolor=col_cbticlbl,
                      labelsize=tick_font,
                      color=col_cbtic,
                      length=cb_ticlen,
                      width=cb_ticwid,
                      direction='out'); 
cb_eof.ax.set_yticklabels(tick_labels_eof)
cb_eof.outline.set_edgecolor(col_cbedg)
cb_eof.outline.set_linewidth(cb_edgthic)

f.savefig(outDIR+'/isimip_gldb_bias_q1_q3.png',bbox_inches='tight',dpi=400)


# %%
