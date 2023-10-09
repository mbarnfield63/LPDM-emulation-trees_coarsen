# Imports
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime

import glob
import dask
import warnings
import cartopy
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import pickle
import imageio
import os

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from trees_emulator.load_data import *
from trees_emulator.training import *
from trees_emulator.predicting import *

import warnings
warnings.filterwarnings('ignore')

'''
Additional functions to assist within notebook testing
'''

def plotting_footprint(LoadData, date, jump):
    '''
    Plotting just the footprint data from a LoadData input.
    '''
    if type(date) == str:
            try:
                date = datetime.strptime(date, "%H:00 %d/%m/%Y")
            except:
                print("Something was wrong with the date. Please pass it in format hour:00 day/month/year (eg 15:00 1/3/2016)")
            
            try:
                idx = pd.DatetimeIndex(LoadData.met.time.values[jump:-3]).get_loc(date)
            except:
                raise KeyError("This date is out of range")

    if type(date) == int:
        idx = date
        try:
            date = pd.DatetimeIndex(LoadData.met.time.values[jump:-3])[idx]
        except:
            raise KeyError("This index is out of range")

    ## create figure and plot
    fig, ax = plt.subplots(1,1,figsize = (8,6), subplot_kw={'projection':cartopy.crs.Mercator()})

    c = ax.pcolormesh(LoadData.fp_lons, LoadData.fp_lats, np.reshape(LoadData.fp_data[idx+jump,:], (LoadData.size, LoadData.size)), transform=cartopy.crs.PlateCarree(), cmap="Reds", vmin=0)

    ## axis
    # set extent defines the domain to plot
    ax.set_extent([LoadData.fp_lons[0]-0.1,LoadData.fp_lons[-1]+0.1, LoadData.fp_lats[0]+0.1,LoadData.fp_lats[-1]+0.1], crs=cartopy.crs.PlateCarree())
    ax.set_xticks(LoadData.fp_lons[::3], crs=cartopy.crs.PlateCarree())

    lon_formatter = LongitudeFormatter(number_format='.1f', degree_symbol='', dateline_direction_label=True)
    ax.xaxis.set_major_formatter(lon_formatter)  
    ax.set_yticks(LoadData.fp_lats[::3], crs=cartopy.crs.PlateCarree())
    lat_formatter = LatitudeFormatter(number_format='.1f',  degree_symbol='',)
    ax.yaxis.set_major_formatter(lat_formatter)             
    ax.tick_params(axis='both', which='major', labelsize=12)   

    # marker for release point
    ax.plot(LoadData.release_lon+0, LoadData.release_lat+0, marker='o', c="w", markeredgecolor = "k", transform=cartopy.crs.PlateCarree(), markersize=5)
    # 50 or 110
    ax.coastlines(resolution='50m', color='black', linewidth=2)

    ax.set_title("Fp_data - "+ LoadData.site + "\n" + date.strftime("%m/%d/%Y, %H:00"), fontsize = 17)

    ## set up cbar
    cbar = plt.colorbar(c, ax=ax, orientation="vertical", aspect = 15, pad = 0.02)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("sensitivity, (mol/mol)/(mol/m2/s)", size = 15, loc="center", labelpad = 16) 

    fig.show()

def plot_xarray(self, date, site, coarsened_factor):
    """
    Plot singular footprint.

    Takes as inputs the date to plot (as a string in format hour:00 day/month/year (eg 15:00 1/3/2016) or as an data index as an int).
    """
    ## check that date is within range, and get index from date (or viceversa)
    if type(date) == str:
        try:
            date = datetime.strptime(date, "%H:00 %d/%m/%Y")
        except:
            print("Something was wrong with the date. Please pass it in format hour:00 day/month/year (eg 15:00 1/3/2016)")
        
        try:
            idx = pd.DatetimeIndex(self.time.values).get_loc(date)
        except:
            raise KeyError("This date is out of range")

    if type(date) == int:
        idx = date
        try:
            date = pd.DatetimeIndex(self.time.values)[idx]
        except:
            raise KeyError("This index is out of range")
    
    ## create figure and plot
    fig, ax = plt.subplots(figsize = (8,6), subplot_kw={'projection':cartopy.crs.Mercator()})

    values = np.array(self[idx,:])
    values = np.log10(values)
    
    # (10,10) can be size, or self.truths[idx,:] sqrt
    c = ax.pcolormesh(self.lon, self.lat, values, transform=cartopy.crs.PlateCarree(), cmap="Reds")

    ## set up axis
    # set extent defines the domain to plot
    ax.set_extent([self.lon[0]-0.1,self.lon[-1]+0.1, self.lat[0]+0.1,self.lat[-1]+0.1], crs=cartopy.crs.PlateCarree())
    
    ax.set_xticks(self.lon[::coarsened_factor*10], crs=cartopy.crs.PlateCarree())
    lon_formatter = LongitudeFormatter(number_format='.1f', degree_symbol='', dateline_direction_label=True)
    ax.xaxis.set_major_formatter(lon_formatter)  
    
    ax.set_yticks(self.lat[::coarsened_factor*12], crs=cartopy.crs.PlateCarree())
    lat_formatter = LatitudeFormatter(number_format='.1f',  degree_symbol='',)
    ax.yaxis.set_major_formatter(lat_formatter)             
    
    ax.tick_params(axis='both', which='major', labelsize=12)   

    # marker for release point
    #point = ax.plot(self.data.release_lon+0, self.data.release_lat+0, marker='o', c="w", markeredgecolor = "k", transform=cartopy.crs.PlateCarree(), markersize=5)
    # 50 or 110
    ax.coastlines(resolution='110m', color='black', linewidth=2)
    
    #ax.set_title("LPDM-generated footprint - "+ self.data.site + "\n" + date.strftime("%m/%d/%Y, %H:00"), fontsize = 17)
    ax.set_title("LPDM w/ Emulator section FP - "+ site + "\n" + date.strftime("%m/%d/%Y, %H:00") + ", " + "\n(Uncoarsening = mean, threshold = 0.0005)", fontsize = 17)


    ## set up cbar
    cbar = plt.colorbar(c, ax=ax, orientation="vertical", aspect = 15, pad = 0.02)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("log10(sensitivity), (mol/mol)/(mol/m2/s)", size = 15, loc="center", labelpad = 16)

    fig.show()

def plot_fp_full(self, date, site):
    """
    Plot singular footprint.

    Takes as inputs the date to plot (as a string in format hour:00 day/month/year (eg 15:00 1/3/2016) or as an data index as an int).
    """
    ## check that date is within range, and get index from date (or viceversa)
    if type(date) == str:
        try:
            date = datetime.strptime(date, "%H:00 %d/%m/%Y")
        except:
            print("Something was wrong with the date. Please pass it in format hour:00 day/month/year (eg 15:00 1/3/2016)")
        
        try:
            idx = pd.DatetimeIndex(self.time.values).get_loc(date)
        except:
            raise KeyError("This date is out of range")

    if type(date) == int:
        idx = date
        try:
            date = pd.DatetimeIndex(self.time.values)[idx]
        except:
            raise KeyError("This index is out of range")

    ## create figure and plot
    fig, ax = plt.subplots(figsize = (8,6), subplot_kw={'projection':cartopy.crs.Mercator()})
    
    # (10,10) can be size, or self.truths[idx,:] sqrt
    c = ax.pcolormesh(self.lon, self.lat, np.log10(np.array(self.fp[:,:,idx])), transform=cartopy.crs.PlateCarree(), cmap="Reds")

    ## set up axis
    # set extent defines the domain to plot
    ax.set_extent([self.lon[0]-0.1,self.lon[-1]+0.1, self.lat[0]+0.1,self.lat[-1]+0.1], crs=cartopy.crs.PlateCarree())
    
    ax.set_xticks(self.lon[::50], crs=cartopy.crs.PlateCarree())
    lon_formatter = LongitudeFormatter(number_format='.1f', degree_symbol='', dateline_direction_label=True)
    ax.xaxis.set_major_formatter(lon_formatter)  
    
    ax.set_yticks(self.lat[::50], crs=cartopy.crs.PlateCarree())
    lat_formatter = LatitudeFormatter(number_format='.1f',  degree_symbol='',)
    ax.yaxis.set_major_formatter(lat_formatter)             
    
    ax.tick_params(axis='both', which='major', labelsize=12)   

    # marker for release point
    #point = ax.plot(self.data.release_lon+0, self.data.release_lat+0, marker='o', c="w", markeredgecolor = "k", transform=cartopy.crs.PlateCarree(), markersize=5)
    # 50 or 110
    ax.coastlines(resolution='110m', color='black', linewidth=2)
    
    #ax.set_title("LPDM-generated footprint - "+ self.data.site + "\n" + date.strftime("%m/%d/%Y, %H:00"), fontsize = 17)
    ax.set_title("LPDM footprint - "+ site + "\n" + date.strftime("%m/%d/%Y, %H:00"), fontsize = 17)


    ## set up cbar
    cbar = plt.colorbar(c, ax=ax, orientation="vertical", aspect = 15, pad = 0.02)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("sensitivity, (mol/mol)/(mol/m2/s)", size = 15, loc="center", labelpad = 16)

    fig.show()

def plotting_OG_within_2x(data1, data2, date):
    '''
    Plotting original resolution grid within the 2x coarsened grid.
    '''
    if type(date) == str:
        try:
            date = datetime.strptime(date, "%H:00 %d/%m/%Y")
        except:
            print("Something was wrong with the date. Please pass it in format hour:00 day/month/year (eg 15:00 1/3/2016)")
        
        try:
            idx = pd.DatetimeIndex(data1.data.met.time.values[data1.jump:-3]).get_loc(date)
        except:
            raise KeyError("This date is out of range")

    if type(date) == int:
        idx = date
        try:
            date = pd.DatetimeIndex(data1.data.met.time.values[data1.jump:-3])[idx]
        except:
            raise KeyError("This index is out of range")

    vmax = np.nanmax(data2.truths[idx,:])
    
    ## create figure and plot
    fig, ax = plt.subplots(figsize = (8,6), subplot_kw={'projection':cartopy.crs.Mercator()})

    ## coarsest first
    ax.pcolormesh(data1.data.fp_lons, data1.data.fp_lats, 
                  np.reshape(data1.predictions[idx,:], (data1.size, data1.size)), 
                  transform=cartopy.crs.PlateCarree(), cmap="Reds", vmax = vmax, vmin=0)
    
    ## OG last
    c = ax.pcolormesh(data2.data.fp_lons, data2.data.fp_lats, 
                  np.reshape(data2.predictions[idx,:], (data2.size, data2.size)), 
                  transform=cartopy.crs.PlateCarree(), cmap="Reds", vmax = vmax, vmin=0)

    ## set up cbar
    cbar = plt.colorbar(c, ax=ax, orientation="vertical", aspect = 15, pad = 0.02)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("sensitivity, (mol/mol)/(mol/m2/s)", size = 15, loc="center", labelpad = 16)

    ## set up axis
    # set extent defines the domain to plot
    ax.set_extent([data1.data.fp_lons[0]-0.1,data1.data.fp_lons[-1]+0.1, data1.data.fp_lats[0]+0.1,data1.data.fp_lats[-1]+0.1], crs=cartopy.crs.PlateCarree())
    ax.set_xticks(data1.data.fp_lons[::2], crs=cartopy.crs.PlateCarree())

    lon_formatter = LongitudeFormatter(number_format='.1f', degree_symbol='', dateline_direction_label=True)
    ax.xaxis.set_major_formatter(lon_formatter)  
    ax.set_yticks(data1.data.fp_lats[::2], crs=cartopy.crs.PlateCarree())
    lat_formatter = LatitudeFormatter(number_format='.1f',  degree_symbol='',)
    ax.yaxis.set_major_formatter(lat_formatter)             
    ax.tick_params(axis='both', which='major', labelsize=12)   

    # marker for release point
    point_coarse = ax.plot(data1.data.release_lon+0, data1.data.release_lat+0, marker='o', c="w", markeredgecolor = "k", transform=cartopy.crs.PlateCarree(), markersize=5)
    point_OG = ax.plot(data2.data.release_lon+0, data2.data.release_lat+0, marker='x', c="k", transform=cartopy.crs.PlateCarree(), markersize=5)

    # 50 or 110
    ax.coastlines(resolution='50m', color='black', linewidth=2)

    # title
    ax.set_title("Emulator-generated footprint - "+ data1.data.site + "\n" + date.strftime("%m/%d/%Y, %H:00"), fontsize = 17)

    fig.show()

def plotting_coarsened(list, date, tickskip):
    '''
    Plotting original resolution grid within the 2x coarsened grid.
    '''
    if type(date) == str:
        try:
            date = datetime.strptime(date, "%H:00 %d/%m/%Y")
        except:
            print("Something was wrong with the date. Please pass it in format hour:00 day/month/year (eg 15:00 1/3/2016)")
        
        try:
            idx = pd.DatetimeIndex(list[0].data.met.time.values[list[0].jump:-3]).get_loc(date)
        except:
            raise KeyError("This date is out of range")

    if type(date) == int:
        idx = date
        try:
            date = pd.DatetimeIndex(list[0].data.met.time.values[list[0].jump:-3])[idx]
        except:
            raise KeyError("This index is out of range")

    vmax = 0.5*np.nanmax(list[-1].truths[idx,:])
    
    ## create figure and plot
    fig, ax = plt.subplots(figsize = (8,6), subplot_kw={'projection':cartopy.crs.Mercator()})

    for x in range(len(list)-1):
        ax.pcolormesh(list[x].data.fp_lons, list[x].data.fp_lats, 
                  np.reshape(list[x].predictions[idx,:], (list[x].size, list[x].size)), 
                  transform=cartopy.crs.PlateCarree(), cmap="Reds", vmax = vmax, vmin=0)
   
    # ## OG last
    c = ax.pcolormesh(list[-1].data.fp_lons, list[-1].data.fp_lats, 
                  np.reshape(list[-1].predictions[idx,:], (list[-1].size, list[-1].size)), 
                  transform=cartopy.crs.PlateCarree(), cmap="Reds", vmax = vmax, vmin=0)  
    
    ## set up cbar
    cbar = plt.colorbar(c, ax=ax, orientation="vertical", aspect = 15, pad = 0.02)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("sensitivity, (mol/mol)/(mol/m2/s)", size = 15, loc="center", labelpad = 16)

    ## set up axis
    # set extent defines the domain to plot   
    ax.set_extent([list[0].data.fp_lons[0]-0.1,list[0].data.fp_lons[-1]+0.1, list[0].data.fp_lats[0]+0.1,list[0].data.fp_lats[-1]+0.1], crs=cartopy.crs.PlateCarree())
    ax.set_xticks(list[0].data.fp_lons[::tickskip], crs=cartopy.crs.PlateCarree())
    lon_formatter = LongitudeFormatter(number_format='.1f', degree_symbol='', dateline_direction_label=True)
    ax.xaxis.set_major_formatter(lon_formatter)  
    
    ax.set_yticks(list[0].data.fp_lats[::tickskip], crs=cartopy.crs.PlateCarree())
    lat_formatter = LatitudeFormatter(number_format='.1f',  degree_symbol='',)
    ax.yaxis.set_major_formatter(lat_formatter)             
    ax.tick_params(axis='both', which='major', labelsize=12)   

    # marker for release point
    point_coarse = ax.plot(list[0].data.release_lon+0, list[0].data.release_lat+0, marker='o', c="w", markeredgecolor = "k", transform=cartopy.crs.PlateCarree(), markersize=5)
    point_OG = ax.plot(list[-1].data.release_lon+0, list[-1].data.release_lat+0, marker='x', c="k", transform=cartopy.crs.PlateCarree(), markersize=5)

    # 50 or 110
    ax.coastlines(resolution='50m', color='black', linewidth=2)

    # title
    ax.set_title("Emulator-generated footprint - "+ list[0].data.site + "\n" + date.strftime("%m/%d/%Y, %H:00"), fontsize = 17)

    fig.show()

def coarsened_vs_footprint(data1_coarsened, data2_uncoarsened, LoadData, date, fixed_cbar=False):
    '''
    Coarsened grid plot (above) against footprint data.
    '''


    if type(date) == str:
        try:
            date = datetime.strptime(date, "%H:00 %d/%m/%Y")
        except:
            print("Something was wrong with the date. Please pass it in format hour:00 day/month/year (eg 15:00 1/3/2016)")
        
        try:
            idx = pd.DatetimeIndex(data1_coarsened.data.met.time.values[data1_coarsened.jump:-3]).get_loc(date)
        except:
            raise KeyError("This date is out of range")

    if type(date) == int:
        idx = date
        try:
            date = pd.DatetimeIndex(data1_coarsened.data.met.time.values[data1_coarsened.jump:-3])[idx]
        except:
            raise KeyError("This index is out of range")

    if fixed_cbar==False:
        vmax = np.nanmax(data2_uncoarsened.truths[idx,:])
    else:
        vmax = fixed_cbar

    ## create figure and plot
    fig, (axr, axp) = plt.subplots(1,2,figsize = (12,7), subplot_kw={'projection':cartopy.crs.Mercator()})
    
    ## coarsest first
    axp.pcolormesh(data1_coarsened.data.fp_lons, data1_coarsened.data.fp_lats, 
                  np.reshape(data1_coarsened.predictions[idx,:], (data1_coarsened.size, data1_coarsened.size)), 
                  transform=cartopy.crs.PlateCarree(), cmap="Reds", vmax = vmax, vmin=0)
    
    ## OG last
    c = axp.pcolormesh(data2_uncoarsened.data.fp_lons, data2_uncoarsened.data.fp_lats, 
                  np.reshape(data2_uncoarsened.predictions[idx,:], (data2_uncoarsened.size, data2_uncoarsened.size)), 
                  transform=cartopy.crs.PlateCarree(), cmap="Reds", vmax = vmax, vmin=0)
    
    axr.pcolormesh(LoadData.fp_lons, LoadData.fp_lats, np.reshape(LoadData.fp_data[idx+data1_coarsened.jump,:], (LoadData.size, LoadData.size)), transform=cartopy.crs.PlateCarree(), cmap="Reds", vmin=0)


    ## set up axis
    for ax in [axr, axp]:
        # set extent defines the domain to plot
        ax.set_extent([data1_coarsened.data.fp_lons[0],data1_coarsened.data.fp_lons[-1], data1_coarsened.data.fp_lats[0],data1_coarsened.data.fp_lats[-1]], crs=cartopy.crs.PlateCarree())
        ax.set_xticks(data1_coarsened.data.fp_lons[::2], crs=cartopy.crs.PlateCarree())
    
        lon_formatter = LongitudeFormatter(number_format='.1f', degree_symbol='', dateline_direction_label=True)
        ax.xaxis.set_major_formatter(lon_formatter)  
        ax.set_yticks(data1_coarsened.data.fp_lats[::2], crs=cartopy.crs.PlateCarree())
        lat_formatter = LatitudeFormatter(number_format='.1f',  degree_symbol='',)
        ax.yaxis.set_major_formatter(lat_formatter)             
        ax.tick_params(axis='both', which='major', labelsize=12)   

        # marker for release point
        ax.plot(data1_coarsened.data.release_lon+0, data1_coarsened.data.release_lat+0, marker='o', c="w", markeredgecolor = "k", transform=cartopy.crs.PlateCarree(), markersize=5)
        # 50 or 110
        ax.coastlines(resolution='50m', color='black', linewidth=2)

    
    axr.set_title("Fp data - "+ data1_coarsened.data.site + "\n" + date.strftime("%m/%d/%Y, %H:00"), fontsize = 17)
    axp.set_title("GBRT w/ coarsening - "+ data1_coarsened.data.site + "\n" + date.strftime("%m/%d/%Y, %H:00"), fontsize = 17)

    ## set up cbar
    cbar = plt.colorbar(c, ax=[axr, axp], orientation="vertical", aspect = 15, pad = 0.02)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("sensitivity, (mol/mol)/(mol/m2/s)", size = 15, loc="center", labelpad = 16)

    fig.show()

def make_footprint_gif_v2(data1, data2, LoadData, start_date, end_date, savepath=None):
    """
    Make gif of footprints (real and predicted side-by-side) for a range of dates.
    """

    if type(start_date) == str and type(end_date) == str:
        try:
            start_date = datetime.strptime(start_date, "%H:00 %d/%m/%Y")
            end_date = datetime.strptime(end_date, "%H:00 %d/%m/%Y")
        except:
            print("Something was wrong with the date. Please pass it in format hour:00 day/month/year (eg 15:00 1/3/2016)")
        
        try:
            start_idx = pd.DatetimeIndex(data1.data.met.time.values[data1.jump:-3]).get_loc(start_date)
            end_idx = pd.DatetimeIndex(data1.data.met.time.values[data1.jump:-3]).get_loc(end_date)
        except KeyError:
            raise KeyError("One of the dates is out of range")

        assert start_idx<end_idx, "Start date should be before end date"

    if type(start_date) == int and type(end_date) == int:
        start_idx, end_idx = start_date, end_date 
        assert start_idx<end_idx, "Start date should be before end date"
        try:
            start_date = pd.DatetimeIndex(data1.data.met.time.values[data1.jump:-3])[start_idx]
            end_date = pd.DatetimeIndex(data1.data.met.time.values[data1.jump:-3])[end_idx]
        except KeyError:
            raise KeyError("One of the dates is out of range")

    filenames = []

    vmax = 0.25*np.nanmax(data1.truths[start_idx:end_idx,:])

    if savepath==None:
        savepath="footprints_"+ start_date.strftime("%H-%d-%m-%Y") + "_" + end_date.strftime("%H-%d-%m-%Y") + ".gif"
        print("saving as ", savepath )

    # plot each figure and save
    for t in range(start_idx, end_idx):
        coarsened_vs_footprint(data1, data2, LoadData, t, fixed_cbar=vmax)
        #data1.plot_footprint(t, fixed_cbar=vmax)
        filename = f'{t}.png'
        filenames.append(filename)       
        plt.savefig(filename)
        plt.close()

    # write gif
    try:
        with imageio.get_writer(savepath, mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
    except Exception as inst:
        print("Gif could not be saved ", inst)
    # Remove files
    for filename in set(filenames):
        os.remove(filename)

def R_squared_plot(self, coarsen_factor, time_back, training_years):
    '''
    Plotting grid of R2 values for the 3x coarsened grid.
    '''
    middle_section = find_center_indexes(self.size, coarsen_factor)
    r2 = []
    r2_mean = []
    for x in range(self.size**2):
        if x in middle_section: r2.append(0)
        else:
            r2.append(r2_score(self.truths[:,x], self.predictions[:,x]))
            r2_mean.append(r2_score(self.truths[:,x], self.predictions[:,x]))
    r2_grid = np.reshape(r2, (self.size, self.size))
    
    plt.imshow(r2_grid, cmap='hot', origin='lower', vmin=0, vmax=0.75)  

    ## set up cbar
    plt.colorbar(orientation="vertical", aspect = 15, pad = 0.02)

    plt.title(self.data.site + ' Trained ' + training_years + ' Coarsened 3x\nR2 Predictions vs Truths - ' + time_back)
    print(f"Mean R2 external area = {np.mean(r2_mean):.4f}")

def compare_times_R2_values(time_a, time_b, coarsen_factor, times_back, training_years):
    '''
    Plotting grid of compared R2 values for evaluating the ideal number of hours for each coarsened factor.
    '''
    assert time_a.size == time_b.size

    middle_section = middle_section = find_center_indexes(time_a.size, coarsen_factor)
        
    time_a_r2 = []
    time_b_r2 = []

    r2_scores = []

    for x in range(time_a.size**2):
        if x in middle_section: 
            time_a_r2.append(0)
            time_b_r2.append(0)
        else: 
            time_a_r2.append(r2_score(time_a.truths[:,x], time_a.predictions[:,x]))
            time_b_r2.append(r2_score(time_b.truths[:,x], time_b.predictions[:,x]))
            r2_scores.append((r2_score(time_a.truths[:,x], time_a.predictions[:,x]))-(r2_score(time_b.truths[:,x], time_b.predictions[:,x])))
    
    r2 = np.array(time_a_r2) - np.array(time_b_r2)
    r2_grid = np.reshape(r2, (time_a.size,time_a.size))

    plt.imshow(r2_grid, cmap='RdYlGn', origin='lower', vmin=-0.25, vmax=0.25)

    ## set up cbar
    plt.colorbar(orientation="vertical", aspect = 15, pad = 0.02)
    mean_score = np.mean(r2_scores).round(3)
    if mean_score > 0: sign = ''
    else: sign = ''
    plt.title(time_a.data.site + ' Trained ' + training_years + ' Coarsened 3x\nDifference in R2 values when training with\n' + times_back + '\nMean difference = ' + sign + str(mean_score))

def get_all_inputs_v3(variables_past, jumps, variables_nopast, size=10):
    # Does things
    if not (0 in jumps):
        jumps.append(0)
    jumps = sorted(jumps)
    n_samples = np.shape(variables_past[0])[-1] - np.max(jumps) - 3
    n_vars = len(variables_past)*len(jumps) + len(variables_nopast)
    all_vars = np.zeros((n_samples, n_vars*(size**2)))

    col = 0

    for v in variables_past:
        v = np.transpose(v, [2,0,1])
        v = np.reshape(v, (np.shape(v)[0], np.shape(v)[1]**2))

        for jump in jumps:
            if jump==0:
                all_vars[:,col*size**2:(col+1)*size**2] = v[jumps[-1]:-3, :]            
            else:
                all_vars[:, col*size**2:(col+1)*size**2] = v[jumps[-1]-jump:-jump-3:, :]  
            col+=1

    for v in variables_nopast:
        if len(np.shape(v))==3:
            v = np.transpose(v, [2,0,1])
            v = np.reshape(v, (np.shape(v)[0], np.shape(v)[1]**2))
            
        all_vars[:,col*size**2:(col+1)*size**2] = v[jumps[-1]:-3]
        col+=1

    return all_vars

def import_coarsened_and_train(filename, coarsened_data, hours_before, size):
    # Loading coarsened trained model
    with open(f'/user/work/eh19374/LPDM-emulation-trees_TESTING/trained_models/' + filename + '.txt', 'rb') as f:
        [info, clfs] = pickle.load(f)
    print("Trained model info:", info)

    ## variables that are passed at the time of the footprint and x hours before
    vars_with_past = [coarsened_data.y_wind, coarsened_data.x_wind, coarsened_data.met.PBLH.values]
    ## variables that are only passed at the time of the footprint
    vars_without_past = [coarsened_data.temp_grad, coarsened_data.x_wind_grad, coarsened_data.y_wind_grad]

    inputs = get_all_inputs_v3(vars_with_past, hours_before, vars_without_past, size)

    return MakePredictions(clfs, coarsened_data, inputs, max(hours_before), coarsened=True)

def Uncoarsen(dataset, dataset_coarsened, dataOG, coarsen_factor, show_plot=False, date=0):
    data = dataset.predictions
    data_fp_coarse = dataset_coarsened.fp_data

    data = data.reshape(len(data), dataset.size, dataset.size)
    data_fp_coarse = data_fp_coarse.reshape(len(data_fp_coarse), dataset.size, dataset.size)
    data_fp_coarse = data_fp_coarse[dataset.jump:-3]
    
    array = xr.DataArray(data, coords=[('time', dataset.data.met.time.values[dataset.jump:-3]),
                                        ('lat', dataset.data.fp_lats),
                                        ('lon', dataset.data.fp_lons)])
    
    fp = xr.DataArray(data_fp_coarse, coords=[('time', dataset.data.met.time.values[dataset.jump:-3]),
                                            ('lat', dataset.data.fp_lats),
                                            ('lon', dataset.data.fp_lons)])
    
    new_lats = dataOG.fp_data_full.lat
    new_lons = dataOG.fp_data_full.lon
    
    array = array.interp(lat=new_lats, lon=new_lons, method='nearest')
    array = xr.where((array > 0) & ~np.isnan(array), array/coarsen_factor, array)

    fp = fp.interp(lat=new_lats, lon=new_lons, method='nearest')
    fp = xr.where((fp > 0) & ~np.isnan(fp), fp/coarsen_factor, fp)

    if show_plot==True:
        plot_xarray(array, date, dataset.data.site, coarsen_factor)

    return array, fp

def combine_all(list_of_xarrays, original_data, return_all=False):
    # combine uncoarsened datasets
    combined = list_of_xarrays[0]
    for x in range(len(list_of_xarrays)-1):
        combined = combined.combine_first(list_of_xarrays[x+1])
    
    combined = xr.where((combined < 0.0005) & ~np.isnan(combined), 0, combined)

    # place OG at centre
    OG_data = original_data.fp_data.reshape(-1, 10, 10)
    OG_data = OG_data[6:-3]

    # Find indices in combined that correspond to the indices in OG_data
    og_lat_indices = np.searchsorted(combined.lat, original_data.fp_lats)
    og_lon_indices = np.searchsorted(combined.lon, original_data.fp_lons)

    # Replace values using the indices
    combined[:, og_lat_indices, og_lon_indices] = OG_data

    # Create copy of OG xarray to use to fill surrounding values
    fp_data = np.array(original_data.fp_data_full.fp)
    fp_data = np.transpose(fp_data, (2,0,1))[6:-3]
    OG_copy = xr.DataArray(fp_data, coords=[('time', np.array(original_data.fp_data_full.time[6:-3])),
                                            ('lat', np.array(original_data.fp_data_full.lat)),
                                            ('lon', np.array(original_data.fp_data_full.lon))])

    final = xr.where(~np.isnan(combined), combined, OG_copy)

    if return_all == True: return final, combined, OG_copy
    else: return final

def trim_coarsened_and_OG(coarsened_combined, OG_data, coarsened_fp, return_data=True):
    combined = coarsened_combined.dropna(dim='lat', how='all')
    combined = combined.dropna(dim='lon', how='all')
    if return_data==False: combined_data = combined
    else: combined_data = combined.data

    lat_mask = OG_data.lat.isin(combined.lat.values)
    lon_mask = OG_data.lon.isin(combined.lon.values)

    filtered_OG = OG_data.where(lat_mask, drop=True).where(lon_mask, drop=True)
    if return_data==False: filtered_OG_data = filtered_OG
    else: filtered_OG_data = filtered_OG.data

    filtered_coarsened = coarsened_fp.where(lat_mask, drop=True).where(lon_mask, drop=True)
    if return_data==False: filtered_coarsened_data = filtered_coarsened
    else: filtered_coarsened_data = filtered_coarsened.data

    return combined_data, filtered_OG_data, filtered_coarsened_data

def binarize_array(array, threshold):
    # Create a new array of the same shape as the input array
    binarized_array = np.zeros_like(array, dtype=np.int)
    
    # Set values greater than or equal to the threshold to 1
    binarized_array[array >= threshold] = 1
    
    return binarized_array

def find_center_indexes(grid_size, border_width):
    center_indexes = []
    for i in range(border_width+1, grid_size - border_width):
        for j in range(border_width+1, grid_size - border_width):
            # Convert 2D index to 1D index
            center_indexes.append(i * grid_size + j)
    return center_indexes

def pad_and_coarsen_dataset(dataset, coarsen_factor):
    # Calculate the padding necessary for coarsening
    lat_pad_left = np.ceil(dataset.dims['lat'] % coarsen_factor).astype(int)
    lat_pad_right = coarsen_factor - lat_pad_left

    lon_pad_left = np.ceil(dataset.dims['lon'] % coarsen_factor).astype(int)
    lon_pad_right = coarsen_factor - lon_pad_left

    # Pad the dataset's dimensions
    if coarsen_factor == 2 or coarsen_factor == 3:
        dataset = dataset.pad(lat=(lat_pad_left, lat_pad_right), mode='constant')
        dataset = dataset.pad(lon=(lon_pad_left-1, lon_pad_right), mode='constant')
    elif coarsen_factor == 4:
        dataset = dataset.pad(lat=(lat_pad_left+1, lat_pad_right), mode='constant')
        dataset = dataset.pad(lon=(lon_pad_left, lon_pad_right), mode='constant')
    else:
        dataset = dataset.pad(lat=(lat_pad_left, lat_pad_right), mode='constant')
        dataset = dataset.pad(lon=(lon_pad_left, lon_pad_right), mode='constant')

    # Coarsen the dataset
    dataset = dataset.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary='pad').sum()

    return dataset

def intersection_over_union(fps, preds, zero=-1):
    assert (np.unique(fps) == np.array([zero,1])).all(), "pass binary footprints, or if they arent -1/1, pass parameter zero="
    assert len(np.shape(fps))<=2, "currently this only supports flattened arrays (of shape (samples x pixels))"
    intersection = np.sum(np.logical_and(fps==1, preds==1, where=1), axis=-1)
    union = np.sum(np.logical_or(fps==1, preds==1, where=1), axis=-1)
    IoU = intersection/union

    return IoU

def dice_similarity(fps, preds, zero=-1):
    assert (np.unique(fps) == np.array([zero,1])).all(), "pass binary footprints, or if they arent -1/1, pass parameter zero="
    assert len(np.shape(fps))<=2, "currently this only supports flattened arrays (of shape (samples x pixels))"
    TP = np.sum(np.logical_and(fps==1, preds==1), axis=-1) 
    FP = np.sum(np.logical_and(fps==zero, preds==1), axis=-1) 
    FN = np.sum(np.logical_and(fps==1, preds==zero), axis=-1)
    dice = 2*TP/(2*TP + FP + FN)

    return dice

def plot_IoU_dice(iou_array, dice_array, site, coarsened_OG=False):
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].hist(iou_array, bins=50)
    ax[0].set_title(f'{site} IoU\n(mean = {iou_array.mean():.2f})')

    ax[1].hist(dice_array, bins=50)
    ax[1].set_title(f'{site} Dice Similarity\n(mean = {dice_array.mean():.2f})')

    if coarsened_OG == True: plt.suptitle(f'{site}\npredictions vs coarsened truths')
    else: plt.suptitle(f'{site}\npredictions vs original truths')
    plt.show()

def predict_fluxes(true_fp, pred_fp, flux, units_transform = "default"):
    ## convolute predicted footprints and fluxes, returns two np arrays, one with the true flux and one with the emulated flux, of shape (n_footprints,)
    ## flux is a 2D array, regridded and cut to the same resolution and size of the footprints
    ## units_transform can be None (use fluxes directly), "default" (performs flux*1e3 / CH4molarmass) or another function (which should return an array of the same shape as the original flux)
    ## true_fp and pred_fp should both have shape (n_footprints, lat, lon)

    if units_transform != None:
        if units_transform == "default":
            molarmass = 16.0425
            flux = flux*1e3 / molarmass
        else:
            flux = units_transform(flux)
    true_concentration = true_fp*flux
    true_flux = np.sum(true_concentration, axis = (1,2))
    pred_concentration = pred_fp*flux
    pred_flux = np.sum(pred_concentration, axis = (1,2))
    
    return true_flux, pred_flux

def mean_bias(truths, preds):
  # returns dataset wise bias. Positive means underprediction, negative overprediction
    return np.mean(truths-preds)

def plot_fluxes(true_flux, pred_flux, fp_data, site, coarsened_LPDM=False):
    '''
    Plots mol fraction from predictions, uses fp_data for date formatting
    '''
    fig, axis = plt.subplots(1,1,figsize = (15,4))

    fontsizes = {"title":17, "labels": 15, "axis":10}

    dates = pd.DatetimeIndex(fp_data.time.values)
    
    if coarsened_LPDM == True: axis.plot(dates, 1e6*true_flux, label = "using coarsened LPDM-generated footprints", linewidth=2 ,c="#2c6dae")
    else: axis.plot(dates, 1e6*true_flux, label = "using original LPDM-generated footprints", linewidth=2 ,c="#2c6dae")
    axis.plot(dates, 1e6*pred_flux, label = "using coarsened emulated footprints", linewidth=2 ,c="#989933")

    axis.yaxis.offsetText.set_fontsize(0)
    
    axis.set_ylim([0,0.0000001+1e6*np.max([pred_flux, true_flux])])

    axis.set_ylabel('Above baseline methane concentration, (micro mol/mol)', fontsize=fontsizes["axis"])
    axis.set_title('Above baseline methane concentration for ' + site)

    axis.legend()

    plt.show()