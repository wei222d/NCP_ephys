# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:19:36 2023

@author: NMZ  adapted from Junhao's code
"""

'''
veriables terminalogy:

Spike:    
    spike_clusters,  timestamps,    spike_time             their row id means the sequence of spikes based on occurancy
    (cluster id)   (sample count)  (realworld time sec)    their value
    
    spike_clusters_stay,timestamps_stay,spiketime_stay    same as above, if you are using speed mask, this means low velocity part, (no stay) means high velocity
    
    clusters_quality,      its row id means cluster id, value means quality output from kilosort or manually edited
    
    spike_pack,            means a pack contains (spike_clusters, timestamps, spiketime), when in spatial task it contains 2 set of previous components devided by high speed and low speed
    
Sync:    
    vsync,                      esync_timestamps              their row id means the sequence of sync pulse
    (PCtimestamp in the video) (sample count in ns6 file)     their value
    
    signal_on_timestamps    basically same as esync_timestamps, except pulse means external input
    
    signal_on_time          convert timestamp(machine sample) to realworld time
    
Behavior:
    dlc_files               a DataFrame structured same as DLC output csv or h5, value means pixel, after convert may mean cm
    
    Session.frame_time      generated after sync cut, its row means when the frame is recorded after sync start 
    (real world time sec)
    
    Session.XXX             if have same length of frame_time, it means the State of this frame, include mouse postion, task phase... other behavior properties

About Map Axis start:
    dlc_files  mouse_pos  mouse_pose_bin  dwell_map      left up
    
    head direction             head up (forward) ↑ = 0 degree 0 radian,  head down (backward) ↓ = ±180 degree, ±π radian

About X,Y indices:
    dlc_files mouse_pos mouse_bin   (X,Y)
    
    dwell_map firing_map rate_map 
    place_field  PF_COM              (Y,X)  for  (row,column)

'''
#%% Main Bundle
# ----------------------------------------------------------------------------
#    LOTS OF THINGS TO BE DONE.
# ----------------------------------------------------------------------------
# later, mind if there are some nans in DLC files.
# jumpy detection, or smooth, Kalman Filter!
# in class unit, furthur work with its quality check like L-ratio and others. May need to load more files from KS&phy2.
# LFP&spike, their binding do not need anything related to videos. Well except for spd thresh, or maybe some relation with its position.
# decoding. some bayesian?

# ----------------------------------------------------------------------------
#                  Packages 
# ----------------------------------------------------------------------------

import brpylib, time, random, pickle, cv2, pywt, numpy as np, pandas as pd, matplotlib.pyplot as plt, numpy_groupies as npg
from scipy import optimize, ndimage, interpolate, stats, signal
from pathlib import Path
from tkinter import filedialog

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#%% Basic Functions
# ----------------------------------------------------------------------------
#                  Functions here
# ----------------------------------------------------------------------------
      
def load_files_newcam(fdir, fn, dlc_tail, tags):                     
    # if Nses > 1, mind the rule of name.
    spike_times = np.load(fdir/fn/'spike_times.npy')
    spike_times = np.squeeze(spike_times)# delete that stupid dimension.
    spike_clusters = np.load(fdir/fn/'spike_clusters.npy')
    clusters_info = pd.read_csv(fdir/fn/'cluster_info.tsv', sep='\t')
    esync_timestamps_load = np.load(fdir/fn/(fn+'_Frame_timestamps'+'.npy'))
    
    
    waveforms = {'raw':0, 'spike_id':0, 'cluster':0, 'timestamp':0, 'channel':0, 'tetrode':0}
    waveforms['raw'] = np.load(fdir/fn/'_phy_spikes_subset.waveforms.npy')[:,11:71,:]  # (randomized select?) n waveforms, 60 for sample length, 16 for 16 channels
    waveforms['spike_id'] = np.load(fdir/fn/'_phy_spikes_subset.spikes.npy')          # waveform belong to which spike  (in order, 1st spike 2nd spike 3rd spike...)
    waveforms['cluster'] = spike_clusters[waveforms['spike_id']]                        # this spike belong to which cluster
    waveforms['timestamp'] = spike_times[waveforms['spike_id']]                         # when happen
    waveforms['channel'] = np.load(fdir/fn/'_phy_spikes_subset.channels.npy')      # waveform happen on which channel, though we use tetrode, only 4 channel contain the useful information, but phy out put continuous 12 channel info
    waveforms['tetrode'] = waveforms['channel'][:,0]//4                                 # first in channel must be the most significant channel which used to calculate the id of tetrode
    
    if len(list((fdir/fn).glob('*_Signal_on_timestamps.*')))==1:
        signal_on_timestamps_load = np.load(fdir/fn/(fn+'_Signal_on_timestamps'+'.npy'))

    if tags['Nses'] == 1:
        timestamps = spike_times
        spike_clusters2 = spike_clusters
        dlch5 = pd.read_hdf(fdir/fn/(fn+dlc_tail))
        
        frame_state = pd.read_csv(fdir/fn/(fn+'.csv'))
        vsync = np.array(frame_state['PCtimestamp'], dtype='float')
                    
        esync_timestamps = esync_timestamps_load
        dlc_files = dlch5
        
        if len(list((fdir/fn).glob('*Signal_on_timestamps.*')))==1:
            signal_on_timestamps = signal_on_timestamps_load


        
# 有多个Session的情况，更改后的具体内容还需要再次检查    
    elif tags['Nses'] > 1:
        filenames = []
        file_temp = 1
        while file_temp is not str():
            file_temp = filedialog.askopenfilename(initialdir=Path(fdir/fn))
            file_temp2 = Path(file_temp)
            filenames.append(file_temp2.name[:-4])
        if filenames[-1] == str():
            filenames = filenames[:-1]
                        
        timestamps = []
        spike_clusters2 = []
        dlc_files = []
        vsync = []
        # if others' dir rule is not like this, use absolute dir from askopenfilename.
        for i in filenames:
            dlch5 = pd.read_hdf(fdir/fn/(i+dlc_tail))
            dlc_files.append(dlch5)
            
            #vsync这里没有进行framestate文件的适配
            
            vsync_csv = pd.read_csv(fdir/fn/(i+'.csv'), names=[0,1,2])
            vsync_temp = np.array(vsync_csv.loc[:,1], dtype='uint')
            vsync.append(np.where((vsync_temp[1:] - vsync_temp[:-1]) ==1)[0])
       
        # arbituarily more than 10s interval would be made when concatenate ephys files.
        # Hmmm....this may meet some problem if the file recording is stopped right after turnning off sync.
        ses_e_end = esync_timestamps_load[np.where((esync_timestamps_load[1:] - esync_timestamps_load[:-1]) > 100000)[0]]
        ses_e_end = np.append(ses_e_end, esync_timestamps_load[-1])# last one sync needed here.
        esync_timestamps = [esync_timestamps_load[np.where(esync_timestamps_load < ses_e_end[0] + 100000)]]
        
        for i in range(1, tags['Nses']):
            esync_temp = esync_timestamps_load[np.where(esync_timestamps_load < ses_e_end[i] + 100000)]
            esync_temp = esync_temp[np.where(esync_temp > ses_e_end[i-1])]
            esync_timestamps.append(esync_temp)
        if len(list((fdir/fn).glob('Signal_on_timestamps_*.*')))==1:
            signal_on_timestamps = []
            for i in range(tags['Nses']):
                signal_on_temp = signal_on_timestamps_load[np.where(signal_on_timestamps_load < esync_timestamps[i][-1])]
                signal_on_temp = signal_on_temp[np.where(signal_on_temp > esync_timestamps[i][0])]
                signal_on_timestamps.append(signal_on_temp)
                
        timestamps.append(spike_times[np.where(spike_times < ses_e_end[0] + 100000)])
        spike_clusters2.append(spike_clusters[np.where(spike_times < ses_e_end[0] + 100000)])
        for i in range(1, tags['Nses']):
            spike_temp = spike_times[np.where(spike_times < ses_e_end[i] + 100000)] 
            cluster_temp = spike_clusters[np.where(spike_times < ses_e_end[i] + 100000)]
            cluster_temp = cluster_temp[np.where(spike_temp > ses_e_end[i-1] + 100000)]
            spike_temp = spike_temp[np.where(spike_temp > ses_e_end[i-1] + 100000)]
            timestamps.append(spike_temp)
            spike_clusters2.append(cluster_temp)
        print('sessions ended at timestamps below')
        print(ses_e_end)


    else:        
        raise Exception('Nses must be a positive integer.')
    

    if len(list((fdir/fn).glob('Signal_on_timestamps_*.*')))==1:
        return spike_clusters2, timestamps, clusters_info, waveforms, vsync, esync_timestamps, dlc_files, frame_state, signal_on_timestamps
    else:
        return spike_clusters2, timestamps, clusters_info, waveforms, vsync, esync_timestamps, dlc_files, frame_state

def sync_check(esync_timestamps, vsync, tags):                            
    if tags['Nses'] == 1:
        if np.size(vsync) != np.size(esync_timestamps):
            raise Exception('N of E&V Syncs do not Equal!!! Problems with Sync!!!')
        else:
            # plot for check.
            fig = plt.figure(figsize=(10,5))
            ax1 = fig.add_subplot(1,2,1)
            ax2 = fig.add_subplot(1,2,2)
            esync_inter = esync_timestamps[1:] - esync_timestamps[:-1]
            vsync_inter = vsync[1:] - vsync[:-1]
            ax1.hist(esync_inter, bins = len(set(esync_inter)))
            ax1.set_title('N samples between Esyncs', fontsize=tags['fontsize']*1.3)
            ax2.hist(vsync_inter, bins = 100)
            ax2.set_title('time(s) between Vsyncs', fontsize=tags['fontsize']*1.3)
            # check interval
            outlier = np.where(esync_inter>np.mean(esync_inter)*1.1)[0]
            if len(outlier) != 0:
                raise Exception('There maybe frame loss')
            print('N of E&V Syncs equal\nThere is no frame loss\nYou may continue.')
            
    else:
        fig = plt.figure(figsize=(10,5))
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)
        ax1.set_title('N samples between Esyncs', fontsize=tags['fontsize']*1.3)
        ax2.set_title('N frames between Vsyncs', fontsize=tags['fontsize']*1.3)
        # legend?
        for i in range(tags['Nses']):
            if np.size(vsync[i]) != np.size(esync_timestamps[i]):
                raise Exception('N of E&V Syncs do not Equal!!! Problems with Sync in ses ', str(i+1), '!!!')
            else:
                print('ses ', str(i+1),' N of E&V Syncs equal. You may continue.')
                esync_inter = esync_timestamps[i][1:] - esync_timestamps[i][:-1]
                vsync_inter = vsync[i][1:] - vsync[i][:-1]
                ax1.hist(esync_inter, bins = len(set(esync_inter)), alpha=0.2)
                ax2.hist(vsync_inter, bins = len(set(vsync_inter)), alpha=0.2)

def sync_cut_head_tail(spike_clusters, timestamps, esync_timestamps):
    spike_clusters = np.delete(spike_clusters, np.where(timestamps > esync_timestamps[-1])[0])
    spike_clusters = np.delete(spike_clusters, np.where(timestamps < esync_timestamps[0])[0])
    timestamps = np.delete(timestamps, np.where(timestamps > esync_timestamps[-1])[0])
    timestamps = np.delete(timestamps, np.where(timestamps < esync_timestamps[0])[0])
    return spike_clusters, timestamps

def timestamps2time(timestamps, esync_timestamps, tags):
    real_time = np.linspace(0, (np.size(esync_timestamps)-1)/tags['sync_rate'], num=np.size(esync_timestamps))
    stamps2time_interp = interpolate.UnivariateSpline(esync_timestamps, real_time, k=1, s=0)
    spiketime = stamps2time_interp(timestamps)
    return spiketime

def signal_stamps2time(esync_timestamps, signal_on_timestamps, tags):
    if tags['Nses'] == 1:
        interp_y = np.linspace(0, (np.size(esync_timestamps)-1)/tags['sync_rate'], num=np.size(esync_timestamps))
        stamps2time_interp = interpolate.UnivariateSpline(esync_timestamps, interp_y, k=1, s=0)
        signal_on_time = stamps2time_interp(signal_on_timestamps)
    
    # if signal is not on in every session, interp would go wrong.
    
    else:
        signal_on_time = [False for i in range(tags['Nses'])]
        for i in range(tags['Nses']):
            if np.size(signal_on_timestamps[i]) > 0:
                interp_y = np.linspace(0, (np.size(esync_timestamps[i])-1)/tags['sync_rate'], num=np.size(esync_timestamps[i]))
                stamps2time_interp = interpolate.UnivariateSpline(esync_timestamps[i], interp_y, k=1, s=0)
                signal_on_time[i] = stamps2time_interp(signal_on_timestamps[i])
            else:
                pass
    return signal_on_time

def smooth_dlc_file(dlc_files):                                                                     # need!!!!!!
    # how to ?
    # in the dlc built-in filter process, no nan value is created, so every block is filled with number, I don't know how it process
    # in this function, will use un-filtered data to discard uncertain value and smooth the trajectory
    return
    
def apply_speed_mask(spike_clusters, timestamps, spiketime, ses, tags, temporal_bin_length=0.02):                 # wait for modify    
    # applying spd_mask means sort spikes into running and staying.
    if tags['theme'] == 'spatial':
        spiketime_bin_id = (spiketime/temporal_bin_length).astype('uint')
        spike_spd_id = ses.spd_mask[spiketime_bin_id]
        high_spd_id = np.where(spike_spd_id==True)[0]
        low_spd_id= np.where(spike_spd_id==False)[0]
        
        spike_clusters_stay = spike_clusters[low_spd_id]
        timestamps_stay = timestamps[low_spd_id]
        spiketime_stay = spiketime[low_spd_id]
        spike_clusters_run = spike_clusters[high_spd_id]
        timestamps_run = timestamps[high_spd_id]
        spiketime_run = spiketime[high_spd_id]
    
        return (spike_clusters_run,timestamps_run,spiketime_run, spike_clusters_stay,timestamps_stay,spiketime_stay)
    else:
        raise Exception('Only spatial related experiment data should be applied with spd mask.')

#%% Functions on 1D Env
# ----------------------------------------------------------------------------
#                  Functions on 1D Env
# ----------------------------------------------------------------------------        
def find_center_circular_track(x,y, fontsize):
    # try another way, circular fitting
    # https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
    def calc_R(xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((x-xc)**2 + (y-yc)**2)

    def f_2(c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = np.mean(x), np.mean(y)
    center_2,ier = optimize.leastsq(f_2, center_estimate)# could get ier or mesg, for more info of output.
    xc_2, yc_2 = center_2
    Ri_2       = calc_R(*center_2)
    R_2        = Ri_2.mean()
    residu_2   = sum((Ri_2 - R_2)**2)
    #plot for check?
    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(111)
    ax1.set_title('scatter of spatial occupancy in pixels.', fontsize=fontsize)
    ax1.scatter(np.append(x, center_2[0]), np.append(y,center_2[1]), s=3, alpha=0.1)
    return center_2, R_2, residu_2       

def boxcar_smooth_1d_circular(arr, kernel_width=20):
    arr_smo = np.convolve(np.array([1/kernel_width]*kernel_width),
                          np.concatenate((arr,arr)))[kernel_width : (arr.shape[0]+kernel_width)]
    arr_smo = np.concatenate((arr_smo[-(kernel_width):], arr_smo[:-(kernel_width)]))
    return arr_smo

def ratemap_1d_circular(spiketime, time2xy_interp, dwell_smo, nspatial_bins):
    
    spk_pol = np.angle(time2xy_interp[0](spiketime) + 1j*time2xy_interp[1](spiketime))
    spk_bin = ((spk_pol+np.pi)/(2*np.pi)*nspatial_bins).astype('uint')
    Nspike_in_bins = npg.aggregate(spk_bin, 1, size=nspatial_bins)
    Nspike_in_bins_smo = boxcar_smooth_1d_circular(Nspike_in_bins)
    ratemap = Nspike_in_bins_smo/dwell_smo
    return ratemap

def positional_information_olypher_1dcircular(spiketime, time2xy_interp, total_time, temporal_bin_length, nspatial_bins):
    t = np.linspace(0, total_time, num=(total_time/temporal_bin_length).astype('uint'), endpoint=False) + 0.5*temporal_bin_length#assign spatial values at midpoint of every time bin.
    pol = np.angle(time2xy_interp[0](t) + 1j*time2xy_interp[1](t))
    pol_temporal_bin = ((pol+np.pi)/(2*np.pi)*nspatial_bins).astype('uint')
    
    spk_time_temp = (spiketime/temporal_bin_length).astype('uint')
    
    spk_count_temporal_bin = npg.aggregate(spk_time_temp, 1, size=(total_time/temporal_bin_length).astype('uint'))
    p_k = npg.aggregate(spk_count_temporal_bin,1)/np.sum(npg.aggregate(spk_count_temporal_bin,1))
    
    pos_info = []
    for i in range(nspatial_bins):
        spk_count_temporal_bin_xi = spk_count_temporal_bin[np.where(pol_temporal_bin==i)]
        p_kxi = npg.aggregate(spk_count_temporal_bin_xi, 1)/np.sum(npg.aggregate(spk_count_temporal_bin_xi, 1))
        pos_info.append(np.sum(p_kxi * np.log2(p_kxi/p_k[:np.size(p_kxi)])))# set range for p_k is that, e.g. some time bin might have 8 spks or more but not in certain spatial bin, then arrays are not the same length. 
    return np.array(pos_info)
         
def shuffle_test_1d_circular(u, session, experiment_tag, temporal_bin_length=0.02, nspatial_bins_spa=360, nspatial_bins_pos=48, p_threshold=0.01):
    # Ref, Monaco2014, head scanning, JJKnierim's paper.
    # Units must pass shuffle test and either their spa_info >1 or max_pos_info >0.4
    
    # not working well so far.
    
    units = []
    for i in u:
        if u.type == 'excitatory':
            units.append(i)
    spatial_info_pool = []
    positional_info_pool = []
    if experiment_tag['Nses'] == 1:
        for i in units:
            spiketime = np.hstack((i.spiketime, i.spiketime_stay))
            spiketime = session.total_time - spiketime# invert
            spiketime.sort()
            for k in range(1000):
                spiketime += (session.total_time-8) * random.random() + 4 #offset should be at least 4s away from start/end of the session.
                spiketime[np.where(spiketime>session.total_time)] = spiketime[np.where(spiketime>session.total_time)] - session.total_time# wrap back.
                # spd_masking
                spiketime_bin_id = (spiketime/temporal_bin_length).astype('uint')
                spiketime_run = spiketime[session.spd_mask[spiketime_bin_id]]
                # spatial info
                ratemap = ratemap_1d_circular(spiketime_run, session.time2xy_interp, session.dwell_smo, nspatial_bins_spa)
                spa_info = spatial_information_skaggs(spiketime_run, ratemap, session.dwell_smo)
                spatial_info_pool.append(spa_info)
                # positional info
                pos_info = positional_information_olypher_1dcircular(spiketime_run, session.time2xy_interp, session.total_time, temporal_bin_length, nspatial_bins_pos)
                positional_info_pool.append(np.nanmax(pos_info))
                
        spatial_info_pool = np.sort(np.array(spatial_info_pool))
        positional_info_pool = np.sort(np.array(positional_info_pool))
        shuffle_bar_spa = spatial_info_pool[int(np.size(spatial_info_pool)*p_threshold) * (-1)]
        shuffle_bar_pos = positional_info_pool[int(np.size(spatial_info_pool)*p_threshold) * (-1)]
        print('Shuffle results: spatial_info {0}, positional_info {1}'.format(round(shuffle_bar_spa,4), round(shuffle_bar_pos,4)))
        for i in units:
            if i.spatial_info>shuffle_bar_spa and i.positional_info>shuffle_bar_pos:
                if i.spatial_info>1 or i.positional_info>0.4:
                    i.is_place_cell = True
    if experiment_tag['Nses'] > 1:
        for i in units:
            for j in session:
                spiketime = np.hstack((i.spiketime[j.id], i.spiketime_stay[j.id]))
                spiketime = j.total_time - spiketime# invert
                spiketime.sort()
                for k in range(1000):
                    spiketime += (j.total_time-8) * random.random() + 4 #offset should be at least 4s away from start/end of the session.
                    spiketime[np.where(spiketime>j.total_time)] = spiketime[np.where(spiketime>j.total_time)] - j.total_time# wrap back.
                    # spd_masking
                    spiketime_bin_id = (spiketime/temporal_bin_length).astype('uint')
                    spiketime_run = spiketime[j.spd_mask[spiketime_bin_id]]
                    # spatial info
                    ratemap = ratemap_1d_circular(spiketime_run, j.time2xy_interp, j.dwell_smo, nspatial_bins_spa)
                    spa_info = spatial_information_skaggs(spiketime_run, ratemap, j.dwell_smo)
                    spatial_info_pool.append(spa_info)
                    # positional info
                    pos_info = positional_information_olypher_1dcircular(spiketime_run, j.time2xy_interp, j.total_time, temporal_bin_length, nspatial_bins_pos)
                    positional_info_pool.append(np.nanmax(pos_info))
                    
                    
        spatial_info_pool = np.sort(np.array(spatial_info_pool))
        positional_info_pool = np.sort(np.array(positional_info_pool))
        shuffle_bar_spa = spatial_info_pool[int(np.size(spatial_info_pool)*p_threshold) * (-1)]
        shuffle_bar_pos = positional_info_pool[int(np.size(spatial_info_pool)*p_threshold) * (-1)]
        print('Shuffle results: spatial_info {0}, positional_info {1}'.format(round(shuffle_bar_spa,4), round(shuffle_bar_pos,4)))
        for i in units:
            for j in session:
                if i.spatial_info[j.id]>shuffle_bar_spa and i.positional_info[j.id]>shuffle_bar_pos:
                    if i.spatial_info[j.id]>1 or i.positional_info[j.id]>0.4:
                        i.is_place_cell[j.id] = True

def cal_rotational_corrcoef(ratemap1, ratemap2, bin_increment=3, nspatial_bins=360):
    rotational_corrcoef = []
    for i in range(0, nspatial_bins, bin_increment):
        ratemap_rotate1 = np.concatenate((ratemap2[i:], ratemap2[:i]))
        rotational_corrcoef.append(np.corrcoef(ratemap1, ratemap_rotate1)[0,1])
    return np.array(rotational_corrcoef)

#%% Functions on 2D Env
# ----------------------------------------------------------------------------
#                  Functions on 2D Env
# ----------------------------------------------------------------------------
def pixel_to_cm_PlusMaze(dlc_files, nodes_set, fdir, fn):
    
    # kernal_size = 40
    kernal_half_size = 20
    search_size = 50
    #### Kernals ####
    
    kernal = {'LU':0, 'LD':0, 'RU':0, 'RD':0, 'anti_LU':0, 'anti_LD':0, 'anti_RU':0, 'anti_RD':0}
    kernal['LU'] = np.ones((kernal_half_size*2 ,kernal_half_size*2 )) * -1
    kernal['LD'] = np.ones((kernal_half_size*2 ,kernal_half_size*2 )) * -1
    kernal['RU'] = np.ones((kernal_half_size*2 ,kernal_half_size*2 )) * -1
    kernal['RD'] = np.ones((kernal_half_size*2 ,kernal_half_size*2 )) * -1
    kernal['LU'][ 0: kernal_half_size, 0: kernal_half_size] = 1
    kernal['LD'][kernal_half_size:kernal_half_size*2 , 0: kernal_half_size] = 1
    kernal['RU'][ 0: kernal_half_size,kernal_half_size:kernal_half_size*2 ] = 1
    kernal['RD'][kernal_half_size:kernal_half_size*2 ,kernal_half_size:kernal_half_size*2 ] = 1
    kernal['anti_LU'] = np.ones((kernal_half_size*2 ,kernal_half_size*2 ))
    kernal['anti_LD'] = np.ones((kernal_half_size*2 ,kernal_half_size*2 ))
    kernal['anti_RU'] = np.ones((kernal_half_size*2 ,kernal_half_size*2 ))
    kernal['anti_RD'] = np.ones((kernal_half_size*2 ,kernal_half_size*2 ))
    kernal['anti_LU'][ 0: kernal_half_size, 0: kernal_half_size] = -1
    kernal['anti_LD'][kernal_half_size:kernal_half_size*2 , 0: kernal_half_size] = -1
    kernal['anti_RU'][ 0: kernal_half_size,kernal_half_size:kernal_half_size*2 ] = -1
    kernal['anti_RD'][kernal_half_size:kernal_half_size*2 ,kernal_half_size:kernal_half_size*2 ] = -1

    ### average frames across the video ###############################################################
    video_path = str(fdir/fn/(fn+'.avi'))
    cap = cv2.VideoCapture(video_path, cv2.IMREAD_GRAYSCALE)
    ret, frame = cap.read()
    height, width, _ = frame.shape
    
    accumulator = np.zeros((height, width), dtype=np.float32)    
    
    print('this will take long, wait patiently', end='')
    t=time.time()
    frame_count = 0
    while frame_count<1500: #True
        ret, frame = cap.read()
        if not ret:    #检查是否读到帧
            break
        accumulator += frame[:,:,0].astype(np.float32)          # 每一帧累加一次, 因为黑白图像，所以只加0位置的颜色
        frame_count += 1
    print('\rcalculating average frame takes', time.time()-t)
    cap.release()
    
    average_frame = (accumulator / frame_count).astype(np.uint8)
    
    cv2.namedWindow('average_frame', cv2.WINDOW_NORMAL)
    cv2.imshow('average_frame', average_frame)
    cv2.waitKey(0)
    
    ### find corner function ############################################################################
    def get_corner_pos(rough_pos, kernal):
        result = np.ones((height,width))*-9999999
        if rough_pos[0]<(search_size+kernal_half_size) or rough_pos[0]>height-(search_size+kernal_half_size):
            raise ValueError('rough pos is set too close to boundary')
        if rough_pos[1]<(search_size+kernal_half_size) or rough_pos[1]>width-(search_size+kernal_half_size):
            raise ValueError('rough pos is set too close to boundary')
            
        for ih in range(rough_pos[0]-search_size,rough_pos[0]+search_size):              # 100 for search area
            for iw in range(rough_pos[1]-search_size,rough_pos[1]+search_size):
                roi = average_frame[ih-kernal_half_size:ih+kernal_half_size,iw-kernal_half_size:iw+kernal_half_size]             # 60 for kernal size
                result[ih,iw] = np.sum(np.multiply(roi,kernal))
        return (np.where(result==np.max(result))[0][0], np.where(result==np.max(result))[1][0])
    
    ### fit the structure nodes to the precise pixel #####################################################
    t=time.time()
    for i in range(len(nodes_set['name'])):
        nodes_set['precise_position'][i] = get_corner_pos(nodes_set['rough_position'][i], kernal[nodes_set['corner_type'][i]])

    for i in range(len(nodes_set['name'])):
        cv2.circle(average_frame, (nodes_set['precise_position'][i][1], nodes_set['precise_position'][i][0]), 3,(255,255,255),-1)
        cv2.circle(average_frame, (nodes_set['rough_position'][i][1], nodes_set['rough_position'][i][0]), kernal_half_size*2, (130,0,0),2)
    print('fitting precise postion takes', time.time()-t)
    
    cv2.namedWindow('average_frame', cv2.WINDOW_NORMAL)
    cv2.imshow('average_frame', average_frame)
    cv2.waitKey(0)
    
    ### generate a convert matrix based on the precise pixel and real world length #######################
    input_matrix  = np.ones((len(nodes_set['name']),3))
    output_matrix = np.ones((len(nodes_set['name']),3))
    for i in range(len(nodes_set['name'])):
        input_matrix[i][0]  = nodes_set['precise_position'][i][1]
        input_matrix[i][1]  = nodes_set['precise_position'][i][0]
        output_matrix[i][0] = nodes_set['real_position'][i][1]
        output_matrix[i][1] = nodes_set['real_position'][i][0]
    # 使用最小二乘法求解线性映射矩阵 A
    convert_matrix, _, _, _ = np.linalg.lstsq(input_matrix, output_matrix, rcond=None)
    # convert the dlc file
    for i in range(0, int(dlc_files.shape[1]/3)):
        bodypart_position = np.ones((dlc_files.shape[0],3))
        bodypart_position[:,0:2] = dlc_files.iloc[:,3*i:3*i+2]
        bodypart_position = np.dot(bodypart_position, convert_matrix)            # 点乘转换矩阵
        dlc_files.iloc[:,3*i:3*i+2] = bodypart_position[:,0:2]
    
    print('convertion done')

def boxcar_smooth_2d(X_map, boxcar_width = 3):
    boxcar_kernel = np.ones((boxcar_width, boxcar_width)) / (boxcar_width * boxcar_width)
    X_map_smooth = signal.convolve2d(X_map, boxcar_kernel, mode='same')  # to keep edge information, use the mode 'same' keep same size of map
    return X_map_smooth

def gaussian_smooth_2d():
    pass   

def shuffle_test_2d():
    # this would be simple, just go with random temporal offset and play around 1000 times. No worries like in 1d.
    pass

def Kalman_filter_2d():#interpreted from Bohua's code.
    pass

#%% Classes detour session
# ----------------------------------------------------------------------------
#                  Classes detour session 
# ----------------------------------------------------------------------------

class DetourSession(object):
    def __init__(self, dlch5, dlc_col_ind_dict, frame_state, esync_timestamps, tags, ses_id=0):
        
        self.id = ses_id
        self.esync_timestamps = esync_timestamps
        self.sync_rate = tags['sync_rate']
        self.fontsize = tags['fontsize']
        
        dlcmodelstr=dlch5.columns[1][0]
        for key in dlc_col_ind_dict:          # for customized need from DLC.
            pos_for_key = np.vstack((np.array(dlch5[(dlcmodelstr,key,'x')]), np.array(dlch5[(dlcmodelstr,key,'y')]))).T    
            setattr(self, key, pos_for_key)
        
        for key in frame_state.columns:
            setattr(self, key, np.array(frame_state[key]).T)
        
        self.frame_length = len(self.Frame)
        
    def framestamps2time(self):
        self.frametime = self.Frame/self.sync_rate                        #because now the sync is all generated by the output with a latency of camera exposure, so here you can directly divide by sync rate
        self.total_time = self.frametime[-1]
    
    def set_mouse_pos_as(self, key1, key2=0):
        if key2 == 0:
            self.mouse_pos = getattr(self, key1)
        else:
            self.mouse_pos = (getattr(self, key1) + getattr(self, key2)) / 2
                
    def get_head_direction(self):
        hd_vector = self.rightbulb - self.leftbulb
        self.hd_radian = np.angle(hd_vector[:,0] + 1j*hd_vector[:,1])   # head up (forward) ↑ = 0 radian,  head down (backward) ↓ = ±π radian
        self.hd_degree = (self.hd_radian)/(2*np.pi)*360                 # head up (forward) ↑ = 0 degree 0 radian,  head down (backward) ↓ = ±180 degree
            
    def generate_interpolater(self):
        self.get = {}
        
        def time2index(spiketime):
            frame_id = np.round(spiketime*self.sync_rate)       # basically nearest interpolate
            return int(frame_id)
        # time2frame = interpolate.interp1d(self.frametime, np.linspace(0, (np.size(self.frametime)-1),num=np.size(self.frametime)), kind='nearest')
        # self.get = {'time2frame':time2frame }
        
        # def time2index(spiketime):
        #     frame_id = self.get['time2frame'](spiketime)
        #     return int(frame_id)       
        def generate_index_func(obj, key):
            def FUNC(spiketime):
                return getattr(obj, key)[time2index(spiketime)]
            return FUNC
        def generate_mergeXY_func(obj, key):
            def FUNC(spiketime):
                return np.vstack((obj.get[key+'X'](spiketime),obj.get[key+'Y'](spiketime))).T
            return FUNC
                
        for key in vars(self):
            FrameData = getattr(self, key)
            if type(FrameData) is np.ndarray and FrameData.shape[0] == self.frame_length:
                if type(FrameData[0]) == np.bool_ :                             #如果是bool值，那么在进行插值的时候应该取最邻近的值，而且bool值可以被数字比大小，所以单独写一个分支
                    self.get[key] = generate_index_func(self, key) 
                elif type(FrameData[0]) == str :                                #如果是字符串，那么在进行插值的时候应该取最邻近的值
                    self.get[key] = generate_index_func(self, key) 
                elif type(FrameData[0]) == np.ndarray :                         #如果是一个数组，那意味着选到了某个位置（x，y），所以分别对x，y做插值，输出一个（x，y）
                    self.get[key+'X'] = interpolate.UnivariateSpline(self.frametime, FrameData[:,0], k=1, s=0)
                    self.get[key+'Y'] = interpolate.UnivariateSpline(self.frametime, FrameData[:,1], k=1, s=0)         # k=1 means linear interpolation, s=0 I don't know
                    self.get[key] = generate_mergeXY_func(self, key) 
                elif FrameData[0] >= -999:                                        #如果是一个数字,-999是因为头朝向可以是-180
                    self.get[key] = interpolate.UnivariateSpline(self.frametime, FrameData, k=1, s=0)
                else:
                    raise Exception('Error in generating interpolater, value type not defined')
                
    def generate_spd_mask(self, body_part_key, threshold=2, temporal_bin_length=0.02):
        t = np.linspace(0, self.total_time, num=(self.total_time/temporal_bin_length +1).astype('uint'))
        body_part_postion = self.get[body_part_key](t)
        x = body_part_postion[:,0]
        y = body_part_postion[:,1]
        dist = np.sqrt((x[1:]-x[:-1])**2 + (y[1:]-y[:-1])**2)
        self.inst_spd = dist/temporal_bin_length
        self.spd_mask = np.where(self.inst_spd > 2, 1, 0)
        self.spd_mask = np.append(self.spd_mask, 0).astype('bool')# for convinience.
    
    def generate_dwell_map_PlusMaze(self, fdir, fn, smooth='boxcar', temporal_bin_length=0.02):
        # in this method, temperal bin is not deliberately modified, which is the frame interval. 20ms or 16.67ms
        self.mouse_pos_bin = ( (self.mouse_pos+1.5)/2 ).astype(int)
        
        self.dwell_map = np.zeros((29,29))
        for pos_bin in self.mouse_pos_bin:
            if 0<=pos_bin[0]<29 and 0<=pos_bin[1]<29:
                self.dwell_map[pos_bin[1],pos_bin[0]] += temporal_bin_length      # reverse between img(X,Y) and martix(row,column)
        
        self.dwell_map_spdmasked = np.zeros((29,29))
        for pos_bin in self.mouse_pos_bin[self.spd_mask]:
            if 0<=pos_bin[0]<29 and 0<=pos_bin[1]<29:
                self.dwell_map_spdmasked[pos_bin[1],pos_bin[0]] += temporal_bin_length
        
        self.dwell_map_smooth = boxcar_smooth_2d(self.dwell_map, boxcar_width = 3)
        self.dwell_map_spdmasked_smooth = boxcar_smooth_2d(self.dwell_map_spdmasked, boxcar_width = 3)

        fig, axs = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        axs[0].set_box_aspect(1)
        axs[0].set_title('Speed', fontsize=self.fontsize*1.3)
        axs[0].hist(self.inst_spd, range = (0,70), bins = 100)
        axs[0].set_xlim(0,70)
        # spd_bin_max = np.max(np.bincount(self.inst_spd.astype('uint'), minlength=100))
        # axs[0].plot(([np.median(self.inst_spd)]*2), [0, spd_bin_max*0.7], color='k')
        axs[0].axvline(x=np.median(self.inst_spd), color='black', linestyle='dashed', linewidth=2, label='median instant speed')
        axs[0].set_xlabel('animal spd cm/s', fontsize=self.fontsize)
        axs[0].set_ylabel('N 20ms temporal bins', fontsize=self.fontsize)
        
        axs[1].set_title('Trajectory', fontsize=self.fontsize*1.3)
        axs[1].plot(self.mouse_pos[:,0], self.mouse_pos[:,1])
        axs[1].set_aspect(1, adjustable='box')
        axs[1].set_xlim(0,58)
        axs[1].set_ylim(0,58)
        axs[1].set_xlabel('cm', fontsize=self.fontsize)
        axs[1].set_ylabel('cm', fontsize=self.fontsize)
        axs[1].invert_yaxis()
        
        heatmap = self.dwell_map_smooth.copy()
        heatmap[heatmap==0] = np.nan
        axs[2].set_title('Dwell Map', fontsize=self.fontsize*1.3)
        pcm = axs[2].pcolormesh(heatmap, cmap='jet')
        axs[2].set_aspect(1, adjustable='box')
        axs[2].set_xlabel('2cm bin', fontsize=self.fontsize)
        axs[2].set_ylabel('2cm bin', fontsize=self.fontsize)
        axs[2].invert_yaxis()
        fig.colorbar(pcm, ax=axs[2], label='occupancy in sec')
        
        plt.savefig(fdir/fn/'plot'/'Dwell.svg')    
        plt.savefig(fdir/fn/'plot'/'Dwell.png')    

#%% Class unit
 # ----------------------------------------------------------------------------
 #                  Classes unit
 # ----------------------------------------------------------------------------
 
class Unit(object):
    def __init__(self, spike_pack, clusters_info, tags):
        self.KScluid = clusters_info.loc['cluster_id']
        self.quality = clusters_info.loc['group']
        self.KSquality = clusters_info.loc['KSLabel']
        self.type = 'unknown'                                #IN or PC  
        self.meanwaveform = False
        self.channel = clusters_info.loc['ch']
        self.tetrode = self.channel//4
        self.tags = tags
                
        if self.channel < 32:
            self.loc = 'right'
        elif 32<= self.channel < 64:
            self.loc = 'left'
        else:
            raise Exception('not coded for higher channel')

        if tags['Nses'] == 1:
            # unpacking spike_pack.
            spike_pick_cluid = np.where(spike_pack[0] == self.KScluid)[0]
            self.timestamps_run = spike_pack[1][spike_pick_cluid]
            self.spiketime_run = spike_pack[2][spike_pick_cluid]
            spike_stay_pick_cluid = np.where(spike_pack[3] == self.KScluid)[0]
            self.timestamps_stay = spike_pack[4][spike_stay_pick_cluid]
            self.spiketime_stay = spike_pack[5][spike_stay_pick_cluid]
            # initialize spa. params.
            self.positional_info = []
            self.stability = []
            self.is_place_cell = False
            self.Nspikes_total = clusters_info.loc['n_spikes']
        elif self.Nses > 1:
            self.timestamps_run = [0 for i in range(tags['Nses'])]
            self.spiketime_run = [0 for i in range(tags['Nses'])]
            self.timestamps_stay = [0 for i in range(tags['Nses'])]
            self.spiketime_stay = [0 for i in range(tags['Nses'])]
            self.Nspikes_total = [0 for i in range(tags['Nses'])] 
            self.ratemap = [0 for i in range(tags['Nses'])]
            self.peakrate = [0 for i in range(tags['Nses'])]
            self.spatial_info = [0 for i in range(tags['Nses'])]
            self.positional_info = [0 for i in range(tags['Nses'])]
            self.stability = [0 for i in range(tags['Nses'])]
            self.mean_rate_global = [0 for i in range(tags['Nses'])]
            self.mean_rate_running = [0 for i in range(tags['Nses'])]
            self.is_place_cell = [False for i in range(tags['Nses'])]
            # unpacking
            for i in range(tags['Nses']):
                spike_pick_cluid = np.where(spike_pack[i][0] == self.KScluid)[0]
                spike_stay_pick_cluid = np.where(spike_pack[i][3] == self.KScluid)[0]
                self.timestamps_run[i] = spike_pack[i][1][spike_pick_cluid]
                self.spiketime_run[i] = spike_pack[i][2][spike_pick_cluid]
                self.timestamps_stay[i] = spike_pack[i][4][spike_stay_pick_cluid]
                self.spiketime_stay[i] = spike_pack[i][5][spike_stay_pick_cluid]
                self.Nspikes_total[i] = np.size(self.timestamps_run[i]) + np.size(self.timestamps_stay[i])
            
        else:
            print('Wrong input for unit.')
            
    def get_mean_rate(self, ses):
        self.mean_rate_global = (np.size(self.timestamps_run)+np.size(self.timestamps_stay)) / ses.total_time
        self.mean_rate_run = np.size(self.timestamps_run) / (np.sum(ses.spd_mask)/ses.sync_rate)
        self.mean_rate_stay = np.size(self.timestamps_stay) / (ses.total_time - (np.sum(ses.spd_mask)/ses.sync_rate))
    
    def get_rate_map_PlusMaze(self, ses):
        # this rate map is calculated in running state
        mouse_pos = ses.get['mouse_pos'](self.spiketime_run)
        mouse_pos_bin = ( (mouse_pos+1.5)/2 ).astype(int)
        
        self.firing_map = np.zeros((29,29))
        for pos_bin in mouse_pos_bin:
            if 0<=pos_bin[0]<29 and 0<=pos_bin[1]<29:
                self.firing_map[pos_bin[1],pos_bin[0]] += 1      # reverse between img(X,Y) and martix(row,column)
        
        np.seterr(all='ignore')
        self.rate_map = np.divide(self.firing_map, ses.dwell_map_spdmasked)
        self.firing_map_smooth = boxcar_smooth_2d(self.firing_map)
        self.rate_map_smooth = np.divide(self.firing_map_smooth, ses.dwell_map_spdmasked_smooth)
        
        self.peak_rate = np.nanmax(self.rate_map_smooth)                       # peak rate is selected in smooth map
        
        np.seterr(all='warn')
        
    def get_place_field_PlusMaze(self, ses, field_edge_threshold=0.1, sub_field_threshold=0.3):

        field_edge_threshold *= self.peak_rate
        sub_field_threshold *= self.peak_rate

        def expand_place_field(firing_peak):
            place_field = {firing_peak}
            add_bins = {firing_peak}
            while add_bins != set():        # search until no more bin to add
                add_bins = set()
                for i in place_field:
                    if (i[0]+1,i[1]) not in place_field and 0<=i[0]+1<=28 and 0<=i[1]<=28:    # add 4 bordered neighbors to check
                        add_bins.add( (i[0]+1,i[1]) )
                    if (i[0]-1,i[1]) not in place_field and 0<=i[0]-1<=28 and 0<=i[1]<=28:
                        add_bins.add( (i[0]-1,i[1]) )
                    if (i[0],i[1]+1) not in place_field and 0<=i[0]<=28 and 0<=i[1]+1<=28:
                        add_bins.add( (i[0],i[1]+1) )
                    if (i[0],i[1]-1) not in place_field and 0<=i[0]<=28 and 0<=i[1]-1<=28:
                        add_bins.add( (i[0],i[1]-1) )
                bins_to_remove = set()
                for i in add_bins:
                    if np.isnan(self.rate_map_smooth[i[0],i[1]]) or self.rate_map_smooth[i[0],i[1]] < field_edge_threshold:   # check if the potential bins are above the threshold
                        bins_to_remove.add(i)
                add_bins = add_bins - bins_to_remove            
                place_field = place_field | add_bins
            return place_field
        
        def fill_place_field(place_field):
            place_field_array=np.array(list(place_field))
            for i in range(min(place_field_array[:,0]),max(place_field_array[:,0])):
                for j in range(min(place_field_array[:,1]),max(place_field_array[:,1])):
                    horizontal_range = place_field_array[place_field_array[:,0]==i]    # the column where the bin locate
                    vertical_range = place_field_array[place_field_array[:,1]==j]       # the row where the bin locate
                    if min(horizontal_range[:,1])< j <max(horizontal_range[:,1]) and min(vertical_range[:,0])< i <max(vertical_range[:,0]): # if for one bin, its left right up down range all in place field, it is in place field
                        place_field.add((i,j))
                        
        sorted_indices = np.argsort(np.nan_to_num(self.rate_map_smooth),axis=None)                        # sorted by firing rate smooth (low to high)
        row_indices, col_indices = np.unravel_index(sorted_indices[::-1], self.rate_map_smooth.shape)     # convert the sorted sequence to indice (high to low)
        potential_peaks = np.vstack((row_indices,col_indices)).T                                      # merge two indices
        potential_peaks = potential_peaks.tolist()                                                # to list
        potential_peaks = [tuple(sublist) for sublist in potential_peaks]                         # turn the bin indices to turple
        potential_peaks = [item for item in potential_peaks if self.rate_map_smooth[item[0],item[1]] > sub_field_threshold]      # exclude the low firing bin

        self.place_field = []
        while potential_peaks != []:
            potential_peaks[0]                                                  # take the currently highest rate bin
            place_field = expand_place_field(potential_peaks[0])                # expand using this high rate bin
            fill_place_field(place_field)                                       # fill 
            potential_peaks = [item for item in potential_peaks if item not in place_field]  # remove this field from potential peaks
            
            if 4 <= len(place_field):                 # place field area size criterion, 4cm^2 to 200.96cm^2, radius 2cm to 8cm     <= 51
                self.place_field.append(place_field)
                
        self.place_field_map = np.ones((29,29))*-99
        field_id = 0
        for place_field in self.place_field:
            for bins in place_field:
                self.place_field_map[bins[0],bins[1]] = field_id
            field_id +=1

    def get_place_field_COM(self):
        self.place_field_COM = []
        for place_field in self.place_field:
            COM = [0,0]
            sum_rate_field = 0
            for bins in place_field:
                COM[0] += bins[0]*self.rate_map_smooth[bins[0],bins[1]]
                COM[1] += bins[1]*self.rate_map_smooth[bins[0],bins[1]]
                sum_rate_field += self.rate_map_smooth[bins[0],bins[1]]
            COM = [COM[0]/sum_rate_field, COM[1]/sum_rate_field]
            self.place_field_COM.append(COM)

    def get_place_field_ellipse_fit(self):
        
        def get_place_field_contour(place_field):
            place_field_array = np.array(list(place_field))
            place_field_contour = place_field.copy()
            for bins in place_field:
                horizontal_range = place_field_array[place_field_array[:,0]==bins[0]]     # the column where the bin locate
                vertical_range = place_field_array[place_field_array[:,1]==bins[1]]       # the row where the bin locate
                if min(horizontal_range[:,1])< bins[1] <max(horizontal_range[:,1]) and min(vertical_range[:,0])< bins[0] <max(vertical_range[:,0]): # if for one bin, its left right up down range all in place field, it is in place field
                    place_field_contour.remove(bins)
            return place_field_contour
        
        # def ellipse_equation(x, a, b, h, k):
        #     return ((x[:, 0] - h) / a)**2 + ((x[:, 1] - k) / b)**2 - 1
        
        # def ellipse_equation(x, X0, Y0, X1, Y1, d):
        #     l = 2*d + np.sqrt( (X0-X1)**2 + (Y0-Y1)**2 )           # 椭圆上任意一点到两焦点的距离之和
        #     return np.sqrt( (x[:,0]-X0)**2 + (x[:,1]-Y0)**2 )  +  np.sqrt( (x[:,0]-X1)**2 + (x[:,1]-Y1)**2 )  - l     # 任意一点到两焦点的距离 = l ； length - l = 0

        def ellipse_equation(x, h, k, a, b, theta):
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            term1 = ((x[:, 0] - h) * cos_theta + (x[:, 1] - k) * sin_theta) / a
            term2 = ((x[:, 0] - h) * sin_theta - (x[:, 1] - k) * cos_theta) / b
            return term1**2 + term2**2 - 1
        
        
# X0=initial_guess[0]
# Y0=initial_guess[1]
# X1=initial_guess[2]
# Y1=initial_guess[3]
# d=initial_guess[4]

# a=ellipse_equation(place_field_contour_array, X0, Y0, X1, Y1, d)


        def fit_ellipse(place_field_contour):
            # 获取椭圆边界的坐标
            place_field_contour_array=np.array(list(place_field_contour))
            place_field_contour_array = place_field_contour_array+0.5 # 
            # 初始猜测值
            # initial_guess = [np.mean(place_field_contour_array[:, 0]), np.mean(place_field_contour_array[:, 1]),
            #                  np.mean(place_field_contour_array[:, 0]), np.mean(place_field_contour_array[:, 1]),
            #                  2.0]
            
            initial_guess = [np.mean(place_field_contour_array[:, 0]), np.mean(place_field_contour_array[:, 1]), 2.0, 2.0, 0]
            # 拟合椭圆
            # popt, _ = optimize.curve_fit(ellipse_equation, place_field_contour_array, np.zeros(place_field_contour_array.shape[0]), p0=initial_guess)    
            # return popt
            try:
                popt, _ = optimize.curve_fit(ellipse_equation, place_field_contour_array, np.zeros(place_field_contour_array.shape[0]), p0=initial_guess)    
                return popt
            except:
                return initial_guess
        
        #### main ####
        self.place_field_ellipse_parameter = []
        for place_field in self.place_field:
            place_field_contour = get_place_field_contour(place_field)
            ellipse_parameters = fit_ellipse(place_field_contour)
            self.place_field_ellipse_parameter.append(ellipse_parameters)
            
    def get_spatial_information_skaggs(self, ses):
        # for now all using un-smoothed data to calculate, but with speed mask
        np.seterr(all='ignore')
        self.spatial_info = np.nansum((ses.dwell_map_spdmasked/np.sum(ses.dwell_map_spdmasked)) * (self.rate_map/self.mean_rate_run) * np.log2((self.rate_map/self.mean_rate_run)))
        np.seterr(all='warn')
        
    def spatial_info_shuffle_PlusMaze(self, ses):
        # Units must pass shuffle test and either their spa_info >1 or max_pos_info >0.4
        
        # if apply to other maze, change the part of rate map, that part are defined by maze structrue and binning set up
        
        time_range = [4,40]
        shuffle_chunk_size = 6
        shuffle_times = 1000
        
        time_offset = np.random.uniform(time_range[0], time_range[1], shuffle_times)
        time_offset[int(len(time_offset)/2):] *= -1                 # second half minus the offset
        time_offset[int(len(time_offset)/2):] += ses.total_time     # make sure the tiem is positive
        
        self.spatial_info_pool = np.zeros(shuffle_times)
        
        for i in range(shuffle_times):
            # add time offset
            shuffled_spiketime = self.spiketime_run + time_offset[i]
            for j in range(np.size(shuffled_spiketime ,0)):
                if shuffled_spiketime[j] > ses.total_time:
                    shuffled_spiketime[j] -= ses.total_time
            # add chunk disruption
            chunk_list = list(range(0,shuffle_chunk_size))
            shuffled_chunk_list = chunk_list.copy()
            random.shuffle(shuffled_chunk_list)
            chunk_time = ses.total_time / shuffle_chunk_size
            temp_array = shuffled_spiketime.copy()
            for ichunk in chunk_list:
                shift_time = (shuffled_chunk_list[ichunk]-chunk_list[ichunk]) * chunk_time    # a radomized shift time among chunks
                temp_array[(ichunk*chunk_time <= shuffled_spiketime) & (shuffled_spiketime < (ichunk+1)*chunk_time)] += shift_time  # for spike in this chunk, add a shift time to their spike time 
            shuffled_spiketime = temp_array
            
            ############################################
            # calculate spatial info
            mouse_pos = ses.get['mouse_pos'](shuffled_spiketime)
            mouse_pos_bin = ( (mouse_pos+1.5)/2 ).astype(int)        # this is defined maze structrue and binning set up
            
            shuffled_firing_map = np.zeros((29,29))
            for pos_bin in mouse_pos_bin:
                if 0<=pos_bin[0]<29 and 0<=pos_bin[1]<29:
                    shuffled_firing_map[pos_bin[1],pos_bin[0]] += 1      # reverse between img(X,Y) and martix(row,column)
            
            np.seterr(all='ignore')
            shuffled_rate_map = np.divide(shuffled_firing_map, ses.dwell_map_spdmasked)
            shuffled_spatial_info = np.nansum((ses.dwell_map_spdmasked/np.sum(ses.dwell_map_spdmasked)) * (shuffled_rate_map/self.mean_rate_run) * np.log2((shuffled_rate_map/self.mean_rate_run)))
            shuffled_spatial_info = round(shuffled_spatial_info, 4)
            np.seterr(all='warn')
            ###########################################
            self.spatial_info_pool[i] = shuffled_spatial_info
       
    def get_positional_information_olypher_PlusMaze(self, ses):
        np.seterr(all='ignore')

        spike_which_frame = (self.spiketime_run*ses.sync_rate).astype('uint')              # which frame the spike occur
        spike_count_frame = npg.aggregate(spike_which_frame, 1, size=(len(ses.Frame)))     # how many spikes in this frame
        nframe_ispike = npg.aggregate(spike_count_frame,1)  # row is how many spikes, value is the number of frame which has specific spike number
        Pk = nframe_ispike/np.sum(nframe_ispike)            # occurence probability for 0 spike; for 1 spike; for 2spikes; 345...
        
        self.positional_info = np.zeros((29,29))
        for i in range(29):
            for j in range(29):
                spike_count_xi = spike_count_frame[np.where((ses.mouse_pos_bin[:, 0] == i) & (ses.mouse_pos_bin[:, 1] == j))[0]]        # len(self) means how many times are animal in this bin, value means how many spikes this time in this bin 
                if np.sum(spike_count_xi)==0 :
                    positional_info_xi = 0
                else:
                    Pk_xi = npg.aggregate(spike_count_xi, 1)/np.sum(npg.aggregate(spike_count_xi, 1))
                    positional_info_xi = np.nansum(Pk_xi * np.log2(Pk_xi/Pk[:np.size(Pk_xi)]))
                self.positional_info[j,i] = positional_info_xi
        np.seterr(all='warn')
        
    def simple_putative_IN_PC_by_firingrate(self, ses):
        # this is a simple way to put IN and PC not by waveform but only global mean firing rate.
        # see Nuenuebel 2013, DR with L/MEC, threshold of mean firing rate is 10Hz.
        if self.tags['theme'] == 'spatial':
            if self.tags['Nses'] == 1:
                if self.mean_rate_global > 10:
                    self.type = 'inhibory'
                else:
                    self.type = 'excitatory'
            else:
                ses_inds = [i for i in range(1, self.tags['Nses'] + 1)]
                if (np.array(self.mean_rate_global)[ses_inds].all() > 10).all() == True:
                    self.type = 'inhibitory'
                elif (np.array(self.mean_rate_global)[ses_inds] < 10).all() == True:
                    self.type = 'excitatory'
                else:
                    self.type = 'unsure'
        else:
            raise Exception('you might used wrong method.')
    
    def report(self, ses, fdir, fn, unit_id, spatial_info_pool):
        
        fig, axs = plt.subplots(2, 3, figsize=(16, 9), tight_layout=True)
        # plt.subplots_adjust(wspace=0.5, hspace=0.5)
        
        # rate map
        heatmap = self.rate_map_smooth.copy()
        axs[0,0].set_title('Firing Rate Map', fontsize=self.tags['fontsize']*1.3)
        pcm = axs[0,0].pcolormesh(heatmap, cmap='jet')
        axs[0,0].set_aspect(1, adjustable='box')
        axs[0,0].set_xlabel('2cm bin', fontsize=self.tags['fontsize'])
        axs[0,0].set_ylabel('2cm bin', fontsize=self.tags['fontsize'])
        axs[0,0].invert_yaxis()
        fig.colorbar(pcm, ax=axs[0,0], label='firing rate (spike/sec)')
        # for parameter in self.place_field_ellipse_parameter:
        #     h,k,a,b,theta = parameter[0],parameter[1],parameter[2],parameter[3],parameter[4]
        #     ellipse = matplotlib.patches.Ellipse((k, h), a, b, angle=theta, edgecolor='w', facecolor='none', linestyle='--', linewidth=2)
        #     axs[0].add_patch(ellipse)
        scale_factor= 10
        place_field_mapx10 = np.repeat(np.repeat(self.place_field_map, scale_factor, axis=0), scale_factor, axis=1)     
        axs[0,0].contour(place_field_mapx10, levels=[-0.9], colors='white', linewidths=2, extent=[0,29,0,29])
        for i in range(len(self.place_field)):
            axs[0,0].plot(self.place_field_COM[i][1], self.place_field_COM[i][0], color='black', marker='+', markersize=15, label='COM',linewidth=2)
        
        # trajectory
        spike_mouse_pos = ses.get['mouse_pos'](self.spiketime_run)
        axs[0,1].set_title('Trajectory', fontsize=self.tags['fontsize']*1.3)
        axs[0,1].plot(ses.mouse_pos[:,0], ses.mouse_pos[:,1])
        axs[0,1].scatter(spike_mouse_pos[:,0],spike_mouse_pos[:,1], marker='o', s=4, color='black', zorder=2)        
        axs[0,1].set_aspect(1, adjustable='box')
        axs[0,1].set_xlim(0,58)
        axs[0,1].set_ylim(0,58)
        axs[0,1].set_xlabel('cm', fontsize=self.tags['fontsize'])
        axs[0,1].set_ylabel('cm', fontsize=self.tags['fontsize'])
        axs[0,1].invert_yaxis()
        
        #positional information
        heatmap2 = self.positional_info.copy()
        heatmap2[ses.dwell_map_smooth==0] = np.nan     # if mouse haven't dwellen in this place, assign nan
        axs[0,2].set_title('Positional Information', fontsize=self.tags['fontsize']*1.3)
        pcm = axs[0,2].pcolormesh(heatmap2, cmap='jet')
        axs[0,2].set_aspect(1, adjustable='box')
        axs[0,2].set_xlabel('2cm bin', fontsize=self.tags['fontsize'])
        axs[0,2].set_ylabel('2cm bin', fontsize=self.tags['fontsize'])
        axs[0,2].invert_yaxis()
        fig.colorbar(pcm, ax=axs[0,2], label='positional information')
        
        # spatial information shuffle test
        spatial_info_pool_sorted=np.sort(np.append(spatial_info_pool,self.spatial_info))
        order=np.where(spatial_info_pool_sorted==self.spatial_info)[0][0]
        self.over_shuffle_percentage = order/len(spatial_info_pool) 
        
        axs[1,0].hist(spatial_info_pool, bins=40)
        axs[1,0].axvline(x=self.spatial_info, color='red', linestyle='dashed', linewidth=2, label='Spatial Information')
        axs[1,0].set_title('Spatial Info. Shuffle test', fontsize=self.tags['fontsize'])
        axs[1,0].set_aspect(1./axs[1,0].get_data_ratio(), adjustable='box')
        axs[1,0].set_xlabel('spatial information', fontsize=self.tags['fontsize'])
        axs[1,0].set_ylabel('count', fontsize=self.tags['fontsize'])
        axs[1,0].text(0.5, -0.2, 'Spatial Information: '+str(round(self.spatial_info, 4))+'  over '+ str(round(self.over_shuffle_percentage*100, 2))+'%', fontsize=12, ha='center', va='center', transform=axs[1,0].transAxes)
          
        # waveforms
        x = np.linspace(0,2,60)
        sub_axs0 = inset_axes(axs[1,1], width='45%', height='45%', loc='upper left')
        sub_axs1 = inset_axes(axs[1,1], width='45%', height='45%', loc='upper right')
        sub_axs2 = inset_axes(axs[1,1], width='45%', height='45%', loc='lower left')
        sub_axs3 = inset_axes(axs[1,1], width='45%', height='45%', loc='lower right')
        sub_axs0.plot(x, self.mean_waveforms[:,0])
        sub_axs1.plot(x, self.mean_waveforms[:,1])
        sub_axs2.plot(x, self.mean_waveforms[:,2])
        sub_axs3.plot(x, self.mean_waveforms[:,3])
        sub_axs0.set_ylim(-1200,300)
        sub_axs1.set_ylim(-1200,300)
        sub_axs2.set_ylim(-1200,300)
        sub_axs3.set_ylim(-1200,300)
        sub_axs0.xaxis.set_visible(False)
        sub_axs1.xaxis.set_visible(False)
        sub_axs1.yaxis.set_visible(False)
        sub_axs3.yaxis.set_visible(False)
            
        axs[1,1].set_aspect(1, adjustable='box')
        axs[1,1].axis('off')
        axs[1,1].set_title('Waveforms', fontsize=self.tags['fontsize'])
        axs[1,1].set_xlabel('ms', fontsize=self.tags['fontsize'])
        axs[1,1].set_ylabel('uV? sample?', fontsize=self.tags['fontsize'])
        
        # ax 
        axs[1,2].set_aspect(1, adjustable='box')
        
        plt.savefig(  fdir/fn/'plot'/ ('Unit'+str(self.id)+'.svg')  )
        plt.savefig(  fdir/fn/'plot'/ ('Unit'+str(self.id)+'.png')  )
        plt.close()
                             
    def get_mean_waveforms(self, waveforms):
        #to properly use this, I extract waveforms from phy2 with 'extract-waveforms' in command prompt.
        # and, the __init__.py is modified with max_nspikes=2000, nchannels=4, though it turns out 12chs still.
     
        unit_KSchannels = waveforms['channel'][np.where(waveforms['cluster'] == self.KScluid)][0]
        unit_waveforms_16ch = waveforms['raw'][np.where(waveforms['cluster'] == self.KScluid)]
        self.waveforms = np.zeros([unit_waveforms_16ch.shape[0], 60, 4])
        for ich in range(4):
            self.waveforms[:,:,ich] = unit_waveforms_16ch[:,:,np.where(unit_KSchannels == (self.tetrode*4+ich))[0][0]]
        
        self.mean_waveforms = np.mean(self.waveforms, axis=0)
                 
    def plot_PSTH(self, ses):
        raise Exception('Not done yet.')
    
#%% Class LFP
 # ----------------------------------------------------------------------------
 #                  Classes LFP
 # ----------------------------------------------------------------------------        
'''
For time variables:

r_s_e:                     ripple_start_end   in 2000Hz Sample point stamp
ripple_start_end           real world time

t_p_t_p                    2000Hz Sample point stamp
theta_peak_trough_peak     real world time

'''
class LFP(object):
    def __init__(self, fdir, fn, esync_timestamps, tags):
        
        self.img_save_path = fdir/fn/'plot'
        
        fs = 2000 # don't change this, or you need to change the resample ratio, below is 15
        nsx = brpylib.NsxFile(str(fdir/fn/(fn+'.ns6')))
        raw_trace = nsx.getdata()['data'][0]
        nsx.close()
        
        dcstop_b, dcstop_a = signal.butter(2,0.5, btype = 'highpass', output = 'ba', fs = fs)       # deal with DC
        notch_b, notch_a = signal.butter(1, [49.8,50.2], btype = 'bandstop', output ='ba', fs = fs) # deal with 50Hz AC    
        
        if  32<= raw_trace.shape[0] <64 :
            trace_all_ch = raw_trace[  0:32, esync_timestamps[0]:esync_timestamps[-1]+1:15  ].copy()    # cut the head and tail without sync pulse and convert to fs 2000Hz
            for i in range(32):                                                                       # filter ephys 32 channels
                trace_all_ch[i,:] = signal.filtfilt(dcstop_b, dcstop_a, trace_all_ch[i,:])
                trace_all_ch[i,:] = signal.filtfilt(notch_b, notch_a, trace_all_ch[i,:])                
                
            self.trace_all_ch = np.array(trace_all_ch)                                      
            self.time = np.linspace(0, (np.size(esync_timestamps)-1)/tags['sync_rate'], num=self.trace_all_ch.shape[1])
            
        elif  64<= raw_trace.shape[0] :       # bilateral, need to load left and right seperately
            trace_all_ch = raw_trace[  0:64, esync_timestamps[0]:esync_timestamps[-1]+1:15  ].copy()    # cut the head and tail without sync pulse and convert to fs 2000Hz
            for i in range(64):                                                                       # filter ephys 64 channels
                trace_all_ch[i,:] = signal.filtfilt(dcstop_b, dcstop_a, trace_all_ch[i,:])
                trace_all_ch[i,:] = signal.filtfilt(notch_b, notch_a, trace_all_ch[i,:])        
            
            self.trace_all_ch_R = trace_all_ch[:32,:]
            self.trace_all_ch_L = trace_all_ch[32:,:]
            self.time = np.linspace(0, (np.size(esync_timestamps)-1)/tags['sync_rate'], num=self.trace_all_ch_R.shape[1])
            
        else:
            raise Exception('not coded for higher channel')
        
    def SWR_detect(self, channel=None):
        fs = 2000
        window_len = 0.017
        ripplepass_b, ripplepass_a = signal.butter(2,[140,230],btype='bandpass',output='ba',fs=fs)     # bandpass=[140,230]
        
        #################### channel select #####################
        if hasattr(self,'trace_all_ch'):
            if channel==None:           # default: calculate and take the highest ripple power channel,  Mizuseki et al_2011
                mean_ripple_power = []
                for i in range(32):
                    ripple_trace = signal.filtfilt(ripplepass_b, ripplepass_a, self.trace_all_ch[i,:])    # this method dierctly calculate the power of trace, and average to length
                    ripple_power = np.sqrt(np.sum(ripple_trace**2/ripple_trace.shape[0])) # root mean square to serve as power
                    mean_ripple_power.append(ripple_power)
                    print(f'running:{i+1}/32', end='   ')
                highest_ripple_power_channel = mean_ripple_power.index(max(mean_ripple_power))
                
                # mean_ripple_power = []       
                # for i in range(32):
                #     ripple_trace = signal.filtfilt(ripplepass_b, ripplepass_a, self.trace_all_ch[i,:])
                #     ripple_windows = np.lib.stride_tricks.sliding_window_view(ripple_trace, int(window_len*fs))    # this method calculate the power of a window and then average all window's power
                #     ripple_power = np.sqrt(np.sum((ripple_windows**2/int(window_len*fs)), axis=1)) # root mean square to serve as power
                #     mean_ripple_power.append(np.mean(ripple_power))
                #     print(f'running:{i+1}/32', end='   ')
                
                # highest_ripple_power_channel = mean_ripple_power.index(max(mean_ripple_power))
                
                self.ripple_channel = highest_ripple_power_channel
            
            elif type(channel)==int:
                self.ripple_channel = channel
            else:
                raise ValueError('channel should be int for a 32ch recording')
                
        elif hasattr(self,'trace_all_ch_R'):
            if channel==None:           # default: calculate and take the highest ripple power channel
                mean_ripple_power_R = []
                mean_ripple_power_L = []
                for i in range(32):
                    ripple_trace_R = signal.filtfilt(ripplepass_b, ripplepass_a, self.trace_all_ch_R[i,:])
                    ripple_power_R = np.sqrt(np.sum(ripple_trace_R**2/ripple_trace_R.shape[0])) 
                    mean_ripple_power_R.append(ripple_power_R)
                    
                    ripple_trace_L = signal.filtfilt(ripplepass_b, ripplepass_a, self.trace_all_ch_L[i,:])
                    ripple_power_L = np.sqrt(np.sum(ripple_trace_L**2/ripple_trace_L.shape[0]))
                    mean_ripple_power_L.append(ripple_power_L)
                    
                    print(f'running:{i+1}/32', end='   ')
                
                highest_ripple_power_channel_R = mean_ripple_power_R.index(max(mean_ripple_power_R))
                highest_ripple_power_channel_L = mean_ripple_power_L.index(max(mean_ripple_power_L))
                self.ripple_channel_R = highest_ripple_power_channel_R
                self.ripple_channel_L = highest_ripple_power_channel_L+32
            elif type(channel)==list:
                self.ripple_channel_R = min(channel)
                self.ripple_channel_L = max(channel)
            else:
                raise ValueError('channel should be a two int list for 64ch recording')
        
        ################ ripple detect ####################
        if hasattr(self,'trace_all_ch'):             # mono lateral recording
            ripple_trace = signal.filtfilt(ripplepass_b, ripplepass_a, self.trace_all_ch[self.ripple_channel,:])
            self.ripple_trace = ripple_trace
            
            ripple_windows = np.lib.stride_tricks.sliding_window_view(ripple_trace, int(window_len*fs))
            ripple_power = np.sqrt(np.sum((ripple_windows**2/int(window_len*fs)), axis=1))# root mean square to serve as power, length offset +(window_len*fs-1)
            ripple_3sd = np.where(ripple_power > np.mean(ripple_power)+3*np.std(ripple_power), 1, 0)# to mark their starts and stops.
            ripple_7sd = np.where(ripple_power > np.mean(ripple_power)+7*np.std(ripple_power), 1, 0)# artificially mark those peaks. what would happend with large noise?
            ripple_boolean =  np.concatenate( (np.atleast_1d(np.where(ripple_3sd[0]==1, 1, 0)), (ripple_3sd[1:]-ripple_3sd[:-1])) )    # if first is true, the first one will be 1
            ripple_start = np.where(ripple_boolean == 1)[0]
            ripple_end = np.where(ripple_boolean == -1)[0]
            self.r_s_e = np.vstack((ripple_start, ripple_end)).T
            
            rows_to_delete = set()
            for i in range(self.r_s_e.shape[0]):
                if self.r_s_e[i][1] - self.r_s_e[i][0] < 0.015*fs:              # if the ripple period shorter than 15ms
                    rows_to_delete.add(i)
                if max(ripple_7sd[self.r_s_e[i][0]:self.r_s_e[i][1]+1]) == 0:   # if don't contains a ripple peak which larger than +7*sd
                    rows_to_delete.add(i)
            self.r_s_e = np.delete(self.r_s_e, list(rows_to_delete), axis=0)
            self.r_s_e += (int(window_len*fs/2))      # due to being windowed, the index need a offset, now the timetamp takes the middle of window
            
            self.r_p = np.zeros(self.r_s_e.shape[0]).astype(int)
            for i in range(self.r_s_e.shape[0]):        
                self.r_p[i] = np.argmax(ripple_power[self.r_s_e[i][0]:self.r_s_e[i][1]+1]) + self.r_s_e[i][0]
    
            #plot
            fig, axs = plt.subplots(self.r_s_e.shape[0]//4+1, 4, figsize=((40,(self.r_s_e.shape[0]//4+1)*3)), tight_layout=True)
            for i in range(self.r_s_e.shape[0]):
                t = np.linspace(0, self.r_s_e[i][1]-self.r_s_e[i][0], num=self.r_s_e[i][1]-self.r_s_e[i][0]+1)/2  # /2 means convert sample point to ms
                axs[i//4,i%4].plot(t,self.trace_all_ch[self.ripple_channel , self.r_s_e[i][0]:self.r_s_e[i][1]+1])
                axs[i//4,i%4].set_xlim(0,200)
            plt.savefig(  self.img_save_path/ ('ripple events raw ch'+str(self.ripple_channel)+'.svg')  )
            plt.close()
            
            fig, axs = plt.subplots(self.r_s_e.shape[0]//4+1, 4, figsize=((40,(self.r_s_e.shape[0]//4+1)*3)), tight_layout=True)
            for i in range(self.r_s_e.shape[0]):
                t = np.linspace(0, self.r_s_e[i][1]-self.r_s_e[i][0], num=self.r_s_e[i][1]-self.r_s_e[i][0]+1)/2
                axs[i//4,i%4].plot(t,ripple_trace[self.r_s_e[i][0]:self.r_s_e[i][1]+1])
                axs[i//4,i%4].set_xlim(0,200)
            plt.savefig(  self.img_save_path/ ('ripple events bandpassed ch'+str(self.ripple_channel)+'.svg')  )
            plt.close()
            
            # generate real world time ripple time
            self.ripple_start_end = self.r_s_e.astype(np.float64)
            self.ripple_peak = self.r_p.astype(np.float64)
            for i in range(len(self.r_s_e)):
                self.ripple_start_end[i,0] =  self.time[self.r_s_e[i,0] ]
                self.ripple_start_end[i,1] =  self.time[self.r_s_e[i,1] ]
                self.ripple_peak[i] =  self.time[self.r_p[i] ]
                
            
        
        elif hasattr(self,'trace_all_ch_R'):       # bi-lateral recording
            ripple_trace = signal.filtfilt(ripplepass_b, ripplepass_a, self.trace_all_ch_R[self.ripple_channel_R,:])
            self.ripple_trace_R = ripple_trace
            
            ripple_windows = np.lib.stride_tricks.sliding_window_view(ripple_trace, int(window_len*fs))
            ripple_power = np.sqrt(np.sum((ripple_windows**2/int(window_len*fs)), axis=1))# root mean square to serve as power, length offset +(window_len*fs-1)
            ripple_3sd = np.where(ripple_power > np.mean(ripple_power)+3*np.std(ripple_power), 1, 0)# to mark their starts and stops.
            ripple_7sd = np.where(ripple_power > np.mean(ripple_power)+7*np.std(ripple_power), 1, 0)# artificially mark those peaks. what would happend with large noise?
            ripple_boolean =  np.concatenate( (np.atleast_1d(np.where(ripple_3sd[0]==1, 1, 0)), (ripple_3sd[1:]-ripple_3sd[:-1])) )    # if first is true, the first one will be 1
            ripple_start = np.where(ripple_boolean == 1)[0]
            ripple_end = np.where(ripple_boolean == -1)[0]
            self.r_s_e_R = np.vstack((ripple_start, ripple_end)).T
            
            rows_to_delete = set()
            for i in range(self.r_s_e_R.shape[0]):
                if self.r_s_e_R[i][1] - self.r_s_e_R[i][0] < 0.015*fs:              # if the ripple period shorter than 15ms
                    rows_to_delete.add(i)
                if max(ripple_7sd[self.r_s_e_R[i][0]:self.r_s_e_R[i][1]+1]) == 0:   # if don't contains a ripple peak which larger than +7*sd
                    rows_to_delete.add(i)
            self.r_s_e_R = np.delete(self.r_s_e_R, list(rows_to_delete), axis=0)
            self.r_s_e_R += (int(window_len*fs/2))      # due to being windowed, the index need a offset, now the timetamp takes the middle of window
            
            self.r_p_R = np.zeros(self.r_s_e_R.shape[0]).astype(int)
            for i in range(self.r_s_e_R.shape[0]):        
                self.r_p_R[i] = np.argmax(ripple_power[self.r_s_e_R[i][0]:self.r_s_e_R[i][1]+1]) + self.r_s_e_R[i][0]
    
            #plot
            fig, axs = plt.subplots(self.r_s_e_R.shape[0]//4+1, 4, figsize=((40,(self.r_s_e_R.shape[0]//4+1)*3)), tight_layout=True)
            for i in range(self.r_s_e_R.shape[0]):
                t = np.linspace(0, self.r_s_e_R[i][1]-self.r_s_e_R[i][0], num=self.r_s_e_R[i][1]-self.r_s_e_R[i][0]+1)/2  # /2 means convert sample point to ms
                axs[i//4,i%4].plot(t,self.trace_all_ch_R[self.ripple_channel_R, self.r_s_e_R[i][0]:self.r_s_e_R[i][1]+1])
                axs[i//4,i%4].set_xlim(0,200)
            plt.savefig(  self.img_save_path/ ('ripple events raw ch'+str(self.ripple_channel_R)+'.svg')  )
            plt.savefig(  self.img_save_path/ ('ripple events raw ch'+str(self.ripple_channel_R)+'.png')  )
            plt.close()
            
            fig, axs = plt.subplots(self.r_s_e_R.shape[0]//4+1, 4, figsize=((40,(self.r_s_e_R.shape[0]//4+1)*3)), tight_layout=True)
            for i in range(self.r_s_e_R.shape[0]):
                t = np.linspace(0, self.r_s_e_R[i][1]-self.r_s_e_R[i][0], num=self.r_s_e_R[i][1]-self.r_s_e_R[i][0]+1)/2
                axs[i//4,i%4].plot(t,ripple_trace[self.r_s_e_R[i][0]:self.r_s_e_R[i][1]+1])
                axs[i//4,i%4].set_xlim(0,200)
            plt.savefig(  self.img_save_path/ ('ripple events bandpassed ch'+str(self.ripple_channel_R)+'.svg')  )
            plt.savefig(  self.img_save_path/ ('ripple events bandpassed ch'+str(self.ripple_channel_R)+'.png')  )
            plt.close()
            
            # generate real world time ripple time
            self.ripple_start_end_R = self.r_s_e_R.astype(np.float64)
            self.ripple_peak_R = self.r_p_R.astype(np.float64)
            for i in range(len(self.r_s_e_R)):
                self.ripple_start_end_R[i,0] =  self.time[self.r_s_e_R[i,0] ]
                self.ripple_start_end_R[i,1] =  self.time[self.r_s_e_R[i,1] ]
                self.ripple_peak_R[i] =  self.time[self.r_p_R[i] ]

                
            ############################ L R line ##############################
        
            ripple_trace = signal.filtfilt(ripplepass_b, ripplepass_a, self.trace_all_ch_L[self.ripple_channel_L-32,:])
            self.ripple_trace_L = ripple_trace
            
            ripple_windows = np.lib.stride_tricks.sliding_window_view(ripple_trace, int(window_len*fs))
            ripple_power = np.sqrt(np.sum((ripple_windows**2/int(window_len*fs)), axis=1))# root mean square to serve as power, length offset +(window_len*fs-1)
            ripple_3sd = np.where(ripple_power > np.mean(ripple_power)+3*np.std(ripple_power), 1, 0)# to mark their starts and stops.
            ripple_7sd = np.where(ripple_power > np.mean(ripple_power)+7*np.std(ripple_power), 1, 0)# artificially mark those peaks. what would happend with large noise?
            ripple_boolean =  np.concatenate( (np.atleast_1d(np.where(ripple_3sd[0]==1, 1, 0)), (ripple_3sd[1:]-ripple_3sd[:-1])) )    # if first is true, the first one will be 1
            ripple_start = np.where(ripple_boolean == 1)[0]
            ripple_end = np.where(ripple_boolean == -1)[0]
            self.r_s_e_L = np.vstack((ripple_start, ripple_end)).T
            
            rows_to_delete = set()
            for i in range(self.r_s_e_L.shape[0]):
                if self.r_s_e_L[i][1] - self.r_s_e_L[i][0] < 0.015*fs:              # if the ripple period shorter than 15ms
                    rows_to_delete.add(i)
                if max(ripple_7sd[self.r_s_e_L[i][0]:self.r_s_e_L[i][1]+1]) == 0:   # if don't contains a ripple peak which larger than +7*sd
                    rows_to_delete.add(i)
            self.r_s_e_L = np.delete(self.r_s_e_L, list(rows_to_delete), axis=0)
            self.r_s_e_L += (int(window_len*fs/2))      # due to being windowed, the index need a offset, now the timetamp takes the middle of window
            
            self.r_p_L = np.zeros(self.r_s_e_L.shape[0]).astype(int)
            for i in range(self.r_s_e_L.shape[0]):        
                self.r_p_L[i] = np.argmax(ripple_power[self.r_s_e_L[i][0]:self.r_s_e_L[i][1]+1]) + self.r_s_e_L[i][0]
    
            #plot
            fig, axs = plt.subplots(self.r_s_e_L.shape[0]//4+1, 4, figsize=((40,(self.r_s_e_L.shape[0]//4+1)*3)), tight_layout=True)
            for i in range(self.r_s_e_L.shape[0]):
                t = np.linspace(0, self.r_s_e_L[i][1]-self.r_s_e_L[i][0], num=self.r_s_e_L[i][1]-self.r_s_e_L[i][0]+1)/2  # /2 means convert sample point to ms
                axs[i//4,i%4].plot(t,self.trace_all_ch_L[self.ripple_channel_L-32 , self.r_s_e_L[i][0]:self.r_s_e_L[i][1]+1])
                axs[i//4,i%4].set_xlim(0,200)
            plt.savefig(  self.img_save_path/ ('ripple events raw ch'+str(self.ripple_channel_L)+'.svg')  )
            plt.savefig(  self.img_save_path/ ('ripple events raw ch'+str(self.ripple_channel_L)+'.png')  )
            plt.close()
            
            fig, axs = plt.subplots(self.r_s_e_L.shape[0]//4+1, 4, figsize=((40,(self.r_s_e_L.shape[0]//4+1)*3)), tight_layout=True)
            for i in range(self.r_s_e_L.shape[0]):
                t = np.linspace(0, self.r_s_e_L[i][1]-self.r_s_e_L[i][0], num=self.r_s_e_L[i][1]-self.r_s_e_L[i][0]+1)/2
                axs[i//4,i%4].plot(t,ripple_trace[self.r_s_e_L[i][0]:self.r_s_e_L[i][1]+1])
                axs[i//4,i%4].set_xlim(0,200)
            plt.savefig(  self.img_save_path/ ('ripple events bandpassed ch'+str(self.ripple_channel_L)+'.svg')  )
            plt.savefig(  self.img_save_path/ ('ripple events bandpassed ch'+str(self.ripple_channel_L)+'.png')  )
            plt.close()    
            
            # generate real world time ripple time
            self.ripple_start_end_L = self.r_s_e_L.astype(np.float64)
            self.ripple_peak_L = self.r_p_L.astype(np.float64)
            for i in range(len(self.r_s_e_L)):
                self.ripple_start_end_L[i,0] =  self.time[self.r_s_e_L[i,0] ]
                self.ripple_start_end_L[i,1] =  self.time[self.r_s_e_L[i,1] ]
                self.ripple_peak_L[i] =  self.time[self.r_p_L[i] ]
               

    def theta_cycle_detect(self, channel=None):
        fs = 2000
        gamma = [40,100]
        theta = [5,11]
        delta = [2,4]
        gammapass_b, gammapass_a = signal.butter(2,gamma,btype='bandpass',output='ba',fs=fs)
        thetapass_b, thetapass_a = signal.butter(2,theta,btype='bandpass',output='ba',fs=fs)
        deltapass_b, deltapass_a = signal.butter(2,delta,btype='bandpass',output='ba',fs=fs)
        
        if hasattr(self,'trace_all_ch'):
            if channel == None:                      # default: take the highest power channel as inspection
                mean_theta_power = []
                for i in range(32):
                    theta_trace = signal.filtfilt(thetapass_b, thetapass_a, self.trace_all_ch[i,:])
                    theta_power = np.sqrt(np.sum(theta_trace**2/theta_trace.shape[0])) # root mean square to serve as power
                    mean_theta_power.append(theta_power)
                    print(f'running:{i+1}/32', end='   ')
                highest_theta_power_channel = mean_theta_power.index(max(mean_theta_power))
                self.theta_channel = highest_theta_power_channel
                self.theta_trace = self.trace_all_ch[self.theta_channel,:]
            elif channel=='mean':
                self.theta_channel = channel
                self.theta_trace = np.mean(self.trace_all_ch, axis=0)
            elif type(channel)==int:
                self.theta_channel = channel
                self.theta_trace = self.trace_all_ch[channel,:]
            else:
                raise ValueError('channel should be int or mean or default choose for a 32ch recording')
            
            ########################### band pass filter ########################
            self.gamma_trace = signal.filtfilt(gammapass_b, gammapass_a, self.theta_trace)
            self.delta_trace = signal.filtfilt(deltapass_b, deltapass_a, self.theta_trace)          # filter the delta part of the same raw trace 
            self.theta_trace = signal.filtfilt(thetapass_b, thetapass_a, self.theta_trace)
            ########################### peak & trough ###########################
            
            peak_trough_detect_windows = np.lib.stride_tricks.sliding_window_view(self.theta_trace, 3 )    # window= 3 sample points to find if peak or trough exist
            max_in_window = np.argmax(peak_trough_detect_windows, axis=1)                             # 找到每一个window中极值的位置
            min_in_window = np.argmin(peak_trough_detect_windows, axis=1)
            peak = np.where(max_in_window==1)[0]+1                                                    # 如果是拐点，那么极值应该是处于中间的位置 index=1 
            trough = np.where(min_in_window==1)[0]+1                                                  # 那么这行的行号加极值的位置号1 即是极值在总trace的位置
            
            peak_peak = np.vstack((peak[:-1],peak[1:])).T
            trough_in = []
            if trough[0]<peak_peak[0,0]:
                j=1  # trough start from 1
            elif peak_peak[0,0]<trough[0]<peak_peak[0,1]:
                j=0  # trough start from 0
            else:
                raise Exception('two peaks in the start??')
            for i in range(peak_peak.shape[0]):
                if peak_peak[i,0]<trough[j]<peak_peak[i,1]:         # find which trough time that are between two peaks
                    trough_in.append(trough[j])                                                                     
                    j += 1
                else:
                    raise Exception('Found no trough between peaks??')
            trough_in = np.array(trough_in)
            peak_trough_peak = np.vstack((peak_peak[:,0], trough_in, peak_peak[:,1])).T
            
            ######################### check discard ###########################
            window_len = 0.5    # I think this window should cover at least a cycle of delta(2~4Hz), but that would take more averaged information than one theta cycle, is that good?
            theta_windows = np.lib.stride_tricks.sliding_window_view(self.theta_trace, int(window_len*fs))
            theta_power = np.sqrt(np.sum((theta_windows**2/int(window_len*fs)), axis=1))    # root mean square to serve as power, length offset +(window_len*fs-1)
            delta_windows = np.lib.stride_tricks.sliding_window_view(self.delta_trace, int(window_len*fs))
            delta_power = np.sqrt(np.sum((delta_windows**2/int(window_len*fs)), axis=1))
            theta_delta_ratio_mean = np.mean(theta_power / delta_power) 
            theta_delta_ratio_std = np.std(theta_power / delta_power)
            
            row_to_delete=[]
            for i, p_t_p in enumerate(peak_trough_peak):
                theta_power_here = np.sqrt(np.sum(self.theta_trace[p_t_p[0]:p_t_p[2]]**2/(p_t_p[2]-p_t_p[0]+1)))
                delta_power_here = np.sqrt(np.sum(self.delta_trace[p_t_p[0]:p_t_p[2]]**2/(p_t_p[2]-p_t_p[0]+1)))
                theta_delta_ratio_here = theta_power_here / delta_power_here
                
                if p_t_p[1]-p_t_p[0] < fs/theta[1]/2  or p_t_p[2]-p_t_p[1] < fs/theta[1]/2:
                    row_to_delete.append(i)
                elif p_t_p[1]-p_t_p[0] > fs/theta[0]/2  or p_t_p[2]-p_t_p[1] > fs/theta[0]/2:
                    row_to_delete.append(i)
                elif theta_delta_ratio_here < theta_delta_ratio_mean-theta_delta_ratio_std:
                    row_to_delete.append(i)
                elif np.any( (self.r_s_e[:,0]<p_t_p[0]) & (p_t_p[0]<self.r_s_e[:,1]) ):  # if the theta cycle(1st peak) happened during the ripple period
                    row_to_delete.append(i)
                elif np.any( (self.r_s_e[:,0]<p_t_p[1]) & (p_t_p[1]<self.r_s_e[:,1]) ):  # if the theta cycle(trough)
                    row_to_delete.append(i)
                elif np.any( (self.r_s_e[:,0]<p_t_p[2]) & (p_t_p[2]<self.r_s_e[:,1]) ):  # if the theta cycle(2st peak)
                    row_to_delete.append(i)
                                        
            peak_trough_peak = np.delete(peak_trough_peak, row_to_delete, axis=0)
            
            self.t_p_t_p = peak_trough_peak
            
            # generate real world time theta cycle time
            self.theta_peak_trough_peak= self.t_p_t_p.astype(np.float64)
            for i in range(len(self.t_p_t_p)):
                self.theta_peak_trough_peak[i,0] =  self.time[self.t_p_t_p[i,0] ]
                self.theta_peak_trough_peak[i,1] =  self.time[self.t_p_t_p[i,1] ]
                self.theta_peak_trough_peak[i,2] =  self.time[self.t_p_t_p[i,2] ]

            
            # st=25
            # t=10
            # plt.figure(figsize=(40, 4))
            # plt.plot(self.time[2000*st:2000*st+2000*t],self.ripple_trace[2000*st:2000*st+2000*t], color='k')
            # # plt.plot(self.time[2000*st:2000*st+2000*t],self.delta_trace[2000*st:2000*st+2000*t], color='b')
            # plt.plot(self.time[2000*st:2000*st+2000*t],theta_trace[2000*st:2000*st+2000*t], color='b')
            # for i in peak_trough_peak:
            #     plt.plot(self.time[i],theta_trace[i], color='y')
            # plt.xlim([st,st+t])
            # plt.savefig(  self.img_save_path/ ('test.svg')  )
            
            

        
        
        elif hasattr(self,'trace_all_ch_R'):
            if channel == None:                      # default: take the highest power channel as inspection
                mean_theta_power_R = []
                mean_theta_power_L = []
                for i in range(32):
                    theta_trace_R = signal.filtfilt(thetapass_b, thetapass_a, self.trace_all_ch_R[i,:])
                    theta_trace_L = signal.filtfilt(thetapass_b, thetapass_a, self.trace_all_ch_L[i,:])
                    theta_power_R = np.sqrt(np.sum(theta_trace_R**2/theta_trace_R.shape[0]))
                    theta_power_L = np.sqrt(np.sum(theta_trace_L**2/theta_trace_L.shape[0])) # root mean square to serve as power
                    mean_theta_power_R.append(theta_power_R)
                    mean_theta_power_L.append(theta_power_L)
                    print(f'running:{i+1}/32', end='   ')
                highest_theta_power_channel_R = mean_theta_power_R.index(max(mean_theta_power_R))
                highest_theta_power_channel_L = mean_theta_power_L.index(max(mean_theta_power_L))
                self.theta_channel_R = highest_theta_power_channel_R
                self.theta_channel_L = highest_theta_power_channel_L+32
                self.theta_trace_R = self.trace_all_ch_R[self.theta_channel_R,:]
                self.theta_trace_L = self.trace_all_ch_L[self.theta_channel_L-32,:]
                
            elif channel=='mean':
                self.theta_channel_R = channel
                self.theta_channel_L = channel
                self.theta_trace_R = np.mean(self.trace_all_ch_R, axis=0)
                self.theta_trace_L = np.mean(self.trace_all_ch_L, axis=0)
                
            elif type(channel)==list:
                self.theta_channel_R = min(channel)
                self.theta_channel_L = max(channel)
                self.theta_trace_R = self.trace_all_ch_R[self.theta_channel_R,:]
                self.theta_trace_L = self.trace_all_ch_L[self.theta_channel_L-32,:]
            
            else:
                raise ValueError('channel should be list of int or mean or default choose for a 32ch recording')

            
            # R detect    
            ########################### band pass filter ########################
            self.gamma_trace_R = signal.filtfilt(gammapass_b, gammapass_a, self.theta_trace_R)
            self.delta_trace_R = signal.filtfilt(deltapass_b, deltapass_a, self.theta_trace_R)          # filter the delta part of the same raw trace 
            self.theta_trace_R = signal.filtfilt(thetapass_b, thetapass_a, self.theta_trace_R)
            ########################### peak & trough ###########################
            
            peak_trough_detect_windows = np.lib.stride_tricks.sliding_window_view(self.theta_trace_R, 3 )    # window= 3 sample points to find if peak or trough exist
            max_in_window = np.argmax(peak_trough_detect_windows, axis=1)                             # 找到每一个window中极值的位置
            min_in_window = np.argmin(peak_trough_detect_windows, axis=1)
            peak = np.where(max_in_window==1)[0]+1                                                    # 如果是拐点，那么极值应该是处于中间的位置 index=1 
            trough = np.where(min_in_window==1)[0]+1                                                  # 那么这行的行号加极值的位置号1 即是极值在总trace的位置
            
            peak_peak = np.vstack((peak[:-1],peak[1:])).T
            trough_in = []
            if trough[0]<peak_peak[0,0]:
                j=1  # trough start from 1
            elif peak_peak[0,0]<trough[0]<peak_peak[0,1]:
                j=0  # trough start from 0
            else:
                raise Exception('two peaks in the start??')
            for i in range(peak_peak.shape[0]):
                if peak_peak[i,0]<trough[j]<peak_peak[i,1]:         # find which trough time that are between two peaks
                    trough_in.append(trough[j])                                                                     
                    j += 1
                else:
                    raise Exception('Found no trough between peaks??')
            trough_in = np.array(trough_in)
            peak_trough_peak = np.vstack((peak_peak[:,0], trough_in, peak_peak[:,1])).T
            
            ######################### check discard ###########################
            window_len = 0.5    # I think this window should cover at least a cycle of delta(2~4Hz), but that would take more averaged information than one theta cycle, is that good?
            theta_windows = np.lib.stride_tricks.sliding_window_view(self.theta_trace_R, int(window_len*fs))
            theta_power = np.sqrt(np.sum((theta_windows**2/int(window_len*fs)), axis=1))    # root mean square to serve as power, length offset +(window_len*fs-1)
            delta_windows = np.lib.stride_tricks.sliding_window_view(self.delta_trace_R, int(window_len*fs))
            delta_power = np.sqrt(np.sum((delta_windows**2/int(window_len*fs)), axis=1))
            theta_delta_ratio_mean = np.mean(theta_power / delta_power) 
            theta_delta_ratio_std = np.std(theta_power / delta_power)
            
            row_to_delete=[]
            for i, p_t_p in enumerate(peak_trough_peak):
                theta_power_here = np.sqrt(np.sum(self.theta_trace_R[p_t_p[0]:p_t_p[2]]**2/(p_t_p[2]-p_t_p[0]+1)))
                delta_power_here = np.sqrt(np.sum(self.delta_trace_R[p_t_p[0]:p_t_p[2]]**2/(p_t_p[2]-p_t_p[0]+1)))
                theta_delta_ratio_here = theta_power_here / delta_power_here
                
                if p_t_p[1]-p_t_p[0] < fs/theta[1]/2  or p_t_p[2]-p_t_p[1] < fs/theta[1]/2:
                    row_to_delete.append(i)
                elif p_t_p[1]-p_t_p[0] > fs/theta[0]/2  or p_t_p[2]-p_t_p[1] > fs/theta[0]/2:
                    row_to_delete.append(i)
                elif theta_delta_ratio_here < theta_delta_ratio_mean-theta_delta_ratio_std:
                    row_to_delete.append(i)
                elif np.any( (self.r_s_e_R[:,0]<p_t_p[0]) & (p_t_p[0]<self.r_s_e_R[:,1]) ):  # if the theta cycle(1st peak) happened during the ripple period
                    row_to_delete.append(i)
                elif np.any( (self.r_s_e_R[:,0]<p_t_p[1]) & (p_t_p[1]<self.r_s_e_R[:,1]) ):  # if the theta cycle(trough)
                    row_to_delete.append(i)
                elif np.any( (self.r_s_e_R[:,0]<p_t_p[2]) & (p_t_p[2]<self.r_s_e_R[:,1]) ):  # if the theta cycle(2st peak)
                    row_to_delete.append(i)
                                        
            peak_trough_peak = np.delete(peak_trough_peak, row_to_delete, axis=0)
            
            self.t_p_t_p_R = peak_trough_peak
            
            # generate real world time theta cycle time
            self.theta_peak_trough_peak_R= self.t_p_t_p_R.astype(np.float64)
            for i in range(len(self.t_p_t_p_R)):
                self.theta_peak_trough_peak_R[i,0] =  self.time[self.t_p_t_p_R[i,0] ]
                self.theta_peak_trough_peak_R[i,1] =  self.time[self.t_p_t_p_R[i,1] ]
                self.theta_peak_trough_peak_R[i,2] =  self.time[self.t_p_t_p_R[i,2] ]
            
            
            # L detect    
            ########################### band pass filter ########################
            self.gamma_trace_L = signal.filtfilt(gammapass_b, gammapass_a, self.theta_trace_L)
            self.delta_trace_L = signal.filtfilt(deltapass_b, deltapass_a, self.theta_trace_L)          # filter the delta part of the same raw trace 
            self.theta_trace_L = signal.filtfilt(thetapass_b, thetapass_a, self.theta_trace_L)
            ########################### peak & trough ###########################
            
            peak_trough_detect_windows = np.lib.stride_tricks.sliding_window_view(self.theta_trace_L, 3 )    # window= 3 sample points to find if peak or trough exist
            max_in_window = np.argmax(peak_trough_detect_windows, axis=1)                             # 找到每一个window中极值的位置
            min_in_window = np.argmin(peak_trough_detect_windows, axis=1)
            peak = np.where(max_in_window==1)[0]+1                                                    # 如果是拐点，那么极值应该是处于中间的位置 index=1 
            trough = np.where(min_in_window==1)[0]+1                                                  # 那么这行的行号加极值的位置号1 即是极值在总trace的位置
            
            peak_peak = np.vstack((peak[:-1],peak[1:])).T
            trough_in = []
            if trough[0]<peak_peak[0,0]:
                j=1  # trough start from 1
            elif peak_peak[0,0]<trough[0]<peak_peak[0,1]:
                j=0  # trough start from 0
            else:
                raise Exception('two peaks in the start??')
            for i in range(peak_peak.shape[0]):
                if peak_peak[i,0]<trough[j]<peak_peak[i,1]:         # find which trough time that are between two peaks
                    trough_in.append(trough[j])                                                                     
                    j += 1
                else:
                    raise Exception('Found no trough between peaks??')
            trough_in = np.array(trough_in)
            peak_trough_peak = np.vstack((peak_peak[:,0], trough_in, peak_peak[:,1])).T
            
            ######################### check discard ###########################
            window_len = 0.5    # I think this window should cover at least a cycle of delta(2~4Hz), but that would take more averaged information than one theta cycle, is that good?
            theta_windows = np.lib.stride_tricks.sliding_window_view(self.theta_trace_L, int(window_len*fs))
            theta_power = np.sqrt(np.sum((theta_windows**2/int(window_len*fs)), axis=1))    # root mean square to serve as power, length offset +(window_len*fs-1)
            delta_windows = np.lib.stride_tricks.sliding_window_view(self.delta_trace_L, int(window_len*fs))
            delta_power = np.sqrt(np.sum((delta_windows**2/int(window_len*fs)), axis=1))
            theta_delta_ratio_mean = np.mean(theta_power / delta_power) 
            theta_delta_ratio_std = np.std(theta_power / delta_power)
            
            row_to_delete=[]
            for i, p_t_p in enumerate(peak_trough_peak):
                theta_power_here = np.sqrt(np.sum(self.theta_trace_L[p_t_p[0]:p_t_p[2]]**2/(p_t_p[2]-p_t_p[0]+1)))
                delta_power_here = np.sqrt(np.sum(self.delta_trace_L[p_t_p[0]:p_t_p[2]]**2/(p_t_p[2]-p_t_p[0]+1)))
                theta_delta_ratio_here = theta_power_here / delta_power_here
                
                if p_t_p[1]-p_t_p[0] < fs/theta[1]/2  or p_t_p[2]-p_t_p[1] < fs/theta[1]/2:
                    row_to_delete.append(i)
                elif p_t_p[1]-p_t_p[0] > fs/theta[0]/2  or p_t_p[2]-p_t_p[1] > fs/theta[0]/2:
                    row_to_delete.append(i)
                elif theta_delta_ratio_here < theta_delta_ratio_mean-theta_delta_ratio_std:
                    row_to_delete.append(i)
                elif np.any( (self.r_s_e_L[:,0]<p_t_p[0]) & (p_t_p[0]<self.r_s_e_L[:,1]) ):  # if the theta cycle(1st peak) happened during the ripple period
                    row_to_delete.append(i)
                elif np.any( (self.r_s_e_L[:,0]<p_t_p[1]) & (p_t_p[1]<self.r_s_e_L[:,1]) ):  # if the theta cycle(trough)
                    row_to_delete.append(i)
                elif np.any( (self.r_s_e_L[:,0]<p_t_p[2]) & (p_t_p[2]<self.r_s_e_L[:,1]) ):  # if the theta cycle(2st peak)
                    row_to_delete.append(i)
                                        
            peak_trough_peak = np.delete(peak_trough_peak, row_to_delete, axis=0)
            
            self.t_p_t_p_L = peak_trough_peak
            
            # generate real world time theta cycle time
            self.theta_peak_trough_peak_L= self.t_p_t_p_L.astype(np.float64)
            for i in range(len(self.t_p_t_p_L)):
                self.theta_peak_trough_peak_L[i,0] =  self.time[self.t_p_t_p_L[i,0] ]
                self.theta_peak_trough_peak_L[i,1] =  self.time[self.t_p_t_p_L[i,1] ]
                self.theta_peak_trough_peak_L[i,2] =  self.time[self.t_p_t_p_L[i,2] ]
        
     

    def CWT(self, uplim_Hz=150, totalscale=30, starttime = 0, t=20):
        
        def pywt_cwt(data, uplim_Hz, totalscale, fs=2000, wavename='cgau8'):  # total_scale means how many range of frequency
            fc = pywt.central_frequency(wavename)
            cparam = (fs/uplim_Hz)*fc*totalscale# so uplim can be 250Hz?
            scale = cparam / np.arange(totalscale, 0, -1)
            coef, frequencie = pywt.cwt(data, scale, wavename, 1/fs)
            return coef, frequencie
        
        if hasattr(self,'trace_all_ch'):
            cwt_data = np.mean(self.trace_all_ch, axis=0)
            coef,freq = pywt_cwt(cwt_data,uplim_Hz,totalscale)
        
        st = starttime*2000
        deltat = t*2000
        plt.contourf(self.time[st:st+deltat], freq, abs(coef[:,st:st+deltat]), cmap='rainbow')
        plt.savefig(  self.img_save_path/ ('cwt full.svg')  )
        plt.close()
        plt.contourf(self.time[st:st+deltat], freq[:15], abs(coef[:15,st:st+deltat]), cmap='rainbow')
        plt.savefig(  self.img_save_path/ ('cwt 2 half.svg')  )
        plt.close()
        plt.contourf(self.time[st:st+deltat], freq[15:], abs(coef[15:,st:st+deltat]), cmap='rainbow')
        plt.savefig(  self.img_save_path/ ('cwt 1 half.svg')  )
        plt.close()

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
