# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 15:50:28 2023

Welcome to NCP lab! The lasy authar so far has not written any informative docstring yet.

@author: Junhao
"""
'''
UPDATES

20230708, for 1d info & single neuron shuffle test, Skaggs info & Olypher info.

20230715. Core reconstructed, by reconstruction of time defined by external clock(Master8).
Which means all V_frames or E_timestamps are translated into time by count of sync pulse,
in other words, full trust was given to Maser8.
Pros, reconstruted time by many interpolations(Scipy, UniviariateSpline) might be easy to use.
Cons, we cannot know if Master8 is really good enough so time-related calculation might be wrong.



'''

#%% Main Bundle


# ----------------------------------------------------------------------------
#    LOTS OF THINGS TO BE DONE.
# ----------------------------------------------------------------------------


# later, mind if there are some nans in DLC files.
# jumpy detection, or smooth, Kalman Filter!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# HD vector? Especially for some non-place-but-HD cells, might be very informative about the internal compass in DR.

# waveform putative IN or PCs. Then optotag, R of waveforms of units.
# in class unit, furthur work with its quality check like L-ratio and others. May need to load more files from KS&phy2.


# Master8 got a bit faster or DAQ slower? e_intervals are mostly 14998 and none greater than 15000. Errors are accumulating!!!!!!!!!!!!!!!!!!!

# better session, coorperate with behavior-tags 2 frames. 

# LFP&spike, their binding do not need anything related to videos. Well except for spd thresh, or maybe some relation with its position.
# func & methods for 2D exp. , smoothing kernels. More and More
# decoding. some bayesian?



# ----------------------------------------------------------------------------
#                  Packages 
# ----------------------------------------------------------------------------

import brpylib, time, random, numpy as np, pandas as pd, matplotlib.pyplot as plt, numpy_groupies as npg
from scipy import optimize, ndimage, interpolate, stats
from pathlib import Path
from tkinter import filedialog

# ----------------------------------------------------------------------------
#                  Functions here
# ----------------------------------------------------------------------------

# 中文就中文吧...天宇他们的filename_tail未必是数字，这个简单，Loadfile这里写灵活一些就行。
# 博华的建议，接口化，输入可以有若干个，但输入的规则由函数定义。这个怎么写还得继续学。




def load_files(fdir, fn, Nses, experiment_tag, dlc_tail):  
    # if Nses > 1, mind the rule of name.
    # as for dlc body parts, mind that it is read by col index.
    spike_times = np.load(fdir/fn/'spike_times.npy')
    spike_times = np.squeeze(spike_times)# delete that stupid dimension.
    spike_clusters = np.load(fdir/fn/'spike_clusters.npy')
    clusters_quality = pd.read_csv(fdir/fn/'cluster_group.tsv', sep='\t')
    esync_timestamps_load = np.load(fdir/fn/('Esync_timestamps_'+fn+'.npy'))  

    if 'signal_on' in experiment_tag :
        signal_on_timestamps_load = np.load(fdir/fn/('Signal_on_timestamps_'+fn+'.npy')) 
    
    if Nses == 1:
        timestamps = spike_times    
        spike_clusters2 = spike_clusters
        dlch5 = pd.read_hdf(fdir/fn/(fn+dlc_tail))
        vsync_csv = pd.read_csv(fdir/fn/(fn+'.csv'), names=[0,1,2])
        vsync_temp = np.array(vsync_csv.loc[:,1], dtype='uint')
        vsync = np.where((vsync_temp[1:] - vsync_temp[:-1]) ==1)[0]
        
        esync_timestamps = esync_timestamps_load
        dlc_files = dlch5
        if 'signal_on' in experiment_tag:
            signal_on_timestamps = signal_on_timestamps_load
    elif Nses > 1:
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
            vsync_csv = pd.read_csv(fdir/fn/(i+'.csv'), names=[0,1,2])
            vsync_temp = np.array(vsync_csv.loc[:,1], dtype='uint')
            vsync.append(np.where((vsync_temp[1:] - vsync_temp[:-1]) ==1)[0])
       
        #arbituarily more than 10s interval would be made when concatenate ephys files.
        # Hmmm....this may meet some problem if the file recording is stopped right after turnning off sync.
        ses_e_end = esync_timestamps_load[np.where((esync_timestamps_load[1:] - esync_timestamps_load[:-1]) > 100000)[0]]
        ses_e_end = np.append(ses_e_end, esync_timestamps_load[-1])# last one sync needed here.
        esync_timestamps = [esync_timestamps_load[np.where(esync_timestamps_load < ses_e_end[0] + 100000)]]
        
        for i in range(1, Nses):
            esync_temp = esync_timestamps_load[np.where(esync_timestamps_load < ses_e_end[i] + 100000)]
            esync_temp = esync_temp[np.where(esync_temp > ses_e_end[i-1])]
            # esync_temp = esync_timestamps_load[np.where(esync_timestamps_load < ses_e_end[i] + 100000 & esync_timestamps_load > ses_e_end[i-1]+100000)]
            esync_timestamps.append(esync_temp)
        if 'signal_on' in experiment_tag:
            signal_on_timestamps = []
            for i in range(Nses):
                signal_on_temp = signal_on_timestamps_load[np.where(signal_on_timestamps_load < esync_timestamps[i][-1])]
                signal_on_temp = signal_on_temp[np.where(signal_on_temp > esync_timestamps[i][0])]
                # signal_on_temp = signal_on_timestamps_load[np.where(signal_on_timestamps > esync_timestamps[i][0])[0][0] : np.where(signal_on_timestamps_load < esync_timestamps[i][-1])[0][0]+1]
                signal_on_timestamps.append(signal_on_temp)

        
        # eventually shape of spk_t & spk_c are different, one is (n,) but the other(n,1). Hmmmm...
        timestamps.append(spike_times[np.where(spike_times < ses_e_end[0] + 100000)])
        spike_clusters2.append(spike_clusters[np.where(spike_times < ses_e_end[0] + 100000)])
        for i in range(1, Nses):
            spike_temp = spike_times[np.where(spike_times < ses_e_end[i] + 100000)] 
            cluster_temp = spike_clusters[np.where(spike_times < ses_e_end[i] + 100000)]
            cluster_temp = cluster_temp[np.where(spike_temp > ses_e_end[i-1] + 100000)]
            spike_temp = spike_temp[np.where(spike_temp > ses_e_end[i-1] + 100000)]
            timestamps.append(spike_temp)
            spike_clusters2.append(cluster_temp)
        #just for check
        print(ses_e_end)
    else:        
        print('Nses must be a positive integer.')
        
    if 'signal_on' in experiment_tag:
        return spike_clusters2, timestamps, clusters_quality, vsync, esync_timestamps, dlc_files, signal_on_timestamps
    else:
        return spike_clusters2, timestamps, clusters_quality, vsync, esync_timestamps, dlc_files
    

def sync_check(esync_timestamps, vsync, Nses, fontsize):
    
    # hold a sec. np.max(Esync_timestamps[1:] - Esync_timestamps[:-1]) == 14998???? this will accu.   not good.!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    if Nses == 1:
        if np.size(vsync) != np.size(esync_timestamps):
            print('N of E&V Syncs do not Equal!!! Problems with Sync!!!')
        else:
            print('N of E&V Syncs equal. You may continue.')
            # plot for check.
            fig = plt.figure(figsize=(20,10))
            ax1 = fig.add_subplot(1,2,1)
            ax2 = fig.add_subplot(1,2,2)
            esync_inter = esync_timestamps[1:] - esync_timestamps[:-1]
            vsync_inter = vsync[1:] - vsync[:-1]
            ax1.hist(esync_inter, bins = len(set(esync_inter)))
            ax1.set_title('N samples between Esyncs', fontsize=fontsize*1.3)
            ax2.hist(vsync_inter, bins = len(set(vsync_inter)))
            ax2.set_title('N frames between Vsyncs', fontsize=fontsize*1.3)
            
    else:
        fig = plt.figure(figsize=(20,10))
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)
        ax1.set_title('N samples between Esyncs', fontsize=fontsize*1.3)
        ax2.set_title('N frames between Vsyncs', fontsize=fontsize*1.3)
        # legend?
        for i in range(Nses):
            if np.size(vsync[i]) != np.size(esync_timestamps[i]):
                print('N of E&V Syncs do not Equal!!! Problems with Sync in ses ', str(i), '!!!')
            else:
                print('ses ', str(i),' N of E&V Syncs equal. You may continue.')
                esync_inter = esync_timestamps[i][1:] - esync_timestamps[i][:-1]
                vsync_inter = vsync[i][1:] - vsync[i][:-1]
                ax1.hist(esync_inter, bins = len(set(esync_inter)), alpha=0.2)
                ax2.hist(vsync_inter, bins = len(set(vsync_inter)), alpha=0.2)


def sync_cut_apply_spd_mask_100msbin(spike_clusters, timestamps, ses, esync_timestamps, sync_rate, experiment_tag):
    
    # head&tail cut here, then transform into frame_id for spd_mask.
    # applying spd_mask means sort spikes into running and staying.
    spike_clusters = np.delete(spike_clusters, np.where(timestamps > esync_timestamps[-1])[0])
    spike_clusters = np.delete(spike_clusters, np.where(timestamps < esync_timestamps[0])[0])
    timestamps = np.delete(timestamps, np.where(timestamps > esync_timestamps[-1])[0])
    timestamps = np.delete(timestamps, np.where(timestamps < esync_timestamps[0])[0])
    # assign time-values from esync_timestamps
    interp_y = np.linspace(0, (np.size(esync_timestamps)-1)/sync_rate, num=np.size(esync_timestamps))
    stamps2time_interp = interpolate.UnivariateSpline(esync_timestamps, interp_y, k=1, s=0)
    spiketime = stamps2time_interp(timestamps)
        
    # below is used with older version of spd_mask
    # if 'spatial' in experiment_tag:
    #     low_spd_spike_id = np.where(spikeframe == ses.spd_mask_low[:, None])[1]
    #     high_spd_spike_id = np.where(spikeframe == ses.spd_mask_high[:, None])[1] 
    #     spike_clusters_stay = spike_clusters[low_spd_spike_id]
    #     timestamps_stay = timestamps[low_spd_spike_id]
    #     spiketime_stay = spiketime[low_spd_spike_id]
    #     spikeframe_stay = spikeframe[low_spd_spike_id]
    #     spike_clusters = spike_clusters[high_spd_spike_id]
    #     timestamps = timestamps[high_spd_spike_id]
    #     spiketime = spiketime[high_spd_spike_id]
    #     spikeframe = spikeframe[high_spd_spike_id]
    if 'spatial' in experiment_tag:
        spiketime_bin_id = (spiketime//0.1).astype('uint')
        spike_spd_id = ses.spd_mask[spiketime_bin_id]
        high_spd_id = np.where(spike_spd_id==1)[0]
        low_spd_id= np.where(spike_spd_id==0)[0]
        
        spike_clusters_stay = spike_clusters[low_spd_id]
        timestamps_stay = timestamps[low_spd_id]
        spiketime_stay = spiketime[low_spd_id]
        spike_clusters = spike_clusters[high_spd_id]
        timestamps = timestamps[high_spd_id]
        spiketime = spiketime[high_spd_id]

        return (spike_clusters,timestamps,spiketime, spike_clusters_stay,timestamps_stay,spiketime_stay)
    else:
        return (spike_clusters,timestamps,spiketime)

def signal_stamps2time(esync_timestamps, signal_on_timestamps, Nses, sync_rate):
    if Nses == 1:
        interp_y = np.linspace(0, (np.size(esync_timestamps)-1)/sync_rate, num=np.size(esync_timestamps))
        stamps2time_interp = interpolate.UnivariateSpline(esync_timestamps, interp_y, k=1, s=0)
        signal_on_time = stamps2time_interp(signal_on_timestamps)
    else:
        signal_on_time = []
        for i in range(Nses):
            interp_y = np.linspace(0, (np.size(esync_timestamps[i])-1)/sync_rate, num=np.size(esync_timestamps[i]))
            stamps2time_interp = interpolate.UnivariateSpline(esync_timestamps[i], interp_y, k=1, s=0)
            signal_on_time.append(stamps2time_interp(signal_on_timestamps))
    return signal_on_time

# ----------------------------------------------------------------------------
#                  Functions 1D
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
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('scatter of spatial occupancy in pixels.', fontsize=fontsize)
    ax1.scatter(np.append(x, center_2[0]), np.append(y,center_2[1]), s=3, alpha=0.1)
    return center_2, R_2, residu_2       


def boxcar_smooth_1d_circular(arr, kernel_width=20):
    arr_smo = np.convolve(np.array([1/kernel_width]*kernel_width),
                          np.concatenate((arr,arr)))[kernel_width : (arr.shape[0]+kernel_width)]
    arr_smo = np.concatenate((arr_smo[-(kernel_width):], arr_smo[:-(kernel_width)]))
    return arr_smo


def ratemap_1d_circular(spike_time, time2pol, dwell_smo, nspatial_bins):
    spk_pol = ses.time2pol(spike_time)
    spk_bin = ((spk_pol+np.pi)/np.pi*nspatial_bins/2).astype('uint')
    Nspike_in_bins = npg.aggregate(spk_bin, 1, size=nspatial_bins)
    Nspike_in_bins_smo = boxcar_smooth_1d_circular(Nspike_in_bins)
    ratemap = Nspike_in_bins_smo/dwell_smo
    return ratemap

# def shuffle_test_1d_circular(units, session, esync_timestamps, Nses, sample_rate, sync_rate):#not that simillar to 2d shuffle. Ref, Monaco2014, head scanning, JJKnierim's paper.
#     shuffle_spk_trains_sessions = []
#     shuffle_Skaggs_info = []
#     shuffle_Olyper_info = []
# # to implement Olyper's info, as they use 100ms bin, however, our camera has framerate around 54.54. It might be proper to merge each 5 frames to get around 91.67 ms bin.
# # 91.67 ms, mind this.
# # so here has to a re-binning.
# # as for positional binning, just like Monaco2014, try 7.5 degree.
 
#     if Nses > 1:
#         shuffle_spk_trains = []
#         for j in range(Nses):
#             pol_temp = session[j].pol
#             pol_temp2 = pol_temp[:-int(np.size(pol_temp)%5)]# 5 frames binning.
#             pol_temp2 = pol_temp2.reshape(int(np.size(pol_temp2)/5),5)
#             pol_temp3 = np.mean(pol_temp2, axis=1)
#             # pol_temp3 = np.vstack((pol_temp3,pol_temp3,pol_temp3,pol_temp3,pol_temp3))
#             # pol_temp3 = pol_temp3.T
#             # pol_temp4 = pol_temp3.reshape(np.size(pol_temp3))
#             pol_rebin = (((pol_temp4/np.pi)+1)*24).astype('uint')
            
#             # re-resample of spk time for binning of 91.67 ms
#             t_end = (np.size(esync_timestamps[j])/sync_rate) - (np.size(esync_timestamps[j])/sync_rate)%0.09167
#             x = np.linspace(0.09167, t_end, num=int((np.size(esync_timestamps[j])/sync_rate)//0.09167), dtype='float32') - 0.01
#             y = np.linspace(0.09167, t_end, num=int((np.size(esync_timestamps[j])/sync_rate)//0.09167), dtype='float32')
#             resample2 = interpolate.interp1d(x, y, kind='nearest-up')
            
#             for i in units:            
#                 spk_temp = i.timestamps[j].extend(i.timestamps_stay[j])# combine high & low spd_spk.
#                 spk_temp2 = esync_timestamps[j] - spk_temp.astype('int64')# invert by the end of ses.
#                 for k in range(1000):# shuffle 1000 times, 0.01. time-shift 4~60s
#                     r = random.uniform(4, 60)
#                     spk_temp3 = spk_temp2 + int(r*sample_rate)# time shift
#                     spk_temp4 = spk_temp3[np.where(spk_temp3<esync_timestamps[j][-1])]
#                     spk_temp5 = spk_temp3[np.where(spk_temp3>esync_timestamps[j][-1])] - esync_timestamps[j][-1]
#                     spk_temp6 = spk_temp4.extend(spk_temp5)# those spks beyond the end of ses, wrapped to the start of ses.
#                     spk_time = (spk_temp6 - esync_timestamps[j][0]) / sample_rate
#                     spk_frame = (session[j].resample(spk_time)).astype('uint64')# resample
#                     spk_frame_run = spk_frame[np.where(spk_frame == session[j].spd_mask_high[:, None])[1]]#spd mask
#                     spk_time_run = spk_time[np.where(spk_frame == session[j].spd_mask_high[:, None])[1]]
                    
#                     #Skaggs spa info
#                     ratemap_temp = ratemap_1d_circular(spk_frame_run, session[j].pol_bin, session[j].dwell_smo, session[j].nspatial_bins)
#                     running_mean_rate = round(np.size(spk_frame_run)/session[j].frame_time[-1],2)
#                     Skaggs_info = round(np.nansum((session[j].dwell_smo/np.sum(session[j].dwell_smo)) * (ratemap_temp/running_mean_rate) * np.log2((ratemap_temp/running_mean_rate))),2)
#                     shuffle_Skaggs_info.append(Skaggs_info)
                    
#                     #Olyper spa info
#                     spk_time_resample_o = (resample2(spk_time_run)//91.76).astype('uint') - 1
#                     spk_t_bin = npg.aggregate(spk_time_resample_o, np.ones(np.size(spk_time_resample_o)), dtype='uint')
                    
            

            
#     elif Nses == 1:
#         pass
#     else:
#         print('Nses should be a positive integer.')
# ----------------------------------------------------------------------------
#                  Functions 2D
# ----------------------------------------------------------------------------
def boxcar_smooth_2d():
    pass

def gaussian_smooth_2d():
    pass   

def shuffle_test_2d():
    # this would be simple, just go with random temporal offset and play around 1000 times. No worries like in 1d.
    pass

def Kalman_filter_2d():#interpreted from Bohua's code.
    pass

def time2xy(ses,t):
    return np.array(ses.time2xy_interp[0](t), (ses.time2xy_interp[1](t)))
# ----------------------------------------------------------------------------
#                  Classes session
# ----------------------------------------------------------------------------

class Session(object):    
    def __init__(self, dlch5, dlc_col_ind_dict, vsync, sync_rate, experiment_tag,
                 ses_id=0, fontsize=15):
        # docstring?
        
        self.left_pos = np.vstack((np.array(dlch5[dlch5.columns[dlc_col_ind_dict['left_pos']]]), np.array(dlch5[dlch5.columns[dlc_col_ind_dict['left_pos']+1]]))).T
        self.right_pos = np.vstack((np.array(dlch5[dlch5.columns[dlc_col_ind_dict['right_pos']]]), np.array(dlch5[dlch5.columns[dlc_col_ind_dict['right_pos']+1]]))).T            
        self.id = ses_id
        self.vsync = vsync
        self.__experiment_tag = experiment_tag
        self.__sync_rate = sync_rate
        self.__fontsize = fontsize
        for key,value in dlc_col_ind_dict:# for extensive need from DLC. Presently a basic attribution.
            if key != 'left_pos' and key != 'right_pos':
                exec('self.'+key+'=np.vstack((np.array(dlch5[dlch5.columns[dlc_col_ind_dict['+str(value+)']]]), np.array(dlch5[dlch5.columns[dlc_col_ind_dict['+str(value)+']+1]]))).T')

        if 'circular' in self.__experiment_tag:
            # need to find center of the circular track                
            self.center = find_center_circular_track(np.vstack((self.left_pos, self.right_pos))[:,0], np.vstack((self.left_pos, self.right_pos))[:,1], fontsize=self.__fontsize)
            self.pixpcm = 2*self.center[1]/65# D == 65. emmm....
            

    def sync_cut_generate_frame_time(self):
        print('sync_cut of Session should only run for once, otherwise you need to reload files. So far, it only works on left and right pos.')
        self.left_pos = self.left_pos[self.vsync[0]:self.vsync[-1]+1, :]
        self.right_pos = self.right_pos[self.vsync[0]:self.vsync[-1]+1, :]
        #assign time values for frames. So far for a single ses, single video.
        frame2time_interp = interpolate.UnivariateSpline(vsync, np.linspace(0, (np.size(vsync)-1)/self.__sync_rate, num=np.size(vsync)),
                                                         k=1, s=0)
        self.frame_time = frame2time_interp(np.arange(self.left_pos.shape[0])).astype('float64')
        self.total_time = self.frame_time[-1]
    
    def remove_nan_merge_pos(self):
        nan_id = np.isnan(self.left_pos) + np.isnan(self.right_pos)
        nan_id = np.where(nana_id == 2, 1, nan_id)
        self.frame_time = self.frame_time[~nan_id+2]
        self.left_pos = self.left_pos[~nan_id+2]
        self.right_pos = self.right_pos[~nan_id+2]
        self.pos_pix = (self.left_pos + self.right_pos)/2
        if 'circular' in self.__experiment_tag:
            self.pos = ((self.left_pos + self.right_pos)/2 - self.center[0])/self.pixpcm
        else:
            print('you need to code your way do define pixels per cm, to go furthur.')
    
    def generate_time2xy_interpolate(self):
        # using scipy.interpolate.UnivariateSpline seperately with x&y might cause some infidelity. Mind this.
        # k = 3, how to set a proper smooth factor s???
        # what about it after Kalman filter?
        # how to check this???
        time2x_interp = interpolate.UnivairateSpline(self.frame_time, self.pos[:,0])
        time2y_interp = interpolate.UnivairateSpline(self.frame_time, self.pos[:,1])
        self.time2xy_interp = (time2x_interp, time2y_interp)
   
    # below is older version of spd_mask which aligns to frames.
    # def generate_spd_mask(self, spd_threshold = 2):
    #     # spd check mask. Hmmm... instant dist., what to smooth? xy? instant dist. vec? spd?. presently on inst dist.
    #     self.inst_dist = np.sqrt((self.pos[1:,:] - self.pos[:-1,:])[:,0]**2 + (self.pos[1:,:] - self.pos[:-1,:])[:,1]**2)
    #     # 1d gaussian filter on dist. Do not know if it is proper. And, what is the relationship between sigma and Nsamples of kernel????????
    #     self.inst_dist_smo = ndimage.gaussian_filter1d(self.inst_dist, sigma = 1)
    #     self.inst_spd = self.inst_dist_smo/(self.frame_time[1:] - self.frame_time[:-1])
    #     self.spd_mask_low = np.where(self.inst_spd < spd_threshold)[0].astype('uint64')# presently used bar is 2cm/s
    #     self.spd_mask_high = np.where(self.inst_spd > spd_threshold)[0].astype('uint64')

    def generate_spd_mask_100ms_bin(self, threshold=2):
        t = np.linspace(0, self.total_time, num=(self.total_time*10 +1))
        x = self.time2xy_interp[0](t)
        y = self.time2xy_interp[1](t)
        dist = np.sqrt((x[1:]-x[:-1])**2 + (y[1:]-y[:-1])**2)
        self.spd_mask = np.where(dist*10 > 2)[0]
    
    def generate_dwell_map_circular(self, nspatial_bins=360, smooth='boxcar'):
        if 'circular' not in self.__experiment_tag:
            print('you choose a wrong method.')
        else:
            #just a repeat after spd_mask.
            self.pol = np.angle(self.pos[:,0] + 1j*self.pos[:,1])
            self.pol_bin = ((self.pol+np.pi)/np.pi*nspatial_bins/2).astype('uint')

            self.time2pol = interpolate.UnivariateSpline(self.frame_time, self.pol, k=1)# could this work for periodic data??

            # emmm....so where is 0 degree???  It is Right.
            frame_time_inter = np.hstack((0, (self.frame_time[1:] - self.frame_time[:-1])))
            dwell = npg.aggregate(self.pol_bin, frame_time_inter)
            # for stability.
            # self.middle_time = self.frame_time[int(self.frame_time.shape[0]/2)]
            self.middle_frame_id = int(self.pos.shape[0]/2)
            dwell_1half = npg.aggregate(self.pol_bin[:int((self.pol_bin).shape[0]/2)], frame_time_inter[:int((self.pol_bin).shape[0]/2)])
            dwell_2half = npg.aggregate(self.pol_bin[int((self.pol_bin).shape[0]/2):], frame_time_inter[int((self.pol_bin).shape[0]/2):])
            if smooth == 'boxcar':
                self.dwell_smo = boxcar_smooth_1d_circular(dwell)
                self.dwell_1half_smo = boxcar_smooth_1d_circular(dwell_1half)
                self.dwell_2half_smo = boxcar_smooth_1d_circular(dwell_2half)
            else:
                print('for other type of kernels... to be continue...')
            
            fig = plt.figure(figsize=(10,30))
            ax1 = fig.add_subplot(311)
            ax1.set_title('spd check', fontsize=self.__fontsize*1.3)
            ax1.hist(self.inst_spd, range = (0,70), bins = 100)
            spd_bin_max = np.max(np.bincount(self.inst_spd.astype('int'), minlength=100))
            ax1.plot(([np.median(self.inst_spd)]*2), [0, spd_bin_max*0.7], color='k')
            ax1.set_xlabel('animal spd cm/s', fontsize=self.__fontsize)
            ax1.set_ylabel('N frames', fontsize=self.__fontsize)
            ax2 = fig.add_subplot(312)
            ax2.set_title('animal spatial occupancy', fontsize=self.__fontsize)
            ax2.plot(np.linspace(1, nspatial_bins, num=nspatial_bins, dtype='int'), self.dwell_smo, color='k')
            ax2.plot(np.linspace(1, nspatial_bins, num=nspatial_bins, dtype='int'), self.dwell_1half_smo, color='r')
            ax2.plot(np.linspace(1, nspatial_bins, num=nspatial_bins, dtype='int'), self.dwell_2half_smo, color='b')
            ax2.set_ylim(0, np.max(self.dwell_smo)*1.1)
            ax2.set_xlabel('spatial degree-bin', fontsize=self.__fontsize)
            ax2.set_ylabel('occupancy in sec', fontsize=self.__fontsize)
            ax3 = fig.add_subplot(313)
            ax3.scatter(self.pol, np.linspace(0, self.total_time, num=np.size(self.pol)), c='k', s=0.1)
            ax3.set_title('animal trajectory', fontsize=self.__fontsize*1.3)
            ax3.set_xlabel('position in radius degree.', fontsize=self.__fontsize)
            ax3.set_ylabel('time in sec', fontsize=self.__fontsize)
            if abs(np.sum(self.dwell_smo) - np.max(self.frame_time)) > 0.5:
                print('something wrong with dwell-map, sum_time not matching with frame_time')
        
    def generate_temperal_spatial_binning_for_positional_info(self):
        if 'circular' not in self.__experiment_tag:
            print('you choose a wrong method.')
        else:
            pass
            

        

 # ----------------------------------------------------------------------------
 #                  Classes unit
 # ----------------------------------------------------------------------------
 
class Unit(object):
    def __init__(self, cluid, spike_pack, quality, Nses, experiment_tag, fontsize):
        self.cluid = cluid
        self.quality = quality
        self.type = 'unknown'#IN or PC  
        self.meanwaveform = []
        self.__Nses = Nses
        self.__fontsize = fontsize
        self.__experiment_tag = experiment_tag


        if Nses == 1 and 'spatial' in self.__experiment_tag:
            # unpacking spike_pack.
            spike_pick_cluid = np.where(spike_pack[0] == self.cluid)[0]
            self.timestamps = spike_pack[1][spike_pick_cluid]
            self.spiketime = spike_pack[2][spike_pick_cluid]
            spike_stay_pick_cluid = np.where(spike_pack[4] == self.cluid)[0]
            self.timestamps_stay = spike_pack[5][spike_stay_pick_cluid]
            self.spiketime_stay = spike_pack[6][spike_stay_pick_cluid]
            self.Nspikes_total = np.size(self.timestamps) + np.size(self.timestamps_stay)
            # initialize spa. params.
            self.ratemap = []
            self.peakrate = []
            self.spatial_info = []
            self.stability = []
            self.global_mean_rate = []
            self.__running_mean_rate = []
            self.is_place_cell = False
        elif Nses > 1 and 'spatial' in self.__experiment_tag:
            self.timestamps = [0 for i in range(Nses)]
            self.spiketime = [0 for i in range(Nses)]
            self.timestamps_stay = [0 for i in range(Nses)]
            self.spiketime_stay = [0 for i in range(Nses)]
            self.Nspikes_total = [0 for i in range(Nses)] 
            self.ratemap = [0 for i in range(Nses)]
            self.peakrate = [0 for i in range(Nses)]
            self.spatial_info = [0 for i in range(Nses)]
            self.stability = [0 for i in range(Nses)]
            self.global_mean_rate = [0 for i in range(Nses)]
            self.__running_mean_rate = [0 for i in range(Nses)]
            self.is_place_cell = False
            # unpacking
            for i in range(Nses):
                spike_pick_cluid = np.where(spike_pack[i][0] == self.cluid)[0]
                spike_stay_pick_cluid = np.where(spike_pack[i][4] == self.cluid)[0]
                self.timestamps[i] = spike_pack[i][1][spike_pick_cluid]
                self.spiketime[i] = spike_pack[i][2][spike_pick_cluid]
                self.timestamps_stay[i] = spike_pack[i][5][spike_stay_pick_cluid]
                self.spiketime_stay[i] = spike_pack[i][6][spike_stay_pick_cluid]
                self.Nspikes_total[i] = np.size(self.timestamps[i]) + np.size(self.timestamps_stay[i])
            
        elif Nses == 1 and 'spatial' not in self.__experiment_tag:
            spike_pick_cluid = np.where(spike_pack[0] == self.cluid)[0]
            self.timestamps = spike_pack[1][spike_pick_cluid]
            self.spiketime = spike_pack[2][spike_pick_cluid]
        
        elif Nses > 1 and 'spatial' not in self.__experiment_tag:
            self.timestamps = [0 for i in range(Nses)]
            self.spiketime = [0 for i in range(Nses)]
            for i in Nses:
                spike_pick_cluid = np.where(spike_pack[i][0] == self.cluid)[0]
                spike_stay_pick_cluid = np.where(spike_pack[i][4] == self.cluid)[0]
                self.timestamps[i] = spike_pack[i][1][spike_pick_cluid]
                self.spiketime[i] = spike_pack[i][2][spike_pick_cluid]
        else:
            print('Wrong input for unit.')


                                

            
    def simple_putative_IN_PC_by_firingrate(self, ses):
        # this is a simple way to put IN and PC not by waveform but only global mean firing rate.
        # see Nuenuebel 2013, DR with L/MEC, threshold of mean firing rate is 10Hz.
        if 'spatial' in self.__experiment_tag:
            if self.__Nses == 1:
                if self.global_mean_rate > 10:
                    self.type = 'inhibory'
                else:
                    self.type = 'excitatory'
            else:
                ses_inds = [i.id for i in ses]
                if (np.array(self.global_mean_rate)[ses_inds].all() > 10).all() == True:
                    self.type = 'inhibitory'
                elif (np.array(self.global_mean_rate)[ses_inds] < 10).all() == True:
                    self.type = 'excitatory'
                else:
                    self.type = 'unsure' 
        else:
            print('you might used wrong method.')
            
    def report_spatial(self):
        if 'spatial' in self.__experiment_tag:
            print('cluster id:', self.cluid, '\n Nspike:', self.Nspikes_total, '\n peakrate:', self.peakrate, '\n mean rate while running:', self.__running_mean_rate, '\n spa. info:', self.spatial_info, '\n stability:', self.stability)
        else:
            print('you might used wrong method.')
    
    
    def raster_plot_peri_stimulus(self, ses, signal_on_time, pre_sec=30, post_sec=30, stim_color='yellow'):
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        ax.set_title('PSTH around laser-on, clu'+str(self.cluid), fontsize=self.__fontsize*1.3)
        ax.set_xlabel('Time in sec', fontsize=self.__fontsize)
        ax.set_xlim(left=-(pre_sec), right=post_sec)
        ax.set_ylabel('Trials', fontsize=self.__fontsize)
        
        if self.__Nses == 1:
            signal_on_temp = signal_on_time
        else:
            signal_on_temp = signal_on_time[ses.id]
        ax.set_ylim(bottom=1, top=np.size(signal_on_temp)+1)
        if 'spatial' in self.__experiment_tag:
            spike_time = np.concatenate((self.spiketime[ses.id], self.spiketime_stay[ses.id]))
        else:
            spike_time = self.spike_time
        for i in range(np.size(signal_on_temp)):
            spike_time_temp = spike_time[np.where(spike_time < (signal_on_temp[i] + post_sec))]
            spike_time_temp = spike_time_temp[np.where(spike_time_temp > (signal_on_temp[i] - pre_sec))].astype('int64')
            spike_time_temp = spike_time_temp - signal_on_temp[i]
            ax.scatter(spike_time_temp, np.array([i+1]*np.size(spike_time_temp)).astype('uint16'), c='k', marker='|', s=40)
        ax.fill_between([0, post_sec], 0, np.size(signal_on_temp)+1, facecolor=stim_color, alpha=0.5)
   
         
    def opto_inhibitory_tagging(self, ses, signal_on_time, mode, p_threshold=0.01, laser_on_sec=30, laser_off_sec=30, shuffle_range_sec=30):
        # it takes laser on as start of a cycle.
        if self.__Nses == 1:
            signal_on_temp = signal_on_time
        else:
            signal_on_temp = signal_on_time[ses.id]
        if 'spatial' in self.__experiment_tag:
            spike_time = np.concatenate((self.spike_time[ses.id], self.spike_time_stay[ses.id]))
        else:
            spike_time = self.spike_time
        if mode == 'ranksum':# in ranksum test mode, shuffle range is not used.
            on_spk_count = []
            off_spk_count = []
            for i in range(np.size(signal_on_temp)):
                spike_time_temp = spike_time[np.where(spike_time < (signal_on_temp[i] + laser_on_sec))]
                spike_time_temp = spike_time_temp[np.where(spike_time_temp > signal_on_temp[i])]
                on_spk_count.append(np.size(spike_time_temp))
                spike_time_temp2 = spike_time[np.where(spike_time < (signal_on_temp[i] + (laser_on_sec+laser_off_sec)))]
                spike_time_temp2 = spike_time_temp2[np.where(spike_time_temp2 > (signal_on_temp[i] + laser_on_sec))]
                off_spk_count.append(np.size(spike_time_temp2))
            statistic, pvalue = stats.ranksums(on_spk_count, off_spk_count, alternative='less')
            if pvalue < p_threshold:
                print('clu'+str(self.cluid), ' is positive, p-value', pvalue)
                self.opto_tag = 'positive'
            # emmm...how to deal with margitial?
            else:
                print('clu'+str(self.cluid), ' is negative, p-value', pvalue)
                self.opto_tag = 'negative'
        elif mode == 'shuffle':
            print('shuffle test is not finished yet')
            # 1000times, 0.01
            pass            
        else:
            print('Wrong mode or the mode has not been coded.')



    def plot_raw_trace(self, ses):
        if type(ses) == list:
            pass
        else:
            fig = plt.figure(figsize=(10,10))
            ax1 = fig.add_subplot(111)
            ax1.plot(ses.left_pos+ses.right_pos)
            print('this method is not finished yet.')
    
    def plot_PSTH(self, ses):
        pass


 # ----------------------------------------------------------------------------
 #            Derived Classes of Unit
 # ----------------------------------------------------------------------------

class Unit1DCircular(Unit):
    def __init__(self, cluid, spike_pack, quality, Nses, experiment_tag, fontsize):
        super().__init__(cluid, spike_pack, quality, Nses, experiment_tag, fontsize)
        

    def get_ratemap_1d_circular(self, ses, nspatial_bins):
        if 'circular' in self.__experiment_tag and 'spatial' in self.__experiment_tag:
            if self.__Nses == 1:
                self.ratemap = ratemap_1d_circular(self.spike_time, ses.time2pol, ses.dwell_smo, nspatial_bins)
                self.peakrate = round(np.max(self.ratemap),2)
            else:
                self.ratemap[ses.id] = ratemap_1d_circular(self.spiketime[ses.id], ses.time2pol, ses.dwell_smo, nspatial_bins)
                self.peakrate[ses.id] = round(np.max(self.ratemap[ses.id]),2)       
        else:
            print('you might used wrong method.')
    
    
    def get_spatial_info_Skaggs(self, ses):
        if 'spatial' in self.__experiment_tag:
            if self.__Nses == 1:
                self.global_mean_rate = round((np.size(self.timestamps)+np.size(self.timestamps_stay))/ses.frame_time[-1],2)
                
                # running time is related to spd_mask with its temporal binning windown length.
                running_time = np.size(np.where(ses.spd_mask==1)[0])*0.1# presently time window for spd_mask is 100ms.
                
                self.__running_mean_rate = round(np.size(self.timestamps)/running_time
                self.spatial_info = round(np.nansum((ses.dwell_smo/np.sum(ses.dwell_smo)) * (self.ratemap/self.__running_mean_rate) * np.log2((self.ratemap/self.__running_mean_rate))),2)
            else:
                self.global_mean_rate[ses.id] = round((np.size(self.timestamps[ses.id])+np.size(self.timestamps_stay[ses.id]))/ses.frame_time[-1],2)
                running_time = np.size(np.where(ses.spd_mask==1)[0])*0.1# presently time window for spd_mask is 100ms.
                self.__running_mean_rate[ses.id] = round(np.size(self.timestamps[ses.id])/running_time
                self.spatial_info[ses.id] = round(np.nansum((ses.dwell_smo/np.sum(ses.dwell_smo)) * (self.ratemap[ses.id]/self.__running_mean_rate[ses.id]) * np.log2((self.ratemap[ses.id]/self.__running_mean_rate[ses.id]))),2)
        else:
            print('you might used wrong method.')
            
    def get_spatial_info_Olyper(self, ses):
        pass



    def get_stability_1d_circular(self, ses, nspatial_bins=360):
        if 'circular' in self.__experiment_tag and 'spatial' in self.__experiment_tag:
            if self.__Nses == 1:
                spike_time_1half = self.spiketime[np.where(self.spike_time < ses.frame_time[-1]/2)]
                spike_time_2half = self.spiketime[np.where(self.spike_time > ses.frame_time[-1]/2)]
                ratemap_1half = ratemap_1d_circular(spike_time_1half, ses.time2pol, ses.dwell_1half_smo, nspatial_bins)
                ratemap_2half = ratemap_1d_circular(spike_time_1half, ses.time2pol, ses.dwell_2half_smo, nspatial_bins)
                self.stability = round(np.corrcoef(np.vstack((ratemap_1half,ratemap_2half)))[1,0],2)
            else:
                spike_time_1half = self.spiketime[ses.id][np.where(self.spike_time[ses.id] < ses.frame_time[-1]/2)]
                spike_time_2half = self.spiketime[ses.id][np.where(self.spike_time[ses.id] > ses.frame_time[-1]/2)]
                ratemap_1half = ratemap_1d_circular(spike_time_1half, ses.time2pol, ses.dwell_1half_smo, nspatial_bins)
                ratemap_2half = ratemap_1d_circular(spike_time_1half, ses.time2pol, ses.dwell_2half_smo, nspatial_bins)
                self.stability[ses.id] = round(np.corrcoef(np.vstack((ratemap_1half,ratemap_2half)))[1,0],2)
        else:
            print('you might used wrong method.')
 
        
    def plot_ratemap_1d_circular_polar(self, ses, nspatial_bins=360
                                       color_list=['k','cyan','grey','b','orange'], 
                                       legend_list=['standard1', '45 degree conflict', 'standard2', '135 degree conflict', 'inhibitary tagging']):
        if 'circular' in self.__experiment_tag and 'spatial' in self.__experiment_tag:
            fig = plt.figure(figsize=(20,10))
            ax1 = fig.add_subplot(121, polar=True)
            ax1.set_theta_direction('clockwise')
            ax2 = fig.add_subplot(122)
            if self.__Nses == 1:
                theta = np.linspace(0, 2*np.pi, num=nspatial_bins)
                ax1.plot(theta, self.ratemap, c=color_list[ses.id])
                ax2.plot(self.ratemap, c=color_list[ses.id])
                ax1.legend(legend_list[0], fontsize=self.__fontsize, loc='lower right')
            else:
                theta = np.linspace(0, 2*np.pi, num=nspatial_bins)
                for i in ses:
                    ax1.plot(theta, self.ratemap[i.id], c=color_list[i.id])
                    ax2.plot(self.ratemap[i.id], c=color_list[i.id])
                ax1.legend(legend_list, fontsize=self.__fontsize, loc='lower right')
            ax1.set_title('ratemap in 5 sessions, clu'+str(self.cluid)+', '+str(self.quality), fontsize=self.__fontsize*1.3)
            ax2.set_title('spa. info ='+str(self.spatial_info))
            ax1.set_xlabel('spatial bin', fontsize=self.__fontsize)
            ax1.set_ylabel('firing rate', fontsize=self.__fontsize)
            ax2.set_xlabel('spatial bins', fontsize=self.__fontsize)
            ax2.set_ylabel('firing rate', fontsize=self.__fontsize)
        else:
            print('you might used wrong method.')

    

    def rotational_correlation_DR(self, ses1, ses2, nspatial_bins=360, bin_increment=5):
        if 'DR' not in self.__experiment_tag:
            print('Wrong method was used.')
        else:
            rotational_corrcoef = [np.corrcoef(self.ratemap[ses1.id], self.ratemap[ses2.id])[0,1]]
            for i in range(bin_increment, nspatial_bins, bin_increment):
                ratemap_rotate = np.concatenate((self.ratemap[ses2.id][i:],self.ratemap[ses2.id][:i]))
                rotational_corrcoef.append(np.corrcoef(self.ratemap[ses1.id], ratemap_rotate)[0,1])
            rotational_corrcoef = np.array(rotational_corrcoef)
            fig = plt.figure(figsize=(10,10))
            ax1 = fig.add_subplot(111)
            x = np.linspace(0, 360, num=int(nspatial_bins/bin_increment), endpoint=False)
            ax1.plot(x, rotational_corrcoef, c='r')
            ax1.plot(x, [0.75]*np.size(x), c='green')
            ax1.set_title('rota. corr. of clu'+str(self.cluid)+' ')
            try:
                self.rotational_correlation_peak.append(np.where(rotational_corrcoef == np.max(rotational_corrcoef))[0][0]*bin_increment)
            except:
                self.rotational_correlation_peak = [np.where(rotational_corrcoef == np.max(rotational_corrcoef))[0][0]*bin_increment]
                
    def plot_ratemap_and_rotational_corr(self, ses, ses_rot, nspatial_bins=360):
        
        color_list=['k','cyan','grey','b','orange']
        legend_list=['standard1', '90 degree conflict', 'standard2', '180 degree conflict', 'inhibitory tagging']
        bin_increment = 5
        
        fig = plt.figure(figsize=(30,10))
        ax1 = fig.add_subplot(131, polar=True)
        ax1.set_theta_direction('clockwise')
        ax2 = fig.add_subplot(132)
        theta = np.linspace(0, 2*np.pi, num=nspatial_bins)
        for i in ses:
            ax1.plot(theta, self.ratemap[i.id], c=color_list[i.id])
            ax2.plot(self.ratemap[i.id], c=color_list[i.id])
        ax1.legend(legend_list, fontsize=self.__fontsize, loc='lower right')
        ax1.set_title('ratemap in 5 sessions, clu'+str(self.cluid)+', '+str(self.quality), fontsize=self.__fontsize*1.3)
        ax2.set_title('spa. info ='+str(self.spatial_info))
        ax1.set_xlabel('spatial bin', fontsize=self.__fontsize)
        ax1.set_ylabel('firing rate', fontsize=self.__fontsize)
        ax2.set_xlabel('spatial bins', fontsize=self.__fontsize)
        ax2.set_ylabel('firing rate', fontsize=self.__fontsize)
        
        
        rotational_corrcoef1 = [np.corrcoef(self.ratemap[ses_rot[0].id], self.ratemap[ses_rot[1].id])[0,1]]
        rotational_corrcoef2 = [np.corrcoef(self.ratemap[ses_rot[2].id], self.ratemap[ses_rot[3].id])[0,1]]
        for i in range(bin_increment, nspatial_bins, bin_increment):
            ratemap_rotate1 = np.concatenate((self.ratemap[ses_rot[1].id][i:],self.ratemap[ses_rot[1].id][:i]))
            rotational_corrcoef1.append(np.corrcoef(self.ratemap[ses_rot[0].id], ratemap_rotate1)[0,1])
            ratemap_rotate2 = np.concatenate((self.ratemap[ses_rot[3].id][i:],self.ratemap[ses_rot[3].id][:i]))
            rotational_corrcoef2.append(np.corrcoef(self.ratemap[ses_rot[2].id], ratemap_rotate2)[0,1])
        rotational_corrcoef1 = np.array(rotational_corrcoef1)
        rotational_corrcoef2 = np.array(rotational_corrcoef2)

        ax3 = fig.add_subplot(133)
        x = np.linspace(0, 360, num=int(nspatial_bins/bin_increment), endpoint=False)
        ax3.plot(x, rotational_corrcoef1, c='cyan')
        ax3.plot(x, rotational_corrcoef2, c='blue')
        ax3.plot([180,180], [-0.2,1], linestyle='dashed', c='grey')
        ax3.plot(x, [0.75]*np.size(x), c='red')
        ax3.set_title('rota. corr. of clu'+str(self.cluid)+' ')
        # as in Knierim 2002 the criteria include peak corr greater than 0.75.
        self.rotational_correlation_peak = []
        if np.max(rotational_corrcoef1) > 0.75:
            self.rotational_correlation_peak.append(np.where(rotational_corrcoef1 == np.max(rotational_corrcoef1))[0][0]*bin_increment)
        if np.max(rotational_corrcoef2) > 0.75:
            self.rotational_correlation_peak.append(np.where(rotational_corrcoef2 == np.max(rotational_corrcoef2))[0][0]*bin_increment)

        

                    
                
            

            



























            
    

        
        
