# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 15:32:33 2023

@author: NCP-LAB NMZ
"""

from NCPnmz import *
import shutil


mouse = 'EM15'
dir_baked = Path(r'D:\NMZ\DataEphys')
date = '20240511'
task = 'ToggleTrain'

dlc_tail='DLC_resnet50_PlusMazeApr29shuffle1_370000_filtered.h5'
dlc_col_ind_dict = ['leftbulb', 'rightbulb']
tags = {'Nses':1,
        'theme':'spatial',
        'maze shape':'PlusMaze',
        'experiment':'ToggleTrain',
        'sync_rate':50,
        'sample_rate':30000,
        'fontsize':15}

# 把文件都放在有task信息的文件夹中处理
if (dir_baked/mouse/date/(date+mouse+task)/'params.py').exists():
    if len(list((dir_baked/mouse/date).glob('*.*')))==0:                       #如果Baked文件夹中没有那些avi，ns6，csv.....
        raise Exception('No file in the baked folder, check pre treatment, maybe already moved')
    else:
        for i in (dir_baked/mouse/date).glob('*'+task+'*.*'):
            shutil.move(i, dir_baked/mouse/date/(date+mouse+task))  #把Baked文件夹里的东西都拷到处理文件夹里
            print(str(i), ' been moved ')
    (dir_baked/mouse/date/(date+mouse+task)/'plot').mkdir(parents=True)
else:
    raise Exception('Please do preprocess and kilosort first')


#%% 初始化数据
# 开始使用NCP package做初步载入
fdir = dir_baked/mouse/date
fn = date+mouse+task
spike_clusters, timestamps, clusters_info, waveforms, vsync, esync_timestamps, dlc_files, frame_state = load_files_newcam(fdir,fn,dlc_tail,tags)
# 有额外同步信号再用这一句 spike_clusters, timestamps, clusters_quality, waveforms, vsync, esync_timestamps, dlc_files, signal_on_timestamps = load_files(fdir,fn,Nses,dlc_tail)

# 同步信号的检查
sync_check(esync_timestamps, vsync, tags)
# 掐头去尾
spike_clusters, timestamps = sync_cut_head_tail(spike_clusters, timestamps, esync_timestamps)
# 生成有真实世界时间的spiketime
spiketime=timestamps2time(timestamps, esync_timestamps, tags)

#%% session
# 把dlc_file转换成真实世界的距离（cm）

# nodes_set={'name':             [     '中心左上',      '中心左下',      '中心右上',       '中心右下', 'Toggle1上', 'Toggle1下', 'Toggle2左', 'Toggle2右', 'Toggle3上', 'Toggle3下',   'reward左',  'reward右'],
#            'rough_position':   [     (753,945),      (875,940),      (757,1069),      (879,1064),   (697,637),   (928,638),  (458,898),  (458,1126),  (705,1364),  (936,1362),   (1174,885), (1174,1111)],
#            'corner_type':      [     'anti_LU',      'anti_LD',       'anti_RU',       'anti_RD',        'LD',        'LU',       'RU',        'LU',        'RD',        'RU',         'RD',        'LD'],
#            'precise_position': [             0,              0,               0,               0,           0,           0,          0,           0,           0,           0,            0,           0],
#            'real_position':    [   (24.5,24.5),    (30.5,24.5),     (24.5,30.5),     (30.5,30.5),     (22,10),     (33,10),    (10,22),     (10,33),     (22,45),     (33,45),      (45,22),     (45,33)]}

nodes_set={'name':             [     '中心左上',      '中心左下',      '中心右上',       '中心右下', 'Toggle1上', 'Toggle1下', 'Toggle2左', 'Toggle2右', 'Toggle3上', 'Toggle3下',   'reward左',  'reward右'],
           'rough_position':   [     (480,470),      (540,470),       (480,535),        (540,535),   (450,290),  (570,290),  (310,445),   (310,570),   (450,705),   (570,705),   (720,445),    (720,570)],
           'corner_type':      [     'anti_LU',      'anti_LD',       'anti_RU',       'anti_RD',        'LD',        'LU',       'RU',        'LU',        'RD',        'RU',         'RD',        'LD'],
           'precise_position': [             0,              0,               0,               0,           0,           0,          0,           0,           0,           0,            0,           0],
           'real_position':    [   (24.5,24.5),    (30.5,24.5),     (24.5,30.5),     (30.5,30.5),     (22,10),     (33,10),    (10,22),     (10,33),     (22,45),     (33,45),      (45,22),     (45,33)]}

pixel_to_cm_PlusMaze(dlc_files, nodes_set, fdir, fn)

# 实例化Session
ses = DetourSession(dlc_files, dlc_col_ind_dict, frame_state, esync_timestamps, tags)
# Ses.sync_cut_head_tail()  no need in new camera setup
ses.framestamps2time()
ses.set_mouse_pos_as('leftbulb', 'rightbulb')
ses.get_head_direction()
ses.generate_interpolater()
ses.generate_spd_mask('mouse_pos', temporal_bin_length=1/tags['sync_rate'])
ses.generate_dwell_map_PlusMaze(fdir, fn, temporal_bin_length=1/tags['sync_rate'])


#%% unit
from NCPnmz import *
spike_pack = apply_speed_mask(spike_clusters, timestamps, spiketime, ses, tags, temporal_bin_length=1/tags['sync_rate'])    # if non-spatial-task spike_pack = (spike_clusters, timestamps, spiketime)

units_set = []
unit_id = 0   # after selection, continuous new id that will present in fig
spatial_info_pool=np.array([])
for i in range(clusters_info.shape[0]):  # 
    if clusters_info.loc[i,'group'] != 'noise'  and True:
        
        unit = Unit(spike_pack, clusters_info.iloc[i], tags)
        unit.id = unit_id
        unit.get_mean_waveforms(waveforms)
        unit.get_mean_rate(ses)
        unit.get_rate_map_PlusMaze(ses)
        unit.get_spatial_information_skaggs(ses)
        unit.get_positional_information_olypher_PlusMaze(ses)
        unit.get_place_field_PlusMaze(ses)
        unit.get_place_field_COM()
        # unit.get_place_field_ellipse_fit()
        unit.simple_putative_IN_PC_by_firingrate(ses)
        
        unit.spatial_info_shuffle_PlusMaze(ses)
        spatial_info_pool = np.append(spatial_info_pool,unit.spatial_info_pool)
        
        units_set.append(unit)
        unit_id += 1
    print(f'\r {i+1}/{clusters_info.shape[0]}',end='')
        
            
for unit in units_set:  
    unit.report(ses, fdir, fn, unit_id, spatial_info_pool)
    

#%% LFP
from NCPnmz import *
lfp = LFP(fdir, fn, esync_timestamps, tags)
lfp.SWR_detect()
lfp.theta_cycle_detect()
# lfp.CWT(uplim_Hz=30, totalscale=30, starttime = 400, t=25)

#%% average ripple trace
averaged_raw_trace = []
for i in range(32):
    lfp.SWR_detect(channel=i)
    for j in range(lfp.ripple_peak.shape[0]):
        averaged_raw_trace.append(lfp.data_all_ch[i, lfp.ripple_peak[j]-160:lfp.ripple_peak[j]+161])
all_averaged_raw_trace = np.vstack(averaged_raw_trace)
all_averaged_raw_trace = np.mean(all_averaged_raw_trace, axis=0)

averaged_ripple_trace = []
for i in range(32):
    lfp.SWR_detect(channel=i)
    for j in range(lfp.ripple_peak.shape[0]):
        averaged_ripple_trace.append(lfp.ripple_data[lfp.ripple_peak[j]-160:lfp.ripple_peak[j]+161])
all_averaged_ripple_trace = np.vstack(averaged_ripple_trace)
all_averaged_ripple_trace = np.mean(all_averaged_ripple_trace, axis=0)

t = np.linspace(-160, 160, num=321)/2

plt.plot(t,all_averaged_raw_trace)
plt.xlim(-80,80)
plt.savefig(  lfp.img_save_path/ ('ripple peak raw '+'all'+'.svg')  )
plt.close()

plt.plot(t,all_averaged_ripple_trace)
plt.xlim(-80,80)
plt.savefig(  lfp.img_save_path/ ('ripple peak bandpassed '+'all'+'.svg')  )
plt.close()

ripple_set = []
for i in range(32):
    lfp.SWR_detect(channel=i)
    ripple_set += lfp.ripple_start_end.tolist()
ripple_start_end = np.array(ripple_set)
ripple_start_end_sort = np.sort(ripple_start_end, axis=0)
plt.scatter(ripple_start_end_sort[:,0],ripple_start_end_sort[:,1])



    

#%% test
unit = Unit(spike_pack, clusters_info.iloc[1], tags)
unit.id = unit_id
unit.get_mean_waveforms(waveforms)
unit.get_mean_rate(ses)
unit.get_rate_map_PlusMaze(ses)
unit.get_spatial_information_skaggs(ses)
unit.get_positional_information_olypher_PlusMaze(ses)
unit.get_place_field_PlusMaze(ses)
unit.get_place_field_COM()
# unit.get_place_field_ellipse_fit()
unit.simple_putative_IN_PC_by_firingrate(ses)

unit.spatial_info_shuffle_PlusMaze(ses)
spatial_info_pool = unit.spatial_info_pool


unit.spatial_info


a=unit.spatial_info_pool
#%%
time_range = [4,40]
shuffle_chunk_size = 6
shuffle_times = 1000

time_offset = np.random.uniform(time_range[0], time_range[1], shuffle_times)
time_offset[int(len(time_offset)/2):] *= -1                 # second half minus the offset
time_offset[int(len(time_offset)/2):] += ses.total_time     # make sure the tiem is positive

unit.spatial_info_pool = np.zeros(shuffle_times)

for i in range(shuffle_times):
    # add time offset
    shuffled_spiketime = unit.spiketime_run + time_offset[i]
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
    
    # delete those new generated timestamps that targeted low speed state and position
    spiketime_bin_id = (shuffled_spiketime*unit.tags['sync_rate']).astype('uint')
    spike_spd_id = ses.spd_mask[spiketime_bin_id]
    high_spd_id = np.where(spike_spd_id==True)[0]

    shuffled_spiketime = shuffled_spiketime[high_spd_id]
    
    shuffled_mean_rate = np.size(shuffled_spiketime) / (np.size(np.unique(spiketime_bin_id))/ses.sync_rate)
    
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
    shuffled_spatial_info = np.nansum((ses.dwell_map_spdmasked/np.sum(ses.dwell_map_spdmasked)) * (shuffled_rate_map/unit.mean_rate_run) * np.log2((shuffled_rate_map/unit.mean_rate_run)))
    shuffled_spatial_info = round(shuffled_spatial_info, 4)
    np.seterr(all='warn')
    ###########################################
    unit.spatial_info_pool[i] = shuffled_spatial_info






















