# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:17:47 2024

@author: NCP-LAB
"""

#%% initialize data
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

spike_pack = apply_speed_mask(spike_clusters, timestamps, spiketime, ses, tags, temporal_bin_length=1/tags['sync_rate'])    # if non-spatial-task spike_pack = (spike_clusters, timestamps, spiketime)

#%% initialize units

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
        
        # bayesian decoding use
        bayes_temporal_bin = 0.02
        unit.expect_spike_map = np.nan_to_num(unit.rate_map * bayes_temporal_bin, nan=0.0)
        ########################
        
        units_set.append(unit)
        unit_id += 1
    print(f'\r {i+1}/{clusters_info.shape[0]}',end='')


#%% class NeuralState
class NeuralState(object):
    def __init__(self, units_set, start_time, temporal_bin = 0.02):
        
        self.start_time = start_time
        self.temporal_bin = temporal_bin
        
        spike_count = np.zeros(len(units_set)).astype(int)
        for ineuron in range(len(units_set)):  # count how many spikes in this temporal bin
            spike_count[ineuron] = len( [i for i in units_set[ineuron].spiketime_run if start_time <= i < start_time+temporal_bin] )
        
        self.spike_count = spike_count
#%% bayesian decoding
bayes_temporal_bin = 0.02
start_time = 0          
curr_neural_state = NeuralState(units_set, start_time, bayes_temporal_bin)
occupancy = ses.dwell_map_spdmasked / ses.dwell_map_spdmasked.sum()   # P(c=C)

''' 
# slow but easy-understanding method
posterior_P_map = np.zeros([29,29])
for ibin in range(29):
    for jbin in range(29):
        posterior_P = occupancy[ibin,jbin]   # P(c=C) (not divide with full time)
        for ineuron in range(len(units_set)):
            P_nNcC = stats.poisson.pmf(curr_neural_state.spike_count[ineuron], units_set[ineuron].expect_spike_map[ibin,jbin] )
            posterior_P *= P_nNcC    # 累乘 cumulative product P(ni=Ni|c=C), assume as poisson distribution, calculate the probablity for occuring these spikes
        posterior_P_map[ibin,jbin] = posterior_P
'''
# fast method

occupancy_ex = occupancy[np.newaxis,:,:]     #P(c=C)                                           #shape 1*29*29

expect_spike_map = np.zeros([len(units_set),29,29])
for ineuron in range(len(units_set)):
    expect_spike_map[ineuron] = units_set[ineuron].expect_spike_map# λ in poisson ditribution  #shape ineuron*

curr_neural_state_spike_count_ex = curr_neural_state.spike_count[:, np.newaxis, np.newaxis]    #shape ineuron*1*1
P_nNcC = stats.poisson.pmf(curr_neural_state_spike_count_ex, expect_spike_map )                #shape ineuron*29*29

concatenated_P = np.concatenate((occupancy_ex, P_nNcC), axis=0)         # P(c=C)П P(n=N,c=C)   #shape (ineuron+1)*29*29
cumproded_P = np.cumprod(concatenated_P, axis=0)[-1,:,:]    # cumulative product all those P

# argmax method to find the most possible bin
decoded_bin = np.full(2, np.nan)
decoded_bin_flat = np.argmax(cumproded_P)
decoded_bin[0], decoded_bin[1] = np.unravel_index(decoded_bin_flat, cumproded_P.shape)
# weighted average method to calculate the continuous position from decoded bins
decoded_pos = np.full(2, np.nan)
decoded_pos[0] = ( cumproded_P.sum(axis = 1) * np.arange(29) ).sum() / cumproded_P.sum()          # simply did the weighted average of all bins
decoded_pos[1] = ( cumproded_P.sum(axis = 0) * np.arange(29) ).sum() / cumproded_P.sum()
decoded_pos = decoded_pos*2 -1.5   # bin to cm 

'''
!!!!!!!
!!!!!!!!!!
在poisson分布中，如果期望是0，那么一定发生0次，这种“确定性”会带来问题： 
训练集中：如果一个神经元在这个bin里面从来没有发放过。  
那么在测试集中，如果此时的population vector中这个神经元有任何发放，那么该bin的可能性直接变成0，emmm这很不合适    
'''    
#%% Bayesian Decoding Function
def bayes_decode(units_set, ses, start_time, bayes_temporal_bin):
    # initialize Neural State
    curr_neural_state = NeuralState(units_set, start_time, bayes_temporal_bin)
    # calculate P(c=C)
    occupancy = ses.dwell_map_spdmasked / ses.dwell_map_spdmasked.sum()   # P(c=C)
    occupancy_ex = occupancy[np.newaxis,:,:]     #P(c=C)                                           #shape 1*29*29
    
    # generate expect spike poisson distribution map
    expect_spike_map = np.zeros([len(units_set),29,29])
    for ineuron in range(len(units_set)):
        expect_spike_map[ineuron] = units_set[ineuron].expect_spike_map# λ in poisson ditribution  #shape ineuron*
    
    # calculate the Probablity under Poisson distribution
    curr_neural_state_spike_count_ex = curr_neural_state.spike_count[:, np.newaxis, np.newaxis]    #shape ineuron*1*1
    P_nNcC = stats.poisson.pmf(curr_neural_state_spike_count_ex, expect_spike_map )                #shape ineuron*29*29
    
    # cumulative product
    concatenated_P = np.concatenate((occupancy_ex, P_nNcC), axis=0)         # P(c=C)П P(n=N,c=C)   #shape (ineuron+1)*29*29
    cumproded_P = np.cumprod(concatenated_P, axis=0)[-1,:,:]    # cumulative product all those P
    cumproded_P_normal = cumproded_P / np.max(cumproded_P)
    
    # argmax method to find the most possible bin
    # decoded_bin = np.full(2, np.nan)
    # decoded_bin_flat = np.argmax(cumproded_P)
    # decoded_bin[0], decoded_bin[1] = np.unravel_index(decoded_bin_flat, cumproded_P.shape)
    
    
    # weighted average method to calculate the continuous position from decoded bins
    decoded_pos = np.full(2, np.nan)
    decoded_pos[1] = ( cumproded_P.sum(axis = 1) * np.arange(29) ).sum() / cumproded_P.sum()          # simply did the weighted average of all bins
    decoded_pos[0] = ( cumproded_P.sum(axis = 0) * np.arange(29) ).sum() / cumproded_P.sum()
    decoded_pos = decoded_pos*2 -1.5   # bin to cm 
    
    return decoded_pos, cumproded_P_normal

#%% main
start_moment = 0     # temporal bins , in 20ms bin AKA frame
end_moment = 33439   # just like python index, this moment is not included
decoded_pos = []
decoded_distribution = []
status = 0
for moment in range(start_moment, end_moment):   
   pos, distribution = bayes_decode(units_set, ses, moment/50, bayes_temporal_bin) 
   decoded_pos.append( pos )
   decoded_distribution.append( distribution )
   status += 1
   print(f'\r {status}/{end_moment-start_moment}',end='')
decoded_pos = np.array(decoded_pos)

#%% prepare cm_to_pixel_PlusMaze(dlc_files, nodes_set, fdir, fn)

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

#%% cm_to_pixel_PlusMaze
# generate a convert matrix based on the precise pixel and real world length #######################
input_matrix  = np.ones((len(nodes_set['name']),3))
output_matrix = np.ones((len(nodes_set['name']),3))
for i in range(len(nodes_set['name'])):        
    input_matrix[i][0] = nodes_set['real_position'][i][1]
    input_matrix[i][1] = nodes_set['real_position'][i][0]
    output_matrix[i][0]  = nodes_set['precise_position'][i][1]
    output_matrix[i][1]  = nodes_set['precise_position'][i][0]
# 使用最小二乘法求解线性映射矩阵 A
convert_matrix, _, _, _ = np.linalg.lstsq(input_matrix, output_matrix, rcond=None)

# convert the decoded position
decoded_pos_ex = np.ones((decoded_pos.shape[0],3))
decoded_pos_ex[:,0:2] = decoded_pos
decoded_pos = np.dot(decoded_pos_ex, convert_matrix)         # 点乘转换矩阵
decoded_pos = np.round(decoded_pos[:,0:2]).astype(int)

#%% Visualization

video_path = str(fdir/fn/(fn+'.avi'))
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
height, width, _ = frame.shape

if not (fdir/fn/'video').exists():
    (fdir/fn/'video').mkdir(parents=True, exist_ok=True)
output_path = str(fdir/fn/'video'/(fn+'_Bayes_Deco.avi'))

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 编解码器
out = cv2.VideoWriter(output_path, fourcc, 50, (width, height))

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

frame_count = 0
while True: #
    ret, frame = cap.read()
    if not ret:    #检查是否读到帧
        break
    
    if start_moment <= frame_count < end_moment:
        cv2.circle(frame, (decoded_pos[frame_count-start_moment, 0], decoded_pos[frame_count-start_moment, 1]), 3,(0,255,255),-1)
        cv2.imshow('frame', frame)
        out.write(frame)
    elif end_moment <= frame_count:
        break
    
    frame_count += 1
    
    # need a buffer time to render the frame, or will it claculate next frame, leading to look like gray screen
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()

#%% Visualization 2

height = 580
width = 580

if not (fdir/fn/'video').exists():
    (fdir/fn/'video').mkdir(parents=True, exist_ok=True)
output_path = str(fdir/fn/'video'/(fn+'_Bayes_Deco_2.avi'))

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 编解码器
out = cv2.VideoWriter(output_path, fourcc, 50, (width, height))

cv2.namedWindow('Preview', cv2.WINDOW_NORMAL)


for imoment in range(start_moment, end_moment):
           
    bayes_cloud = decoded_distribution[imoment-start_moment]
    # bayes_mask[bayes_mask<0.5] = 0
    zoom_factor=20
    bayes_cloud_resized = ndimage.zoom(bayes_cloud, zoom_factor, order=1)
    bayes_cloud_resized = (bayes_cloud_resized*255).astype(np.uint8)
    frame = np.stack((bayes_cloud_resized, bayes_cloud_resized, bayes_cloud_resized), axis=2)
    
    
    curr_mouse_pos = ((ses.mouse_pos[imoment]+1.5)*10).astype(int)    
    curr_body1_pos = ((np.array(dlc_files.iloc[imoment,12:14])+1.5)*10).astype(int)
    curr_body2_pos = ((np.array(dlc_files.iloc[imoment,15:17])+1.5)*10).astype(int)
    curr_body3_pos = ((np.array(dlc_files.iloc[imoment,18:])+1.5)*10).astype(int)
    # plot body line    
    cv2.line(frame, (curr_mouse_pos[0], curr_mouse_pos[1]), (curr_body1_pos[0],curr_body1_pos[1]), (255,50,0), thickness=16, lineType=cv2.LINE_AA, shift=None)
    cv2.line(frame, (curr_body1_pos[0],curr_body1_pos[1]), (curr_body2_pos[0],curr_body2_pos[1]), (255,50,0), thickness=20, lineType=cv2.LINE_AA, shift=None)
    cv2.line(frame, (curr_body2_pos[0],curr_body2_pos[1]), (curr_body3_pos[0],curr_body3_pos[1]), (255,50,0), thickness=10, lineType=cv2.LINE_AA, shift=None)
    # plot head
    cv2.circle(frame, (curr_mouse_pos[0], curr_mouse_pos[1]), 8, (255,150,150),-1)
    
    
    cv2.imshow('Preview', frame)
    out.write(frame)    
    
    # need a buffer time to render the frame, or will it claculate next frame, leading to look like gray screen
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
out.release()
cv2.destroyAllWindows()

#%% Function Visualization
# example: bayes_cloud_video(5, 30, units_set, ses, frame_rate=5, bayes_temporal_bin=0.02)
def bayes_cloud_video(start_time, end_time, units_set, ses, frame_rate=50, bayes_temporal_bin=0.02):
    decoded_pos = []
    decoded_distribution = []
    # convert real world time to NeuralState temporal bin
    start_moment = int(start_time/bayes_temporal_bin)
    end_moment = int(end_time/bayes_temporal_bin)
    
    status = 0
    for moment in range(start_moment, end_moment):   
       pos, distribution = bayes_decode(units_set, ses, moment/50, bayes_temporal_bin) 
       decoded_pos.append( pos )
       decoded_distribution.append( distribution )
       status += 1
       print(f'\r {status}/{end_moment-start_moment}',end='')
    decoded_pos = np.array(decoded_pos)
    
    height = 580
    width = 580
    
    if not (fdir/fn/'video').exists():
        (fdir/fn/'video').mkdir(parents=True, exist_ok=True)
    output_path = str(fdir/fn/'video'/('Bayes_cloud_'+str(start_moment)+'-'+str(end_moment)+'.avi'))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 编解码器
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    cv2.namedWindow('Preview', cv2.WINDOW_NORMAL)


    for imoment in range(start_moment, end_moment):
               
        bayes_cloud = decoded_distribution[imoment-start_moment]
        # bayes_mask[bayes_mask<0.5] = 0
        zoom_factor=20
        bayes_cloud_resized = ndimage.zoom(bayes_cloud, zoom_factor, order=1)
        bayes_cloud_resized = (bayes_cloud_resized*255).astype(np.uint8)
        frame = np.stack((bayes_cloud_resized, bayes_cloud_resized, bayes_cloud_resized), axis=2)
        
        
        curr_mouse_pos = ((ses.mouse_pos[imoment]+1.5)*10).astype(int)    
        curr_body1_pos = ((np.array(dlc_files.iloc[imoment,12:14])+1.5)*10).astype(int)
        curr_body2_pos = ((np.array(dlc_files.iloc[imoment,15:17])+1.5)*10).astype(int)
        curr_body3_pos = ((np.array(dlc_files.iloc[imoment,18:])+1.5)*10).astype(int)
        # plot body line    
        cv2.line(frame, (curr_mouse_pos[0], curr_mouse_pos[1]), (curr_body1_pos[0],curr_body1_pos[1]), (255,50,0), thickness=16, lineType=cv2.LINE_AA, shift=None)
        cv2.line(frame, (curr_body1_pos[0],curr_body1_pos[1]), (curr_body2_pos[0],curr_body2_pos[1]), (255,50,0), thickness=20, lineType=cv2.LINE_AA, shift=None)
        cv2.line(frame, (curr_body2_pos[0],curr_body2_pos[1]), (curr_body3_pos[0],curr_body3_pos[1]), (255,50,0), thickness=10, lineType=cv2.LINE_AA, shift=None)
        # plot head
        cv2.circle(frame, (curr_mouse_pos[0], curr_mouse_pos[1]), 8, (255,150,150),-1)
        
        
        cv2.imshow('Preview', frame)
        out.write(frame)    
        
        # need a buffer time to render the frame, or will it claculate next frame, leading to look like gray screen
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    out.release()
    cv2.destroyAllWindows()
    
#%% theta cycle decoding
'''
below analyse is on Right lateral signal, check before appling to Left
'''
lfp = LFP(fdir, fn, esync_timestamps, tags)
lfp.SWR_detect()
lfp.theta_cycle_detect()

select_time_range = [14.5,21.5]        # real world time
theta_cycle = lfp.theta_peak_trough_peak_R[:,(0,2)]
selected_theta_cycle = theta_cycle[(theta_cycle[:, 0] > select_time_range[0]) & (theta_cycle[:, 1] < select_time_range[1])]

for i in selected_theta_cycle:
    bayes_cloud_video(i[0], i[1], units_set, ses, frame_rate=5, bayes_temporal_bin=0.02)
    
    theta_cycle_index = np.where(theta_cycle[:,0]==i[0])
    p_t_p = lfp.t_p_t_p_R[theta_cycle_index]
    # plt.plot(lfp.time[p_t_p[0,0]:p_t_p[0,2]], lfp.trace_all_ch_R[lfp.theta_channel_R,:][p_t_p[0,0]:p_t_p[0,2]], color='k')  # when appling to Left, lfp.trace_all_ch_L[lfp.theta_channel_L-32,:]
    plt.plot(lfp.time[p_t_p[0,0]:p_t_p[0,2]],lfp.gamma_trace_R[p_t_p[0,0]:p_t_p[0,2]], color='g')
    plt.plot(lfp.time[p_t_p[0,0]:p_t_p[0,2]],lfp.theta_trace_R[p_t_p[0,0]:p_t_p[0,2]], color='b')
    
    start_moment = int(i[0]/bayes_temporal_bin)
    end_moment = int(i[1]/bayes_temporal_bin)
    plt.savefig(  str(fdir/fn/'video'/('Bayes_cloud_'+str(start_moment)+'-'+str(end_moment)+'.png'))  )
    plt.savefig(  str(fdir/fn/'video'/('Bayes_cloud_'+str(start_moment)+'-'+str(end_moment)+'.svg'))  )
    plt.close()
    
#%% SWR decoding
lfp = LFP(fdir, fn, esync_timestamps, tags)
lfp.SWR_detect()

for i in lfp.ripple_start_end_R:
    bayes_cloud_video(i[0], i[1], units_set, ses, frame_rate=5, bayes_temporal_bin=0.02)

#%%
bayes_cloud_video(1018/50, 1026/50, units_set, ses, frame_rate=5, bayes_temporal_bin=0.02)


a = lfp.time
aa = lfp.theta_trace_R[199:455]
aaa = lfp.trace_all_ch_R
















