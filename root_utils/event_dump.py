"""
Python 3 script for processing a list of ROOT files into .npz files

To keep references to the original ROOT files, the file path is stored in the output.
An index is saved for every event in the output npz file corresponding to the event index within that ROOT file (ev).

Authors: Nick Prouse, modifications by Felix Cormier
"""

import argparse
from urllib.parse import non_hierarchical
from DataTools.root_utils.root_file_utils import *
from DataTools.root_utils.pos_utils import *
import h5py
import matplotlib.pyplot as plt
import math

ROOT.gROOT.SetBatch(True)


def get_args():
    parser = argparse.ArgumentParser(description='dump WCSim data into numpy .npz file')
    parser.add_argument('input_files', type=str, nargs='+')
    parser.add_argument('-d', '--output_dir', type=str, default=None)
    args = parser.parse_args()
    return args

def clean_up_coord(array, uniques, col):
    for i, value in enumerate(uniques): 
        if i >= len(uniques)-1:
            continue
        if abs(value-uniques[i+1]) < 0.00001:
            temp = array[:,col]
            temp[temp==value] = uniques[i+1]
            array[:,col] = temp
            uniques = np.delete(uniques, uniques==value)
    return array, uniques


def project_barrel(array,col):
    array = array[np.argsort(array[:,col])]
    unique = np.flip(np.unique(array[:,2]))
    array, unique = clean_up_coord(array,unique,col)
    new_array = []
    for z_value in unique:
        temp_array = array[array[:,2]==z_value]
        temp_array_pos = temp_array[temp_array[:,1] > 0]
        temp_array_pos = temp_array_pos[np.argsort(temp_array_pos[:,0])]
        temp_array_neg = temp_array[temp_array[:,1] < 0]
        temp_array_neg = temp_array_neg[np.flip(np.argsort(temp_array_neg[:,0]))]
        temp_array = np.concatenate((temp_array_neg, temp_array_pos))
        new_array.append(temp_array.tolist())
    return np.array(new_array)

def project_endcap(array, flip_x):
    array = array[np.argsort(array[:,0])]
    unique = np.unique(array[:,0])
    if flip_x:
        unique = np.flip(unique)
    new_array = []
    for x_value in unique:
        temp_array = array[array[:,0]==x_value]
        temp_array = temp_array[np.argsort(temp_array[:,1])]
        new_array.append(temp_array)
    return np.array(new_array, dtype=object)
    #radius = np.sqrt((np.add(np.square(array[:,0]),np.square(array[:,1]))))
    #unique, counts = np.unique(radius, return_counts=True)

def pad_endcap(array, length):
    channels = array[0].shape[1]
    new_array = []
    for line in array:
        current_length=line.shape[0]
        pad_length = length-current_length
        half_length = pad_length/2
        left = math.floor(half_length)
        right = math.ceil(half_length)
        #print(f'current_length: {current_length}, pad_length: {pad_length}, half_length: {half_length}, left: {left}, right: {right}')
        line = np.pad(line, ((left,right),(0,0)), 'constant', constant_values=-1)
        new_array.append(line)
    return np.array(new_array)

    


def image_file(geo_dict):

    data = np.array(list(geo_dict.values()))[:,0] 
    pmts = np.array(list(geo_dict.keys()))
    data = np.column_stack((data,pmts))

    endcap_min = data[data[:,2]==np.amin(data[:,2])]
    endcap_max = data[data[:,2]==np.amax(data[:,2])]
    barrel = data[ (data[:,2]!=np.amax(data[:,2])) & (data[:,2]!=np.amin(data[:,2]))]
    barrel = project_barrel(barrel,2)
    endcap_min = project_endcap(endcap_min, flip_x=False)
    endcap_max = project_endcap(endcap_max, flip_x=True)

    endcap_min = pad_endcap(endcap_min, barrel.shape[1])
    endcap_max = pad_endcap(endcap_max, barrel.shape[1])

    final_array = np.concatenate((endcap_max, barrel, endcap_min))
    plt.imshow(final_array[:,:,0])
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('X value', rotation=90)
    plt.savefig('output/final_array_x_projection.png', format='png')
    plt.close()
    plt.clf()
    plt.imshow(final_array[:,:,1])
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Y value', rotation=90)
    plt.savefig('output/final_array_y_projection.png', format='png')
    plt.close()
    plt.clf()
    plt.imshow(final_array[:,:,2])
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Z value', rotation=90)
    plt.savefig('output/final_array_z_projection.png', format='png')
    plt.close()
    plt.clf()
    plt.imshow(final_array[:,:,3])
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Index', rotation=90)
    plt.savefig('output/final_array_index_projection.png', format='png')
    plt.close()
    plt.clf()
    final_array = final_array[:,:,3]
    print(final_array.shape)
    index_array = []

    for i in range(final_array.shape[0]):
        for j in range(final_array.shape[1]):
            index_array.append([int(i),int(j),int(final_array[i,j])])
    
    index_array = np.array(index_array)[ np.argsort(np.array(index_array)[:,2])]
    index_array = np.array(index_array)[np.array(index_array)[:,2] != -1]
    index_array = index_array[:,[0,1]].astype(int)

    print(index_array.dtype)
    np.save('output/sk_wcsim_imagefile.npy', index_array)

    return 0

def geo_file(geo_dict):
    positions = np.empty((len(geo_dict),3))
    orientations = np.empty((len(geo_dict),3))
    for i in range(len(geo_dict)):
        positions[i] = geo_dict[i][0]
        orientations[i] = geo_dict[i][1]

    print(f'Positions: {positions}')
    print(f'Orientations: {orientations}')

    np.savez('data/geofile',position=positions, orientation=orientations)

    return 0

def dump_file_skdetsim(infile, outfile, save_npz=False, radius = 1690, half_height = 1810, create_image_file=False, create_geo_file=False):

    # All data arrays are initialized here
    skdetsim = SKDETSIM(infile)
    nevents = skdetsim.nevent

    skgeofile = np.load('data/geofile_skdetsim.npz')

    event_id = np.empty(nevents, dtype=np.int32)
    root_file = np.empty(nevents, dtype=object)

    pid = np.empty(nevents, dtype=np.int32)
    position = np.empty((nevents, 3), dtype=np.float64)
    gamma_start_vtx = np.empty((nevents, 3), dtype=np.float64)
    isConversion = np.empty(nevents, dtype=np.float64)
    direction = np.empty((nevents, 3), dtype=np.float64)
    energy = np.empty(nevents,dtype=np.float64)
    electron_direction = np.empty((nevents, 3), dtype=np.float64)
    electron_energy = np.empty(nevents,dtype=np.float64)
    positron_direction = np.empty((nevents, 3), dtype=np.float64)
    positron_energy = np.empty(nevents,dtype=np.float64)

    digi_hit_pmt = np.empty(nevents, dtype=object)
    digi_hit_pmt_pos = np.empty(nevents, dtype=object)
    digi_hit_pmt_or = np.empty(nevents, dtype=object)
    digi_hit_charge = np.empty(nevents, dtype=object)
    digi_hit_time = np.empty(nevents, dtype=object)
    digi_hit_trigger = np.empty(nevents, dtype=object)

    track_id = np.empty(nevents, dtype=object)
    track_pid = np.empty(nevents, dtype=object)
    track_start_time = np.empty(nevents, dtype=object)
    track_energy = np.empty(nevents, dtype=object)
    track_start_position = np.empty(nevents, dtype=object)
    track_stop_position = np.empty(nevents, dtype=object)
    track_parent = np.empty(nevents, dtype=object)
    track_flag = np.empty(nevents, dtype=object)

    trigger_time = np.empty(nevents, dtype=object)
    trigger_type = np.empty(nevents, dtype=object)

    for ev in range(skdetsim.nevent):
        skdetsim.get_event(ev)

        event_info = skdetsim.get_event_info()
        pid[ev] = event_info["pid"]
        if event_info["isConversion"]:
            gamma_start_vtx[ev] = event_info["gamma_start_vtx"]
            electron_direction[ev] = event_info["direction_electron"]
            electron_energy[ev] = event_info["energy_electron"]
            positron_direction[ev] = event_info["direction_positron"]
            positron_energy[ev] = event_info["energy_positron"]
        else:
            gamma_start_vtx[ev]= np.array([-99999,-99999,-99999])
            electron_direction[ev] = np.array([-99999,-99999,-99999])
            electron_energy[ev] = -999
            positron_direction[ev] = np.array([-99999,-99999,-99999])
            positron_energy[ev] = -999
        isConversion[ev] = event_info["isConversion"]
        position[ev] = event_info["position"]
        direction[ev] = event_info["direction"]
        energy[ev] = event_info["energy"]

        track_pid[ev] = event_info["pid"]
        track_energy[ev] = event_info["energy"]
        track_start_position[ev] = event_info["position"]
        track_stop_position[ev] = event_info["position"]


        digi_hits = skdetsim.get_digitized_hits()
        digi_hit_pmt[ev] = digi_hits["pmt"]
        '''
        for i,pmt_no in enumerate(digi_hit_pmt[ev]):
            if i==0:
                print(f"position: {skgeofile['position'][pmt_no]}, orientation: {skgeofile['orientation'][pmt_no]}")
                digi_hit_pmt_pos[ev] = [(skgeofile['position'][pmt_no])]
                digi_hit_pmt_or[ev] = [(skgeofile['orientation'][pmt_no])]
            else:
                digi_hit_pmt_pos[ev].append((skgeofile['position'][pmt_no]))
                digi_hit_pmt_or[ev].append((skgeofile['orientation'][pmt_no]))
        '''
        digi_hit_charge[ev] = digi_hits["charge"]
        digi_hit_time[ev] = digi_hits["time"]
        digi_hit_trigger[ev] = digi_hits["trigger"]

        #Add fake triggers since we don't get that info in SKDETSIM (yet?)
        trigger_time[ev] = np.asarray([0],dtype=np.float32)
        trigger_type[ev] = np.asarray([0],dtype=np.int32)

        event_id[ev] = ev
        root_file[ev] = infile

    dump_digi_hits(outfile, root_file, radius, half_height, event_id, pid, position, isConversion, gamma_start_vtx, direction, energy, electron_energy, electron_direction, positron_energy, positron_direction, digi_hit_pmt, digi_hit_pmt_pos, digi_hit_pmt_or, digi_hit_charge, digi_hit_time, digi_hit_trigger, track_pid, track_energy, track_start_position, track_stop_position, trigger_time, trigger_type, save_tracks=False)

    del skdetsim


def dump_file(infile, outfile, save_npz=False, radius = 1690, half_height = 1810, create_image_file=False, create_geo_file=False):
    """Takes in root file, outputs h5py file

    Args:
        infile (_type_): Path and name of input root file
        outfile (_type_): Path and name of output h5py file
        save_npz (bool, optional): Whether to save npz. Defaults to False.
        radius (int, optional): General radius of detector, for veto. Defaults to 20.
        half_height (int, optional): General half height of detector, for veto. Defaults to 20.
        create_image_file (bool, optional): Script to create a 2D image file from geo file. For ML ResNet. Defaults to False.
    """

    wcsim = WCSimFile(infile)
    nevents = wcsim.nevent

    geo = wcsim.geo

    geo_num_pmts = geo.GetWCNumPMT()

    geo_dict = {}

    #Make conversion between PMT number and position in Water Cherenkov Detector
    for i in range(geo_num_pmts):
        pmt = geo.GetPMT(i)
        #Seems to be off by 1? Should cross-check
        #Apparently SK starts at 1, so this works
        tube_no = pmt.GetTubeNo()-1
        geo_dict[tube_no] = [[0,0,0],[0,0,0]]
        for j in range(3):
            geo_dict[tube_no][0][j] = pmt.GetPosition(j)
            geo_dict[tube_no][1][j] = pmt.GetOrientation(j)

    if create_image_file:
        return image_file(geo_dict)

    if create_geo_file:
        return geo_file(geo_dict)

    # All data arrays are initialized here

    event_id = np.empty(nevents, dtype=np.int32)
    root_file = np.empty(nevents, dtype=object)

    pid = np.empty(nevents, dtype=np.int32)
    position = np.empty((nevents, 3), dtype=np.float64)
    gamma_start_vtx = np.empty((nevents, 3), dtype=np.float64)
    isConversion = np.empty(nevents, dtype=np.float64)
    direction = np.empty((nevents, 3), dtype=np.float64)
    energy = np.empty(nevents,dtype=np.float64)
    electron_direction = np.empty((nevents, 3), dtype=np.float64)
    electron_energy = np.empty(nevents,dtype=np.float64)
    positron_direction = np.empty((nevents, 3), dtype=np.float64)
    positron_energy = np.empty(nevents,dtype=np.float64)

    digi_hit_pmt = np.empty(nevents, dtype=object)
    digi_hit_pmt_pos = np.empty(nevents, dtype=object)
    digi_hit_pmt_or = np.empty(nevents, dtype=object)
    digi_hit_charge = np.empty(nevents, dtype=object)
    digi_hit_time = np.empty(nevents, dtype=object)
    digi_hit_trigger = np.empty(nevents, dtype=object)

    true_hit_pmt = np.empty(nevents, dtype=object)
    true_hit_pmt_pos = np.empty(nevents, dtype=object)
    true_hit_pmt_or = np.empty(nevents, dtype=object)
    true_hit_time = np.empty(nevents, dtype=object)
    true_hit_pos = np.empty(nevents, dtype=object)
    true_hit_start_time = np.empty(nevents, dtype=object)
    true_hit_start_pos = np.empty(nevents, dtype=object)
    true_hit_parent = np.empty(nevents, dtype=object)

    track_id = np.empty(nevents, dtype=object)
    track_pid = np.empty(nevents, dtype=object)
    track_start_time = np.empty(nevents, dtype=object)
    track_energy = np.empty(nevents, dtype=object)
    track_start_position = np.empty(nevents, dtype=object)
    track_stop_position = np.empty(nevents, dtype=object)
    track_parent = np.empty(nevents, dtype=object)
    track_flag = np.empty(nevents, dtype=object)

    trigger_time = np.empty(nevents, dtype=object)
    trigger_type = np.empty(nevents, dtype=object)

    part_array = []

    for ev in range(wcsim.nevent):
        wcsim.get_event(ev)

        event_info = wcsim.get_event_info()
        pid[ev] = event_info["pid"]
        if event_info["isConversion"]:
            gamma_start_vtx[ev] = event_info["gamma_start_vtx"]
            electron_direction[ev] = event_info["direction_electron"]
            electron_energy[ev] = event_info["energy_electron"]
            positron_direction[ev] = event_info["direction_positron"]
            positron_energy[ev] = event_info["energy_positron"]
        else:
            gamma_start_vtx[ev]= np.array([-99999,-99999,-99999])
            electron_direction[ev] = np.array([-99999,-99999,-99999])
            electron_energy[ev] = -999
            positron_direction[ev] = np.array([-99999,-99999,-99999])
            positron_energy[ev] = -999
        isConversion[ev] = event_info["isConversion"]
        position[ev] = event_info["position"]
        direction[ev] = event_info["direction"]
        energy[ev] = event_info["energy"]

        true_hits = wcsim.get_hit_photons()
        true_hit_pmt[ev] = true_hits["pmt"]
        for i,pmt_no in enumerate(true_hit_pmt[ev]):
            if i==0:
                true_hit_pmt_pos[ev] = [(geo_dict[pmt_no][0])]
                true_hit_pmt_or[ev] = [(geo_dict[pmt_no][1])]
            else:
                true_hit_pmt_pos[ev].append((geo_dict[pmt_no][0]))
                true_hit_pmt_or[ev].append((geo_dict[pmt_no][1]))
        true_hit_time[ev] = true_hits["end_time"]
        true_hit_pos[ev] = true_hits["end_position"]
        true_hit_start_time[ev] = true_hits["start_time"]
        true_hit_start_pos[ev] = true_hits["start_position"]
        true_hit_parent[ev] = true_hits["track"]

        digi_hits = wcsim.get_digitized_hits()
        digi_hit_pmt[ev] = digi_hits["pmt"]
        for i,pmt_no in enumerate(digi_hit_pmt[ev]):
            if i==0:
                digi_hit_pmt_pos[ev] = [(geo_dict[pmt_no][0])]
                digi_hit_pmt_or[ev] = [(geo_dict[pmt_no][1])]
            else:
                digi_hit_pmt_pos[ev].append((geo_dict[pmt_no][0]))
                digi_hit_pmt_or[ev].append((geo_dict[pmt_no][1]))
        digi_hit_charge[ev] = digi_hits["charge"]
        digi_hit_time[ev] = digi_hits["time"]
        digi_hit_trigger[ev] = digi_hits["trigger"]

        tracks = wcsim.get_tracks()
        track_id[ev] = tracks["id"]
        track_pid[ev] = tracks["pid"]
        track_start_time[ev] = tracks["start_time"]
        track_energy[ev] = tracks["energy"]
        track_start_position[ev] = tracks["start_position"]
        track_stop_position[ev] = tracks["stop_position"]
        track_parent[ev] = tracks["parent"]
        track_flag[ev] = tracks["flag"]

        triggers = wcsim.get_triggers()
        trigger_time[ev] = triggers["time"]
        trigger_type[ev] = triggers["type"]

        event_id[ev] = ev
        root_file[ev] = infile
    

    dump_digi_hits(outfile, root_file, radius, half_height, event_id, pid, position, isConversion, gamma_start_vtx, direction, energy, electron_energy, electron_direction, positron_energy, positron_direction, digi_hit_pmt, digi_hit_pmt_pos, digi_hit_pmt_or, digi_hit_charge, digi_hit_time, digi_hit_trigger, track_pid, track_energy, track_start_position, track_stop_position, trigger_time, trigger_type)
    dump_true_hits(outfile, root_file, radius, half_height, event_id, pid, position, isConversion, gamma_start_vtx, direction, energy, true_hit_pmt, true_hit_pmt_pos, true_hit_pmt_or, digi_hit_trigger, track_pid, track_energy, track_start_position, track_stop_position, true_hit_time, true_hit_parent)



    if (save_npz):
        np.savez_compressed(outfile+'.npz',
                            event_id=event_id,
                            root_file=root_file,
                            pid=pid,
                            position=position,
                            gamma_start_vtx=gamma_start_vtx,
                            direction=direction,
                            energy=energy,
                            digi_hit_pmt=digi_hit_pmt,
                            digi_hit_charge=digi_hit_charge,
                            digi_hit_time=digi_hit_time,
                            digi_hit_trigger=digi_hit_trigger,
                            true_hit_pmt=true_hit_pmt,
                            true_hit_time=true_hit_time,
                            true_hit_pos=true_hit_pos,
                            true_hit_start_time=true_hit_start_time,
                            true_hit_start_pos=true_hit_start_pos,
                            true_hit_parent=true_hit_parent,
                            track_id=track_id,
                            track_pid=track_pid,
                            track_start_time=track_start_time,
                            track_energy=track_energy,
                            track_start_position=track_start_position,
                            track_stop_position=track_stop_position,
                            track_parent=track_parent,
                            track_flag=track_flag,
                            trigger_time=trigger_time,
                            trigger_type=trigger_type
                            )
    del wcsim

def dump_digi_hits(outfile, infile, radius, half_height, event_id, pid, position, isConversion, gamma_start_vtx, direction, energy, electron_energy, electron_direction, positron_energy, positron_direction, digi_hit_pmt, digi_hit_pmt_pos, digi_hit_pmt_or, digi_hit_charge, digi_hit_time, digi_hit_trigger, track_pid, track_energy, track_start_position, track_stop_position, trigger_time, trigger_type, save_tracks=True):
    """Save the digi hits, event variables

    Args:
        Inputs
    """
    f = h5py.File(outfile+'_digi.hy', 'w')
    print("IN DUMP DIGI HITS")

    hit_triggers = digi_hit_trigger
    total_rows = hit_triggers.shape[0]
    event_triggers = np.full(hit_triggers.shape[0], np.nan)
    min_hits=1
    offset = 0
    offset_next = 0
    hit_offset = 0
    hit_offset_next = 0
    total_hits=0
    good_hits=0
    good_rows=0

    for i, (times, types, hit_trigs) in enumerate(zip(trigger_time, trigger_type, hit_triggers)):
        print(f"i: {i}, times: {times}, types: {types}, hit_trigs: {hit_trigs}")
        good_triggers = np.where(types == 0)[0]
        print(good_triggers)
        if len(good_triggers) == 0:
            continue
        first_trigger = good_triggers[np.argmin(times[good_triggers])]
        nhits = np.count_nonzero(hit_trigs == first_trigger)
        total_hits += nhits
        print(f"first trig: {first_trigger}, good_trigger: {good_triggers}, nhits: {nhits}")
        if nhits >= min_hits:
            event_triggers[i] = first_trigger
            good_hits += nhits
            good_rows += 1
    file_event_triggers = event_triggers
    
    dset_labels = f.create_dataset("labels",
                                   shape=(total_rows,),
                                   dtype=np.int32)
    dset_IDX = f.create_dataset("event_ids",
                                shape=(total_rows,),
                                dtype=np.int32)
    dset_PATHS=f.create_dataset("root_files",
                                shape=(total_rows,),
                                dtype=h5py.special_dtype(vlen=str))
    dset_hit_time = f.create_dataset("hit_time",
                                     shape=(good_hits, ),
                                     dtype=np.float32)
    dset_hit_charge = f.create_dataset("hit_charge",
                                       shape=(good_hits, ),
                                       dtype=np.float32)
    dset_hit_pmt = f.create_dataset("hit_pmt",
                                    shape=(good_hits, ),
                                    dtype=np.int32)
    '''
    dset_hit_pmt_pos = f.create_dataset("hit_pmt_pos",
                                    shape=(good_hits, 3),
                                    dtype=np.float32)
    dset_hit_pmt_or = f.create_dataset("hit_pmt_or",
                                    shape=(good_hits, 3),
                                    dtype=np.float32)
    '''
    dset_event_hit_index = f.create_dataset("event_hits_index",
                                            shape=(total_rows,),
                                            dtype=np.int64)  # int32 is too small to fit large indices
    dset_energies = f.create_dataset("energies",
                                     shape=(total_rows, 1),
                                     dtype=np.float32)
    dset_electron_energies = f.create_dataset("energies_electron",
                                     shape=(total_rows, 1),
                                     dtype=np.float32)
    dset_positron_energies = f.create_dataset("energies_positron",
                                     shape=(total_rows, 1),
                                     dtype=np.float32)
    dset_positions = f.create_dataset("positions",
                                      shape=(total_rows, 1, 3),
                                      dtype=np.float32)
    dset_directions=f.create_dataset("directions",
                                    shape=(total_rows, 1, 3),
                                    dtype=np.float32)
    dset_electron_directions=f.create_dataset("directions_electron",
                                    shape=(total_rows, 1, 3),
                                    dtype=np.float32)
    dset_positron_directions=f.create_dataset("directions_positron",
                                    shape=(total_rows, 1, 3),
                                    dtype=np.float32)
    dset_angles = f.create_dataset("angles",
                                   shape=(total_rows, 2),
                                   dtype=np.float32)
    dset_veto = f.create_dataset("veto",
                                 shape=(total_rows,),
                                 dtype=np.bool_)
    dset_veto2 = f.create_dataset("veto2",
                                  shape=(total_rows,),
                                  dtype=np.bool_)
    dset_gamma_start_vtx = f.create_dataset("gamma_start_vtx",
                                    shape=(total_rows, 1, 3),
                                    dtype=np.float32)
                        

    good_events = ~np.isnan(file_event_triggers)

    offset_next = event_id.shape[0]

    dset_IDX[offset:offset_next] = event_id
    dset_PATHS[offset:offset_next] = infile
    dset_energies[offset:offset_next, :] = energy.reshape(-1, 1)
    dset_electron_energies[offset:offset_next, :] = electron_energy.reshape(-1, 1)
    dset_positron_energies[offset:offset_next, :] = positron_energy.reshape(-1, 1)
    dset_positions[offset:offset_next, :, :] = position.reshape(-1, 1, 3)
    dset_directions[offset:offset_next, :, :] = direction.reshape(-1, 1, 3)
    dset_electron_directions[offset:offset_next, :, :] = electron_direction.reshape(-1, 1, 3)
    dset_positron_directions[offset:offset_next, :, :] = positron_direction.reshape(-1, 1, 3)

    labels = np.full(pid.shape[0], -1)
    label_map = {13: 0, 11: 1, 22: 2}
    for k, v in label_map.items():
        labels[pid == k] = v
    dset_labels[offset:offset_next] = labels
    dset_gamma_start_vtx[offset:offset_next, :, :] = gamma_start_vtx.reshape(-1, 1, 3)


    polars = np.arccos(direction[:, 1])
    azimuths = np.arctan2(direction[:, 2], direction[:, 0])
    dset_angles[offset:offset_next, :] = np.hstack((polars.reshape(-1, 1), azimuths.reshape(-1, 1)))

    if save_tracks:
        for i, (pids, energies, starts, stops) in enumerate(zip(track_pid, track_energy, track_start_position, track_stop_position)):
            muons_above_threshold = (np.abs(pids) == 13) & (energies > 166)
            electrons_above_threshold = (np.abs(pids) == 11) & (energies > 2)
            gammas_above_threshold = (np.abs(pids) == 22) & (energies > 2)
            above_threshold = muons_above_threshold | electrons_above_threshold | gammas_above_threshold
            outside_tank = (np.linalg.norm(stops[:, (0, 1)], axis=1) > radius) | (np.abs(stops[:, 2]) > half_height)
            dset_veto[offset+i] = np.any(above_threshold & outside_tank)
            end_energies_estimate = energies - np.linalg.norm(stops - starts, axis=1)*2
            muons_above_threshold = (np.abs(pids) == 13) & (end_energies_estimate > 166)
            electrons_above_threshold = (np.abs(pids) == 11) & (end_energies_estimate > 2)
            gammas_above_threshold = (np.abs(pids) == 22) & (end_energies_estimate > 2)
            above_threshold = muons_above_threshold | electrons_above_threshold | gammas_above_threshold
            dset_veto2[offset+i] = np.any(above_threshold & outside_tank)

    for i, (trigs, times, charges, pmts, pmt_pos, pmt_or) in enumerate(zip(hit_triggers, digi_hit_time, digi_hit_charge, digi_hit_pmt, digi_hit_pmt_pos, digi_hit_pmt_or)):
        dset_event_hit_index[offset+i] = hit_offset
        hit_indices = np.where(trigs == event_triggers[i])[0]
        hit_offset_next += len(hit_indices)
        dset_hit_time[hit_offset:hit_offset_next] = times[hit_indices]
        dset_hit_charge[hit_offset:hit_offset_next] = charges[hit_indices]
        dset_hit_pmt[hit_offset:hit_offset_next] = pmts[hit_indices]
        pmt_pos = np.array(pmt_pos)
        pmt_or = np.array(pmt_or)
        #This somehow breaks if there's no PMTs, so skip if that happens
        if hit_offset == hit_offset_next:
            pass
        hit_offset = hit_offset_next

    offset = offset_next
    f.close()


def dump_true_hits(outfile, infile, radius, half_height, event_id, pid, position, isConversion, gamma_start_vtx, direction, energy, true_hit_pmt, true_hit_pmt_pos, true_hit_pmt_or, digi_hit_trigger, track_pid, track_energy, track_start_position, track_stop_position, hit_times, hit_parents):
    """Save the true hits, event variables

    Args:
        Inputs
    """
    f = h5py.File(outfile+'_truth.hy', 'w')

    hit_triggers = digi_hit_trigger
    total_rows = hit_triggers.shape[0]
    event_triggers = np.full(hit_triggers.shape[0], np.nan)
    min_hits=1
    offset = 0
    offset_next = 0
    hit_offset = 0
    hit_offset_next = 0
    total_hits=0
    good_hits=0
    good_rows=0

    hit_pmts = true_hit_pmt
    total_rows += hit_pmts.shape[0]
    for h in hit_pmts:
        total_hits += h.shape[0]
    file_event_triggers = event_triggers
    
    dset_labels=f.create_dataset("labels",
                                 shape=(total_rows,),
                                 dtype=np.int32)
    dset_PATHS=f.create_dataset("root_files",
                                shape=(total_rows,),
                                dtype=h5py.special_dtype(vlen=str))
    dset_IDX=f.create_dataset("event_ids",
                              shape=(total_rows,),
                              dtype=np.int32)
    dset_hit_time=f.create_dataset("hit_time",
                                 shape=(total_hits, ),
                                 dtype=np.float32)
    dset_hit_pmt=f.create_dataset("hit_pmt",
                                  shape=(total_hits, ),
                                  dtype=np.int32)
    '''
    dset_hit_pmt_pos=f.create_dataset("hit_pmt_pos",
                                  shape=(total_hits, 3),
                                  dtype=np.int32)
    dset_hit_pmt_or=f.create_dataset("hit_pmt_or",
                                  shape=(total_hits, 3),
                                  dtype=np.int32)
    '''
    dset_hit_parent=f.create_dataset("hit_parent",
                                  shape=(total_hits, ),
                                  dtype=np.int32)
    dset_event_hit_index=f.create_dataset("event_hits_index",
                                          shape=(total_rows,),
                                          dtype=np.int64) # int32 is too small to fit large indices
    dset_energies=f.create_dataset("energies",
                                   shape=(total_rows, 1),
                                   dtype=np.float32)
    dset_positions=f.create_dataset("positions",
                                    shape=(total_rows, 1, 3),
                                    dtype=np.float32)
    dset_directions=f.create_dataset("directions",
                                    shape=(total_rows, 1, 3),
                                    dtype=np.float32)
    dset_angles=f.create_dataset("angles",
                                 shape=(total_rows, 2),
                                 dtype=np.float32)
    dset_veto = f.create_dataset("veto",
                                 shape=(total_rows,),
                                 dtype=np.bool_)
    dset_veto2 = f.create_dataset("veto2",
                                  shape=(total_rows,),
                                  dtype=np.bool_)
                        

    good_events = ~np.isnan(file_event_triggers)

    offset_next = event_id.shape[0]

    dset_IDX[offset:offset_next] = event_id
    dset_PATHS[offset:offset_next] = infile
    dset_energies[offset:offset_next, :] = energy.reshape(-1, 1)
    dset_positions[offset:offset_next, :, :] = position.reshape(-1, 1, 3)
    dset_directions[offset:offset_next, :, :] = direction.reshape(-1, 1, 3)

    labels = np.full(pid.shape[0], -1)
    label_map = {13: 0, 11: 1, 22: 2}
    for k, v in label_map.items():
        labels[pid == k] = v
    dset_labels[offset:offset_next] = labels

    polars = np.arccos(direction[:, 1])
    azimuths = np.arctan2(direction[:, 2], direction[:, 0])
    dset_angles[offset:offset_next, :] = np.hstack((polars.reshape(-1, 1), azimuths.reshape(-1, 1)))

    for i, (pids, energies, starts, stops) in enumerate(zip(track_pid, track_energy, track_start_position, track_stop_position)):
        muons_above_threshold = (np.abs(pids) == 13) & (energies > 166)
        electrons_above_threshold = (np.abs(pids) == 11) & (energies > 2)
        gammas_above_threshold = (np.abs(pids) == 22) & (energies > 2)
        above_threshold = muons_above_threshold | electrons_above_threshold | gammas_above_threshold
        outside_tank = (np.linalg.norm(stops[:, (0, 1)], axis=1) > radius) | (np.abs(stops[:, 2]) > half_height)
        dset_veto[offset+i] = np.any(above_threshold & outside_tank)
        end_energies_estimate = energies - np.linalg.norm(stops - starts, axis=1)*2
        muons_above_threshold = (np.abs(pids) == 13) & (end_energies_estimate > 166)
        electrons_above_threshold = (np.abs(pids) == 11) & (end_energies_estimate > 2)
        gammas_above_threshold = (np.abs(pids) == 22) & (end_energies_estimate > 2)
        above_threshold = muons_above_threshold | electrons_above_threshold | gammas_above_threshold
        dset_veto2[offset+i] = np.any(above_threshold & outside_tank)

    for i, (times, pmts, pmt_pos, pmt_or, parents) in enumerate(zip(hit_times, hit_pmts, true_hit_pmt_pos, true_hit_pmt_or, hit_parents)):
        dset_event_hit_index[offset+i] = hit_offset
        hit_offset_next += times.shape[0]
        dset_hit_time[hit_offset:hit_offset_next] = times
        dset_hit_pmt[hit_offset:hit_offset_next] = pmts
        pmt_pos = np.array(pmt_pos)
        pmt_or = np.array(pmt_or)
        dset_hit_parent[hit_offset:hit_offset_next] = parents
        hit_offset = hit_offset_next

    offset = offset_next
    f.close()

if __name__ == '__main__':

    config = get_args()
    if config.output_dir is not None:
        print("output directory: " + str(config.output_dir))
        if not os.path.exists(config.output_dir):
            print("                  (does not exist... creating new directory)")
            os.mkdir(config.output_dir)
        if not os.path.isdir(config.output_dir):
            raise argparse.ArgumentTypeError("Cannot access or create output directory" + config.output_dir)
    else:
        print("output directory not provided... output files will be in same locations as input files")

    file_count = len(config.input_files)
    current_file = 0

    for input_file in config.input_files:
        if os.path.splitext(input_file)[1].lower() != '.root':
            print("File " + input_file + " is not a .root file, skipping")
            continue
        input_file = os.path.abspath(input_file)

        if config.output_dir is None:
            output_file = os.path.splitext(input_file)[0] + '.npz'
        else:
            output_file = os.path.join(config.output_dir, os.path.splitext(os.path.basename(input_file))[0] + '.npz')

        print("\nNow processing " + input_file)
        print("Outputting to " + output_file)

        dump_file(input_file, output_file)

        current_file += 1
        print("Finished converting file " + output_file + " (" + str(current_file) + "/" + str(file_count) + ")")

    print("\n=========== ALL FILES CONVERTED ===========\n")
