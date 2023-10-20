import os
import inspect
import math

import numpy as np
import ROOT

print(os.path.abspath(ROOT.__file__))



class fiTQun:
    def __init__(self,infile) -> None:
        inFile = ROOT.TFile.Open(infile ,"READ")
        #print(dir(inFile))
        self.file = inFile
        tree = inFile.fiTQun
        self.tree = tree
        self.nevent = tree.GetEntries()
        #self.tree.GetEvent(0)
        #print(f'num trigs?: {len(self.tree.T)}')
        #print(f'nvc: {self.tree.nvc}, 0: {self.tree.ID[0]}, 1: {self.tree.ID[1]}, 3: {self.tree.ID[2]}')

    def get_event(self, ev):
        # Delete previous triggers to prevent memory leak (only if file does not change)
        print(ev)
        self.tree.GetEvent(ev)
        self.current_event = ev

class SKDETSIM:
    def __init__(self,infile) -> None:
        inFile = ROOT.TFile.Open(infile ,"READ")
        #print(dir(inFile))
        self.file = inFile
        tree = inFile.T
        self.tree = tree
        self.nevent = tree.GetEntries()
        self.tree.GetEvent(0)
        #print(f'num trigs?: {len(self.tree.T)}')
        #print(f'nvc: {self.tree.nvc}, 0: {self.tree.ID[0]}, 1: {self.tree.ID[1]}, 3: {self.tree.ID[2]}')

    def get_event(self, ev):
        # Delete previous triggers to prevent memory leak (only if file does not change)
        print(ev)
        self.tree.GetEvent(ev)
        self.nhits = len(self.tree.ID)
        self.current_event = ev

    def get_event_info(self):
        self.tree.GetEvent(self.current_event)
        mass_correction = 0
        if self.tree.ipvc[0] == 11:
            mass_correction = 0.5
        elif self.tree.ipvc[0] == 13:
            mass_correction = 105.7
        elif self.tree.ipvc[0] == 211:
            mass_correction = 139.584
        total_p = math.sqrt(math.pow(self.tree.pvc[0],2)+math.pow(self.tree.pvc[1],2)+math.pow(self.tree.pvc[2],2))
        if self.current_event == 0 or self.current_event == 1 or self.current_event == 2:
            print(f'nevent: {self.current_event}, x mom : {self.tree.pvc[0]}, y mom: {self.tree.pvc[1]}, z mom: {self.tree.pvc[2]}, total p: {total_p} ')
        return {
            "pid": self.tree.ipvc[0],
            #End for photon decay vertex, position for photon production vertex
            "position": [self.tree.posivc[0], self.tree.posivc[1], self.tree.posivc[2]],# e+ / e- should have same position
            "gamma_start_vtx": [-999,-999,-999], # e+ / e- should have same position
            "isConversion": False,
            "energy_electron": -999,
            "energy_positron": -999,
            "direction_electron": [-999,-999,-999],
            "direction_positron": [-999,-999,-999],
            "direction": [self.tree.pvc[0]/total_p, self.tree.pvc[1]/total_p, self.tree.pvc[2]/total_p],
            "energy": math.sqrt( math.pow(total_p,2) + math.pow(mass_correction,2))
        }
    def get_digitized_hits(self):
        position = []
        charge = []
        time = []
        pmt = []
        trigger = []
        for hit in range(self.nhits):
            pmt_id = self.tree.ID[hit]-1
            position.append([self.tree.X[hit], self.tree.Y[hit], self.tree.Z[hit]])
            charge.append(self.tree.Q[hit])
            time.append(self.tree.T[hit])
            pmt.append(pmt_id)
            trigger.append(0)
        hits = {
            "position": np.asarray(position, dtype=np.float32),
            "charge": np.asarray(charge, dtype=np.float32),
            "time": np.asarray(time, dtype=np.float32),
            "pmt": np.asarray(pmt, dtype=np.int32),
            "trigger": np.asarray(trigger, dtype=np.int32),
        }
        return hits




class WCSim:
    def __init__(self, tree):
        ROOT.gSystem.Load("/home/fcormier/t2k/t2k_ml_base/t2k_ml/DataTools/libWCSimRoot.so")
        print("number of entries in the geometry tree: " + str(self.geotree.GetEntries()))
        self.geotree.GetEntry(0)
        self.geo = self.geotree.wcsimrootgeom
        self.num_pmts = self.geo.GetWCNumPMT()
        self.tree = tree
        self.nevent = self.tree.GetEntries()
        print("number of entries in the tree: " + str(self.nevent))
        # Get first event and trigger to prevent segfault when later deleting trigger to prevent memory leak
        self.tree.GetEvent(0)
        self.current_event = 0
        self.event = self.tree.wcsimrootevent
        self.ntrigger = self.event.GetNumberOfEvents()
        self.trigger = self.event.GetTrigger(0)
        self.current_trigger = 0

    def get_event(self, ev):
        # Delete previous triggers to prevent memory leak (only if file does not change)
        triggers = [self.event.GetTrigger(i) for i in range(self.ntrigger)]
        oldfile = self.tree.GetCurrentFile()
        self.tree.GetEvent(ev)
        if self.tree.GetCurrentFile() == oldfile:
            [t.Delete() for t in triggers]
        self.current_event = ev
        self.event = self.tree.wcsimrootevent
        self.ntrigger = self.event.GetNumberOfEvents()

    def get_trigger(self, trig):
        self.trigger = self.event.GetTrigger(trig)
        self.current_trigger = trig
        return self.trigger

    def get_first_trigger(self):
        first_trigger = 0
        first_trigger_time = 9999999.0
        for index in range(self.ntrigger):
            self.get_trigger(index)
            trigger_time = self.trigger.GetHeader().GetDate()
            if trigger_time < first_trigger_time:
                first_trigger_time = trigger_time
                first_trigger = index
        return self.get_trigger(first_trigger)

    def get_truth_info(self):  # deprecated: should now use get_event_info instead, leaving here for use with old files
        self.get_trigger(0)
        tracks = self.trigger.GetTracks()
        energy = []
        position = []
        direction = []
        pid = []
        for i in range(self.trigger.GetNtrack()):
            if tracks[i].GetParenttype() == 0 and tracks[i].GetFlag() == 0 and tracks[i].GetIpnu() in [22, 11, -11, 13,
                                                                                                       -13, 111]:
                pid.append(tracks[i].GetIpnu())
                position.append([tracks[i].GetStart(0), tracks[i].GetStart(1), tracks[i].GetStart(2)])
                direction.append([tracks[i].GetDir(0), tracks[i].GetDir(1), tracks[i].GetDir(2)])
                energy.append(tracks[i].GetE())
        return direction, energy, pid, position

    def get_event_info(self):
        self.get_trigger(0)
        tracks = self.trigger.GetTracks()
        # Primary particles with no parent are the initial simulation
        particles = [t for t in tracks if t.GetFlag() == 0 and t.GetParenttype() == 0]
        pid_array = []
        decay_e_energy = 0
        decay_electron = None
        isConversion=False
        for part in tracks:
            test_energy = part.GetE()
            test_Start = [part.GetStart(i) for i in range(3)]
            test_Dir = [part.GetDir(i) for i in range(3)]
            test_time = part.GetTime()
            test_pid = part.GetIpnu()
            pid_array.append(test_pid)
            if test_pid == 11:
                if part.GetParenttype() == 13:
                    if part.GetE() > decay_e_energy:
                        decay_e_energy = part.GetE()
                        decay_electron = part
        decay_electron_exists = False
        decay_electron_e = -999
        decay_electron_time = -999
        if decay_electron is None:
            pass
        elif decay_electron.GetE() > 10:
            decay_electron_exists = True
            decay_electron_e = decay_electron.GetE()
            decay_electron_time = decay_electron.GetTime()

            
        #with open('decay_e_data.txt', 'a') as f:
        #    f.write(f'{decay_electron.GetE()}, {decay_electron.GetTime()}\n') 
        # Check there is exactly one particle with no parent:
        if len(particles) == 1:
            # Only one primary, this is the particle being simulated
            print(f'primary time: {particles[0].GetTime()}')
            return {
                "decayElectronExists":decay_electron_exists,
                "decayElectronEnergy": decay_electron_e,
                "decayElectronTime": decay_electron_time,
                "pid": particles[0].GetIpnu(),
                "position": [particles[0].GetStart(i) for i in range(3)],
                "direction": [particles[0].GetDir(i) for i in range(3)],
                "energy": particles[0].GetE(),
                "isConversion": isConversion,
                "gamma_start_vtx": [-99999, -99999, -99999],
                "range": -999,
                "energy_electron": -999,
                "energy_positron": -999,
                "direction_electron": [-99999, -99999, -99999],
                "direction_positron": [-99999, -99999, -99999]
            }
        # Particle with flag -1 is the incoming neutrino or 'dummy neutrino' used for gamma
        # WCSim saves the gamma details (except position) in the neutrino track with flag -1
        neutrino = [t for t in tracks if t.GetFlag() == -1]
        # Check for dummy neutrino that actually stores a gamma that converts to e+ / e-
        isConversion = len(particles) == 2 and {p.GetIpnu() for p in particles} == {11, -11}
        if isConversion and len(neutrino) == 1 and neutrino[0].GetIpnu() == 22:
            print("WRONG 1")
            return {
                "decayElectronExists":decay_electron_exists,
                "decayElectronEnergy": decay_electron_e,
                "decayElectronTime": decay_electron_time,
                "pid": 22,
                "position": [particles[0].GetStart(i) for i in range(3)], # e+ / e- should have same position
                "direction": [neutrino[0].GetDir(i) for i in range(3)],
                "energy": neutrino[0].GetE(),
                "isConversion": isConversion,
                "gamma_start_vtx": [-99999, -99999, -99999],
                "range": -999,
                "energy_electron": -999,
                "energy_positron": -999,
                "direction_electron": [-99999, -99999, -99999],
                "direction_positron": [-99999, -99999, -99999]
            }
        # Check for dummy neutrino from old gamma simulations that didn't save the gamma info
        if isConversion and len(neutrino) == 1 and neutrino[0].GetIpnu() == 12 and neutrino[0].GetE() < 0.0001:
            # Should be a positron/electron pair from a gamma simulation (temporary hack since no gamma truth saved)
            momentum = [sum(p.GetDir(i) * p.GetP() for p in particles) for i in range(3)]
            norm = np.sqrt(sum(p ** 2 for p in momentum))
            print("WRONG 2")
            return {
                "decayElectronExists":decay_electron_exists,
                "decayElectronEnergy": decay_electron_e,
                "decayElectronTime": decay_electron_time,
                "pid": 22,
                "position": [particles[0].GetStart(i) for i in range(3)],  # e+ / e- should have same position
                "direction": [p / norm for p in momentum],
                "isConversion": isConversion,
                "energy": sum(p.GetE() for p in particles),
                "gamma_start_vtx": [-99999, -99999, -99999],
                "range": -999,
                "energy_electron": -999,
                "energy_positron": -999,
                "direction_electron": [-99999, -99999, -99999],
                "direction_positron": [-99999, -99999, -99999]
            }
        # Otherwise something else is going on... guess info from the primaries
        #EDIT: Return info of particle with highest energy
        momentum = [sum(p.GetDir(i) * p.GetP() for p in particles) for i in range(3)]
        norm = np.sqrt(sum(p ** 2 for p in momentum))
        final_particle = None
        max_E=0
        for part in particles:
           test_1 = part.GetIpnu()
           test_2 = part.GetE()
           test_3 = [part.GetStart(i) for i in range(3)]
           test_4 = [part.GetStop(i) for i in range(3)]
           test_5 = [part.GetDir(i) for i in range(3)]
           if part.GetE() > max_E:
                final_particle = part 
                max_E = part.GetE()
        #If particle with highest energy is gamma, find e+/e-
        test_final_particle_ipnu = final_particle.GetIpnu() 
        if final_particle.GetIpnu() == 22:
            neutrino = [t for t in tracks if t.GetFlag() == -1]
            neutrino_ipnu = neutrino[0].GetIpnu()
            test_position_neutrino = [neutrino[0].GetStart(i) for i in range(3)]
            test_end_neutrino = [neutrino[0].GetStop(i) for i in range(3)]
            test_direction_neutrino = [neutrino[0].GetDir(i) for i in range(3)] 
            test_energy_neutrino = neutrino[0].GetE()
            for part in particles:
                test_1 = part.GetIpnu()
                test_2 = part.GetE()
                test_3 = [part.GetStart(i) for i in range(3)]
                test_4 = [part.GetStop(i) for i in range(3)]
                test_5 = [part.GetDir(i) for i in range(3)]

        #Either has two particles with one being an electron and the other photon, or two photons? WCSIm is weird
        isConversion = len(particles) == 2 and (particles[0].GetIpnu() == 11 or (particles[0].GetIpnu() == 22 and particles[1].GetIpnu() == 22 )) 
            
        #Sometimes final_particle (the one with most energy) is not the photon during photon generation (how???)
        #So look at tell-tale signs of conversion to see if it's a photon
        #TODO: test if this affects e-/mu-
        if final_particle.GetIpnu()==22 or isConversion:
            neutrino = [t for t in tracks if t.GetFlag() == -1]
            neutrino_ipnu = neutrino[0].GetIpnu()
            all_particles = [t for t in tracks]
            all_particles_ipnu = []
            '''
            for p_all in all_particles:
                test_all_ipnu = p_all.GetIpnu()
                test_all_position = [p_all.GetStart(i) for i in range(3)]  
                test_all_direction = [p_all.GetDir(i) for i in range(3)]  
                test_all_flag = p_all.GetFlag()
                test_all_parenttype = p_all.GetParenttype()
                test_all_energy = p_all.GetE()
                if test_all_ipnu != 11:
                    print(test_all_ipnu)
                all_particles_ipnu.append(test_all_ipnu)
            unique_all_ipnu, all_unique_counts = np.unique(all_particles_ipnu, return_counts=True)
            '''
            # Check for dummy neutrino that actually stores a gamma that converts to e+ / e-

            stop_photon_pos = [-9999,-9999,-9999]


            #for n in neutrino:
            #    test_n = n.GetIpnu()
            #    test_position_gamma = [p.GetStart(i) for i in range(3)] 
            #    test_end_neutrino = [n.GetStop(i) for i in range(3)]
            #    test_direction_neutrino = [n.GetDir(i) for i in range(3)] 
            #    test_flag_neutrino = n.GetFlag()
            #    test_energy_n= n.GetE()
            #    print(f'NEUTRINO position: {test_position_neutrino}, stop: {test_end_neutrino}, direction: {test_direction_neutrino}, pid: {test_n}, energy: {test_energy_n}, flag: {test_flag_neutrino}')
            #Only shows the electron (not positron?)
            #Also photon with flag -1? And it has different particle directions?
            if isConversion and len(neutrino) == 1 and neutrino[0].GetIpnu() == 22:
                for p in particles:
                    test_p = p.GetIpnu()
                    test_position_gamma = [p.GetStart(i) for i in range(3)] 
                    test_end_gamma = [p.GetStop(i) for i in range(3)] 
                    test_direction_gamma = [p.GetDir(i) for i in range(3)] 
                    test_energy= p.GetE()
                    test_flag_gamma = p.GetFlag()
                    #print(f'position: {test_position_gamma}, stop: {test_end_gamma}, direction: {test_direction_gamma}, pid: {test_p}, energy: {test_energy}, flag: {test_flag_gamma}')
                    #This is the photon we want
                    #If position is exactly 0 something weird happened and the usual pid 11 got pid 22, skip it
                    electron_energy = -999
                    electron_start = -999
                    electron_end = -999
                    positron_energy = -999
                    primary_range = -999
                    electron_direction = [-999,-999,-999]
                    positron_direction = [-999,-999,-999]
                    found_electron=False
                    found_positron=False
                    if test_p==22 and test_flag_gamma==0 and test_position_gamma[0] != 0:
                        test_direction_t=[-999,-999,-999]
                        #Find electron and positron
                        for t in tracks:
                            test_position_track = [t.GetStart(i) for i in range(3)]
                            if (test_position_track[0]==test_end_gamma[0]) and (test_position_track[1]==test_end_gamma[1]) and (test_position_track[2]==test_end_gamma[2]):
                                test_t = t.GetIpnu()
                                test_start_t = [t.GetStart(i) for i in range(3)] 
                                test_end_t = [t.GetStop(i) for i in range(3)] 
                                test_direction_t = [t.GetDir(i) for i in range(3)] 
                                test_energy_t= t.GetE()
                                test_flag_t = t.GetFlag()
                                #Arbitrarily choose electron to get direction
                                if test_t == 11:
                                    electron_energy = test_energy_t
                                    electron_start = test_start_t
                                    electron_end = test_end_t
                                    e_range = math.sqrt( math.pow(electron_start[0] - electron_end[0],2) + math.pow(electron_start[1] - electron_end[1],2) + math.pow(electron_start[2] - electron_end[2],2))
                                    if e_range > primary_range:
                                        primary_range = e_range
                                    electron_direction = test_direction_t
                                    found_electron=True
                                elif test_t == -11:
                                    positron_energy = test_energy_t
                                    positron_direction = test_direction_t
                                    positron_start = test_start_t
                                    positron_end = test_end_t
                                    p_range = math.sqrt( math.pow(positron_start[0] - positron_end[0],2) + math.pow(positron_start[1] - positron_end[1],2) + math.pow(positron_start[2] - positron_end[2],2))
                                    if p_range > primary_range:
                                        primary_range = p_range
                                    found_positron=True
                                if found_electron and found_positron:
                                    break
                        return {
                            "decayElectronExists":decay_electron_exists,
                            "decayElectronEnergy": decay_electron_e,
                            "decayElectronTime": decay_electron_time,
                            "pid": 22,
                            #End for photon decay vertex, position for photon production vertex
                            "position": test_end_gamma, # e+ / e- should have same position
                            "gamma_start_vtx": test_position_gamma, # e+ / e- should have same position
                            "isConversion": isConversion,
                            "energy_electron": electron_energy,
                            "range": primary_range,
                            "energy_positron": positron_energy,
                            "direction_electron": electron_direction,
                            "direction_positron": positron_direction,
                            "direction": test_direction_gamma,
                            "energy": test_energy
                        }
            else:
                print(f'primary time: {particles[0].GetTime()}')
                return {
                    "decayElectronExists":decay_electron_exists,
                    "decayElectronEnergy": decay_electron_e,
                    "decayElectronTime": decay_electron_time,
                    "pid": final_particle.GetIpnu(),
                    "position": [final_particle.GetStart(i) for i in range(3)],
                    "end": [final_particle.GetStop(i) for i in range(3)],
                    "stopVol": final_particle.GetStopvol(),
                    "startVol": final_particle.GetStartvol(),
                    "gamma_start_vtx": [-99999, -99999, -99999],
                    "range": -999,
                    "energy_electron": -999,
                    "energy_positron": -999,
                    "direction_electron": [-99999, -99999, -99999],
                    "direction_positron": [-99999, -99999, -99999],
                    "direction": [final_particle.GetDir(i) for i in range(3)],
                    "isConversion": isConversion,
                    "energy": final_particle.GetE()
            }
        else:
            start = [final_particle.GetStart(i) for i in range(3)]
            end = [final_particle.GetStop(i) for i in range(3)]
            primary_range = math.sqrt( math.pow(start[0] - end[0],2) + math.pow(start[1] - end[1],2) + math.pow(start[2] - end[2],2))
            return {
                "decayElectronExists":decay_electron_exists,
                "decayElectronEnergy": decay_electron_e,
                "decayElectronTime": decay_electron_time,
                "pid": final_particle.GetIpnu(),
                "position": [final_particle.GetStart(i) for i in range(3)],
                "end": [final_particle.GetStop(i) for i in range(3)],
                "range": primary_range,
                "energy_electron": -999,
                "energy_positron": -999,
                "direction_electron": [-99999, -99999, -99999],
                "direction_positron": [-99999, -99999, -99999],
                "stopVol": final_particle.GetStopvol(),
                "startVol": final_particle.GetStartvol(),
                "direction": [final_particle.GetDir(i) for i in range(3)],
                "isConversion": isConversion,
                "energy": final_particle.GetE()
        }

    def get_digitized_hits(self):
        position = []
        charge = []
        time = []
        pmt = []
        trigger = []
        for t in range(self.ntrigger):
            self.get_trigger(t)
            for hit in self.trigger.GetCherenkovDigiHits():
                pmt_id = hit.GetTubeId() - 1
                position.append([self.geo.GetPMT(pmt_id).GetPosition(j) for j in range(3)])
                charge.append(hit.GetQ())
                time.append(hit.GetT())
                pmt.append(pmt_id)
                trigger.append(t)
        hits = {
            "position": np.asarray(position, dtype=np.float32),
            "charge": np.asarray(charge, dtype=np.float32),
            "time": np.asarray(time, dtype=np.float32),
            "pmt": np.asarray(pmt, dtype=np.int32),
            "trigger": np.asarray(trigger, dtype=np.int32)
        }
        return hits

    def get_true_hits(self):
        position = []
        track = []
        pmt = []
        PE = []
        trigger = []
        for t in range(self.ntrigger):
            self.get_trigger(t)
            for hit in self.trigger.GetCherenkovHits():
                pmt_id = hit.GetTubeID() - 1
                tracks = set()
                for j in range(hit.GetTotalPe(0), hit.GetTotalPe(0)+hit.GetTotalPe(1)):
                    pe = self.trigger.GetCherenkovHitTimes().At(j)
                    tracks.add(pe.GetParentID())
                position.append([self.geo.GetPMT(pmt_id).GetPosition(k) for k in range(3)])
                track.append(tracks.pop() if len(tracks) == 1 else -2)
                pmt.append(pmt_id)
                PE.append(hit.GetTotalPe(1))
                trigger.append(t)
        hits = {
            "position": np.asarray(position, dtype=np.float32),
            "track": np.asarray(track, dtype=np.int32),
            "pmt": np.asarray(pmt, dtype=np.int32),
            "PE": np.asarray(PE, dtype=np.int32),
            "trigger": np.asarray(trigger, dtype=np.int32)
        }
        return hits

    def get_hit_photons(self):
        start_position = []
        end_position = []
        start_time = []
        end_time = []
        track = []
        pmt = []
        trigger = []
        for t in range(self.ntrigger):
            self.get_trigger(t)
            n_photons = self.trigger.GetNcherenkovhittimes()
            trigger.append(np.full(n_photons, t, dtype=np.int32))
            counts = [h.GetTotalPe(1) for h in self.trigger.GetCherenkovHits()]
            hit_pmts = [h.GetTubeID()-1 for h in self.trigger.GetCherenkovHits()]
            pmt.append(np.repeat(hit_pmts, counts))
            end_time.append(np.zeros(n_photons, dtype=np.float32))
            track.append(np.zeros(n_photons, dtype=np.int32))
            start_time.append(np.zeros(n_photons, dtype=np.float32))
            start_position.append(np.zeros((n_photons, 3), dtype=np.float32))
            end_position.append(np.zeros((n_photons, 3), dtype=np.float32))
            photons = self.trigger.GetCherenkovHitTimes()
            end_time[t][:] = [p.GetTruetime() for p in photons]
            track[t][:] = [p.GetParentID() for p in photons]
            try:  # Only works with new tracking branch of WCSim
                start_time[t][:] = [p.GetPhotonStartTime() for p in photons]
                for i in range(3):
                    start_position[t][:,i] = [p.GetPhotonStartPos(i)/10 for p in photons]
                    end_position[t][:,i] = [p.GetPhotonEndPos(i)/10 for p in photons]
            except AttributeError: # leave as zeros if not using tracking branch
                pass
        photons = {
            "start_position": np.concatenate(start_position),
            "end_position": np.concatenate(end_position),
            "start_time": np.concatenate(start_time),
            "end_time": np.concatenate(end_time),
            "track": np.concatenate(track),
            "pmt": np.concatenate(pmt),
            "trigger": np.concatenate(trigger)
        }
        return photons

    def get_tracks(self):
        track_id = []
        pid = []
        start_time = []
        energy = []
        start_position = []
        stop_position = []
        parent = []
        flag = []
        for t in range(self.ntrigger):
            self.get_trigger(t)
            for track in self.trigger.GetTracks():
                track_id.append(track.GetId())
                pid.append(track.GetIpnu())
                start_time.append(track.GetTime())
                energy.append(track.GetE())
                start_position.append([track.GetStart(i) for i in range(3)])
                stop_position.append([track.GetStop(i) for i in range(3)])
                parent.append(track.GetParenttype())
                flag.append(track.GetFlag())
        tracks = {
            "id": np.asarray(track_id, dtype=np.int32),
            "pid": np.asarray(pid, dtype=np.int32),
            "start_time": np.asarray(start_time, dtype=np.float32),
            "energy": np.asarray(energy, dtype=np.float32),
            "start_position": np.asarray(start_position, dtype=np.float32),
            "stop_position": np.asarray(stop_position, dtype=np.float32),
            "parent": np.asarray(parent, dtype=np.int32),
            "flag": np.asarray(flag, dtype=np.int32)
        }
        return tracks

    def get_triggers(self):
        trigger_times = np.empty(self.ntrigger, dtype=np.float32)
        trigger_types = np.empty(self.ntrigger, dtype=np.int32)
        for t in range(self.ntrigger):
            self.get_trigger(t)
            trigger_times[t] = self.trigger.GetHeader().GetDate()
            trig_type = self.trigger.GetTriggerType()
            if trig_type > np.iinfo(np.int32).max:
                trig_type = -1;
            trigger_types[t] = trig_type
        triggers = {
                "time": trigger_times,
                "type": trigger_types
        }
        return triggers


class WCSimFile(WCSim):
    def __init__(self, filename):
        self.file = ROOT.TFile(filename, "read")
        tree = self.file.Get("wcsimT")
        self.geotree = self.file.Get("wcsimGeoT")
        super().__init__(tree)

    def __del__(self):
        self.file.Close()


class WCSimChain(WCSim):
    def __init__(self, filenames):
        self.chain = ROOT.TChain("wcsimT")
        for file in filenames:
            self.chain.Add(file)
        self.file = self.GetFile()
        self.geotree = self.file.Get("wcsimGeoT")
        super().__init__(self.chain)

def get_label(infile):
    if "_gamma" in infile:
        label = 0
    elif "_e" in infile:
        label = 1
    elif "_mu" in infile:
        label = 2
    elif "_pi0" in infile:
        label = 3
    else:
        print("Unknown input file particle type")
        raise SystemExit
    return label
