import utils_data 



def time_overlap(start1, end1, start2, end2):
    if end2 is None:
        return start1 >= start2
    return max(start1, start2) <= min(end1, end2)




class Language:
    def __init__(self, ):
        pass
    def segs2atoms(self,target, segments, frames=None):
        atoms = []
        if target == "av":
            print("Processing AV segments")
            for seg_id, segment in segments.items():
                label_id = segment["label_id"]
                frames = segment["frames"]
                start_frame = frames[0]
                end_frame = frames[-1]
                atoms.append({
                    "target": target,
                    "seg_id": seg_id,
                    "label_id": label_id,
                    "frames": frames,
                    "start_frame": start_frame,
                    "end_frame": end_frame
                })
        elif target == "agents":
            print("Processing agent segments")
            action_loc_pairs = utils_data.build_agent_frame_action_loc_pairs(segments, frames)
            for seg_id, segment in segments.items():
                label_id = segment["label_id"]
                frames = list(segment["annos"].keys())
                start_frame = frames[0]
                end_frame = frames[-1]
                atoms.append({
                    "target": target,
                    "seg_id": seg_id,
                    "label_id": label_id,
                    "frames": frames,
                    'frame-action-location': action_loc_pairs.get(seg_id, None),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                })
        else:
            raise ValueError(f"Unknown target: {target}")
        
        return atoms 

    def _atoms_by_time(self, atoms):
        time_lists = sorted(set(int(atom["start_frame"]) for atom in atoms))
        atoms_by_time = {}
        for t, t_1 in zip(time_lists, time_lists[1:] + [None]):
            for atom in atoms:
                atom_start = int(atom["start_frame"])
                atom_end = int(atom["end_frame"])
                if time_overlap(atom_start, atom_end, t, t_1):
                    if t not in atoms_by_time:
                        atoms_by_time[t] = {
                            'start_frame': t,
                            'end_frame': t_1,
                            'atoms': [],
                        }
                    atoms_by_time[t]['atoms'].append(atom)
        return atoms_by_time

    
    def atoms2facts(self, atoms):
        atoms_by_time = self._atoms_by_time(atoms)
        facts = []
        for t, atoms_at_time in atoms_by_time.items():
            start_frame = int(atoms_at_time['start_frame'])
            end_frame = atoms_at_time['end_frame']
            if end_frame is not None:
                end_frame = int(end_frame)
            else:
                end_frame = float('inf')

            fact = {
                'start_frame': start_frame,
                'end_frame': end_frame,
                'agents': [],

            }
            for atom in atoms_at_time['atoms']:
                
                # Process each atom as needed
                if atom['target']=='av':
                    fact['av_action_id'] = atom['label_id']

                elif atom['target']=='agents':
                    agent_class = atom['label_id']
                    frame_action_location = [pair for pair in atom['frame-action-location'] 
                                             if pair["frame"] >= start_frame 
                                             and pair["frame"] <= end_frame]
                    

                    agent_behavior = {
                        'class': agent_class,
                        'frame-action-location': frame_action_location,
                    }
                    fact['agents'].append(agent_behavior)
            facts.append(fact)
        return facts

    def facts2rules(self, facts):
        rules = []
        for fact in facts:
            rule = {
                'start_frame': fact['start_frame'],
                'end_frame': fact['end_frame'],
                'agents': fact['agents'],
                'av_action_id': fact.get('av_action_id', None),
            }
            rules.append(rule)
        return rules


