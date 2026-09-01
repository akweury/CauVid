import utils_data 



def time_overlap(start1, end1, start2, end2):
    if end2 is None:
        return start1 >= start2
    return max(start1, start2) <= min(end1, end2)


class Rule:
    head: None
    body: None
    def __init__(self, fact_tuple):
        self.head = {'av_action_id': fact_tuple[0]}
        self.body = {'agent_class': fact_tuple[1], 
                     'action': fact_tuple[2], 
                     'location': fact_tuple[3]
                     }
    


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
        fact_tuples = []
        for fact in facts:
            if 'av_action_id' not in fact:
                continue
            head = fact['av_action_id']
            for agent in fact['agents']:
                actions = [pair['action_ids'] for pair in agent['frame-action-location']]
                locs = [pair['loc_ids'] for pair in agent['frame-action-location']]
                agent_class = agent['class']
                
                agent_fact_tuples = [(head, agent_class, tuple(a), tuple(l)) for a in actions for l in locs]
                fact_tuples.extend(agent_fact_tuples)
        fact_tuples = list(set(fact_tuples))
        rules = [Rule(fact_tuple) for fact_tuple in fact_tuples]
        return rules


