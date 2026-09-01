from collections import Counter

import utils_data 



def time_overlap(start1, end1, start2, end2):
    if end2 is None:
        return start1 >= start2
    return max(start1, start2) <= min(end1, end2)


class Rule:
    head: None
    body: None
    def __init__(self, fact_tuple, support=1, total_support=1, evidence_count=1):
        self.head = {'av_action_id': fact_tuple[0]}
        self.body = {
            'agent_class': fact_tuple[1],
            'action': tuple(fact_tuple[2]) if isinstance(fact_tuple[2], (list, tuple)) else (fact_tuple[2],),
            'location': tuple(fact_tuple[3]) if isinstance(fact_tuple[3], (list, tuple)) else (fact_tuple[3],),
        }
        self.support = int(support)
        self.total_support = max(1, int(total_support))
        self.evidence_count = max(1, int(evidence_count))

    @property
    def confidence(self):
        return float(self.support / self.evidence_count)

    @property
    def coverage(self):
        return float(self.support / self.total_support)

    @property
    def rank_key(self):
        return [
            -int(self.support),
            -round(self.coverage, 12),
            -round(self.confidence, 12),
            int(self.head['av_action_id']),
            int(self.body['agent_class']),
            list(self.body['action']),
            list(self.body['location']),
        ]

    def to_dict(self):
        return {
            'head': self.head,
            'body': {
                'agent_class': self.body['agent_class'],
                'action': list(self.body['action']),
                'location': list(self.body['location']),
            },
            'support': self.support,
            'total_support': self.total_support,
            'coverage': self.coverage,
            'confidence': self.confidence,
            'rank_key': list(self.rank_key),
        }
    


class Language:
    def __init__(self, ):
        pass

    @staticmethod
    def _flatten_ids(values):
        flattened = []
        for value in values:
            if isinstance(value, (list, tuple, set)):
                flattened.extend(value)
            else:
                flattened.append(value)
        return tuple(flattened)

    @staticmethod
    def _rule_signature(head, agent_class, action_ids, loc_ids):
        return (head, agent_class, tuple(action_ids), tuple(loc_ids))

    def evaluate_rule(self, rule, support, total_support, evidence_count):
        return Rule(rule, support=support, total_support=total_support, evidence_count=evidence_count).to_dict()

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
        rule_supports = Counter()
        rule_evidence = Counter()
        for fact in facts:
            if 'av_action_id' not in fact:
                continue
            head = fact['av_action_id']
            for agent in fact['agents']:
                frame_action_location = agent.get('frame-action-location', []) or []
                if not frame_action_location:
                    continue
                agent_class = agent['class']

                action_options = []
                loc_options = []
                for pair in frame_action_location:
                    action_ids = self._flatten_ids(pair.get('action_ids', []))
                    loc_ids = self._flatten_ids(pair.get('loc_ids', []))
                    if not action_ids or not loc_ids:
                        continue
                    action_options.append(action_ids)
                    loc_options.append(loc_ids)

                if not action_options or not loc_options:
                    continue

                evidence_count = len(action_options) * len(loc_options)
                for action_ids in action_options:
                    for loc_ids in loc_options:
                        signature = self._rule_signature(head, agent_class, action_ids, loc_ids)
                        rule_supports[signature] += 1
                        rule_evidence[signature] += evidence_count

        total_support_by_head = Counter()
        for (head, _agent_class, _action_ids, _loc_ids), support in rule_supports.items():
            total_support_by_head[head] += support

        rules = []
        for signature, support in rule_supports.items():
            head, agent_class, action_ids, loc_ids = signature
            rule = self.evaluate_rule(
                (head, agent_class, action_ids, loc_ids),
                support=support,
                total_support=total_support_by_head[head],
                evidence_count=rule_evidence[signature],
            )
            rules.append(rule)

        rules.sort(key=lambda row: tuple(row['rank_key']))
        return rules


