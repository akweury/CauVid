
import torch 
from tqdm import tqdm 

from collections import Counter
from src.exp_roadpp import utils_data 
from src.exp_roadpp.logic import Clause, Atom, Predicate 


def time_overlap(start1, end1, start2, end2):
    if end2 is None:
        return start1 >= start2
    return max(start1, start2) <= min(end1, end2)



class Rule:
    head: None
    body: None
    def __init__(self, fact_tuple, support=1, total_support=1, evidence_count=1):
        head_key, head_value, agent_class, action_ids, loc_ids = fact_tuple
        self.head = {head_key: head_value}
        self.body = {
            'agent_class': agent_class,
            'action': tuple(action_ids) if isinstance(action_ids, (list, tuple)) else (action_ids,),
            'location': tuple(loc_ids) if isinstance(loc_ids, (list, tuple)) else (loc_ids,),
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
        head_key, head_value = next(iter(self.head.items()))
        return [
            -int(self.support),
            -round(self.coverage, 12),
            -round(self.confidence, 12),
            str(head_key),
            int(head_value),
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
    def __init__(self, device):
        self.device= device
        self.predicates = {
            'action': Predicate(
                'action', arity=5, dtypes=[str, str, int, int, str],
                field_names = ("action_id", "agent_class", "start_frame", "end_frame", "tube_uid")),
            'location': Predicate(
                'location', arity=5, dtypes=[str, str, int, int, str],
                field_names = ("location_name", "agent_class", "start_frame", "end_frame", "tube_uid")),
        }

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
    @staticmethod
    def _body_signature(agent_class, action_ids, loc_ids):
        return (agent_class, tuple(action_ids), tuple(loc_ids))
    @staticmethod
    def _lookup_tube_uid(segment, frames):
        for frame_id, box_id in segment["annos"].items():
            box = (frames or {}).get(str(frame_id), {}).get("annos", {}).get(box_id, {})
            if box.get("tube_uid"):
                return box["tube_uid"]
        return None
    @staticmethod
    def _rle_intervals(frame_pairs, key):
        """Collapse consecutive frames sharing the same value under `key`
    ('action_ids' or 'loc_ids') into (value_tuple, start_frame, end_frame)."""
        intervals = []
        current_value = None 
        current_start = None 
        current_end = None 
        for pair in frame_pairs:
            value = tuple(sorted(pair[key]))
            frame = pair["frame"]
            if value != current_value:
                if current_value is not None:
                    intervals.append((current_value, current_start, current_end))
                current_value = value
                current_start = frame
            current_end = frame 
        if current_value is not None:
            intervals.append((current_value, current_start, current_end))
        return intervals
    
    def evaluate_rule(self, rule, support, total_support, evidence_count):
        return Rule(rule, support=support, total_support=total_support, evidence_count=evidence_count).to_dict()

    def video2atoms(self, target, segments, frames=None):
        atoms = []
        if target == "av":
            for seg_id, segment in segments.items():
                # av from start to end of the segment, with action
                av_action_id = segment["label_id"]
                seg_frames = sorted(segment["frames"], key=int)
                start_frame = seg_frames[0]
                end_frame = seg_frames[-1]
                atom = self.predicates['action'].make_atom(
                    action_id=av_action_id, 
                    agent_class="av", 
                    start_frame=start_frame, 
                    end_frame=end_frame, 
                    tube_uid=None).to_dict()
                atoms.append(atom)
        elif target == "agents":
            action_loc_pairs = utils_data.build_agent_frame_action_loc_pairs(segments, frames)
            for seg_id, segment in segments.items():
                agent_class = segment["label_id"]
                start_frame, end_frame, tube_uid = utils_data.get_start_end_frame(segment, frames)
                action_intervals = self._rle_intervals(action_loc_pairs[seg_id], "action_ids")
                location_intervals = self._rle_intervals(action_loc_pairs[seg_id], "loc_ids")

                for action_ids, seg_start, seg_end in action_intervals:
                    for action_id in action_ids:
                        atoms.append(self.predicates['action'].make_atom(
                            action_id=action_id,
                            agent_class=agent_class,
                            start_frame=seg_start,
                            end_frame=seg_end,
                            tube_uid=tube_uid,
                        ).to_dict())
                for loc_ids, seg_start, seg_end in location_intervals:
                    for loc_id in loc_ids:
                        atoms.append(self.predicates['location'].make_atom(
                            location_name=loc_id,
                            agent_class=agent_class,
                            start_frame=seg_start,
                            end_frame=seg_end,
                            tube_uid=tube_uid,
                        ).to_dict())
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

    def _to_ungrounded_atom(self, atom):
        if "action_id" in atom:
            ungrounded_atom = self.predicates.get(atom['pred']).make_atom(
                action_id=atom["action_id"],
                agent_class=atom["agent_class"],
                start_frame=None,
                end_frame=None,
                tube_uid=None,
            )
        elif "location_name" in atom:
            ungrounded_atom = self.predicates.get(atom['pred']).make_atom(
                location_name=atom["location_name"],
                agent_class=atom["agent_class"],
                start_frame=None,
                end_frame=None,
                tube_uid=None,
            )
        else:
            raise ValueError(f"Cannot convert atom to ungrounded form: {atom}")
        return ungrounded_atom


    
    def atoms2atom_clauses(self, atoms_by_videos, head_ungrounded_atoms):
        clauses = set()
        for video_id, atoms in tqdm(atoms_by_videos.items()):
            atoms_by_time = {}
            for atom in atoms:
                start_frame = int(atom["start_frame"])
                ungrounded_atom = self._to_ungrounded_atom(atom)
                if start_frame not in atoms_by_time:
                    atoms_by_time[start_frame] = []
                atoms_by_time[start_frame].append(ungrounded_atom)
            sorted_times = sorted(atoms_by_time)
            for i, body_time in enumerate(sorted_times):
                body_atoms = atoms_by_time[body_time]
                for head_time in sorted_times[i:]:
                    head_atoms = atoms_by_time[head_time]
                    for head_atom in head_atoms:
                        for body_atom in body_atoms:
                            clause = Clause(head_atom,[body_atom])
                            if clause.is_tautology():
                                continue
                            clauses.add(clause)
        return [clause.to_dict() for clause in clauses]
            # start_frame = int(atoms_at_time['start_frame'])
            # end_frame = atoms_at_time['end_frame']
            # if end_frame is not None:
            #     end_frame = int(end_frame)
            # else:
            #     continue
            # fact = {
            #     'start_frame': start_frame,
            #     'end_frame': end_frame,
            #     'agents': {},

            # }
            # for atom in atoms_at_time['atoms']:
                
            #     # Process each atom as needed
            #     if atom['target']=='av':
            #         fact['av_action_id'] = atom['label_id']

            #     tube_uid = atom.get("tube_uid")
            #     if tube_uid is None:
            #         if atom["target"] != "av":
            #             print(f"Warning: tube_uid is None for atom {atom}, skipping this atom.")
            #         continue
            #     agent_record = fact["agents"].setdefault(tube_uid, {})

            #     if atom['target']=='agents':
            #         agent_class = atom['label_id']
            #         frame_action_location = [pair for pair in atom['frame-action-location'] 
            #                                  if pair["frame"] >= start_frame 
            #                                  and pair["frame"] <= end_frame]
            #         agent_record["class"] = agent_class
            #         agent_record["frame-action-location"] = frame_action_location
            #         # agent_behavior = {
            #         #     'class': agent_class,
            #         #     'frame-action-location': frame_action_location,
            #         # }
            #         # fact['agents'].append(agent_behavior)
            #     elif atom["target"] == "action":
            #         agent_record["action_id"] = atom['label_id']
            #     elif atom['target'] == 'location':
            #         agent_record["loc_id"] = atom['label_id']
            #     elif atom['target'] == 'duplex':
            #         agent_record["duplex_id"] = atom['label_id']
            #     elif atom['target'] == 'triplet':
            #         agent_record["triplet_id"] = atom['label_id']


            # if 'av_action_id' not in fact:
            #     continue
            # fact['agents'] = list(fact['agents'].values())
            # facts.append(fact)
        return clauses

    @staticmethod
    def _fact_head_candidates(fact):
        """Every scalar predicate in a fact that can serve as a rule head, paired
        with the agent index it was derived from (None for fact-level predicates
        such as av_action_id) so that agent can be excluded from the rule body."""
        candidates = []
        if fact.get('av_action_id') is not None:
            candidates.append(('av_action_id', fact['av_action_id'], None))
        for agent_index, agent in enumerate(fact.get('agents', []) or []):
            for agent_key, head_key in (
                ('class', 'agent_class'),
                ('action_id', 'action_id'),
                ('loc_id', 'loc_id'),
                ('duplex_id', 'duplex_id'),
                ('triplet_id', 'triplet_id'),
            ):
                value = agent.get(agent_key)
                if value is not None:
                    candidates.append((head_key, value, agent_index))
        return candidates

    def _agent_body_candidates(self, agent):
        frame_action_location = agent.get('frame-action-location', []) or []
        agent_class = agent.get('class')
        if not frame_action_location or agent_class is None:
            return agent_class, []

        action_options = []
        loc_options = []
        for pair in frame_action_location:
            action_ids = self._flatten_ids(pair["action_ids"])
            loc_ids = self._flatten_ids(pair['loc_ids'])
            if not action_ids or not loc_ids:
                continue
            action_options.append(action_ids)
            loc_options.append(loc_ids)

        unique_pairs = list(dict.fromkeys(zip(action_options, loc_options)))
        return agent_class, unique_pairs

    def fact2rules(self, fact):
        rule_supports = Counter()
        head_supports = Counter()
        rules = []
        head_lookup = {}


        agent_bodies = [self._agent_body_candidates(agent) for agent in fact['agents'] or []]

        for head_key, head_value, source_agent_index in self._fact_head_candidates(fact):
            head_id = f"{head_key}:{head_value}"
            head_lookup[head_id] = (head_key, head_value)
            for agent_index, (agent_class, unique_pairs) in enumerate(agent_bodies):
                if agent_index == source_agent_index or agent_class is None:
                    continue
                for action_ids, loc_ids in unique_pairs:
                    body_signature = self._body_signature(agent_class, action_ids, loc_ids)
                    if body_signature not in rule_supports:
                        rule_supports[body_signature] = {}
                    if head_id not in rule_supports[body_signature]:
                        rule_supports[body_signature][head_id] = 0
                    rule_supports[body_signature][head_id] += 1

                    if head_id not in head_supports:
                        head_supports[head_id] = 0
                    head_supports[head_id] += 1



        for body_signature, support in rule_supports.items():
            agent_class, action_ids, loc_ids = body_signature
            for head_id, count in support.items():
                head_key, head_value = head_lookup[head_id]
                rule = self.evaluate_rule(
                    (head_key, head_value, agent_class, action_ids, loc_ids),
                    support=count,
                    total_support=head_supports[head_id],
                    evidence_count=sum(support.values()),
                )
                rules.append(rule)

        rules.sort(key=lambda row: tuple(row['rank_key']))
        return rules, rule_supports, head_supports

