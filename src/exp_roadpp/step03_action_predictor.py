
import torch 

from src.exp_roadpp.utils_data import atom_to_signature, same_signature_no_tube_uid, build_signature
from src.exp_roadpp.step03_visual import visual_2d_heatmap

class ActionPredictor:
    def __init__(self, clauses,dataset_labels, device):
        self.device = "cpu"
        self.coarse_pruned_clauses = clauses
        self.head_signatures = [atom_to_signature(clause["head"]) for clause in clauses]
        self.body_signatures = [[atom_to_signature(body) for body in clause["bodies"]] for clause in clauses]
        self.dataset_labels = dataset_labels
        self.agent_action_num = len(self.dataset_labels["action_labels"])
        self.av_action_num = len(self.dataset_labels["av_action_labels"])
        self.location_name_num = len(self.dataset_labels["loc_labels"])
        self.action_num = self.agent_action_num + self.av_action_num + self.location_name_num

    @staticmethod
    def _fact_signature(signature):
        if len(signature) >= 12 and signature[-2] == "tube_uid":
            return tuple(signature[:-2])
        return tuple(signature)

    @staticmethod
    def _entity_key(atom, index):
        tube_uid = atom[-1]
        if tube_uid is not None:
            return ("tube_uid", tube_uid)
        if atom[7] == "av":
            return ("agent_class", "av")
        return ("atom", index)

    @staticmethod
    def _matches_entity_key(atom, entity_key, index=None):
        if entity_key[0] == "tube_uid":
            return atom[-1] == entity_key[1]
        if entity_key[0] == "agent_class":
            return atom[7] == entity_key[1]
        return entity_key[0] == "atom" and index == entity_key[1]

    @staticmethod
    def _same_identified_entity(atom_a, atom_b):
        if atom_a[-1] is not None or atom_b[-1] is not None:
            return atom_a[-1] is not None and atom_a[-1] == atom_b[-1]
        return atom_a[7] == "av" and atom_b[7] == "av"


    def _atom_to_signature(self, atom):
        if "action_id" in atom:
            prediction = ("action_id_label", atom["action_id_label"],
                          "agent_class_label", atom["agent_class_label"]
                          )
        elif "location_name" in atom:
            prediction = ("location_name_label", atom["location_name_label"],
                          "agent_class_label", atom["agent_class_label"]
                          )
        else:
            raise ValueError("Unknown head type in clause")
        return prediction

    
    def find_matching_bodies(self, atom, clause_body_signatures):
        matching_indices = []
        matching_scores = []
        for i, body_signature in enumerate(clause_body_signatures):
            if atom == body_signature[0]:
                matching_indices.append(i)
                matching_scores.append(body_signature[1])
        return matching_indices

    def find_matching_head_signatures(self, matching_body_signature_indices, clause_head_signatures, th = 0.9):
        all_matches = [clause_head_signatures[i] for i in matching_body_signature_indices]
        filtered_matches = [match for match in all_matches if match[1] >= th]
        return filtered_matches

    def atom_signature_to_index(self, atom_signature):
        if "location_name_label" in atom_signature:
            index = self.agent_action_num + self.av_action_num + self.dataset_labels["loc_labels"].index(atom_signature[5])
        elif "action_id_label" in atom_signature and "av" in atom_signature:
            index = self.agent_action_num + self.dataset_labels["av_action_labels"].index(atom_signature[5])
        elif "action_id_label" in atom_signature and "av" not in atom_signature:
            index = self.dataset_labels["action_labels"].index(atom_signature[5])
        else:
            raise ValueError("Unknown atom signature type")
        return index 
    
    def predict_atom(self, as_in, as_forzen):
        """
        as_in: list of atom signatures in the current frame
        as_forzen: list of atom signatures from the rule head to match against

        # for each atom in as_in, we need to give it a
        """
        pred_actions = torch.zeros(self.action_num).to(self.device)
        
        for atom_signature in as_in:
            for frozen_signature, score in as_forzen:
                if not same_signature_no_tube_uid(atom_signature, frozen_signature):
                    continue
                if "location_name_label" in atom_signature:
                    index = self.agent_action_num + self.av_action_num + self.dataset_labels["loc_labels"].index(atom_signature[5])
                elif "action_id_label" in atom_signature and "av" in atom_signature:
                    index = self.agent_action_num + self.dataset_labels["av_action_labels"].index(atom_signature[5])
                elif "action_id_label" in atom_signature and "av" not in atom_signature:
                    index = self.dataset_labels["action_labels"].index(atom_signature[5])
                else:
                    raise ValueError("Unknown atom signature type")
                pred_actions[index] = score
        
        return pred_actions

    def sis_to_sos(self, sis, label_index):
        """
        Convert a Scene Input Signature (SIS) to a Scene Output Signature (SOS) based on the predicted label index.
        """
        
        predicate, label, label_id = self.index_to_label(label_index)
        
        agent_class = sis[7]
        agent_class_label = sis[9]
        tube_uid = sis[-1]

        return build_signature(predicate, label, label_id, agent_class, agent_class_label, tube_uid)
    
    def connect_pred_to_tube_uid(self, siss, fim, added_atoms):
        """
        For each sis, only its action id and action label is changing.
        The action is determined by the highest scoring prediction in the frame influence matrix for that atom.

        LV: label vector, representing the predicted labels for the atom based on the frame influence matrix.
        """
        
        soss = []
        fim_mean = fim.mean(dim=0)
        
        for atom_i in range(fim_mean.shape[0]):
            sis = siss[atom_i]
            lv = fim_mean[atom_i]
            if lv.sum()>0:
                label_index = (lv.argmax().item())
                soss.append(sis)

        for atom in added_atoms:
            soss.append(atom)
        return soss

    def calc_influence_vector(self, siss, fhs):
        """
        Calculate the Influence Vector (IV) for a given Frozen Head Signature (FHS) on the next frame's atoms.
        fhs: Frozen Head Signature
        next_frame_atoms: List of atoms in the next frame
        Returns a Nx1 vector representing the influence of the FHS on each atom in the next frame.
        """
        def calculate_influence(fhs, sis):
            remain_label = fhs[:10] == sis[:10]
            return int(remain_label)
        
        iv = torch.zeros(len(siss))
        label_index = self.atom_signature_to_index(fhs[0])
        for i, sis in enumerate(siss):
            iv[i] = calculate_influence(fhs[0], sis)
        return iv, label_index
    def create_new_atom(self, fhs):
        """
        Create a new atom based on the given Frozen Head Signature (FHS).
        """
        # Implementation for creating a new atom goes here
        return fhs[0]
    def build_fim(self, siss):
        """
        # how to determine whether the atom will persist to the next frame?
        # A should either disappear, or persist, or change its action in the next frame.
        # change because of rules
        # disappear because it position is out of the frame
        # persist if it remains in the frame without changing its action
        """
        fbs = self.body_signatures
        fhs = self.head_signatures
        # frame influence matrix (FIM) initialization
        fim = torch.zeros(len(siss),len(siss), self.action_num)
        # frame added atoms
        added_atoms = []


        for sis_index, sis in enumerate(siss):
            matched_fbs_indices = self.find_matching_bodies(sis, fbs)
            matched_fhs = self.find_matching_head_signatures(matched_fbs_indices, fhs)
            if len(matched_fhs) == 0:
                # the atom is unchanged in the next frame
                label_index = self.atom_signature_to_index(sis)
                fim[sis_index, sis_index, label_index] = 1                
            else:
                for k in range(len(matched_fhs)):
                    iv, label_index = self.calc_influence_vector(siss, matched_fhs[k])
                    # found matching sis
                    if iv.sum() > 0:
                        fim[sis_index, :, label_index] += iv.squeeze()
                    else:
                        # no matching sis found for this frozen head signature, create it as a new atom
                        new_atom = self.create_new_atom(matched_fhs[k])
                        added_atoms.append(new_atom) 
        return fim, added_atoms
    
    
    def _body_conditions_match(self, body_signatures, body_scopes, entity_key, siss):
        """Return whether every learned condition matches the current scene.

        `body_scopes` entries are integer object-slot variables (or None):
          - 0: the condition atom must refer to the same object as the clause
            anchor (`entity_key`).
          - None: the condition atom is not tied to any particular object.
          - any other integer: the condition atom must refer to some object
            other than the anchor; atoms that share the same slot number must
            resolve to the very same concrete object (variable co-reference).
        """
        slot_bindings = {0: entity_key}
        for body_index, body_signature in enumerate(body_signatures[1:], start=1):
            required_fact = self._fact_signature(body_signature)
            slot = body_scopes[body_index] if body_index < len(body_scopes) else None

            for atom_index, atom in enumerate(siss):
                if self._fact_signature(atom) != required_fact:
                    continue
                if slot is None:
                    break
                condition_entity = self._entity_key(atom, atom_index)
                bound_entity = slot_bindings.get(slot)
                if bound_entity is not None:
                    if condition_entity == bound_entity:
                        break
                    continue
                if condition_entity == entity_key:
                    continue
                slot_bindings[slot] = condition_entity
                break
            else:
                return False

        return True

    def _fired_clause_heads(self, siss):
        """Return each applicable learned clause with its entity-bound head."""
        fired_heads = []
        
        for clause_index, body_signatures in enumerate(self.body_signatures):
            if not body_signatures:
                continue

            anchor_fact = self._fact_signature(body_signatures[0])
            head_fact = self._fact_signature(self.head_signatures[clause_index])
            body_scopes = self.coarse_pruned_clauses[clause_index].get("body_scopes", [])
            for anchor_index, anchor in enumerate(siss):
                if self._fact_signature(anchor) != anchor_fact:
                    continue
                entity_key = self._entity_key(anchor, anchor_index)
                if not self._body_conditions_match(
                    body_signatures, body_scopes, entity_key, siss
                ):
                    continue

                fired_heads.append(
                    (
                        clause_index,
                        entity_key,
                        head_fact + tuple(anchor[-2:]),
                    )
                )

        return fired_heads

    def _collect_clause_proposals(self, siss):
        """Find the highest-confidence fired head per entity and predicate."""
        proposals = {}
        for clause_index, entity_key, predicted_atom in self._fired_clause_heads(siss):
            head_fact = self._fact_signature(predicted_atom)
            confidence = self.coarse_pruned_clauses[clause_index].get("confidence", 0.0)
            proposal_key = (entity_key, head_fact[1])
            existing = proposals.get(proposal_key)
            if existing is None or confidence > existing[0]:
                proposals[proposal_key] = (
                    confidence,
                    predicted_atom,
                )

        return proposals

    def _apply_clause_proposals(self, siss, proposals):
        """Replace matching entity facts with proposed heads; append missing heads."""
        soss = list(siss)
        for (entity_key, predicate), proposal in proposals.items():
            predicted_atom = proposal[1]
            matching_indices = []
            for index, atom in enumerate(soss):
                if atom[1] != predicate:
                    continue
                candidate_entity = self._entity_key(atom, index)
                if candidate_entity == entity_key:
                    matching_indices.append(index)

            if matching_indices:
                for index in matching_indices:
                    soss[index] = predicted_atom
            else:
                soss.append(predicted_atom)

        return soss

    def _visualize_predictions(self, soss, meta_data):
        """Render a one-hot action matrix for the predicted atoms."""
        prediction_matrix = torch.zeros(len(soss), self.action_num)
        for index, atom in enumerate(soss):
            prediction_matrix[index, self.atom_signature_to_index(atom)] = 1
        visual_2d_heatmap(
            prediction_matrix,
            output_dir=meta_data["output_dir"],
            title=f"Frame {meta_data['frame_id']} predictions",
            x_label="Action",
            y_label="Atoms",
            filename=f"{meta_data['video_id']}_frame_{meta_data['frame_id']}_predictions",
        )

    def predict(self, siss, meta_data=None):
        """Match complete rule bodies, resolve heads, and return next-frame atoms."""
        proposals = self._collect_clause_proposals(siss)
        soss = self._apply_clause_proposals(siss, proposals)
        if meta_data is not None:
            self._visualize_predictions(soss, meta_data)
        return soss

    def soss_to_fam(self, soss):
        # Convert the soss (predicted actions/locations) to a Frame Action Matrix (FAM)
        # soss: List of predicted actions/locations for the next frame
        # fam: Initialized Frame Action Matrix to be filled
        fam = torch.zeros(len(soss), self.action_num)
        for i, sos in enumerate(soss):
            action_index = self.atom_signature_to_index(sos)
            fam[i, action_index] = 1
        return fam

    def get_same_id_atoms(self,atom_signatures, tube_uid):
        indices = []
        for i, atom_signature in enumerate(atom_signatures):
            if atom_signature[-1]==tube_uid:
                indices.append(i)
        return indices

    def classify_transition(self, current_atoms, next_atoms):
        """Classify facts by exact membership in the current and next frames.

        Every occurrence of a fact present in both frames is unchanged.
        Duplicates do not become additions or removals merely because their
        counts differ between frames.
        """
        unchanged = [atom for atom in next_atoms if atom in current_atoms]
        return {
            "added": [atom for atom in next_atoms if atom not in current_atoms],
            "removed": [atom for atom in current_atoms if atom not in next_atoms],
            "unchanged": unchanged,
        }

    def _count_event_matches(
        self,
        all_events,
        all_events_gt,
        event_type,
        current_atoms,
        predicted_atoms,
        ground_truth_atoms,
    ):
        """Count one-to-one exact fact matches, including duplicate occurrences."""
        ground_truth_events = all_events_gt[event_type]
        predicted_events = all_events[event_type]

        ground_truth_matched = [False] * len(ground_truth_events)
        fired_heads = self._fired_clause_heads(current_atoms)
        true_positive = 0
        for predicted_event in predicted_events:
            matched = False
            for index, ground_truth_event in enumerate(ground_truth_events):
                if (
                    not ground_truth_matched[index]
                    and predicted_event == ground_truth_event 
                    and event_type in ("unchanged", "removed")
                ):
                    ground_truth_matched[index] = True
                    true_positive += 1
                    matched = True
                    break
                if (
                    not ground_truth_matched[index]
                    and predicted_event[:-1]==ground_truth_event[:-1]
                    and event_type == "added"
                ):
                    ground_truth_matched[index] = True
                    true_positive += 1
                    matched = True
                    break
            if not matched and event_type in ("added", "removed"):
                self._log_false_positive_event(
                    predicted_event,
                    all_events,
                    event_type,
                    current_atoms,
                    predicted_atoms,
                    ground_truth_atoms,
                    fired_heads,
                )
                print(f"  False positive event")

        return {
            "tp": true_positive,
            "fp": len(predicted_events) - true_positive,
            "fn": len(ground_truth_events) - true_positive,
        }

    def _rule_body_true_in_ground_truth(self, clause_index, entity_key, ground_truth_atoms):
        body_signatures = self.body_signatures[clause_index]
        if not body_signatures:
            return False

        anchor_fact = self._fact_signature(body_signatures[0])
        body_scopes = self.coarse_pruned_clauses[clause_index].get("body_scopes", [])
        for anchor_index, anchor in enumerate(ground_truth_atoms):
            if self._fact_signature(anchor) != anchor_fact:
                continue
            if (
                entity_key[0] != "atom"
                and not self._matches_entity_key(anchor, entity_key)
            ):
                continue
            ground_truth_entity = self._entity_key(anchor, anchor_index)
            if self._body_conditions_match(
                body_signatures, body_scopes, ground_truth_entity, ground_truth_atoms
            ):
                return True
        return False

    def _matched_body_atoms(self, clause_index, entity_key, atoms):
        """Return current-frame atoms satisfying each literal of a fired clause."""
        body_signatures = self.body_signatures[clause_index]
        if not body_signatures:
            return []

        anchor_fact = self._fact_signature(body_signatures[0])
        body_scopes = self.coarse_pruned_clauses[clause_index].get("body_scopes", [])
        for anchor_index, anchor in enumerate(atoms):
            if self._fact_signature(anchor) != anchor_fact:
                continue
            if not self._matches_entity_key(anchor, entity_key, anchor_index):
                continue

            matched_atoms = [anchor]
            for body_index, body_signature in enumerate(body_signatures[1:], start=1):
                required_fact = self._fact_signature(body_signature)
                scope = (
                    body_scopes[body_index]
                    if body_index < len(body_scopes)
                    else "scene"
                )
                if scope not in ("entity", "scene"):
                    raise ValueError(f"Unknown extended-clause body scope: {scope}")

                for atom_index, atom in enumerate(atoms):
                    if self._fact_signature(atom) != required_fact:
                        continue
                    same_entity = (
                        self._entity_key(atom, atom_index) == entity_key
                    )
                    if (scope == "entity" and same_entity) or (
                        scope == "scene" and not same_entity
                    ):
                        matched_atoms.append(atom)
                        break
                else:
                    break
            else:
                return matched_atoms

        return []

    def _log_false_positive_event(
        self,
        predicted_event,
        all_events,
        event_type,
        current_atoms,
        predicted_atoms,
        ground_truth_atoms,
        fired_heads,
    ):
        """Print prediction provenance and compare its body with the GT frame."""
        print(f"[Event mismatch: {event_type}]")
        print(f"[predicted {event_type}]: {predicted_event}")
        present_current = predicted_event in current_atoms
        present_ground_truth_next = predicted_event in ground_truth_atoms
        if present_current and present_ground_truth_next:
            ground_truth_status = "unchanged"
        elif present_ground_truth_next:
            ground_truth_status = "added"
        elif present_current:
            ground_truth_status = "removed"
        else:
            ground_truth_status = "absent"
        print(f"[GT {ground_truth_status}] {predicted_event}")

        event_entity = None
        if event_type == "removed":
            for current_index, atom in enumerate(current_atoms):
                if atom == predicted_event:
                    event_entity = self._entity_key(atom, current_index)
                    break
            replacement_facts = [
                (atom, atom_index)
                for atom_index, atom in enumerate(predicted_atoms)
                if atom[1] == predicted_event[1]
                and self._entity_key(atom, atom_index) == event_entity
                and self._fact_signature(atom) != self._fact_signature(predicted_event)
            ]
        else:
            replacement_facts = [
                (atom, atom_index)
                for atom_index, atom in enumerate(predicted_atoms)
                if self._fact_signature(atom) == self._fact_signature(predicted_event)
            ]
            if replacement_facts:
                replacement, replacement_index = replacement_facts[0]
                for current_index, atom in enumerate(current_atoms):
                    if self._matches_entity_key(
                        atom,
                        self._entity_key(replacement, replacement_index),
                        current_index,
                    ):
                        event_entity = self._entity_key(atom, current_index)
                        break

        relevant_rules = [
            (clause_index, entity_key, head)
            for clause_index, entity_key, head in fired_heads
            if (
                event_entity is None
                or entity_key == event_entity
            )
            and any(
                self._fact_signature(head) == self._fact_signature(replacement)
                for replacement, _ in replacement_facts
            )
        ]
        if not relevant_rules:
            print("  Fired rule: no matching learned clause head was found.")
            return

        clause_index, entity_key, head = max(
            relevant_rules,
            key=lambda fired_rule: self.coarse_pruned_clauses[fired_rule[0]].get(
                "confidence", 0.0
            ),
        )
        clause = self.coarse_pruned_clauses[clause_index]
        print(f"\nFired rule #{clause_index} (confidence={clause.get('confidence', 0.0):.4f})")
        for body_i, body in enumerate(clause["bodies"]):
            print(f"[{clause['body_scopes'][body_i]}]  Body: {body}")
        print(f"Head: {head}")
        for body_index, body_atom in enumerate(
            self._matched_body_atoms(clause_index, entity_key, current_atoms)
        ):
            print(f"  Current-frame body atom #{body_index}: {body_atom}")
        print(
            "  Full body true in ground truth="
            f"{self._rule_body_true_in_ground_truth(clause_index, entity_key, ground_truth_atoms)}"
        )

    def stat_pred_frames_atoms(self, siss, soss, soss_gt):
        predicted_events = self.classify_transition(siss, soss)
        ground_truth_events = self.classify_transition(siss, soss_gt)
        # print all the duplicate tube_uid atoms in the soss_gt by tube_uids
        tube_uids = set([atom[-1] for atom in soss_gt])
        for tube_uid in tube_uids:
            duplicate_atoms = [atom for atom in soss_gt if atom[-1] == tube_uid]
            if len(duplicate_atoms) > 1:
                for atom in duplicate_atoms:
                    print(f"{atom}")
                print(f"\n\n")
                

        return {
            event_type: self._count_event_matches(
                predicted_events,
                ground_truth_events,
                event_type,
                siss,
                soss,
                soss_gt,
            )
            for event_type in ("added", "removed", "unchanged")
        }

    @staticmethod
    def _scores_from_counts(counts):
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        )
        return {**counts, "precision": precision, "recall": recall, "f1": f1}

    def _rule_matches(self, siss, soss, soss_gt):
        """Count fired rules with correct heads and ground-truth transitions missed."""
        fired_heads = self._fired_clause_heads(siss)

        correct_fired_rule_count = sum(
            any(
                self._matches_entity_key(ground_truth_atom, entity_key)
                and self._fact_signature(ground_truth_atom)
                == self._fact_signature(predicted_atom)
                for ground_truth_atom in soss_gt
            )
            for _, entity_key, predicted_atom in fired_heads
        )

        predicted_transitions = self.classify_transition(siss, soss)["added"]
        ground_truth_transitions = self.classify_transition(siss, soss_gt)["added"]
        matched_ground_truth = [False] * len(ground_truth_transitions)
        predicted_transition_count = 0
        for predicted_atom in predicted_transitions:
            for ground_truth_index, ground_truth_atom in enumerate(ground_truth_transitions):
                if (
                    not matched_ground_truth[ground_truth_index]
                    and self._same_identified_entity(predicted_atom, ground_truth_atom)
                    and self._fact_signature(predicted_atom)
                    == self._fact_signature(ground_truth_atom)
                ):
                    matched_ground_truth[ground_truth_index] = True
                    predicted_transition_count += 1
                    break

        fired_rule_count = len(fired_heads)
        transition_count = len(ground_truth_transitions)
        return {
            "fired_rule_count": fired_rule_count,
            "correct_fired_rule_count": correct_fired_rule_count,
            "incorrect_fired_rule_count": fired_rule_count - correct_fired_rule_count,
            "rule_accuracy": (
                correct_fired_rule_count / fired_rule_count
                if fired_rule_count
                else None
            ),
            "transition_count": transition_count,
            "predicted_transition_count": predicted_transition_count,
            "unpredicted_transition_count": transition_count - predicted_transition_count,
            "transition_recall": (
                predicted_transition_count / transition_count
                if transition_count
                else None
            ),
        }

    def eval(self, siss, soss, soss_gt, meta_data=None):
        counts = self.stat_pred_frames_atoms(siss, soss, soss_gt)
        rule_matches = self._rule_matches(siss, soss, soss_gt)
        return {
            "events": {
                event_type: self._scores_from_counts(event_counts)
                for event_type, event_counts in counts.items()
            },
            "rule_matches": rule_matches,
        }
        
        