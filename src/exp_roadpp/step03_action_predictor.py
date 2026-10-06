
import torch 


from src.exp_roadpp.utils_data import atom_to_signature, add_labels_to_atom, same_signature_no_tube_uid, build_signature


class ActionPredictor:
    def __init__(self, coarse_pruned_clauses,dataset_labels):
        self.device = "cpu"
        self.coarse_pruned_clauses = coarse_pruned_clauses
        self.head_signatures = [(atom_to_signature(clause[0]["clause"]["head"]), clause[1]) for clause in coarse_pruned_clauses]
        self.body_signatures = [(atom_to_signature(clause[0]["clause"]["body"][0]), clause[1]) for clause in coarse_pruned_clauses]
        self.dataset_labels = dataset_labels
        self.agent_action_num = len(self.dataset_labels["action_labels"])
        self.av_action_num = len(self.dataset_labels["av_action_labels"])
        self.location_name_num = len(self.dataset_labels["loc_labels"])
        self.action_num = self.agent_action_num + self.av_action_num + self.location_name_num
        

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
    
    def connect_pred_to_tube_uid(self, siss, fim):
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

    
    def predict(self, siss):
        """
        AIM:    Atom Influence Matrix, a NxM matrix representing the influence of a single atom on all atoms in the next frame.
        FIM:    Frame Influence Matrix, a NxNxM matrix representing the influence of all atoms in the current frame on all atoms in the next frame.
        IV:     Influence Vector, a Nx1 vector representing the influence of a single frozen head signature on all atoms in the next frame.
        FHS:    Frozen Head Signature, a representation of a frozen head that can influence the next frame.
        FBS:    Frozen Body Signature, a representation of a frozen body that can match with frozen head signatures to influence the next frame.
        SIS:    Scene Input Signature, a representation of a single atom in the current frame.
        SOS:    Scene Output Signature, a representation of the predicted action/location for a single atom in the next frame.

        Predict the actions/locations for the next frame based on the current frame's atoms.

        Each atom in as_in, should produce a influnce to every atom in as_in, and that influnce determines the predicted actions/locations for the next frame.
        The influnce of one atom to others can be represented as a 2D matrix, 
        where each row corresponds to an atom in the current frame and 
        each column corresponds to a possible action/location for the next frame.

        Let N be the number of atoms in the current frame,
        and M be the number of possible actions/locations for the next frame.
        For the influence of all the atoms, we can use a 3D matrix of shape (N, N, M), 
        where the entry at (i, j, k) represents the influence of the i-th atom in the current frame 
        on the j-th atom in the next frame for the k-th possible action/location.
        For each atom's matrix NxM, it is named as the atom's influence matrix, AIM.
        For the AIM of all the atoms, named as the frame influence matrix, FIM.

        To build this matrix, we iterate over each atom in the current frame, 
        if the atom matches any of the frozen body signatures, the corresponding frozen head signatures should give a influnce to the AIM, a N x M matrix.

        How to calculate the AIM:
        for each valid FHS (Frozen Head Signature), we need to determine which atoms in the next frame it influences.
        The action of the FHS is determined, thus it is a Nx1 vector indicating the influence of the FHS on each atom in the next frame. 
        The Nx1 vector is denoted as IV (Influence Vector).
        Assume there are K valid FHSs for the current atom, then the AIM can be calculated by aggregating the IV of all K FHSs.
        In other words, the AIM is updated K times, once for each valid FHS.
        The IV is the core measurement for determining which atoms is influnced by the FHS.
        In short, to calculate AIM, we calculate IV from K valid FHSs and aggregate them.

        How to calculate the FIM (Frame Influence Matrix):
        The FIM is obtained by stacking the AIMs of all atoms in the current frame.
        If there are N atoms in the current frame, the FIM will have a shape of (N, N, M), 
        where each slice along the first dimension corresponds to the AIM of a particular atom.

        Returns:
            as_out: List of predicted actions/locations for the next frame based on the current frame's atoms.
        """

        # frame influence matrix (FIM) initialization
        fim = torch.zeros(len(siss),len(siss), self.action_num)
        fbs = self.body_signatures
        fhs = self.head_signatures
        
        # calculate the AIM iteratively for each atom in the current frame
        for sis_index, sis in enumerate(siss):
            matched_fbs_indices = self.find_matching_bodies(sis, fbs)
            matched_fhs = self.find_matching_head_signatures(matched_fbs_indices, fhs)
            for k in range(len(matched_fhs)):
                iv, label_index = self.calc_influence_vector(siss, matched_fhs[k])
                fim[sis_index, :, label_index] += iv.squeeze()
        
        soss = self.connect_pred_to_tube_uid(siss, fim)
        print(f"FIM SUM: {fim.sum().item()}")
        return soss,fim





    def eval(self, pred, labels):
        # Implement the evaluation logic for the predictions
        pass