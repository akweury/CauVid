import torch
from scipy.optimize import linear_sum_assignment
from src.exp_roadpp import utils_data



class CauVid:
    def __init__(self, cfg):
        self.clauses = utils_data.load_json(cfg["extended_clauses_file"])
        self.unique_atoms = utils_data.load_json(cfg["unique_atoms_file"])
        self.delay_windows = cfg["delay_windows"]
        self.clauses = self._unique_clauses()

        self.clause_matrices = self._build_clause_matrix()
        self.threshold = cfg["threshold"]
    def _unique_clauses(self):
        unique_clauses = []
        seen = set()
        for c in self.clauses:
            c_tuple = (tuple(c["head"]), tuple(c["body_scopes"]), tuple(tuple(b) for b in c["bodies"]))
            if c_tuple not in seen:
                seen.add(c_tuple)
                unique_clauses.append(c)
        print(f'(Found {len(self.clauses) - len(unique_clauses)} repeat clauses, removed them.)')
        print()

        return unique_clauses

    
    def _reorder_clauses(self, clauses):
        reordered_clauses = []
        for c in clauses:
            reordered_body = []
            for b in c["bodies"]:
                assert len(b) == 3, "Each body should have exactly 3 elements"
                if b[2] == "now":
                    window = 0
                elif "appeared" in b[2]:
                    window = int(b[2].split("_")[2])
                else:
                    raise ValueError(f"Unknown window type: {b[2]}")
                reordered_body.append([b[1],window,b[0]])
            reordered_c = {
                "head": c["head"],
                "body_scopes": c["body_scopes"],
                "bodies": reordered_body
            }
            reordered_clauses.append(reordered_c)

        return reordered_clauses
    def _build_clause_matrix(self):
        self.clauses = self._reorder_clauses(self.clauses)
        clause_matrices = []
        for c in self.clauses:
            scope_size = max(c["body_scopes"])+1
            window_size = max([b[1] for b in c["bodies"]])+1
            matrix = torch.zeros((scope_size, window_size, len(self.unique_atoms)))
            for b_i, body in enumerate(c["bodies"]):
                matrix[body[0], body[1], body[2]] = True
            clause_matrices.append(matrix)
        return clause_matrices
    
    def predict(self, task_input):
        # -------- substep functions ----------------------------------
        def _compress_task_matrix(task_matrix):
            return task_matrix.sum(dim=1)
        
        def _pad_rows(m, obj):
            if m.shape[0] < obj:
                padding = torch.zeros((obj - m.shape[0], m.shape[1]), dtype=m.dtype)
                m = torch.cat([m, padding], dim=0)
            return m

        def _fit_clause_to_task(c_compressed, task_compressed):
            # Best object grounding = max-weight assignment between task objects and clause scopes
            # (equivalent to trying all permutations, but O(obj^3) instead of O(obj!))
            obj = max(c_compressed.shape[0], task_compressed.shape[0])
            c = _pad_rows(c_compressed, obj)
            t = _pad_rows(task_compressed, obj).to(c.dtype)
            # match[i, j] = number of clause-j atoms satisfied by task object i
            # (an atom required k times is satisfied at most k times)
            match = torch.minimum(t[:, None, :], c[None, :, :]).sum(dim=-1)
            rows, cols = linear_sum_assignment(match.numpy(), maximize=True)
            satisfied = match[rows, cols].sum().item()
            total = c.sum().item()
            return satisfied / total if total > 0 else 0.0
        
        def _build_task_matrix(task_input):
            task_window = len(task_input)
            obj_ids = list(set(atom[-1] for frame in task_input for atom in frame))
            task_obj_num = len(obj_ids)
            atom_num = len(self.unique_atoms)
            task_matrix = torch.zeros((task_obj_num, task_window,atom_num), dtype=torch.bool) 
            for f_i, frame in enumerate(task_input):
                for atom in frame:
                    atom_idx = atom[0]
                    obj_idx = obj_ids.index(atom[-1])
                    task_matrix[obj_idx, f_i, atom_idx] = True
            return task_matrix
        
        def _compute_clause_score(clause_matrices, task_matrix):
            clause_matches = {}
            task_compressed = _compress_task_matrix(task_matrix)
            for c_i, c_matrix in enumerate(clause_matrices):
                best_score = _fit_clause_to_task(c_matrix.sum(dim=1), task_compressed)
                clause_matches[c_i] = best_score
            return clause_matches

        def _select_matching_clauses(clause_matches):
            selected_clauses = []
            for c_i, score in clause_matches.items():
                if score >= self.threshold:
                    head_atom = self.unique_atoms[self.clauses[c_i]["head"][0]]
                    if "action_id" in head_atom:
                        selected_clauses.append({
                            'score': score,
                            "pred": head_atom[1],
                            "action_id": head_atom[3],
                            "action_id_label": head_atom[5],
                            "agent_class": head_atom[7],
                            "agent_class_label": head_atom[9]
                        })
                    elif "location" in head_atom:
                        selected_clauses.append({
                            'score': score,
                            "pred": head_atom[1],
                            "location_name": head_atom[3],
                            "location_name_label": head_atom[5],
                            "agent_class": head_atom[7],
                            "agent_class_label": head_atom[9]
                        })
            return selected_clauses
        
        # -------- main prediction steps ----------------------------------
        task_matrix = _build_task_matrix(task_input)
        # clauses are last for n frames as temporal clauses
        # the input can match part of the temporal clauses
        # if the input match the first few frames of a temporal clause, it can still be considered a match, 
        # and we can estimate the delay of the head atom accordingly.
        clause_matches = _compute_clause_score(self.clause_matrices, task_matrix)
        selected_clauses = _select_matching_clauses(clause_matches)
        return selected_clauses

        
    

class TemporalRuleNet(torch.nn.Module):
    """Layer-per-frame rule network: predicts which atoms switch on in frame t
    from the K frames before it.

    Every agent runs the same weights. The frames t-1..t-K are grouped into B lag
    bins (lo, hi], e.g. edges (1, 2, 4, 8, ..., 100) -> bins 1, 2, 3-4, 5-8, ...,
    65-100 frames ago. Each bin is one input layer with two groups of A atom
    nodes: slot 0 "self" (the agent holds the atom in some frame of the bin) and
    slot 1 "others" (some other agent holds it in some frame of the bin), so
    M = 2 * B * A input nodes. R rule nodes are soft ANDs over a sparse
    hard-concrete mask of the inputs; the bins of a rule's inputs give the order
    of its body atoms. Each head atom is a weighted OR of the rule nodes.
    """

    def __init__(self, num_atoms, lag_edges=(1, 2, 4, 8, 16, 32, 64, 100), num_rules=256, init_log_alpha=-6.0):
        super().__init__()
        self.num_atoms = num_atoms
        # Bin i covers lo_i..hi_i frames ago, hi_i = lag_edges[i], lo_i = hi_{i-1} + 1.
        self.lag_bins = [(lo + 1, hi) for lo, hi in zip((0, *lag_edges[:-1]), lag_edges)]
        self.num_lags = lag_edges[-1]
        self.num_bins = len(self.lag_bins)
        self.num_inputs = 2 * self.num_bins * num_atoms
        # Very negative so only a few random inputs per rule start non-zero.
        self.log_alpha = torch.nn.Parameter(torch.full((num_rules, self.num_inputs), init_log_alpha))
        self.head_weight = torch.nn.Parameter(0.01 * torch.randn(num_atoms, num_rules))
        self.head_bias = torch.nn.Parameter(torch.full((num_atoms,), -4.0))
        # Inputs always in their rule (mask 1, no sparsity cost): the seeded anchors.
        self.register_buffer("fixed_inputs", torch.zeros((num_rules, self.num_inputs), dtype=torch.bool))

    def input_index(self, atom, slot, bin_index):
        """Input node of `decode_input`'s (atom, slot, lag bin); slots > 1 are "others"."""
        return min(slot, 1) * self.num_bins * self.num_atoms + bin_index * self.num_atoms + atom

    @torch.no_grad()
    def seed_rules(self, anchors, head_weight=1.0):
        """Seed one rule node per (anchor body, lag bin) from the anchor clauses.

        anchors: [((atom, slot), [head atom])]. Rule i * B + k holds the body atom
        in lag bin k as a fixed input, and starts with weight head_weight to the
        anchor's heads and 0 to the others; training adds the extra body atoms.
        Returns the number of seeded rules, the first ones of the R rule nodes.
        """
        num_seeded = len(anchors) * self.num_bins
        if num_seeded > self.log_alpha.shape[0]:
            raise ValueError(f"{num_seeded} seeded rules do not fit into {self.log_alpha.shape[0]} rule nodes.")
        for i, ((atom, slot), heads) in enumerate(anchors):
            for k in range(self.num_bins):
                r = i * self.num_bins + k
                self.fixed_inputs[r, self.input_index(atom, slot, k)] = True
                self.head_weight[:, r] = 0.0
                self.head_weight[list(heads), r] = head_weight
        return num_seeded

    # -------- inputs and targets of one video ----------------------------------
    def layer_inputs(self, X, frames):
        """(F * N, M) float input layers of every agent at each target frame.

        X: (T, N, A) bool atom tensor of `step05.build_video_tensor`; frames: (F,)
        long target frames t (may be T, one past the video, to predict from it).
        Row f * N + n is agent n at frames[f]; frames before 0 are empty layers.
        """
        K, B, A = self.num_lags, self.num_bins, self.num_atoms
        N = X.shape[1]
        past = frames[:, None] - torch.arange(1, K + 1, device=X.device)   # (F, K) lag k = column k - 1
        own = X[past.clamp(min=0)] & (past >= 0)[..., None, None]             # (F, K, N, A)
        own = torch.stack([own[:, lo - 1:hi].any(dim=1) for lo, hi in self.lag_bins], dim=1)  # (F, B, N, A)
        holders = own.sum(dim=2, keepdim=True)
        others = (holders - own.int()) > 0
        inputs = torch.stack([own, others], dim=1)                            # (F, 2, B, N, A)
        return inputs.permute(0, 3, 1, 2, 4).reshape(len(frames) * N, 2 * B * A).float()

    def training_rows(self, X, frames, min_off_frames=1, horizon=1):
        """Inputs and onset targets of the agents visible at each target frame.

        Returns inputs (B, M) float; targets (B, A) bool, the at-risk atoms that
        switch on (false -> true) for the agent in one of the frames t..t + horizon - 1
        (horizon 1: exactly at t); at_risk (B, A) bool, the atoms off in all of
        t - min_off_frames..t - 1, the only ones that can switch on (an atom back on
        after a shorter gap is a re-onset, not a new event); rows (B, 2) long, (t, n).
        """
        N = X.shape[1]
        inputs = self.layer_inputs(X, frames)
        t = frames.repeat_interleave(N)
        n = torch.arange(N, device=X.device).repeat(len(frames))
        keep = X[t, n].any(dim=1)
        t, n, inputs = t[keep], n[keep], inputs[keep]
        past = t[:, None] - torch.arange(1, min_off_frames + 1, device=X.device)     # (B, k)
        held = (X[past.clamp(min=0), n[:, None]] & (past >= 0)[..., None]).any(dim=1)
        at_risk = ~held
        targets = at_risk & self.future_onsets(X, t, n, horizon, min_off_frames)
        return inputs, targets, at_risk, torch.stack([t, n], dim=1)

    @staticmethod
    def future_onsets(X, t, n, horizon, min_off_frames):
        """(B, A) bool: the atom switches on for agent n in some frame f of
        t..t + horizon - 1: on in f, off in f - min_off_frames..f - 1, and the agent
        visible in f - 1 (not an agent entering the scene)."""
        T = X.shape[0]
        frames = t[:, None] + torch.arange(-min_off_frames, horizon, device=X.device)     # (B, L)
        window = X[frames.clamp(0, T - 1), n[:, None]] & ((frames >= 0) & (frames < T))[..., None]
        onsets = torch.zeros((len(t), X.shape[2]), dtype=torch.bool, device=X.device)
        for i in range(min_off_frames, min_off_frames + horizon):      # window index of f
            held = window[:, i - min_off_frames:i].any(dim=1)
            seen = window[:, i - 1].any(dim=1, keepdim=True)
            onsets |= window[:, i] & ~held & seen
        return onsets

    def hard_concrete_mask(self, log_alpha, beta=2 / 3, gamma=-0.1, zeta=1.1, training=True):
        if training:
            u = torch.rand_like(log_alpha).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / beta)
        else:
            s = torch.sigmoid(log_alpha)
        s_bar = s * (zeta - gamma) + gamma
        return s_bar.clamp(0, 1)
    # -------- forward ----------------------------------
    def rule_mask(self):
        """(R, M) mask in [0, 1]: stochastic while training, deterministic in eval;
        1 for the fixed inputs."""
        mask = self.hard_concrete_mask(self.log_alpha, training=self.training)
        return mask.masked_fill(self.fixed_inputs, 1.0)

    def forward(self, inputs):
        """Head logits (B, A) and rule activations (B, R) of input layers (B, M).

        A rule is the product over its masked inputs of (1 - mask * (1 - x)): a
        missing input with mask 1 turns the rule off. In the log domain this is one
        matmul, (1 - x) @ log(1 - mask).T.
        """
        mask = self.rule_mask().clamp(max=1 - 1e-6)
        rules = torch.exp((1 - inputs) @ torch.log1p(-mask).T)
        logits = rules @ self.head_weight.T + self.head_bias
        return logits, rules

    def loss(self, logits, targets, at_risk, pos_weight=20.0):
        """Mean binary cross-entropy over the at-risk atoms; onsets weighted up."""
        weight = torch.where(targets, pos_weight, 1.0) * at_risk
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        return (bce * weight).sum() / weight.sum().clamp(min=1)

    def sparsity_loss(self, beta=2 / 3, gamma=-0.1, zeta=1.1):
        """Expected number of active inputs per rule (L0 of the hard-concrete mask),
        the fixed inputs left out, plus the L1 of the head weights."""
        shift = beta * torch.log(torch.tensor(-gamma / zeta))
        l0 = (torch.sigmoid(self.log_alpha - shift) * ~self.fixed_inputs).sum(dim=1).mean()
        return l0 + self.head_weight.abs().sum(dim=1).mean()

    # -------- use after training ----------------------------------
    @torch.no_grad()
    def predict(self, X):
        """Onset probability (N, A) of each atom for each agent in the frame after X.

        Agents not visible in the last frame of X get zeros.
        """
        T, N, _A = X.shape
        inputs = self.layer_inputs(X, torch.tensor([T], device=X.device))
        probabilities = torch.sigmoid(self(inputs)[0])
        return probabilities * X[-1].any(dim=1, keepdim=True)

    @torch.no_grad()
    def partial_activation(self, inputs, threshold=0.5):
        """(B, R) share of each rule's body inputs that hold, in [0, 1]: a rule with
        3 of its 4 body atoms seen is 0.75 active."""
        body = (self.rule_mask() > threshold).float()
        return (inputs @ body.T) / body.sum(dim=1).clamp(min=1)

    def decode_input(self, m):
        """Input node m -> (atom, slot, lo, hi): slot 0 self, 1 others; the atom
        holds lo..hi frames before the head."""
        B, A = self.num_bins, self.num_atoms
        lo, hi = self.lag_bins[(m // A) % B]
        return m % A, m // (B * A), lo, hi

    @torch.no_grad()
    def extract_rules(self, mask_threshold=0.5, weight_threshold=0.1):
        """Rules [{"rule": r, "bodies": [(atom, slot, lo, hi)], "heads": [(atom, weight)]}].

        Bodies are sorted earliest first (largest lag bin), the order the atoms happen.
        """
        body = self.rule_mask() > mask_threshold
        rules = []
        for r in range(body.shape[0]):
            inputs = torch.nonzero(body[r]).flatten().tolist()
            if not inputs:
                continue
            heads = torch.nonzero(self.head_weight[:, r] > weight_threshold).flatten().tolist()
            if not heads:
                continue
            bodies = sorted((self.decode_input(m) for m in inputs), key=lambda b: -b[2])
            rules.append({
                "rule": r,
                "bodies": bodies,
                "heads": [(h, self.head_weight[h, r].item()) for h in heads],
            })
        return rules
