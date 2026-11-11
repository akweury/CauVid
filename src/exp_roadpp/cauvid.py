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
    """Slot rule network: predicts, for every agent visible in frame t, how likely
    each atom switches on within the next W frames (W = 10, 20, ..., one window
    per future slot), from whatever part of the last num_lags (slot_size * num_slots) frames is known.

    Time is cut into slots of slot_size (10) frames: the memory holds the
    num_slots (5) past slots p = 1..5 (p = 1: the 10 frames before t), the
    future the slots f = 0..F-1 (f = 0: frames t..t+9), one per window.

    Rules are temporal patterns: rule r is a soft AND of body inputs (atom, slot
    type, j): the agent itself (slot type 0) or some other agent (1) holds the
    atom in the slot j = 1..num_slots before the head's slot. To predict the head
    in future slot f the rule is aligned so that its head lies in f: a body with
    j > f lies in past slot p = j - f and is looked up, a body with j <= f has not
    happened yet and is unknown. Every input of the memory is three-valued:
    true (seen), false (its slot fully observed and the atom absent) or unknown
    (a slot older than the known history, e.g. before the video start or after a
    truncation). A body contributes
      true: 1,   false: 1 - m,   unknown: 1 - m (1 - pi),
    with m its mask and pi = sigmoid(unknown_logit) the learned chance that an
    unknown input holds; the rule activation is the product over its inputs, the
    chance that its whole body holds given what is known. So the confidence of a
    rule rises as more of its bodies are confirmed true, falls to 0 when one is
    confirmed false, and stays in between while they are unknown: a far-ahead
    head (large f) relies on the early bodies only, and its confidence grows as
    time passes and the later bodies come true.

    The hazard of future slot f and atom a is sigmoid(sum_r w_ar rule_r(f) + b_fa);
    the probability of an onset within W_k = (k + 1) * slot_size frames is
    1 - prod_{f <= k} (1 - h_fa), never decreasing with the window.

    Inputs are packed as (B, 2M): the truth values x of the M = scopes * num_slots * A
    memory inputs, then their known flags (`split_inputs`). Scopes: 0 the agent
    itself, 1 some other agent, 2.. one per role of `roles` (e.g. "ahead": the
    agent ahead of it, `roles.role_partners`), so rules can name a specific
    related agent instead of any agent of the class.

    mask_type "hard_concrete": a stochastic L0 mask while training. "sigmoid": the
    deterministic mask sigmoid(log_alpha) of `LayeredRuleNet`, started near 0 so
    every input gets a gradient, kept sparse by an L1 loss.
    """

    def __init__(self, num_atoms, slot_size=10, num_slots=5, num_rules=256, init_log_alpha=-6.0,
                 mask_type="hard_concrete", windows=(10,), roles=(), input_mask=None, require_self=False):
        super().__init__()
        if mask_type not in ("hard_concrete", "sigmoid"):
            raise ValueError(f"Unknown mask_type {mask_type}.")
        windows = list(windows)
        if windows != [slot_size * (f + 1) for f in range(len(windows))]:
            raise ValueError(f"windows {windows} must be slot_size, 2 * slot_size, ... (slot_size {slot_size}).")
        self.mask_type = mask_type
        self.num_atoms = num_atoms
        self.slot_size, self.num_slots = slot_size, num_slots
        self.num_bins = num_slots                      # rule seeds per anchor (one per slot offset j)
        self.num_lags = slot_size * num_slots          # frames of memory
        self.roles = tuple(roles)
        self.num_scopes = 2 + len(self.roles)
        self.num_inputs = self.num_scopes * num_slots * num_atoms
        self.windows = windows
        num_future = len(windows)
        # Very negative so only a few random inputs per rule start non-zero.
        self.log_alpha = torch.nn.Parameter(torch.full((num_rules, self.num_inputs), init_log_alpha))
        self.unknown_logit = torch.nn.Parameter(torch.full((self.num_inputs,), -4.0))
        self.head_weight = torch.nn.Parameter(0.01 * torch.randn(num_atoms, num_rules))
        self.head_bias = torch.nn.Parameter(torch.full((num_future, num_atoms), -6.0))
        # Inputs always in their rule (mask 1, no sparsity cost): the seeded anchors.
        self.register_buffer("fixed_inputs", torch.zeros((num_rules, self.num_inputs), dtype=torch.bool))
        # (M,) bool: the memory inputs a rule may learn (None: all), e.g. not the grid position
        # atoms of the other agents (`step05.rule_input_mask`). Not saved: rebuilt from the data.
        if input_mask is not None and input_mask.shape != (self.num_inputs,):
            raise ValueError(f"input_mask {tuple(input_mask.shape)} does not match the {self.num_inputs} inputs.")
        self.register_buffer("input_mask", torch.ones(self.num_inputs, dtype=torch.bool) if input_mask is None
                             else input_mask.bool().clone(), persistent=False)
        # require_self: every rule must hold an atom of the agent itself (scope 0). Its activation is
        # multiplied by the self gate, the chance that at least one of its own-agent body atoms holds
        # (`forward`): ~1 for a rule with own-agent atoms (they all hold when it fires), ~0 for a rule
        # of the other agents only, which would otherwise fire for every agent of the scene alike.
        self.require_self = require_self
        self.register_buffer("self_inputs", torch.arange(self.num_inputs) < num_slots * num_atoms,
                             persistent=False)
        # Slot offset j of every input; and for each future slot f, the memory input each rule
        # input (atom, slot type, j) reads (past slot j - f), or M (unknown) for j <= f.
        j = (torch.arange(self.num_inputs) // num_atoms) % num_slots + 1
        self.register_buffer("input_offset", j)
        shift = []
        for f in range(min(num_future, num_slots)):
            shift.append(torch.where(j > f, torch.arange(self.num_inputs) - f * num_atoms, self.num_inputs))
        self.register_buffer("shift_index", torch.stack(shift))

    def input_index(self, atom, slot, offset):
        """Input node of `decode_input`'s (atom, slot type (scope), slot offset j = 1..num_slots)."""
        if not 0 <= slot < self.num_scopes:
            raise ValueError(f"Scope {slot} outside 0..{self.num_scopes - 1}.")
        return slot * self.num_slots * self.num_atoms + (offset - 1) * self.num_atoms + atom

    def split_inputs(self, inputs):
        """Packed inputs (B, 2M) -> truth values x (B, M), known flags k (B, M)."""
        return inputs[:, :self.num_inputs], inputs[:, self.num_inputs:]

    @torch.no_grad()
    def seed_rules(self, anchors, head_weight=1.0):
        """Seed one rule node per (anchor body, slot offset j) from the anchor clauses.

        anchors: [((atom, slot), [head atom])]. Rule i * J + (j - 1) holds the body
        atom j slots before the head as a fixed input, and starts with weight
        head_weight to the anchor's heads and 0 to the others; training adds the
        extra body atoms.
        Returns the number of seeded rules, the first ones of the R rule nodes.
        """
        num_seeded = len(anchors) * self.num_slots
        if num_seeded > self.log_alpha.shape[0]:
            raise ValueError(f"{num_seeded} seeded rules do not fit into {self.log_alpha.shape[0]} rule nodes.")
        for i, ((atom, slot), heads) in enumerate(anchors):
            for j in range(1, self.num_slots + 1):
                r = i * self.num_slots + j - 1
                # The anchors name the head agent (0) or other agents (>= 1: any other agent).
                self.fixed_inputs[r, self.input_index(atom, min(slot, 1), j)] = True
                self.head_weight[:, r] = 0.0
                self.head_weight[list(heads), r] = head_weight
        return num_seeded

    # -------- inputs and targets of one video ----------------------------------
    def layer_inputs(self, X, frames, history=None, roles=None, keep=None):
        """(F * N, 2M) packed memory inputs of every agent at each target frame.

        X: (T, N, A) bool atom tensor of `step05.build_video_tensor`; frames: (F,)
        long target frames t (may be T, one past the video, to predict from it);
        history: (F,) long, how many frames before t are known (default all of
        them, t; capped at t). Row f * N + n is agent n at frames[f]. An input is
        true if the atom was seen in a known frame of its slot, false if the slot
        is fully known and it was not, unknown otherwise.
        Scopes (slot types): 0 the agent itself, 1 some other agent, then one per
        role of self.roles: the atoms of the agent filling that role for this agent
        in frame t - 1 (the last observed frame), read over the whole memory.
        roles (T, N, K) long (`roles.role_partners`, -1: none) is needed if the
        model has roles; an agent without a partner gets false (where known).
        keep (F, N) bool: if given, only the rows of these (frame, agent) pairs, in the
        same order (row f * N + n of the full output, the kept ones), saving memory.
        """
        J, S, A = self.num_slots, self.slot_size, self.num_atoms
        N, F = X.shape[1], len(frames)
        history = frames if history is None else torch.minimum(history.to(frames.device), frames)
        lags = torch.arange(1, J * S + 1, device=X.device)
        past = frames[:, None] - lags                                          # (F, L)
        observed = lags[None, :] <= history.clamp(min=0)[:, None]              # (F, L), implies past >= 0
        # The atoms seen in each slot, one slot at a time ((F, S, N, A) at most, not (F, L, N, A)).
        own = torch.zeros((F, J, N, A), dtype=torch.bool, device=X.device)
        for j in range(J):
            lag = slice(j * S, (j + 1) * S)
            own[:, j] = (X[past[:, lag].clamp(min=0)] & observed[:, lag, None, None]).any(dim=1)
        holders = own.sum(dim=2, keepdim=True)
        others = (holders - own.int()) > 0
        channels = [own, others]
        if self.roles:
            if roles is None:
                raise ValueError(f"This model reads the agent roles {self.roles}: pass roles (T, N, K).")
            last = frames - 1
            partner = roles.to(X.device)[last.clamp(min=0)]                     # (F, N, K)
            partner = torch.where((last >= 0)[:, None, None], partner, -1)
            for k in range(len(self.roles)):
                p = partner[:, :, k]                                           # (F, N)
                index = p.clamp(min=0)[:, None, :, None].expand(F, J, N, A)
                channels.append(own.gather(2, index) & (p >= 0)[:, None, :, None])
        x = torch.stack(channels, dim=1)                                       # (F, scopes, J, N, A)
        slot_known = (torch.arange(1, J + 1, device=X.device) * S)[None, :] <= history.clamp(min=0)[:, None]
        k = x | slot_known[:, None, :, None, None]
        x = x.permute(0, 3, 1, 2, 4)                                           # (F, N, scopes, J, A)
        k = k.permute(0, 3, 1, 2, 4)
        if keep is not None:
            x, k = x[keep], k[keep]
        x = x.reshape(-1, self.num_inputs)
        k = k.reshape(-1, self.num_inputs)
        return torch.cat([x, k], dim=1).float()

    def training_rows(self, X, frames, min_off_frames=1, history=None, roles=None):
        """Inputs and window targets of the agents visible at each target frame.
        roles: the agent roles of `layer_inputs` (only models with roles take them).

        Returns a dict:
          inputs (B, 2M) float (`layer_inputs`, with the known history per frame);
          rows (B, 2) long, (t, n);
          at_risk (B, A) bool: the atoms off in all of t - min_off_frames..t - 1,
              the only ones that can switch on (an atom back on after a shorter gap
              is a re-onset, not a new event);
          first (B, A) long: frames from t to the atom's first onset for the agent
              (0 = in t itself), max(windows) if none within the largest window;
          targets (B, W, A) bool: an at-risk atom switches on within window k
              (first < W_k);
          known (B, W, A) bool: the target of window k is observed, the video
              runs on for W_k frames after t or the onset was seen (rows near the
              video end without an onset are censored, not negatives).
        """
        N = X.shape[1]
        t = frames.repeat_interleave(N)
        n = torch.arange(N, device=X.device).repeat(len(frames))
        keep = X[t, n].any(dim=1)
        inputs = (self.layer_inputs(X, frames, history, keep=keep.view(len(frames), N)) if roles is None
                  else self.layer_inputs(X, frames, history, roles=roles, keep=keep.view(len(frames), N)))
        t, n = t[keep], n[keep]
        past = t[:, None] - torch.arange(1, min_off_frames + 1, device=X.device)     # (B, k)
        held = (X[past.clamp(min=0), n[:, None]] & (past >= 0)[..., None]).any(dim=1)
        at_risk = ~held
        windows = torch.tensor(self.windows, device=X.device)
        first = self.first_onsets(X, t, n, max(self.windows), min_off_frames)
        targets = at_risk[:, None, :] & (first[:, None, :] < windows[None, :, None])
        known = ((t[:, None] + windows[None, :]) <= X.shape[0])[..., None] | targets
        return {"inputs": inputs, "rows": torch.stack([t, n], dim=1), "at_risk": at_risk, "first": first,
                "targets": targets, "known": known}

    @staticmethod
    def first_onsets(X, t, n, horizon, min_off_frames):
        """(B, A) long: frames from t to the first frame f of t..t + horizon - 1 where
        the atom switches on for agent n (on in f, off in f - min_off_frames..f - 1,
        the agent visible in f - 1, so not an agent entering the scene); horizon if none."""
        T = X.shape[0]
        frames = t[:, None] + torch.arange(-min_off_frames, horizon, device=X.device)     # (B, L)
        window = X[frames.clamp(0, T - 1), n[:, None]] & ((frames >= 0) & (frames < T))[..., None]
        held_count = torch.cat([window.new_zeros((len(t), 1, X.shape[2]), dtype=torch.int32),
                                window.int().cumsum(dim=1)], dim=1)          # [j]: frames held in window[:j]
        i = torch.arange(min_off_frames, min_off_frames + horizon, device=X.device)     # window index of f
        held = (held_count[:, i] - held_count[:, i - min_off_frames]) > 0
        seen = window[:, i - 1].any(dim=2, keepdim=True)
        onset = window[:, i] & ~held & seen                                   # (B, horizon, A)
        return torch.where(onset.any(dim=1), onset.int().argmax(dim=1), horizon)

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
        """(R, M) mask in [0, 1]; 1 for the fixed inputs. hard_concrete: stochastic
        while training, deterministic in eval; sigmoid: sigmoid(log_alpha) always."""
        if self.mask_type == "sigmoid":
            mask = torch.sigmoid(self.log_alpha)
        else:
            mask = self.hard_concrete_mask(self.log_alpha, training=self.training)
        return (mask * self.input_mask).masked_fill(self.fixed_inputs, 1.0)

    def forward(self, inputs):
        """Hazard logits (B, W, A) of the future slots and the rule activations
        (B, R) of the head slot f = 0 (every body in the memory), of packed inputs
        (B, 2M); `window_probs` turns the hazards into onset-within-window probabilities.

        In the log domain a rule is two matmuls: the confirmed-false inputs times
        log(1 - m) and the unknown inputs times log(1 - m (1 - pi)).
        """
        x, k = self.split_inputs(inputs)
        B = len(inputs)
        mask = self.rule_mask().clamp(max=1 - 1e-6)
        log_false = torch.log1p(-mask)                                          # (R, M)
        log_unknown = torch.log1p(-mask * (1 - torch.sigmoid(self.unknown_logit)))
        false = torch.cat([(1 - x) * k, x.new_zeros((B, 1))], dim=1)            # column M: a future body
        unknown = torch.cat([1 - k, x.new_ones((B, 1))], dim=1)
        if self.require_self:
            # Self gate 1 - prod_i (1 - m_i p_i) over the own-agent inputs, p_i = 1 if true, pi if unknown, 0 if false.
            mask_self = mask * self.self_inputs
            log_self_true = torch.log1p(-mask_self)
            log_self_unknown = torch.log1p(-mask_self * torch.sigmoid(self.unknown_logit))
            true = torch.cat([x * k, x.new_zeros((B, 1))], dim=1)
        hazards, rules_now, all_unknown = [], None, None
        for f in range(len(self.windows)):
            if f < self.num_slots:
                index = self.shift_index[f]
                rules = torch.exp(false[:, index] @ log_false.T + unknown[:, index] @ log_unknown.T)
                if self.require_self:
                    rules = rules * -torch.expm1(true[:, index] @ log_self_true.T
                                                 + unknown[:, index] @ log_self_unknown.T)
            else:
                # Every body lies in the future: the same activation for every row.
                if all_unknown is None:
                    all_unknown = torch.exp(log_unknown.sum(dim=1)).expand(B, -1)
                    if self.require_self:
                        all_unknown = all_unknown * -torch.expm1(log_self_unknown.sum(dim=1))
                rules = all_unknown
            if f == 0:
                rules_now = rules
            hazards.append(rules @ self.head_weight.T + self.head_bias[f])
        return torch.stack(hazards, dim=1), rules_now

    @staticmethod
    def window_log_probs(hazards):
        """log P(onset within W_k) and log P(no onset within W_k), each (B, W, A),
        of hazard logits (B, W, A): log P(none) = -sum_{j <= k} softplus(h_j)."""
        log_none = -torch.nn.functional.softplus(hazards).cumsum(dim=1)
        log_onset = torch.log(-torch.expm1(log_none.clamp(max=-1e-7)))
        return log_onset, log_none

    @staticmethod
    def window_probs(hazards):
        """(B, W, A) onset probability within each window, non-decreasing in W."""
        return -torch.expm1(-torch.nn.functional.softplus(hazards).cumsum(dim=1))

    @classmethod
    def loss(cls, hazards, targets, at_risk, known, pos_weight=20.0):
        """Mean binary cross-entropy of the onset-within-window probabilities over
        the at-risk atoms and the observed windows; onsets weighted up."""
        log_onset, log_none = cls.window_log_probs(hazards)
        weight = torch.where(targets, pos_weight, 1.0) * (at_risk[:, None, :] & known)
        bce = -torch.where(targets, log_onset, log_none)
        return (bce * weight).sum() / weight.sum().clamp(min=1)

    def sparsity_loss(self, beta=2 / 3, gamma=-0.1, zeta=1.1):
        """Expected number of active inputs per rule (L0 of the hard-concrete mask;
        L1 of the sigmoid mask), the fixed inputs left out, plus the L1 of the head weights."""
        shift = beta * torch.log(torch.tensor(-gamma / zeta)) if self.mask_type == "hard_concrete" else 0.0
        l0 = (torch.sigmoid(self.log_alpha - shift) * (~self.fixed_inputs & self.input_mask)).sum(dim=1).mean()
        return l0 + self.head_weight.abs().sum(dim=1).mean()

    # -------- use after training ----------------------------------
    @torch.no_grad()
    def predict(self, X, history=None, roles=None):
        """(N, W, A) probability that each atom switches on for each agent within
        each window after X, knowing the last `history` frames of X (default all).
        Agents not visible in the last frame of X get zeros. roles: see `layer_inputs`."""
        T, N, _A = X.shape
        frames = torch.tensor([T], device=X.device)
        known = None if history is None else torch.tensor([history], device=X.device)
        probabilities = self.window_probs(self(self.layer_inputs(X, frames, known, roles))[0])
        return probabilities * X[-1].any(dim=1)[:, None, None]

    @torch.no_grad()
    def partial_activation(self, inputs, threshold=0.5):
        """(B, R) share of each rule's body inputs that hold in the memory (the
        head in the next slot), in [0, 1]: a rule with 3 of its 4 bodies seen is 0.75 active."""
        x, _k = self.split_inputs(inputs)
        body = (self.rule_mask() > threshold).float()
        return (x @ body.T) / body.sum(dim=1).clamp(min=1)

    def decode_input(self, m):
        """Input node m -> (atom, slot type, j): slot type 0 the agent itself, 1
        another agent, 2 + k the agent of role self.roles[k]; the atom holds in
        the slot j slots before the head's slot."""
        J, A = self.num_slots, self.num_atoms
        return m % A, m // (J * A), (m // A) % J + 1

    @torch.no_grad()
    def extract_rules(self, mask_threshold=0.5, weight_threshold=0.1):
        """Rules [{"rule": r, "bodies": [(atom, slot type, j)], "heads": [(atom, weight)]}].

        Bodies are sorted earliest first (largest j), the order the atoms happen.
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
            if self.require_self and not self.self_inputs[inputs].any():
                continue                                    # muted by the self gate (`forward`)
            bodies = sorted((self.decode_input(m) for m in inputs), key=lambda b: -b[2])
            rules.append({"rule": r, "bodies": bodies,
                          "heads": [(h, self.head_weight[h, r].item()) for h in heads]})
        return rules


class LayeredRuleNet(torch.nn.Module):
    """Layer-per-time-step rule network: predicts the full atom state of every
    agent in layer t from the K layers t-1..t-K before it.

    A layer is one block of frames (`step05.coarsen_video`). Each of the K past
    layers gives two groups of A input nodes per agent: slot 0 "self" (the agent
    holds the atom in that layer) and slot 1 "others" (some other agent holds it),
    so M = 2 * K * A inputs, each at an exact layer offset 1..K. States may be
    soft (predicted probabilities), so a predicted layer can be fed back as the
    input of the next one and rules chain over layers.

    R rule nodes are soft ANDs over a sigmoid mask of the inputs, deterministic
    and started near 0: with thousands of mostly-0 inputs, a noisy (hard-concrete)
    mask opens a few of them in every sample and turns the rule off, so it gets no
    gradient. An L1 of the mask keeps it sparse. The next state is a transition
    of the previous one: each atom switches on (off -> on) with the weighted OR
    of the rules of its "on" head, and off with its "off" head:
        s_t = s_{t-1} * (1 - off) + (1 - s_{t-1}) * on.
    """

    def __init__(self, num_atoms, num_layers=20, num_rules=256, init_log_alpha=-11.0):
        super().__init__()
        self.num_atoms = num_atoms
        self.num_layers = num_layers
        self.num_inputs = 2 * num_layers * num_atoms
        self.log_alpha = torch.nn.Parameter(torch.full((num_rules, self.num_inputs), init_log_alpha))
        self.on_weight = torch.nn.Parameter(0.01 * torch.randn(num_atoms, num_rules))
        self.off_weight = torch.nn.Parameter(0.01 * torch.randn(num_atoms, num_rules))
        self.on_bias = torch.nn.Parameter(torch.full((num_atoms,), -4.0))
        self.off_bias = torch.nn.Parameter(torch.full((num_atoms,), -3.0))
        # Inputs always in their rule (mask 1, no sparsity cost): the seeded anchors.
        self.register_buffer("fixed_inputs", torch.zeros((num_rules, self.num_inputs), dtype=torch.bool))
        # Inputs a rule may learn; all False for the anchor-only rules.
        self.register_buffer("learnable", torch.ones((num_rules, self.num_inputs), dtype=torch.bool))

    def input_index(self, atom, slot, offset):
        """Input node of (atom, slot, layer offset 1..K); slots > 1 are "others"."""
        return min(slot, 1) * self.num_layers * self.num_atoms + (offset - 1) * self.num_atoms + atom

    def decode_input(self, m):
        """Input node m -> (atom, slot, offset): slot 0 self, 1 others; the atom
        holds offset layers before the head."""
        K, A = self.num_layers, self.num_atoms
        return m % A, m // (K * A), (m // A) % K + 1

    @torch.no_grad()
    def seed_rules(self, anchors, offsets, num_specialized=2, head_weight=1.0, noise=0.5):
        """Seed 1 + num_specialized rule nodes per (anchor body, layer offset).

        anchors: [((atom, slot), [head atom])]. All copies hold the body atom at
        the offset as a fixed input and start with weight head_weight to the
        anchor's "on" heads. Copy 0 keeps only the anchor (no learnable inputs), so
        the general rule keeps its coverage; copies 1.. may add atoms, their gates
        jittered by noise so they specialize in different directions.
        Returns the number of seeded rules, the first ones of the R rule nodes.
        """
        copies = 1 + num_specialized
        num_seeded = len(anchors) * len(offsets) * copies
        if num_seeded > self.log_alpha.shape[0]:
            raise ValueError(f"{num_seeded} seeded rules do not fit into {self.log_alpha.shape[0]} rule nodes.")
        r = 0
        for (atom, slot), heads in anchors:
            for offset in offsets:
                for copy in range(copies):
                    m = self.input_index(atom, slot, offset)
                    self.fixed_inputs[r, m] = True
                    self.learnable[r] = copy > 0
                    self.learnable[r, m] = False
                    self.log_alpha[r] += noise * torch.randn(self.num_inputs) * (copy > 0)
                    self.on_weight[:, r] = 0.0
                    self.on_weight[list(heads), r] = head_weight
                    self.off_weight[:, r] = 0.0
                    r += 1
        return num_seeded

    def layer_inputs(self, context):
        """(W * N, M) float inputs of every agent from its context of K layers.

        context: (W, K, N, A) states in [0, 1] of W windows, oldest layer first.
        Row w * N + n is agent n of window w. "Others" is the soft OR over the
        other agents, 1 - prod(1 - s), so soft states pass their gradient.
        """
        W, K, N, A = context.shape
        own = context.flip(1)                                         # offset 1 first
        log_free = torch.log1p(-own.clamp(max=1 - 1e-6))
        others = 1 - torch.exp(log_free.sum(dim=2, keepdim=True) - log_free)
        inputs = torch.stack([own, others], dim=2)                    # (W, K, 2, N, A)
        return inputs.permute(0, 3, 2, 1, 4).reshape(W * N, 2 * K * A)

    # -------- forward ----------------------------------
    def rule_mask(self):
        """(R, M) mask sigmoid(log_alpha) in [0, 1]; 1 for the fixed inputs, 0 for
        the inputs a rule may not learn."""
        return (torch.sigmoid(self.log_alpha) * self.learnable).masked_fill(self.fixed_inputs, 1.0)

    def forward(self, inputs):
        """On logits (B, A), off logits (B, A) and rule activations (B, R) of
        inputs (B, M); rules as in `TemporalRuleNet.forward`."""
        mask = self.rule_mask().clamp(max=1 - 1e-6)
        rules = torch.exp((1 - inputs) @ torch.log1p(-mask).T)
        on = rules @ self.on_weight.T + self.on_bias
        off = rules @ self.off_weight.T + self.off_bias
        return on, off, rules

    @staticmethod
    def next_state(prev, on, off):
        """(B, A) atom probabilities of the next layer from the previous states."""
        return prev * (1 - torch.sigmoid(off)) + (1 - prev) * torch.sigmoid(on)

    @staticmethod
    def state_loss(pred, prev, target, pos_weight=20.0):
        """Sum and total weight of the binary cross-entropy of the predicted states
        against the true ones; transitions (target != true previous state) weighted
        up, since most atoms keep their state."""
        changed = (target > 0.5) != (prev > 0.5)
        weight = torch.where(changed, pos_weight, 1.0)
        p = pred.clamp(1e-6, 1 - 1e-6)
        bce = -(target * torch.log(p) + (1 - target) * torch.log(1 - p))
        return (bce * weight).sum(), weight.sum()

    def rule_heads(self):
        """(R,) index into [on heads, off heads] (0..2A-1) of each rule's largest head weight."""
        return torch.cat([self.on_weight, self.off_weight]).argmax(dim=0)

    def precision_loss(self, rules, prev, target, counts, k=10.0, decay=0.9):
        """-log of the m-estimate precision (hits + k * p0) / (fires + k) of each
        rule with learnable inputs, for its main head (`rule_heads`).

        fires: soft firings where the head can happen (atom off for an "on" head,
        on for an "off" head); hits: those where it happens; p0: its base rate. A
        rule hitting few examples gets a precision near p0, so precise rules must
        also cover data. counts {"hits", "fires"} accumulate over the recent
        batches (decayed, detached), so k is in about 1 / (1 - decay) batches.
        """
        prev, target = prev > 0.5, target > 0.5
        risk = torch.cat([~prev, prev], dim=1)                        # (B, 2A)
        happens = torch.cat([~prev & target, prev & ~target], dim=1)
        h = self.rule_heads()
        m, y = risk[:, h].float(), happens[:, h].float()              # (B, R)
        hits = decay * counts.get("hits", 0.0) + (rules * y).sum(dim=0)
        fires = decay * counts.get("fires", 0.0) + (rules * m).sum(dim=0)
        p0 = (y * m).sum(dim=0) / m.sum(dim=0).clamp(min=1)
        counts["hits"], counts["fires"] = hits.detach(), fires.detach()
        precision = (hits + k * p0) / (fires + k)
        trained = self.learnable.any(dim=1)
        return -(torch.log(precision.clamp(min=1e-6)) * trained).sum() / trained.sum().clamp(min=1)

    def sparsity_loss(self):
        """L1 of the learnable mask per rule with any, plus the L1 of the on and
        off head weights."""
        trained = self.learnable.any(dim=1)
        l1 = (torch.sigmoid(self.log_alpha) * self.learnable).sum(dim=1)
        l1 = (l1 * trained).sum() / trained.sum().clamp(min=1)
        return l1 + self.on_weight.abs().sum(dim=1).mean() + self.off_weight.abs().sum(dim=1).mean()

    # -------- use after training ----------------------------------
    @torch.no_grad()
    def rollout(self, X, steps=1):
        """(steps, N, A) atom probabilities of the layers after X, each predicted
        from the K layers before it, the predicted ones included.

        X: (T, N, A) layer states (bool or float), T >= 1; layers before X are empty.
        The agents visible in the last layer of X stay visible, the others empty.
        """
        K = self.num_layers
        states = [X.new_zeros(X.shape[1:], dtype=torch.float)] * K + list(X.float().unbind(0))
        visible = (X[-1].float().amax(dim=1, keepdim=True) > 0).float()
        for _ in range(steps):
            on, off, _rules = self(self.layer_inputs(torch.stack(states[-K:])[None]))
            states.append(self.next_state(states[-1], on, off) * visible)
        return torch.stack(states[-steps:])

    @torch.no_grad()
    def partial_activation(self, inputs, threshold=0.5):
        """(B, R) share of each rule's body inputs that hold, in [0, 1]."""
        body = (self.rule_mask() > threshold).float()
        return (inputs @ body.T) / body.sum(dim=1).clamp(min=1)

    @torch.no_grad()
    def extract_rules(self, mask_threshold=0.5, weight_threshold=0.1):
        """Rules [{"rule": r, "bodies": [(atom, slot, offset)], "heads": [(atom, kind, weight)]}],
        kind "on" or "off". Bodies are sorted earliest first (largest offset)."""
        body = self.rule_mask() > mask_threshold
        rules = []
        for r in range(body.shape[0]):
            inputs = torch.nonzero(body[r]).flatten().tolist()
            heads = [(h, kind, weights[h, r].item())
                     for kind, weights in (("on", self.on_weight), ("off", self.off_weight))
                     for h in torch.nonzero(weights[:, r] > weight_threshold).flatten().tolist()]
            if not inputs or not heads:
                continue
            rules.append({
                "rule": r,
                "bodies": sorted((self.decode_input(m) for m in inputs), key=lambda b: -b[2]),
                "heads": heads,
            })
        return rules
