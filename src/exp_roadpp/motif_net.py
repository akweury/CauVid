"""Two-level rule network: short motifs on 2-frame blocks, long-range rules on 10-frame slots.

Level 1, motifs: the memory (60 frames before t) is cut into fine blocks of
fine_size (2) frames. Per block, atom and slot type (the agent itself / some
other agent) there are three fine inputs: the atom holds in the block, switches
on in it (onset) or switches off in it (offset). Events of different agents lie
close together (median 10 frames apart), so their order needs this resolution.
A motif is a soft AND over the fine inputs of motif_span (5) consecutive blocks,
"A switches on, then 4 frames later B switches on": it is evaluated at every
block position of the memory (one 1D convolution over the blocks).

Level 2, rules: the memory is also cut into num_slots (6) coarse slots of
slot_size (10) frames. A node of a slot is a motif (it occurred somewhere in the
slot: the soft OR of its activations there) or a raw hold input (the atom held in
the slot, self / other). A rule is a soft AND over nodes in the slots j = 1..6
before the head's slot, as in `cauvid.TemporalRuleNet`: to predict the head in
future slot f the rule is aligned so that the bodies with j > f lie in the memory
(past slot j - f) and those with j <= f have not happened yet (unknown).

Every input is three-valued: true, false (its block or slot fully known and the
atom absent) or unknown (older than the known history, or in the future); an
unknown input counts as true with a learned prior chance pi. With v the chance an
input holds and m its mask, a soft AND is prod v^m (v floored at eps): a body
(m = 1) seen true gives 1, seen false about 0, unknown pi, so a rule with some
bodies unknown keeps a low, non-zero activation that rises as they are seen;
m = 0 leaves the input out. In the log domain it is one matmul (or convolution)
of log v with the masks. Motif activations enter level 2 as such chances.

Heads, noisy-OR: every clause (rule r, head atom a) has a confidence
c_ar = sigmoid(conf_logit) in [0, 1]; the hazard of future slot f is
  h_fa = 1 - (1 - b_fa) prod_r (1 - c_ar)^(rule_r(f)),
so a rule firing alone makes the head switch on in the slot with chance about
c_ar: the confidence reads as the clause's precision, and the rules together
cover the onsets as a union. `precision_loss` pushes every clause with a
non-negligible confidence to a precision of at least p_min (or out). The onset
probability within W_k = (k + 1) * slot_size frames is 1 - prod_{f <= k} (1 - h_fa).

Inputs are packed per row as [fine truth (C1 * B1), fine known, coarse truth
(2A * J), coarse known], C1 = 3 features x 2 slot types x A atoms, B1 blocks
oldest first, J slots oldest first.
"""
import torch

from src.exp_roadpp.cauvid import TemporalRuleNet

FEATURES = ("hold", "onset", "offset")


class MotifRuleNet(torch.nn.Module):
    # step 06 builds this model's own inputs (they differ from TemporalRuleNet's).
    own_inputs = True
    # Rows, targets, window probabilities and loss are TemporalRuleNet's.
    training_rows = TemporalRuleNet.training_rows
    first_onsets = staticmethod(TemporalRuleNet.first_onsets)
    window_log_probs = staticmethod(TemporalRuleNet.window_log_probs)
    window_probs = staticmethod(TemporalRuleNet.window_probs)
    loss = classmethod(TemporalRuleNet.loss.__func__)

    def __init__(self, num_atoms, fine_size=2, motif_span=5, slot_size=10, num_slots=6, num_motifs=1024,
                 num_rules=1024, windows=(10,), init_log_alpha=-11.0, eps=1e-4):
        super().__init__()
        windows = list(windows)
        if windows != [slot_size * (f + 1) for f in range(len(windows))]:
            raise ValueError(f"windows {windows} must be slot_size, 2 * slot_size, ... (slot_size {slot_size}).")
        if slot_size % fine_size:
            raise ValueError(f"slot_size {slot_size} must be a multiple of fine_size {fine_size}.")
        A = num_atoms
        self.num_atoms, self.windows, self.eps = A, windows, eps
        self.fine_size, self.motif_span = fine_size, motif_span
        self.slot_size, self.num_slots = slot_size, num_slots
        self.num_lags = slot_size * num_slots                      # frames of memory
        self.num_blocks = self.num_lags // fine_size
        self.fine_channels = len(FEATURES) * 2 * A
        self.coarse_channels = 2 * A
        self.num_positions = self.num_blocks - motif_span + 1     # motif positions in the memory
        self.num_motifs, self.num_nodes = num_motifs, num_motifs + 2 * A
        self.num_rule_inputs = num_slots * self.num_nodes
        num_future = len(windows)

        # Level 1.
        self.motif_alpha = torch.nn.Parameter(torch.full((num_motifs, self.fine_channels, motif_span), init_log_alpha))
        self.fine_prior = torch.nn.Parameter(torch.full((self.fine_channels,), -4.0))
        self.register_buffer("motif_fixed", torch.zeros_like(self.motif_alpha, dtype=torch.bool))
        # Level 2.
        self.rule_alpha = torch.nn.Parameter(torch.full((num_rules, self.num_rule_inputs), init_log_alpha))
        self.node_prior = torch.nn.Parameter(torch.full((self.num_nodes,), -4.0))
        self.register_buffer("rule_fixed", torch.zeros_like(self.rule_alpha, dtype=torch.bool))
        # Heads: clause confidences and the hazard of no rule (per future slot).
        self.conf_logit = torch.nn.Parameter(torch.full((A, num_rules), -6.0))
        self.base_logit = torch.nn.Parameter(torch.full((num_future, A), -7.0))

        # Motif output position (oldest first) -> the coarse slot (oldest first) it ends in.
        block_of_position = torch.arange(self.num_positions) + motif_span - 1            # its newest block
        slot_of_block = block_of_position // (slot_size // fine_size)
        pool = torch.zeros((self.num_positions, num_slots))
        pool[torch.arange(self.num_positions), slot_of_block] = 1.0
        self.register_buffer("pool", pool)
        # Rule input i = (j - 1) * num_nodes + node, j the slots before the head's slot; for a head
        # in future slot f it reads past slot j - f (oldest-first slot index num_slots - (j - f)).
        j = torch.arange(self.num_rule_inputs) // self.num_nodes + 1
        node = torch.arange(self.num_rule_inputs) % self.num_nodes
        self.register_buffer("input_offset", j)
        shift = [torch.where(j > f, (num_slots - (j - f)) * self.num_nodes + node, -1)
                 for f in range(min(num_future, num_slots))]
        self.register_buffer("shift_index", torch.stack(shift))

    # -------- indices ----------------------------------
    def fine_channel(self, feature, slot, atom):
        return FEATURES.index(feature) * 2 * self.num_atoms + min(slot, 1) * self.num_atoms + atom

    def rule_input(self, node, offset):
        return (offset - 1) * self.num_nodes + node

    def raw_node(self, slot, atom):
        return self.num_motifs + min(slot, 1) * self.num_atoms + atom

    def decode_motif_input(self, channel, i):
        """(feature, slot type, atom, frames before the motif's newest block end)."""
        A = self.num_atoms
        feature, rest = divmod(channel, 2 * A)
        slot, atom = divmod(rest, A)
        return FEATURES[feature], slot, atom, (self.motif_span - 1 - i) * self.fine_size

    def decode_rule_input(self, m):
        """("motif", id, j) or ("hold", slot type, atom, j): the node in the slot j before the head's."""
        node, j = m % self.num_nodes, m // self.num_nodes + 1
        if node < self.num_motifs:
            return ("motif", node, j)
        slot, atom = divmod(node - self.num_motifs, self.num_atoms)
        return ("hold", slot, atom, j)

    @torch.no_grad()
    def seed(self, anchors, motif_conf=0.1):
        """Seed from the anchor clauses [((atom, slot), [heads])]: per anchor body one
        motif "the atom switches on (in the newest block)" and per slot offset j one
        rule holding the atom in slot j, with confidence motif_conf to the anchor's
        heads. Returns (motifs seeded, rules seeded)."""
        if len(anchors) > self.num_motifs or len(anchors) * self.num_slots > self.rule_alpha.shape[0]:
            raise ValueError("too few motifs / rules for the anchors")
        logit = torch.logit(torch.tensor(motif_conf))
        for i, ((atom, slot), heads) in enumerate(anchors):
            self.motif_fixed[i, self.fine_channel("onset", slot, atom), self.motif_span - 1] = True
            for j in range(1, self.num_slots + 1):
                r = i * self.num_slots + j - 1
                self.rule_fixed[r, self.rule_input(self.raw_node(slot, atom), j)] = True
                self.conf_logit[list(heads), r] = logit
        return len(anchors), len(anchors) * self.num_slots

    # -------- inputs ----------------------------------
    def layer_inputs(self, X, frames, history=None):
        """(F * N, D) packed inputs of every agent at each target frame (see the module doc)."""
        S, J, b = self.slot_size, self.num_slots, self.fine_size
        L, A = self.num_lags, self.num_atoms
        N, F = X.shape[1], len(frames)
        history = frames if history is None else torch.minimum(history.to(frames.device), frames)
        history = history.clamp(min=0)
        lags = torch.arange(1, L + 2, device=X.device)                           # one more for onsets
        past = frames[:, None] - lags
        observed = lags[None, :] <= history[:, None]                             # (F, L + 1)
        own = X[past.clamp(min=0)] & observed[..., None, None]                    # (F, L + 1, N, A)
        hold, before = own[:, :-1], own[:, 1:]
        onset, offset = hold & ~before, ~hold & before
        seen_pair = observed[:, 1:]                                               # lag and lag + 1 known

        def others(x):
            return (x.sum(dim=2, keepdim=True) - x.int()) > 0

        def blocks(x, size):
            # (F, L, N, A) by lag (newest first) -> (F, L / size, N, A), oldest block first.
            return x.view(F, L // size, size, N, A).any(dim=2).flip(1)

        fine = []
        for feature in (hold, onset, offset):
            for x in (feature, others(feature)):
                fine.append(blocks(x, b))
        x1 = torch.stack(fine, dim=1)                                             # (F, 6, B1, N, A)
        known_hold = observed[:, :-1].view(F, L // b, b).all(dim=2).flip(1)        # (F, B1)
        known_pair = seen_pair.view(F, L // b, b).all(dim=2).flip(1)
        k_block = torch.stack([known_hold, known_hold, known_pair, known_pair, known_pair, known_pair], dim=1)
        k1 = x1 | k_block[..., None, None]
        x1 = x1.permute(0, 3, 1, 4, 2).reshape(F * N, self.fine_channels * self.num_blocks)
        k1 = k1.permute(0, 3, 1, 4, 2).reshape(F * N, self.fine_channels * self.num_blocks)

        x2 = torch.stack([blocks(hold, S), blocks(others(hold), S)], dim=1)          # (F, 2, J, N, A)
        k_slot = observed[:, :-1].view(F, J, S).all(dim=2).flip(1)
        k2 = x2 | k_slot[:, None, :, None, None]
        x2 = x2.permute(0, 3, 1, 4, 2).reshape(F * N, self.coarse_channels * J)
        k2 = k2.permute(0, 3, 1, 4, 2).reshape(F * N, self.coarse_channels * J)
        return torch.cat([x1, k1, x2, k2], dim=1).float()

    def split_inputs(self, inputs):
        """Packed inputs -> fine x, k (B, C1, B1) and coarse x, k (B, 2A, J)."""
        B = len(inputs)
        n1, n2 = self.fine_channels * self.num_blocks, self.coarse_channels * self.num_slots
        x1, k1, x2, k2 = torch.split(inputs, [n1, n1, n2, n2], dim=1)
        return (x1.view(B, self.fine_channels, self.num_blocks), k1.view(B, self.fine_channels, self.num_blocks),
                x2.view(B, self.coarse_channels, self.num_slots), k2.view(B, self.coarse_channels, self.num_slots))

    # -------- forward ----------------------------------
    def motif_mask(self):
        return torch.sigmoid(self.motif_alpha).masked_fill(self.motif_fixed, 1.0)

    def rule_mask(self):
        return torch.sigmoid(self.rule_alpha).masked_fill(self.rule_fixed, 1.0)

    def memory_nodes(self, inputs):
        """(B, J * nodes) chance that each level-2 node holds in each memory slot, oldest slot first."""
        x1, k1, x2, k2 = self.split_inputs(inputs)
        v1 = torch.where(k1 > 0.5, x1, torch.sigmoid(self.fine_prior)[None, :, None])
        log_v1 = torch.log(v1.clamp(min=self.eps))
        motifs = torch.exp(torch.nn.functional.conv1d(log_v1, self.motif_mask()))   # (B, K1, positions)
        log_absent = torch.log1p(-motifs.clamp(max=1 - 1e-6))
        pooled = 1 - torch.exp(log_absent @ self.pool)                            # (B, K1, J)
        raw_prior = torch.sigmoid(self.node_prior[self.num_motifs:])
        v2 = torch.where(k2 > 0.5, x2, raw_prior[None, :, None])                  # (B, 2A, J)
        nodes = torch.cat([pooled, v2], dim=1)                                    # (B, nodes, J)
        return nodes.transpose(1, 2).reshape(len(inputs), -1)

    def forward(self, inputs):
        """Hazard logits (B, W, A) of the future slots and the rule activations
        (B, R) with the head in the next slot (every body in the memory)."""
        nodes = self.memory_nodes(inputs)
        B = len(inputs)
        mask = self.rule_mask()                                                     # (R, M2)
        log_nodes = torch.log(nodes.clamp(min=self.eps))
        log_prior = torch.log(torch.sigmoid(self.node_prior).clamp(min=self.eps)).repeat(self.num_slots)  # by rule input
        log_no_conf = torch.log1p(-torch.sigmoid(self.conf_logit).clamp(max=1 - 1e-6))   # (A, R)
        log_no_base = torch.log1p(-torch.sigmoid(self.base_logit).clamp(max=1 - 1e-6))   # (W, A)
        log_survive, rules_now, all_unknown = [], None, None
        for f in range(len(self.windows)):
            if f < self.num_slots:
                index = self.shift_index[f]
                log_values = torch.where(index >= 0, log_nodes[:, index.clamp(min=0)], log_prior[None])
                rules = torch.exp(log_values @ mask.T)
            else:
                if all_unknown is None:
                    all_unknown = torch.exp(log_prior @ mask.T).expand(B, -1)
                rules = all_unknown
            if f == 0:
                rules_now = rules
            log_survive.append(log_no_base[f] + rules @ log_no_conf.T)
        log_survive = torch.stack(log_survive, dim=1).clamp(max=-1e-7)            # (B, W, A)
        hazards = torch.log(-torch.expm1(log_survive)) - log_survive               # logit of 1 - e^log_survive
        return hazards, rules_now

    # -------- regularization ----------------------------------
    def sparsity_loss(self):
        """Mean L1 of the learnable masks per motif and per rule, plus the mean
        summed confidence per head (fewer, sparser clauses)."""
        motif = (torch.sigmoid(self.motif_alpha) * ~self.motif_fixed).flatten(1).sum(dim=1).mean()
        rule = (torch.sigmoid(self.rule_alpha) * ~self.rule_fixed).sum(dim=1).mean()
        conf = torch.sigmoid(self.conf_logit).sum(dim=1).mean()
        return motif + rule + conf

    def precision_loss(self, rules_now, rows, counts, p_min=0.05, k=10.0, decay=0.9):
        """Mean shortfall below p_min of the m-estimate precision (hits + k p0) /
        (fires + k) of the clauses (rule, head) for an onset in the next slot,
        weighted by their confidence: an imprecise clause must sharpen its body
        or lose its confidence. Counts accumulate over the recent batches (decayed,
        detached), so k is in about 1 / (1 - decay) batches."""
        risk = (rows["at_risk"] & rows["known"][:, 0]).float()                       # (B, A)
        onset = rows["targets"][:, 0].float()
        hits = decay * counts.get("hits", 0.0) + rules_now.T @ onset                # (R, A)
        fires = decay * counts.get("fires", 0.0) + rules_now.T @ risk
        p0 = onset.sum(dim=0) / risk.sum(dim=0).clamp(min=1)
        counts["hits"], counts["fires"] = hits.detach(), fires.detach()
        precision = (hits + k * p0) / (fires + k)
        conf = torch.sigmoid(self.conf_logit).T                                    # (R, A)
        return (conf * torch.relu(p_min - precision)).sum() / conf.sum().clamp(min=1e-6)

    # -------- hard rules (extraction, counts) ----------------------------------
    @torch.no_grad()
    def hard_rules(self, inputs, threshold=0.5):
        """(B, R) bool: every body of the rule holds in the memory (head in the next
        slot), motifs and rules with their masks above threshold and every input
        taken as known (full history)."""
        x1, _k1, x2, _k2 = self.split_inputs(inputs)
        body1 = (self.motif_mask() > threshold).float()
        size1 = body1.flatten(1).sum(dim=1)
        motifs = (torch.nn.functional.conv1d(x1, body1) >= size1[None, :, None] - 0.5) & (size1[None, :, None] > 0)
        pooled = (motifs.float() @ self.pool) > 0                                   # (B, K1, J)
        nodes = torch.cat([pooled.float(), x2], dim=1).transpose(1, 2).reshape(len(inputs), -1)
        nodes = nodes[:, self.shift_index[0]]                       # memory slots -> rule inputs (head next slot)
        body2 = (self.rule_mask() > threshold).float()
        size2 = body2.sum(dim=1)
        return (nodes @ body2.T >= size2 - 0.5) & (size2 > 0)

    @torch.no_grad()
    def extract(self, mask_threshold=0.5, conf_threshold=0.02):
        """Rules [{"rule": r, "bodies": [decoded rule inputs, earliest first], "heads":
        [(atom, confidence)]}] and the motifs they use {id: [(feature, slot, atom, frames before)]}."""
        body2 = self.rule_mask() > mask_threshold
        body1 = self.motif_mask() > mask_threshold
        conf = torch.sigmoid(self.conf_logit)
        rules, motifs = [], {}
        for r in range(body2.shape[0]):
            inputs = torch.nonzero(body2[r]).flatten().tolist()
            heads = [(h, conf[h, r].item()) for h in torch.nonzero(conf[:, r] > conf_threshold).flatten().tolist()]
            if not inputs or not heads:
                continue
            bodies = sorted((self.decode_rule_input(m) for m in inputs), key=lambda b: -b[-1])
            for b in bodies:
                if b[0] == "motif" and b[1] not in motifs:
                    c, i = torch.nonzero(body1[b[1]], as_tuple=True)
                    motifs[b[1]] = sorted((self.decode_motif_input(ci, ii) for ci, ii in zip(c.tolist(), i.tolist())),
                                          key=lambda x: -x[3])
            rules.append({"rule": r, "bodies": bodies, "heads": heads})
        return rules, motifs
