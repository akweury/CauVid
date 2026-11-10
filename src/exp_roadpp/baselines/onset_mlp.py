import torch

from src.exp_roadpp.cauvid import TemporalRuleNet


class OnsetMLP(torch.nn.Module):
    """Black-box baseline of the TemporalRuleNet: the same input layers, window
    targets, hazard heads and loss, with a dense hidden layer in place of the
    sparse AND rules.

    Input (B, 2M): the packed memory inputs of `TemporalRuleNet.layer_inputs`
    (per past slot and atom, does the agent itself / some other agent hold it,
    and is that known).
    Output (B, W, A): hazard logits of each window bin and atom
    (`TemporalRuleNet.window_probs` turns them into onset-within-window probabilities).
    Hidden units are unconstrained (any mix of present and absent inputs), so the
    gap to the TemporalRuleNet is the cost of the rule structure. activation "gelu"
    (default) or "relu": with ReLU and a large learning rate every hidden unit can
    die in the first epochs, leaving one constant score per head (the prior).
    """

    # The input layers, targets and loss are the TemporalRuleNet's, shared code.
    layer_inputs = TemporalRuleNet.layer_inputs
    training_rows = TemporalRuleNet.training_rows
    first_onsets = staticmethod(TemporalRuleNet.first_onsets)
    window_log_probs = staticmethod(TemporalRuleNet.window_log_probs)
    window_probs = staticmethod(TemporalRuleNet.window_probs)
    loss = classmethod(TemporalRuleNet.loss.__func__)

    ACTIVATIONS = {"gelu": torch.nn.functional.gelu, "relu": torch.relu}

    def __init__(self, num_atoms, slot_size=10, num_slots=5, hidden_size=256, dropout=0.2, windows=(10,),
                 activation="gelu", roles=()):
        super().__init__()
        self.num_atoms = num_atoms
        # The memory slots of TemporalRuleNet.
        self.slot_size, self.num_slots = slot_size, num_slots
        self.num_lags = slot_size * num_slots
        # The scopes of TemporalRuleNet: the agent itself, some other agent, one per role.
        self.roles = tuple(roles)
        self.num_scopes = 2 + len(self.roles)
        self.num_inputs = self.num_scopes * num_slots * num_atoms
        self.windows = list(windows)
        self.hidden = torch.nn.Linear(2 * self.num_inputs, hidden_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = self.ACTIVATIONS[activation]
        self.head = torch.nn.Linear(hidden_size, len(self.windows) * num_atoms)
        # Onsets are rare: start every hazard at a low probability, as TemporalRuleNet does.
        torch.nn.init.constant_(self.head.bias, -6.0)

    def forward(self, inputs):
        """Hazard logits (B, W, A) and hidden activations (B, H) of input layers (B, M)."""
        hidden = self.dropout(self.activation(self.hidden(inputs)))
        return self.head(hidden).view(len(inputs), len(self.windows), self.num_atoms), hidden

    @torch.no_grad()
    def predict(self, X, history=None, roles=None):
        """(N, W, A) probability that each atom switches on for each agent within
        each window after X, knowing the last `history` frames of X (default all).
        Agents not visible in the last frame of X get zeros."""
        T, N, _A = X.shape
        known = None if history is None else torch.tensor([history], device=X.device)
        inputs = self.layer_inputs(X, torch.tensor([T], device=X.device), known, roles)
        probabilities = self.window_probs(self(inputs)[0])
        return probabilities * X[-1].any(dim=1)[:, None, None]
