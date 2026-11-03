import torch

from src.exp_roadpp.cauvid import TemporalRuleNet


class OnsetMLP(torch.nn.Module):
    """Black-box baseline of the TemporalRuleNet: the same input layers, onset
    targets and loss, with a dense hidden layer in place of the sparse AND rules.

    Input (B, M): the M = 2 * B * A lag-bin nodes of `TemporalRuleNet.layer_inputs`
    (per lag bin and atom, does the agent itself / some other agent hold it).
    Output (B, A): onset logits of each atom for the agent in the target frame.
    Hidden units are unconstrained (any mix of present and absent inputs), so the
    gap to the TemporalRuleNet is the cost of the rule structure. With
    hidden_size = R the parameter count matches a TemporalRuleNet of R rules.
    """

    # The input layers and targets are the TemporalRuleNet's, shared code.
    layer_inputs = TemporalRuleNet.layer_inputs
    training_rows = TemporalRuleNet.training_rows
    future_onsets = staticmethod(TemporalRuleNet.future_onsets)
    loss = TemporalRuleNet.loss

    def __init__(self, num_atoms, lag_edges=(1, 2, 4, 8, 16, 32, 64, 100), hidden_size=256, dropout=0.2):
        super().__init__()
        self.num_atoms = num_atoms
        # Bin i covers lo_i..hi_i frames ago, as in TemporalRuleNet.
        self.lag_bins = [(lo + 1, hi) for lo, hi in zip((0, *lag_edges[:-1]), lag_edges)]
        self.num_lags = lag_edges[-1]
        self.num_bins = len(self.lag_bins)
        self.num_inputs = 2 * self.num_bins * num_atoms
        self.hidden = torch.nn.Linear(self.num_inputs, hidden_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(hidden_size, num_atoms)
        # Onsets are rare: start every head at a low probability, as TemporalRuleNet does.
        torch.nn.init.constant_(self.head.bias, -4.0)

    def forward(self, inputs):
        """Head logits (B, A) and hidden activations (B, H) of input layers (B, M)."""
        hidden = self.dropout(torch.relu(self.hidden(inputs)))
        return self.head(hidden), hidden

    @torch.no_grad()
    def predict(self, X):
        """Onset probability (N, A) of each atom for each agent in the frame after X.

        Agents not visible in the last frame of X get zeros.
        """
        T, N, _A = X.shape
        inputs = self.layer_inputs(X, torch.tensor([T], device=X.device))
        probabilities = torch.sigmoid(self(inputs)[0])
        return probabilities * X[-1].any(dim=1, keepdim=True)
