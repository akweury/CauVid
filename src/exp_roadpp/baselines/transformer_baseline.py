import copy
import math
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from src.exp_roadpp import utils_data

"""
The transformer baseline model is implemented in this file. 
The model takes the facts of the objects of each video segment as input and predicts the av_action_id, 
i.e. the action of the autonomous vehicle in the segment.

The input includes the object class, action_ids, and loc_ids of the objects in the segment. 
The model is trained on the training set and evaluated on the test set.

Each fact (segment) has the shape:
    {
        "start_frame": int,
        "end_frame": int,
        "av_action_id": int,
        "agents": [
            {
                "class": int,
                "frame-action-location": [
                    {"frame": int, "action_ids": [...], "loc_ids": [...]},
                    ...
                ],
            },
            ...
        ],
    }

Every agent in a segment is encoded as a single token: its class is embedded and the
(possibly multi-label, possibly time-varying) action/location ids it was observed with
during the segment are aggregated into multi-hot vectors and projected into the same
embedding space. The resulting bag of agent tokens (which may be empty for segments
without agents) is fed, together with a learnable [CLS] token, into a standard
Transformer encoder. The final av_action_id is predicted from the [CLS] token.
"""


def _flatten_ids(values):
    flattened = []
    for value in values or []:
        if isinstance(value, (list, tuple, set)):
            flattened.extend(value)
        else:
            flattened.append(value)
    return flattened


def _agent_action_loc_ids(agent):
    action_ids = set()
    loc_ids = set()
    for pair in agent.get("frame-action-location", []) or []:
        action_ids.update(_flatten_ids(pair.get("action_ids", [])))
        loc_ids.update(_flatten_ids(pair.get("loc_ids", [])))
    return action_ids, loc_ids


def _scan_vocab_sizes(*fact_lists):
    max_agent_class = -1
    max_action_id = -1
    max_loc_id = -1
    max_av_action_id = -1
    for facts in fact_lists:
        for fact in facts or []:
            if "av_action_id" in fact:
                max_av_action_id = max(max_av_action_id, int(fact["av_action_id"]))
            for agent in fact.get("agents", []) or []:
                if agent.get("class") is not None:
                    max_agent_class = max(max_agent_class, int(agent["class"]))
                action_ids, loc_ids = _agent_action_loc_ids(agent)
                if action_ids:
                    max_action_id = max(max_action_id, max(int(a) for a in action_ids))
                if loc_ids:
                    max_loc_id = max(max_loc_id, max(int(l) for l in loc_ids))
    return {
        "num_agent_classes": max(1, max_agent_class + 1),
        "num_action_classes": max(1, max_action_id + 1),
        "num_loc_classes": max(1, max_loc_id + 1),
        "num_av_actions": max(1, max_av_action_id + 1),
    }


def _encode_fact(fact, vocab):
    agent_classes = []
    action_ids_list = []
    loc_ids_list = []
    for agent in fact.get("agents", []) or []:
        agent_class = agent.get("class")
        if agent_class is None:
            continue
        agent_class = min(max(0, int(agent_class)), vocab["num_agent_classes"] - 1)
        action_ids, loc_ids = _agent_action_loc_ids(agent)
        agent_classes.append(agent_class)
        action_ids_list.append(action_ids)
        loc_ids_list.append(loc_ids)

    num_agents = len(agent_classes)
    agent_class_tensor = torch.tensor(agent_classes, dtype=torch.long) if num_agents else torch.zeros(0, dtype=torch.long)

    action_multihot = torch.zeros(num_agents, vocab["num_action_classes"])
    loc_multihot = torch.zeros(num_agents, vocab["num_loc_classes"])
    for i, ids in enumerate(action_ids_list):
        for action_id in ids:
            action_id = int(action_id)
            if 0 <= action_id < vocab["num_action_classes"]:
                action_multihot[i, action_id] = 1.0
    for i, ids in enumerate(loc_ids_list):
        for loc_id in ids:
            loc_id = int(loc_id)
            if 0 <= loc_id < vocab["num_loc_classes"]:
                loc_multihot[i, loc_id] = 1.0

    label = int(fact["av_action_id"])
    return agent_class_tensor, action_multihot, loc_multihot, label


class _SegmentDataset(Dataset):
    def __init__(self, facts, vocab):
        self.examples = [_encode_fact(fact, vocab) for fact in facts]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def _collate_batch(batch):
    batch_size = len(batch)
    action_dim = batch[0][1].shape[1]
    loc_dim = batch[0][2].shape[1]
    max_agents = max(item[0].shape[0] for item in batch)

    agent_class_ids = torch.zeros(batch_size, max_agents, dtype=torch.long)
    action_multihot = torch.zeros(batch_size, max_agents, action_dim)
    loc_multihot = torch.zeros(batch_size, max_agents, loc_dim)
    padding_mask = torch.ones(batch_size, max_agents, dtype=torch.bool)  # True marks padded positions
    labels = torch.zeros(batch_size, dtype=torch.long)

    for i, (agent_classes, action_mh, loc_mh, label) in enumerate(batch):
        num_agents = agent_classes.shape[0]
        if num_agents:
            agent_class_ids[i, :num_agents] = agent_classes
            action_multihot[i, :num_agents] = action_mh
            loc_multihot[i, :num_agents] = loc_mh
            padding_mask[i, :num_agents] = False
        labels[i] = label

    return agent_class_ids, action_multihot, loc_multihot, padding_mask, labels


class _TransformerNet(nn.Module):
    def __init__(self, vocab, embed_dim=32, num_heads=4, num_layers=1, ff_dim=64, dropout=0.3):
        super().__init__()
        self.agent_class_embedding = nn.Embedding(vocab["num_agent_classes"], embed_dim)
        self.action_proj = nn.Linear(vocab["num_action_classes"], embed_dim, bias=False)
        self.loc_proj = nn.Linear(vocab["num_loc_classes"], embed_dim, bias=False)
        self.token_norm = nn.LayerNorm(embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, vocab["num_av_actions"])

    def forward(self, agent_class_ids, action_multihot, loc_multihot, padding_mask):
        batch_size = agent_class_ids.shape[0]
        agent_tokens = (
            self.agent_class_embedding(agent_class_ids)
            + self.action_proj(action_multihot)
            + self.loc_proj(loc_multihot)
        )
        agent_tokens = self.token_norm(agent_tokens)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, agent_tokens], dim=1)

        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=tokens.device)
        full_mask = torch.cat([cls_mask, padding_mask], dim=1)

        encoded = self.encoder(tokens, src_key_padding_mask=full_mask)
        cls_output = self.dropout(encoded[:, 0, :])
        return self.classifier(cls_output)


class TransformerModel:

    def __init__(self, vocab, embed_dim=32, num_heads=4, num_layers=1, ff_dim=64, dropout=0.3,
                 lr=1e-3, weight_decay=1e-2, label_smoothing=0.1, grad_clip_norm=1.0,
                 batch_size=16, max_epochs=50, patience=5, seed=7, device=None):
        self.vocab = vocab
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_clip_norm = grad_clip_norm
        self.device = device

        torch.manual_seed(seed)
        self.model = _TransformerNet(
            vocab, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
            ff_dim=ff_dim, dropout=dropout,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def _make_loader(self, facts, shuffle):
        dataset = _SegmentDataset(facts, self.vocab)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, collate_fn=_collate_batch)

    def _run_epoch(self, loader, train_mode):
        self.model.train(train_mode)
        total_loss = 0.0
        total_examples = 0
        all_preds = []
        all_labels = []
        with torch.enable_grad() if train_mode else torch.no_grad():
            for agent_class_ids, action_mh, loc_mh, padding_mask, labels in loader:
                agent_class_ids = agent_class_ids.to(self.device)
                action_mh = action_mh.to(self.device)
                loc_mh = loc_mh.to(self.device)
                padding_mask = padding_mask.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(agent_class_ids, action_mh, loc_mh, padding_mask)
                loss = self.criterion(logits, labels)

                if train_mode:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.optimizer.step()

                batch_size = labels.shape[0]
                total_loss += loss.item() * batch_size
                total_examples += batch_size
                all_preds.extend(logits.argmax(dim=-1).detach().cpu().tolist())
                all_labels.extend(labels.detach().cpu().tolist())

        avg_loss = total_loss / total_examples if total_examples else 0.0
        accuracy = accuracy_score(all_labels, all_preds) if total_examples else 0.0
        return avg_loss, accuracy

    def train(self, train_facts, val_facts):
        """
        Train the Transformer model on the training facts and validate on the validation facts.

        Args:
            train_facts (list): List of training facts.
            val_facts (list): List of validation facts.
        """
        if not train_facts:
            print("No training facts provided; skipping Transformer training.")
            return

        train_loader = self._make_loader(train_facts, shuffle=True)
        val_loader = self._make_loader(val_facts, shuffle=False) if val_facts else None

        best_val_loss = math.inf
        best_state = copy.deepcopy(self.model.state_dict())
        epochs_without_improvement = 0

        for epoch in range(1, self.max_epochs + 1):
            train_loss, train_acc = self._run_epoch(train_loader, train_mode=True)

            if val_loader is not None:
                val_loss, val_acc = self._run_epoch(val_loader, train_mode=False)
                print(f"[Transformer] Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                      f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(self.model.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.patience:
                        print(f"[Transformer] Early stopping at epoch {epoch}.")
                        break
            else:
                print(f"[Transformer] Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f}")
                best_state = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(best_state)

    def evaluate(self, test_facts, dataset_labels):
        """
        Evaluate the Transformer model on the test facts.

        Args:
            test_facts (list): List of test facts.

        Returns:
            dict: Evaluation results.
        """
        loader = self._make_loader(test_facts, shuffle=False)
        av_action_label_map = dataset_labels["av_action_labels"]
        self.model.eval()
        all_preds = []
        all_label_ids = []
        
        with torch.no_grad():
            for agent_class_ids, action_mh, loc_mh, padding_mask, labels in loader:
                agent_class_ids = agent_class_ids.to(self.device)
                action_mh = action_mh.to(self.device)
                loc_mh = loc_mh.to(self.device)
                padding_mask = padding_mask.to(self.device)

                logits = self.model(agent_class_ids, action_mh, loc_mh, padding_mask)
                all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
                all_label_ids.extend(labels.tolist())

        test_accuracy = float(accuracy_score(all_label_ids, all_preds))
        test_f1_macro = float(f1_score(all_label_ids, all_preds, average="macro"))
        
        test_accuracy_per_class = {}
        for class_label in sorted(set(all_label_ids)):
            class_indices = [i for i, label in enumerate(all_label_ids) if label == class_label]
            class_correct = sum(1 for i in class_indices if all_preds[i] == class_label)
            action_name = av_action_label_map[int(class_label)]
            test_accuracy_per_class[action_name] = float(class_correct) / len(class_indices)
            test_accuracy_per_class[f"{action_name}_Num"] = len(class_indices)
            test_accuracy_per_class[f"{action_name}_Correct"] = class_correct
        print(f"[Transformer] Test accuracy: {test_accuracy:.4f}, F1 macro: {test_f1_macro:.4f}")
        for action_name, acc in test_accuracy_per_class.items():
            if not action_name.endswith("_Num") and not action_name.endswith("_Correct"):
                print(f"  {action_name}: accuracy={acc:.4f}, Num={test_accuracy_per_class[action_name + '_Num']}, "
                      f"Correct={test_accuracy_per_class[action_name + '_Correct']}")
        return {
            "test_label_count": len(set(all_label_ids)),
            "test_accuracy": test_accuracy,
            "test_f1_macro": test_f1_macro,
            "test_accuracy_per_class": test_accuracy_per_class,
        }

    def save(self, model_path):
        torch.save({"model_state": self.model.state_dict(), "vocab": self.vocab}, model_path)


def prepare_dataset(train_facts, val_facts, test_facts):
    """
    Prepare the dataset for the Transformer model, i.e. compute the vocabulary sizes
    (agent classes, action ids, location ids and av_action ids) shared across the
    train/val/test splits so they can all be encoded into the same embedding space.

    Args:
        train_facts (list): List of training facts.
        val_facts (list): List of validation facts.
        test_facts (list): List of test facts.

    Returns:
        dict: Vocabulary sizes (num_agent_classes, num_action_classes, num_loc_classes, num_av_actions).
    """
    return _scan_vocab_sizes(train_facts, val_facts, test_facts)


def run(all_facts, dataset_labels, output_dir, result_summary, device):
    """
    Run the Transformer baseline on the given train, validation, and test sets.

    Args:
        all_facts (dict): Dictionary containing 'train', 'val', and 'test' facts.
        dataset_labels (dict): Dictionary containing dataset labels.
        output_dir (Path): Directory to save the results.
        result_summary (dict): Dictionary to store the results summary.
        device (str): Device to run the model on (e.g., "cuda:0" or "cpu").
    """
    print("\n--------- Running Transformer Baseline ----------------------\n")
    model_output_dir = output_dir / "transformer"
    os.makedirs(model_output_dir, exist_ok=True)

    train_facts, val_facts, test_facts = all_facts["train"], all_facts["val"], all_facts["test"]
    transformer_results = {}
    transformer_model_file = model_output_dir / "transformer_model.pth"  # Path to save the trained model
    transformer_result_file = model_output_dir / "transformer_results.json"  # Path to save the results
    if transformer_model_file.exists() and transformer_result_file.exists():
        print("Transformer model and results already exist. Loading them...")
        # Load the model and results if they already exist
        transformer_results = utils_data.load_json(transformer_result_file)
    else:
        vocab = prepare_dataset(train_facts, val_facts, test_facts)
        transformer_model = TransformerModel(vocab, device=device)  # Initialize the Transformer model
        transformer_model.train(train_facts, val_facts)  # Train the model
        transformer_model.save(transformer_model_file)  # Save the trained model weights
        transformer_results = transformer_model.evaluate(test_facts, dataset_labels)  # Evaluate on test set
        utils_data.save_json(transformer_results, transformer_result_file)  # Save the results
    result_summary["transformer"] = transformer_results
    print("\n--------- Transformer Baseline Done! ------------------------\n")