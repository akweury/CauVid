

class Pruner:
    def __init__(self, prune_args, dataset_labels=None):
        self.freq_th = prune_args.get("frequency_threshold", 1)
        self.support_coverage_ratio_th = prune_args.get("support_coverage_ratio_threshold", 0.5)
        self.dataset_labels = dataset_labels

    def _is_debug_target_rule(self, clause):
        if self.dataset_labels is None:
            return False

        action_labels = self.dataset_labels["action_labels"]
        agent_labels = self.dataset_labels["agent_labels"]
        if "Stop" not in action_labels or "Wait2X" not in action_labels or "Car" not in agent_labels:
            return False

        body = clause["clause"]["body"][0]
        head = clause["clause"]["head"]
        return (
            body.get("pred") == "action"
            and head.get("pred") == "action"
            and body.get("action_id") == action_labels.index("Stop")
            and head.get("action_id") == action_labels.index("Wait2X")
            and body.get("agent_class") == agent_labels.index("Car")
            and head.get("agent_class") == agent_labels.index("Car")
        )

    def coarse_prune(self, init_clauses):
        print(f"Total clauses before coarse pruning: {len(init_clauses)}")

        support_per_clause = {key: sum([support_coverage["support"] 
                                        for support_coverage in value["support_coverage"].values()]) 
                                        for key, value in init_clauses.items()}
        coverage_per_clause = {key: sum([support_coverage["coverage"] 
                                         for support_coverage in value["support_coverage"].values()]) 
                                         for key, value in init_clauses.items()}
        
        target_rules = [
            key for key, clause in init_clauses.items()
            if self._is_debug_target_rule(clause)
        ]
        if self.dataset_labels is not None:
            if not target_rules:
                print("[Pruner debug] Car Stop -> Wait2X rule not found before pruning.")
            for key in target_rules:
                support = support_per_clause[key]
                passes_frequency = support >= self.freq_th
                print(
                    "[Pruner debug] Car Stop -> Wait2X: "
                    f"support={support}"
                    f"frequency_threshold={self.freq_th}, "
                    f"passes_frequency={passes_frequency}, "
                )

        pruned_init_clauses = {key: value for key, value in init_clauses.items() 
                               if support_per_clause[key] >= self.freq_th}
        print(f"[Pruner] C_NUM after frequency pruning: {len(pruned_init_clauses)}")

        return pruned_init_clauses