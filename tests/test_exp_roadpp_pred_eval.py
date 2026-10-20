import unittest

import torch

from src.exp_roadpp.step03_pred_eval import collect_clause_proposals


class ClauseProposalTest(unittest.TestCase):
    def test_proposes_clause_when_all_bodies_are_present_with_extra_facts(self) -> None:
        video_matrix = torch.tensor([
            [True, True, False],
            [True, False, True],
        ])
        clause_matrix = torch.tensor([
            [True, False, False],
            [True, False, True],
        ])

        proposals = collect_clause_proposals(video_matrix, clause_matrix)

        expected = torch.tensor([
            [True, False],
            [True, True],
        ])
        self.assertTrue(torch.equal(proposals, expected))

    def test_does_not_propose_clause_when_a_body_is_missing(self) -> None:
        video_matrix = torch.tensor([[True, False]])
        clause_matrix = torch.tensor([[True, True]])

        proposals = collect_clause_proposals(video_matrix, clause_matrix)

        self.assertTrue(torch.equal(proposals, torch.tensor([[False]])))


if __name__ == "__main__":
    unittest.main()
