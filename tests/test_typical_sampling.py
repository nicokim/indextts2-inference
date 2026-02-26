import torch

from indextts.utils.typical_sampling import TypicalLogitsWarper


class TestTypicalLogitsWarper:
    def test_filters_low_probability_tokens(self):
        warper = TypicalLogitsWarper(mass=0.5)
        input_ids = torch.zeros(1, 1, dtype=torch.long)
        # Create scores where one token is dominant
        scores = torch.tensor([[10.0, -10.0, -10.0, -10.0]])
        result = warper(input_ids, scores)
        # The dominant token should remain, others filtered to -inf
        assert result[0, 0].item() == 10.0
        assert (result[0, 1:] == float("-inf")).all()

    def test_respects_min_tokens_to_keep(self):
        warper = TypicalLogitsWarper(mass=0.01, min_tokens_to_keep=3)
        input_ids = torch.zeros(1, 1, dtype=torch.long)
        scores = torch.tensor([[10.0, 5.0, 2.0, -5.0, -10.0]])
        result = warper(input_ids, scores)
        # At least 3 tokens should survive (not be -inf)
        surviving = (result[0] != float("-inf")).sum().item()
        assert surviving >= 3

    def test_output_shape_matches_input(self):
        warper = TypicalLogitsWarper(mass=0.9)
        input_ids = torch.zeros(2, 5, dtype=torch.long)
        scores = torch.randn(2, 100)
        result = warper(input_ids, scores)
        assert result.shape == scores.shape

    def test_batch_processing(self):
        warper = TypicalLogitsWarper(mass=0.9)
        input_ids = torch.zeros(3, 1, dtype=torch.long)
        scores = torch.randn(3, 50)
        result = warper(input_ids, scores)
        assert result.shape == (3, 50)
        # Each row should have at least 1 non-filtered token
        for i in range(3):
            assert (result[i] != float("-inf")).any()
