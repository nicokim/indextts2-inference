from unittest.mock import patch

import torch

from indextts.emotion import QwenEmotion, find_most_similar_cosine


def _make_emotion_instance():
    """Create a QwenEmotion without loading the actual model."""
    with patch.object(QwenEmotion, "__init__", lambda self, *a, **kw: None):
        obj = QwenEmotion.__new__(QwenEmotion)
        obj.max_score = 1.2
        obj.min_score = 0.0
        obj.cn_key_to_en = {
            "高兴": "happy",
            "愤怒": "angry",
            "悲伤": "sad",
            "恐惧": "afraid",
            "反感": "disgusted",
            "低落": "melancholic",
            "惊讶": "surprised",
            "自然": "calm",
        }
        obj.desired_vector_order = ["高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "自然"]
        return obj


class TestClampScore:
    def setup_method(self):
        self.emo = _make_emotion_instance()

    def test_within_range(self):
        assert self.emo.clamp_score(0.5) == 0.5

    def test_below_min(self):
        assert self.emo.clamp_score(-0.3) == 0.0

    def test_above_max(self):
        assert self.emo.clamp_score(2.0) == 1.2

    def test_at_boundaries(self):
        assert self.emo.clamp_score(0.0) == 0.0
        assert self.emo.clamp_score(1.2) == 1.2


class TestConvert:
    def setup_method(self):
        self.emo = _make_emotion_instance()

    def test_basic_conversion(self):
        content = {"高兴": 0.8, "愤怒": 0.2}
        result = self.emo.convert(content)
        assert result["happy"] == 0.8
        assert result["angry"] == 0.2
        assert result["sad"] == 0.0  # missing key defaults to 0.0
        assert list(result.keys()) == [
            "happy",
            "angry",
            "sad",
            "afraid",
            "disgusted",
            "melancholic",
            "surprised",
            "calm",
        ]

    def test_clamps_values(self):
        content = {"高兴": 2.0, "愤怒": -0.5}
        result = self.emo.convert(content)
        assert result["happy"] == 1.2
        assert result["angry"] == 0.0

    def test_fallback_to_calm(self):
        content = {"高兴": 0.0, "愤怒": 0.0}
        result = self.emo.convert(content)
        assert result["calm"] == 1.0

    def test_no_fallback_when_positive(self):
        content = {"高兴": 0.1}
        result = self.emo.convert(content)
        assert result["calm"] == 0.0  # not set to 1.0


class TestFindMostSimilarCosine:
    def test_finds_correct_index(self):
        query = torch.tensor([[1.0, 0.0, 0.0]])
        matrix = torch.tensor(
            [
                [0.0, 1.0, 0.0],  # orthogonal
                [1.0, 0.1, 0.0],  # most similar
                [0.0, 0.0, 1.0],  # orthogonal
            ]
        )
        idx = find_most_similar_cosine(query, matrix)
        assert idx.item() == 1

    def test_identical_vector(self):
        query = torch.tensor([[0.5, 0.5, 0.5]])
        matrix = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.5, 0.5, 0.5],  # identical
                [1.0, 0.0, 0.0],
            ]
        )
        idx = find_most_similar_cosine(query, matrix)
        assert idx.item() == 1

    def test_single_row_matrix(self):
        query = torch.tensor([[1.0, 2.0]])
        matrix = torch.tensor([[3.0, 4.0]])
        idx = find_most_similar_cosine(query, matrix)
        assert idx.item() == 0
