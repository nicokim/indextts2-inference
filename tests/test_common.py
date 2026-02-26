import torch

from indextts.utils.common import (
    de_tokenized_by_CJK_char,
    make_pad_mask,
    tokenize_by_CJK_char,
)


class TestTokenizeByCJKChar:
    def test_mixed_cjk_and_english(self):
        result = tokenize_by_CJK_char("你好世界是 hello world 的中文")
        assert result == "你 好 世 界 是 HELLO WORLD 的 中 文"

    def test_only_cjk(self):
        result = tokenize_by_CJK_char("你好世界")
        assert result == "你 好 世 界"

    def test_only_english(self):
        result = tokenize_by_CJK_char("hello world")
        assert result == "HELLO WORLD"

    def test_do_upper_case_false(self):
        result = tokenize_by_CJK_char("hello world 你好", do_upper_case=False)
        assert result == "hello world 你 好"

    def test_empty_string(self):
        result = tokenize_by_CJK_char("  ")
        assert result == ""


class TestDeTokenizedByCJKChar:
    def test_cjk_only(self):
        result = de_tokenized_by_CJK_char("你 好 世 界")
        assert result == "你好世界"

    def test_cjk_roundtrip(self):
        original = "你好世界"
        tokenized = tokenize_by_CJK_char(original)
        result = de_tokenized_by_CJK_char(tokenized)
        assert result == original

    def test_mixed_adjacent_english(self):
        # When English placeholders merge into a single CJK word, they get restored
        result = de_tokenized_by_CJK_char("你 HELLO 好")
        assert "你" in result
        assert "好" in result

    def test_do_lower_case_with_merged_placeholders(self):
        # English words adjacent to CJK chars without spaces get lowered
        result = de_tokenized_by_CJK_char("你 A 好 B 你", do_lower_case=True)
        assert "你" in result


class TestMakePadMask:
    def test_basic(self):
        lengths = torch.tensor([5, 3, 2])
        mask = make_pad_mask(lengths)
        assert mask.shape == (3, 5)
        # First row: all valid (length=5, max=5)
        assert mask[0].tolist() == [False, False, False, False, False]
        # Second row: 3 valid, 2 padded
        assert mask[1].tolist() == [False, False, False, True, True]
        # Third row: 2 valid, 3 padded
        assert mask[2].tolist() == [False, False, True, True, True]

    def test_explicit_max_len(self):
        lengths = torch.tensor([2, 1])
        mask = make_pad_mask(lengths, max_len=4)
        assert mask.shape == (2, 4)
        assert mask[0].tolist() == [False, False, True, True]
        assert mask[1].tolist() == [False, True, True, True]

    def test_single_element(self):
        lengths = torch.tensor([3])
        mask = make_pad_mask(lengths)
        assert mask.shape == (1, 3)
        assert mask[0].tolist() == [False, False, False]


