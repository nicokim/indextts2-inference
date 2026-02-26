from indextts.utils.front import TextNormalizer, TextTokenizer


class TestTextNormalizerMatchEmail:
    def setup_method(self):
        self.norm = TextNormalizer()

    def test_valid_email(self):
        assert self.norm.match_email("user@example.com") is True

    def test_valid_email_with_numbers(self):
        assert self.norm.match_email("user123@mail456.org") is True

    def test_invalid_no_at(self):
        assert self.norm.match_email("userexample.com") is False

    def test_invalid_no_domain(self):
        assert self.norm.match_email("user@") is False

    def test_invalid_special_chars(self):
        assert self.norm.match_email("user+tag@example.com") is False


class TestTextNormalizerUseChinese:
    def setup_method(self):
        self.norm = TextNormalizer()

    def test_chinese_text(self):
        assert self.norm.use_chinese("你好世界") is True

    def test_english_text(self):
        assert self.norm.use_chinese("hello world") is False

    def test_pinyin_with_tone(self):
        assert self.norm.use_chinese("xuan4") is True

    def test_email_returns_true(self):
        assert self.norm.use_chinese("user@example.com") is True

    def test_numbers_only(self):
        # No alpha, no Chinese → True (the `not has_alpha` branch)
        assert self.norm.use_chinese("12345") is True

    def test_mixed_chinese_english(self):
        assert self.norm.use_chinese("你好hello") is True


class TestTextNormalizerCorrectPinyin:
    def setup_method(self):
        self.norm = TextNormalizer()

    def test_ju_to_jv(self):
        assert self.norm.correct_pinyin("ju3") == "JV3"

    def test_que_to_qve(self):
        assert self.norm.correct_pinyin("que4") == "QVE4"

    def test_xu_to_xv(self):
        assert self.norm.correct_pinyin("xun2") == "XVN2"

    def test_non_jqx_unchanged(self):
        result = self.norm.correct_pinyin("zhu4")
        assert result == "zhu4"  # not jqx, returned as-is

    def test_lv_unchanged(self):
        result = self.norm.correct_pinyin("lu4")
        assert result == "lu4"


class TestTextNormalizerSaveRestoreNames:
    def setup_method(self):
        self.norm = TextNormalizer()

    def test_roundtrip(self):
        original = "由克里斯托弗·诺兰执导"
        saved, name_list = self.norm.save_names(original)
        assert "克里斯托弗·诺兰" not in saved
        assert name_list is not None
        restored = self.norm.restore_names(saved, name_list)
        assert restored == original

    def test_no_names(self):
        text = "今天天气真好"
        saved, name_list = self.norm.save_names(text)
        assert saved == text
        assert name_list is None

    def test_multiple_names(self):
        original = "克里斯托弗·诺兰，莱昂纳多·迪卡普里奥"
        saved, name_list = self.norm.save_names(original)
        assert name_list is not None
        assert len(name_list) == 2
        restored = self.norm.restore_names(saved, name_list)
        assert restored == original

    def test_restore_with_none_list(self):
        result = self.norm.restore_names("hello", None)
        assert result == "hello"


class TestTextNormalizerSaveRestoreTechTerms:
    def setup_method(self):
        self.norm = TextNormalizer()

    def test_roundtrip(self):
        original = "GPT-5-nano is great"
        saved, tech_list = self.norm.save_tech_terms(original)
        assert "-" not in saved or saved.count("-") < original.count("-")
        assert "<H>" in saved
        restored = self.norm.restore_tech_terms(saved, tech_list)
        assert restored == original

    def test_no_tech_terms(self):
        text = "hello world"
        saved, tech_list = self.norm.save_tech_terms(text)
        assert saved == text
        assert tech_list is None

    def test_multiple_terms(self):
        original = "F5-TTS and Fish-Speech"
        saved, tech_list = self.norm.save_tech_terms(original)
        assert len(tech_list) == 2
        restored = self.norm.restore_tech_terms(saved, tech_list)
        assert restored == original

    def test_restore_with_none_list(self):
        result = self.norm.restore_tech_terms("hello", None)
        assert result == "hello"

    def test_restore_cleans_spaces_around_placeholder(self):
        # Simulate normalizer adding spaces around <H>
        result = self.norm.restore_tech_terms("GPT <H> 5 <H> nano", ["GPT-5-nano"])
        assert result == "GPT-5-nano"


class TestTextNormalizerSaveRestorePinyinTones:
    def setup_method(self):
        self.norm = TextNormalizer()

    def test_roundtrip_xuan4(self):
        original = "晕xuan4是一种感觉"
        saved, pinyin_list = self.norm.save_pinyin_tones(original)
        assert "xuan4" not in saved
        assert pinyin_list is not None
        restored = self.norm.restore_pinyin_tones(saved, pinyin_list)
        # x is in jqx, so xu → xv: xuan4 → XVAN4
        assert "XVAN4" in restored

    def test_roundtrip_ju3(self):
        original = "读ju3的音"
        saved, pinyin_list = self.norm.save_pinyin_tones(original)
        restored = self.norm.restore_pinyin_tones(saved, pinyin_list)
        # ju3 → JV3 via correct_pinyin
        assert "JV3" in restored

    def test_no_pinyin(self):
        text = "hello world"
        saved, pinyin_list = self.norm.save_pinyin_tones(text)
        assert saved == text
        assert pinyin_list is None

    def test_restore_with_none_list(self):
        result = self.norm.restore_pinyin_tones("hello", None)
        assert result == "hello"

    def test_does_not_match_beta1(self):
        text = "beta1 is here"
        saved, pinyin_list = self.norm.save_pinyin_tones(text)
        # beta1 should NOT be matched as pinyin
        assert pinyin_list is None


class TestTextNormalizerApplyGlossaryTerms:
    def setup_method(self):
        self.norm = TextNormalizer(enable_glossary=True)
        self.norm.term_glossary = {
            "C++": {"en": "C plus plus", "zh": "C 加加"},
            "C#": "C sharp",
        }

    def test_apply_zh(self):
        result = self.norm.apply_glossary_terms("学习C++编程", lang="zh")
        assert result == "学习C 加加编程"

    def test_apply_en(self):
        result = self.norm.apply_glossary_terms("Learn C++ programming", lang="en")
        assert result == "Learn C plus plus programming"

    def test_apply_string_value(self):
        result = self.norm.apply_glossary_terms("Play C# note", lang="en")
        assert result == "Play C sharp note"

    def test_empty_glossary(self):
        norm = TextNormalizer()
        result = norm.apply_glossary_terms("C++ is great")
        assert result == "C++ is great"

    def test_case_insensitive(self):
        self.norm.term_glossary = {"tts": "T T S"}
        result = self.norm.apply_glossary_terms("TTS is cool", lang="zh")
        assert result == "T T S is cool"


class TestTextTokenizerSplitSegmentsByToken:
    def test_split_by_punctuation(self):
        # Use max_text_tokens_per_segment small enough so segments don't merge
        tokens = list("abcdefghij") + ["."] + list("klmnopqrst") + ["!"]
        result = TextTokenizer.split_segments_by_token(tokens, [".", "!"], max_text_tokens_per_segment=15)
        assert len(result) == 2
        assert result[0][-1] == "."
        assert result[1][-1] == "!"

    def test_respects_max_tokens(self):
        tokens = ["a", "b", "c", "d", "e", "f"]
        result = TextTokenizer.split_segments_by_token(tokens, ["."], max_text_tokens_per_segment=3)
        for seg in result:
            assert len(seg) <= 3

    def test_merges_short_segments(self):
        tokens = ["a", ".", "b", "."]
        result = TextTokenizer.split_segments_by_token(tokens, ["."], max_text_tokens_per_segment=120)
        # Segments are short enough to merge
        assert len(result) == 1
        assert result[0] == ["a", ".", "b", "."]

    def test_empty_input(self):
        result = TextTokenizer.split_segments_by_token([], ["."], max_text_tokens_per_segment=120)
        assert result == []

    def test_no_split_tokens_present(self):
        tokens = ["hello", "world"]
        result = TextTokenizer.split_segments_by_token(tokens, ["."], max_text_tokens_per_segment=120)
        assert result == [["hello", "world"]]

    def test_comma_fallback_split(self):
        tokens = ["a", "b", ",", "c", "d", "."]
        result = TextTokenizer.split_segments_by_token(tokens, ["."], max_text_tokens_per_segment=120)
        # All tokens should be present in the result
        flat = [t for seg in result for t in seg]
        assert flat == tokens
