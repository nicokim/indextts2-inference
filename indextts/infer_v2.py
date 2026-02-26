import os
import random
import time
import warnings
from subprocess import CalledProcessError

import librosa
import safetensors
import torch
import torchaudio
from huggingface_hub import hf_hub_download, snapshot_download
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from transformers import SeamlessM4TFeatureExtractor

from indextts.emotion import QwenEmotion, find_most_similar_cosine
from indextts.gpt.model_v2 import UnifiedVoice
from indextts.logging import get_logger
from indextts.s2mel.modules.audio import mel_spectrogram
from indextts.s2mel.modules.bigvgan import bigvgan
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.front import TextNormalizer, TextTokenizer
from indextts.utils.maskgct_utils import build_semantic_codec, build_semantic_model

logger = get_logger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_HF_REPO = "IndexTeam/IndexTTS-2"


class IndexTTS2:
    def __init__(
        self,
        model_dir=None,
        cfg_path=None,
        use_fp16=False,
        device=None,
        use_cuda_kernel=None,
        use_deepspeed=False,
        use_accel=False,
        use_torch_compile=False,
        language=None,
        attn_backend=None,
    ):
        """
        Args:
            model_dir (str | None): path to the model directory. If None, auto-downloads from HuggingFace.
            cfg_path (str | None): path to the config file. If None, uses model_dir/config.yaml.
            use_fp16 (bool): whether to use fp16.
            device (str): device to use (e.g., 'cuda:0', 'cpu'). If None, it will be set automatically based on the availability of CUDA or MPS.
            use_cuda_kernel (None | bool): whether to use BigVGan custom fused activation CUDA kernel, only for CUDA device.
            use_deepspeed (bool): whether to use DeepSpeed or not.
            use_accel (bool): whether to use acceleration engine for GPT2 or not.
            use_torch_compile (bool): whether to use torch.compile for optimization or not.
            language (str | None): language for text normalization ('es', 'zh', 'en'). If None, auto-detects between Chinese and English.
            attn_backend (str | None): attention backend to use.
                - None: PyTorch native SDPA (default, auto-selects best backend)
                - "sage": SageAttention (2-5x faster, requires sageattention package)
                - "flash": Flash Attention v2 acceleration engine with KV cache and CUDA graphs
                           (requires flash-attn package)
        """
        if attn_backend == "sage":
            try:
                from sageattention import sageattn
            except ImportError:
                raise ImportError(
                    "sageattention is required for attn_backend='sage'. "
                    "Install with: pip install indextts2-inference[sage-attn]"
                )
            _original_sdpa = torch.nn.functional.scaled_dot_product_attention

            def _sage_sdpa(query, key, value, *args, **kwargs):
                if query.dtype in (torch.float16, torch.bfloat16):
                    return sageattn(query, key, value, *args, **kwargs)
                return _original_sdpa(query, key, value, *args, **kwargs)

            torch.nn.functional.scaled_dot_product_attention = _sage_sdpa
            logger.info("SageAttention backend enabled")
        elif attn_backend == "flash":
            use_accel = True
        elif attn_backend is not None:
            raise ValueError(f"Unknown attn_backend={attn_backend!r}. Must be None, 'sage', or 'flash'.")

        self._resolve_device(device, use_fp16, use_cuda_kernel)
        self._load_config(model_dir, cfg_path)
        self.use_accel = use_accel
        self.use_torch_compile = use_torch_compile
        self._load_gpt(use_deepspeed)
        self._load_semantic()
        self._load_s2mel()
        self._load_vocoder()
        self._load_text(language)
        self._load_emotion_matrices()
        self._init_cache()

    def _resolve_device(self, device, use_fp16, use_cuda_kernel):
        if device is not None:
            self.device = device
            self.use_fp16 = False if device == "cpu" else use_fp16
            self.use_cuda_kernel = use_cuda_kernel is not None and use_cuda_kernel and device.startswith("cuda")
        elif torch.cuda.is_available():
            self.device = "cuda:0"
            self.use_fp16 = use_fp16
            self.use_cuda_kernel = use_cuda_kernel is None or use_cuda_kernel
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            self.device = "xpu"
            self.use_fp16 = use_fp16
            self.use_cuda_kernel = False
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            self.use_fp16 = False  # Use float16 on MPS is overhead than float32
            self.use_cuda_kernel = False
        else:
            self.device = "cpu"
            self.use_fp16 = False
            self.use_cuda_kernel = False
            logger.warning("Be patient, it may take a while to run in CPU mode.")
        self.dtype = torch.float16 if self.use_fp16 else None

    def _load_config(self, model_dir, cfg_path):
        if model_dir is None:
            model_dir = snapshot_download(DEFAULT_HF_REPO)
        if cfg_path is None:
            cfg_path = os.path.join(model_dir, "config.yaml")
        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.stop_mel_token = self.cfg.gpt.stop_mel_token
        self.model_version = self.cfg.version if hasattr(self.cfg, "version") else None

    def _load_gpt(self, use_deepspeed):
        self.qwen_emo = QwenEmotion(os.path.join(self.model_dir, self.cfg.qwen_emo_path))

        self.gpt = UnifiedVoice(**self.cfg.gpt, use_accel=self.use_accel)
        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        load_checkpoint(self.gpt, self.gpt_path)
        self.gpt = self.gpt.to(self.device)
        if self.use_fp16:
            self.gpt.eval().half()
        else:
            self.gpt.eval()
        logger.info("GPT weights restored from: %s", self.gpt_path)

        if use_deepspeed:
            try:
                import deepspeed  # noqa: F401
            except (ImportError, OSError, CalledProcessError) as e:
                use_deepspeed = False
                logger.warning("Failed to load DeepSpeed. Falling back to normal inference. Error: %s", e)

        self.gpt.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=True, half=self.use_fp16)

        if self.use_cuda_kernel:
            try:
                from indextts.s2mel.modules.bigvgan.alias_free_activation.cuda import activation1d

                logger.info("Preload custom CUDA kernel for BigVGAN %s", activation1d.anti_alias_activation_cuda)
            except Exception as e:
                logger.warning("Failed to load custom CUDA kernel for BigVGAN. Falling back to torch. %r", e)
                self.use_cuda_kernel = False

    def _load_semantic(self):
        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
            os.path.join(self.model_dir, self.cfg.w2v_stat)
        )
        self.semantic_model = self.semantic_model.to(self.device)
        self.semantic_model.eval()
        self.semantic_mean = self.semantic_mean.to(self.device)
        self.semantic_std = self.semantic_std.to(self.device)

        semantic_codec = build_semantic_codec(self.cfg.semantic_codec)
        semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
        safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
        self.semantic_codec = semantic_codec.to(self.device)
        self.semantic_codec.eval()
        logger.info("semantic_codec weights restored from: %s", semantic_code_ckpt)

    def _load_s2mel(self):
        s2mel_path = os.path.join(self.model_dir, self.cfg.s2mel_checkpoint)
        s2mel = MyModel(self.cfg.s2mel, use_gpt_latent=True)
        s2mel, _, _, _ = load_checkpoint2(
            s2mel,
            None,
            s2mel_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        self.s2mel = s2mel.to(self.device)
        self.s2mel.models["cfm"].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

        if self.use_torch_compile:
            logger.info("Enabling torch.compile optimization")
            self.s2mel.enable_torch_compile()
            logger.info("torch.compile optimization enabled successfully")

        self.s2mel.eval()
        logger.info("s2mel weights restored from: %s", s2mel_path)

        campplus_ckpt_path = hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin")
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        self.campplus_model = campplus_model.to(self.device)
        self.campplus_model.eval()
        logger.info("campplus_model weights restored from: %s", campplus_ckpt_path)

    def _load_vocoder(self):
        bigvgan_name = self.cfg.vocoder.name
        self.bigvgan = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=self.use_cuda_kernel)
        self.bigvgan = self.bigvgan.to(self.device)
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()
        logger.info("bigvgan weights restored from: %s", bigvgan_name)

    def _load_text(self, language):
        self.bpe_path = os.path.join(self.model_dir, self.cfg.dataset["bpe_model"])
        self.normalizer = TextNormalizer(enable_glossary=True, preferred_language=language)
        self.normalizer.load()
        logger.info("TextNormalizer loaded (language=%s)", language or "auto")
        self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)
        logger.info("bpe model loaded from: %s", self.bpe_path)

        self.glossary_path = os.path.join(self.model_dir, "glossary.yaml")
        if os.path.exists(self.glossary_path):
            self.normalizer.load_glossary_from_yaml(self.glossary_path)
            logger.info("Glossary loaded from: %s", self.glossary_path)

    def _load_emotion_matrices(self):
        emo_matrix = torch.load(os.path.join(self.model_dir, self.cfg.emo_matrix))
        self.emo_matrix = emo_matrix.to(self.device)
        self.emo_num = list(self.cfg.emo_num)

        spk_matrix = torch.load(os.path.join(self.model_dir, self.cfg.spk_matrix))
        self.spk_matrix = spk_matrix.to(self.device)

        self.emo_matrix = torch.split(self.emo_matrix, self.emo_num)
        self.spk_matrix = torch.split(self.spk_matrix, self.emo_num)

        mel_fn_args = {
            "n_fft": self.cfg.s2mel["preprocess_params"]["spect_params"]["n_fft"],
            "win_size": self.cfg.s2mel["preprocess_params"]["spect_params"]["win_length"],
            "hop_size": self.cfg.s2mel["preprocess_params"]["spect_params"]["hop_length"],
            "num_mels": self.cfg.s2mel["preprocess_params"]["spect_params"]["n_mels"],
            "sampling_rate": self.cfg.s2mel["preprocess_params"]["sr"],
            "fmin": self.cfg.s2mel["preprocess_params"]["spect_params"].get("fmin", 0),
            "fmax": None if self.cfg.s2mel["preprocess_params"]["spect_params"].get("fmax", "None") == "None" else 8000,
            "center": False,
        }
        self.mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)

    def _init_cache(self):
        self.cache_spk_cond = None
        self.cache_s2mel_style = None
        self.cache_s2mel_prompt = None
        self.cache_spk_audio_prompt = None
        self.cache_emo_cond = None
        self.cache_emo_audio_prompt = None
        self.cache_mel = None

    @torch.no_grad()
    def get_emb(self, input_features, attention_mask):
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        feat = (feat - self.semantic_mean) / self.semantic_std
        return feat

    def remove_long_silence(self, codes: torch.Tensor, silent_token=52, max_consecutive=30):
        """
        Shrink special tokens (silent_token and stop_mel_token) in codes
        codes: [B, T]
        """
        code_lens = []
        codes_list = []
        device = codes.device
        isfix = False
        for i in range(0, codes.shape[0]):
            code = codes[i]
            if not torch.any(code == self.stop_mel_token).item():
                len_ = code.size(0)
            else:
                stop_mel_idx = (code == self.stop_mel_token).nonzero(as_tuple=False)
                len_ = stop_mel_idx[0].item() if len(stop_mel_idx) > 0 else code.size(0)

            count = torch.sum(code == silent_token).item()
            if count > max_consecutive:
                ncode_idx = []
                n = 0
                for k in range(len_):
                    assert code[k] != self.stop_mel_token, (
                        f"stop_mel_token {self.stop_mel_token} should be shrinked here"
                    )
                    if code[k] != silent_token:
                        ncode_idx.append(k)
                        n = 0
                    elif code[k] == silent_token and n < 10:
                        ncode_idx.append(k)
                        n += 1
                len_ = len(ncode_idx)
                codes_list.append(code[ncode_idx])
                isfix = True
            else:
                # shrink to len_
                codes_list.append(code[:len_])
            code_lens.append(len_)
        if isfix:
            if len(codes_list) > 1:
                codes = pad_sequence(codes_list, batch_first=True, padding_value=self.stop_mel_token)
            else:
                codes = codes_list[0].unsqueeze(0)
        else:
            # unchanged
            pass
        # clip codes to max length
        max_len = max(code_lens)
        if max_len < codes.shape[1]:
            codes = codes[:, :max_len]
        code_lens = torch.tensor(code_lens, dtype=torch.long, device=device)
        return codes, code_lens

    def interval_silence(self, wavs, sampling_rate=22050, interval_silence=200):
        """
        Silences to be insert between generated segments.
        """

        if not wavs or interval_silence <= 0:
            return wavs

        # get channel_size
        channel_size = wavs[0].size(0)
        # get silence tensor
        sil_dur = int(sampling_rate * interval_silence / 1000.0)
        return torch.zeros(channel_size, sil_dur)

    def insert_interval_silence(self, wavs, sampling_rate=22050, interval_silence=200):
        """
        Insert silences between generated segments.
        wavs: List[torch.tensor]
        """

        if not wavs or interval_silence <= 0:
            return wavs

        # get channel_size
        channel_size = wavs[0].size(0)
        # get silence tensor
        sil_dur = int(sampling_rate * interval_silence / 1000.0)
        sil_tensor = torch.zeros(channel_size, sil_dur)

        wavs_list = []
        for i, wav in enumerate(wavs):
            wavs_list.append(wav)
            if i < len(wavs) - 1:
                wavs_list.append(sil_tensor)

        return wavs_list

    def _load_and_cut_audio(self, audio_path, max_audio_length_seconds, verbose=False, sr=None):
        if not sr:
            audio, sr = librosa.load(audio_path)
        else:
            audio, _ = librosa.load(audio_path, sr=sr)
        audio = torch.tensor(audio).unsqueeze(0)
        max_audio_samples = int(max_audio_length_seconds * sr)

        if audio.shape[1] > max_audio_samples:
            if verbose:
                logger.debug("Audio too long (%d samples), truncating to %d samples", audio.shape[1], max_audio_samples)
            audio = audio[:, :max_audio_samples]
        return audio, sr

    def normalize_emo_vec(self, emo_vector, apply_bias=True):
        # apply biased emotion factors for better user experience,
        # by de-emphasizing emotions that can cause strange results
        if apply_bias:
            # [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
            emo_bias = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]
            emo_vector = [vec * bias for vec, bias in zip(emo_vector, emo_bias)]

        # the total emotion sum must be 0.8 or less
        emo_sum = sum(emo_vector)
        if emo_sum > 0.8:
            scale_factor = 0.8 / emo_sum
            emo_vector = [vec * scale_factor for vec in emo_vector]

        return emo_vector

    def _get_speaker_conditioning(self, spk_audio_prompt, verbose=False):
        """Returns (spk_cond_emb, style, prompt_condition, ref_mel) with caching."""
        if self.cache_spk_cond is None or self.cache_spk_audio_prompt != spk_audio_prompt:
            if self.cache_spk_cond is not None:
                self.cache_spk_cond = None
                self.cache_s2mel_style = None
                self.cache_s2mel_prompt = None
                self.cache_mel = None
                torch.cuda.empty_cache()
            audio, sr = self._load_and_cut_audio(spk_audio_prompt, 15, verbose)
            audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
            audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

            inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
            input_features = inputs["input_features"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            spk_cond_emb = self.get_emb(input_features, attention_mask)

            _, S_ref = self.semantic_codec.quantize(spk_cond_emb)
            ref_mel = self.mel_fn(audio_22k.to(spk_cond_emb.device).float())
            ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)
            feat = torchaudio.compliance.kaldi.fbank(
                audio_16k.to(ref_mel.device), num_mel_bins=80, dither=0, sample_frequency=16000
            )
            feat = feat - feat.mean(dim=0, keepdim=True)
            style = self.campplus_model(feat.unsqueeze(0))

            prompt_condition = self.s2mel.models["length_regulator"](
                S_ref, ylens=ref_target_lengths, n_quantizers=3, f0=None
            )[0]

            self.cache_spk_cond = spk_cond_emb
            self.cache_s2mel_style = style
            self.cache_s2mel_prompt = prompt_condition
            self.cache_spk_audio_prompt = spk_audio_prompt
            self.cache_mel = ref_mel
        else:
            spk_cond_emb = self.cache_spk_cond
            style = self.cache_s2mel_style
            prompt_condition = self.cache_s2mel_prompt
            ref_mel = self.cache_mel

        return spk_cond_emb, style, prompt_condition, ref_mel

    def _get_emotion_conditioning(self, emo_audio_prompt, verbose=False):
        """Returns emo_cond_emb with caching."""
        if self.cache_emo_cond is None or self.cache_emo_audio_prompt != emo_audio_prompt:
            if self.cache_emo_cond is not None:
                self.cache_emo_cond = None
                torch.cuda.empty_cache()
            emo_audio, _ = self._load_and_cut_audio(emo_audio_prompt, 15, verbose, sr=16000)
            emo_inputs = self.extract_features(emo_audio, sampling_rate=16000, return_tensors="pt")
            emo_input_features = emo_inputs["input_features"].to(self.device)
            emo_attention_mask = emo_inputs["attention_mask"].to(self.device)
            emo_cond_emb = self.get_emb(emo_input_features, emo_attention_mask)

            self.cache_emo_cond = emo_cond_emb
            self.cache_emo_audio_prompt = emo_audio_prompt
        else:
            emo_cond_emb = self.cache_emo_cond

        return emo_cond_emb

    def infer(
        self,
        spk_audio_prompt,
        text,
        output_path,
        emo_audio_prompt=None,
        emo_alpha=1.0,
        emo_vector=None,
        use_emo_text=False,
        emo_text=None,
        use_random=False,
        interval_silence=200,
        verbose=False,
        max_text_tokens_per_segment=120,
        stream_return=False,
        more_segment_before=0,
        **generation_kwargs,
    ):
        if stream_return:
            return self.infer_generator(
                spk_audio_prompt,
                text,
                output_path,
                emo_audio_prompt,
                emo_alpha,
                emo_vector,
                use_emo_text,
                emo_text,
                use_random,
                interval_silence,
                verbose,
                max_text_tokens_per_segment,
                stream_return,
                more_segment_before,
                **generation_kwargs,
            )
        else:
            try:
                return list(
                    self.infer_generator(
                        spk_audio_prompt,
                        text,
                        output_path,
                        emo_audio_prompt,
                        emo_alpha,
                        emo_vector,
                        use_emo_text,
                        emo_text,
                        use_random,
                        interval_silence,
                        verbose,
                        max_text_tokens_per_segment,
                        stream_return,
                        more_segment_before,
                        **generation_kwargs,
                    )
                )[0]
            except IndexError:
                return None

    def infer_generator(
        self,
        spk_audio_prompt,
        text,
        output_path,
        emo_audio_prompt=None,
        emo_alpha=1.0,
        emo_vector=None,
        use_emo_text=False,
        emo_text=None,
        use_random=False,
        interval_silence=200,
        verbose=False,
        max_text_tokens_per_segment=120,
        stream_return=False,
        quick_streaming_tokens=0,
        **generation_kwargs,
    ):
        logger.info("starting inference...")
        if verbose:
            logger.debug(
                "origin text:%s, spk_audio_prompt:%s, emo_audio_prompt:%s, emo_alpha:%s, "
                "emo_vector:%s, use_emo_text:%s, emo_text:%s",
                text,
                spk_audio_prompt,
                emo_audio_prompt,
                emo_alpha,
                emo_vector,
                use_emo_text,
                emo_text,
            )
        start_time = time.perf_counter()

        if use_emo_text or emo_vector is not None:
            # we're using a text or emotion vector guidance; so we must remove
            # "emotion reference voice", to ensure we use correct emotion mixing!
            emo_audio_prompt = None

        if use_emo_text:
            # automatically generate emotion vectors from text prompt
            if emo_text is None:
                emo_text = text  # use main text prompt
            emo_dict = self.qwen_emo.inference(emo_text)
            logger.debug("detected emotion vectors from text: %s", emo_dict)
            # convert ordered dict to list of vectors; the order is VERY important!
            emo_vector = list(emo_dict.values())

        if emo_vector is not None:
            # we have emotion vectors; they can't be blended via alpha mixing
            # in the main inference process later, so we must pre-calculate
            # their new strengths here based on the alpha instead!
            emo_vector_scale = max(0.0, min(1.0, emo_alpha))
            if emo_vector_scale != 1.0:
                # scale each vector and truncate to 4 decimals (for nicer printing)
                emo_vector = [int(x * emo_vector_scale * 10000) / 10000 for x in emo_vector]
                logger.debug("scaled emotion vectors to %sx: %s", emo_vector_scale, emo_vector)

        if emo_audio_prompt is None:
            # we are not using any external "emotion reference voice"; use
            # speaker's voice as the main emotion reference audio.
            emo_audio_prompt = spk_audio_prompt
            # must always use alpha=1.0 when we don't have an external reference voice
            emo_alpha = 1.0

        spk_cond_emb, style, prompt_condition, ref_mel = self._get_speaker_conditioning(spk_audio_prompt, verbose)

        if emo_vector is not None:
            weight_vector = torch.tensor(emo_vector, device=self.device)
            if use_random:
                random_index = [random.randint(0, x - 1) for x in self.emo_num]
            else:
                random_index = [find_most_similar_cosine(style, tmp) for tmp in self.spk_matrix]

            emo_matrix = [tmp[index].unsqueeze(0) for index, tmp in zip(random_index, self.emo_matrix)]
            emo_matrix = torch.cat(emo_matrix, 0)
            emovec_mat = weight_vector.unsqueeze(1) * emo_matrix
            emovec_mat = torch.sum(emovec_mat, 0)
            emovec_mat = emovec_mat.unsqueeze(0)

        emo_cond_emb = self._get_emotion_conditioning(emo_audio_prompt, verbose)

        text_tokens_list = self.tokenizer.tokenize(text)
        segments = self.tokenizer.split_segments(
            text_tokens_list, max_text_tokens_per_segment, quick_streaming_tokens=quick_streaming_tokens
        )
        segments_count = len(segments)

        text_token_ids = self.tokenizer.convert_tokens_to_ids(text_tokens_list)
        if self.tokenizer.unk_token_id in text_token_ids:
            logger.warning(
                "Input text contains %d unknown tokens (id=%d): %s. "
                "Consider updating the BPE model or modifying the text to avoid unknown tokens.",
                text_token_ids.count(self.tokenizer.unk_token_id),
                self.tokenizer.unk_token_id,
                [t for t, id in zip(text_tokens_list, text_token_ids) if id == self.tokenizer.unk_token_id],
            )

        if verbose:
            logger.debug("text_tokens_list: %s", text_tokens_list)
            logger.debug("segments count: %d", segments_count)
            logger.debug("max_text_tokens_per_segment: %d", max_text_tokens_per_segment)
            for seg in segments:
                logger.debug("%s", seg)
        generation_kwargs.pop("do_sample", None)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 0.8)
        autoregressive_batch_size = 1
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 1500)
        sampling_rate = 22050

        wavs = []
        gpt_gen_time = 0
        gpt_forward_time = 0
        s2mel_time = 0
        bigvgan_time = 0
        has_warned = False
        silence = None  # for stream_return
        for seg_idx, sent in enumerate(segments):
            text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
            text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)
            if verbose:
                logger.debug("%s", text_tokens)
                logger.debug("text_tokens shape: %s, text_tokens type: %s", text_tokens.shape, text_tokens.dtype)
                # debug tokenizer
                text_token_syms = self.tokenizer.convert_ids_to_tokens(text_tokens[0].tolist())
                logger.debug("text_token_syms is same as segment tokens: %s", text_token_syms == sent)

            m_start_time = time.perf_counter()
            with torch.no_grad():
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    emovec = self.gpt.merge_emovec(
                        spk_cond_emb,
                        emo_cond_emb,
                        torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                        torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                        alpha=emo_alpha,
                    )

                    if emo_vector is not None:
                        emovec = emovec_mat + (1 - torch.sum(weight_vector)) * emovec

                    codes, speech_conditioning_latent = self.gpt.inference_speech(
                        spk_cond_emb,
                        text_tokens,
                        emo_cond_emb,
                        cond_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_cond_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_vec=emovec,
                        do_sample=True,
                        top_p=top_p,
                        top_k=top_k,
                        temperature=temperature,
                        num_return_sequences=autoregressive_batch_size,
                        length_penalty=length_penalty,
                        num_beams=num_beams,
                        repetition_penalty=repetition_penalty,
                        max_generate_length=max_mel_tokens,
                        **generation_kwargs,
                    )

                gpt_gen_time += time.perf_counter() - m_start_time
                if not has_warned and (codes[:, -1] != self.stop_mel_token).any():
                    warnings.warn(
                        f"WARN: generation stopped due to exceeding `max_mel_tokens` ({max_mel_tokens}). "
                        f"Input text tokens: {text_tokens.shape[1]}. "
                        f"Consider reducing `max_text_tokens_per_segment`({max_text_tokens_per_segment}) or increasing `max_mel_tokens`.",
                        category=RuntimeWarning,
                    )
                    has_warned = True

                code_lens = []
                max_code_len = 0
                for code in codes:
                    if self.stop_mel_token not in code:
                        code_len = len(code)
                    else:
                        len_ = (code == self.stop_mel_token).nonzero(as_tuple=False)[0]
                        code_len = len_[0].item() if len_.numel() > 0 else len(code)
                    code_lens.append(code_len)
                    max_code_len = max(max_code_len, code_len)
                codes = codes[:, :max_code_len]
                code_lens = torch.LongTensor(code_lens)
                code_lens = code_lens.to(self.device)
                if verbose:
                    logger.debug("%s %s", codes, type(codes))
                    logger.debug("fix codes shape: %s, codes type: %s", codes.shape, codes.dtype)
                    logger.debug("code len: %s", code_lens)

                m_start_time = time.perf_counter()
                use_speed = torch.zeros(spk_cond_emb.size(0)).to(spk_cond_emb.device).long()
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    latent = self.gpt(
                        speech_conditioning_latent,
                        text_tokens,
                        torch.tensor([text_tokens.shape[-1]], device=text_tokens.device),
                        codes,
                        torch.tensor([codes.shape[-1]], device=text_tokens.device),
                        emo_cond_emb,
                        cond_mel_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_cond_mel_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_vec=emovec,
                        use_speed=use_speed,
                    )
                    gpt_forward_time += time.perf_counter() - m_start_time

                dtype = None
                with torch.amp.autocast(text_tokens.device.type, enabled=dtype is not None, dtype=dtype):
                    m_start_time = time.perf_counter()
                    diffusion_steps = 25
                    inference_cfg_rate = 0.7
                    latent = self.s2mel.models["gpt_layer"](latent)
                    S_infer = self.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))
                    S_infer = S_infer.transpose(1, 2)
                    S_infer = S_infer + latent
                    target_lengths = (code_lens * 1.72).long()

                    cond = self.s2mel.models["length_regulator"](
                        S_infer, ylens=target_lengths, n_quantizers=3, f0=None
                    )[0]
                    cat_condition = torch.cat([prompt_condition, cond], dim=1)
                    vc_target = self.s2mel.models["cfm"].inference(
                        cat_condition,
                        torch.LongTensor([cat_condition.size(1)]).to(cond.device),
                        ref_mel,
                        style,
                        None,
                        diffusion_steps,
                        inference_cfg_rate=inference_cfg_rate,
                    )
                    vc_target = vc_target[:, :, ref_mel.size(-1) :]
                    s2mel_time += time.perf_counter() - m_start_time

                    m_start_time = time.perf_counter()
                    wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0)
                    logger.debug("wav shape: %s", wav.shape)
                    bigvgan_time += time.perf_counter() - m_start_time
                    wav = wav.squeeze(1)

                wav = torch.clamp(wav, -1.0, 1.0)
                if verbose:
                    logger.debug("wav shape: %s min: %s max: %s", wav.shape, wav.min(), wav.max())
                wavs.append(wav.cpu())
                if stream_return:
                    yield wav.cpu()
                    if silence is None:
                        silence = self.interval_silence(
                            wavs, sampling_rate=sampling_rate, interval_silence=interval_silence
                        )
                    yield silence
        end_time = time.perf_counter()

        wavs = self.insert_interval_silence(wavs, sampling_rate=sampling_rate, interval_silence=interval_silence)
        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / sampling_rate
        logger.debug("gpt_gen_time: %.2f seconds", gpt_gen_time)
        logger.debug("gpt_forward_time: %.2f seconds", gpt_forward_time)
        logger.debug("s2mel_time: %.2f seconds", s2mel_time)
        logger.debug("bigvgan_time: %.2f seconds", bigvgan_time)
        logger.debug("Total inference time: %.2f seconds", end_time - start_time)
        logger.debug("Generated audio length: %.2f seconds", wav_length)
        logger.debug("RTF: %.4f", (end_time - start_time) / wav_length)

        # save audio
        wav = wav.cpu()  # to cpu
        if output_path:
            if os.path.isfile(output_path):
                os.remove(output_path)
                logger.debug("remove old wav file: %s", output_path)
            if os.path.dirname(output_path) != "":
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav.float(), sampling_rate, bits_per_sample=16)
            logger.info("wav file saved to: %s", output_path)
            if stream_return:
                return None
            yield output_path
        else:
            if stream_return:
                return None
            wav_data = (wav.float() * 32767).clamp(-32767, 32767).to(torch.int16)
            wav_data = wav_data.numpy().T
            yield (sampling_rate, wav_data)
