from indextts import IndexTTS2

MODEL_DIR = "/home/kimox/Projects/finetune-indextts2/checkpoints"
CFG_PATH = f"{MODEL_DIR}/config_es.yaml"
SPK_AUDIO = "/home/kimox/Projects/finetune-indextts2/emotion_refs/default.wav"

tts = IndexTTS2(model_dir=MODEL_DIR, cfg_path=CFG_PATH, language="es")
tts.infer(
    spk_audio_prompt=SPK_AUDIO,
    text="Hola, esto es una prueba del paquete index tts inference, usando el modelo finetuneado en español.",
    output_path="test_output.wav",
)
print("Done! Output saved to test_output.wav")
