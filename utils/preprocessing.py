from phonemizer import phonemize
import re
import torchaudio
import soundfile as sf
import librosa
import numpy as np


# text to phoneme
class Text2Phoneme:
    def __init__(self, language):
        self.language = language

    def text2phoneme(self, batch):
        """Convert text to phoneme."""
        batch["sentence"] = phonemize(batch["sentence"], language=self.language, backend="espeak")
        return batch


# Preprocessing
class Preprocess:
    def __init__(self, chars_to_ignore_regex=None, is_torch=True, source_sampling=48_000, target_sampling=16_000):
        self.chars_to_ignore_regex = chars_to_ignore_regex
        self.is_torch = is_torch
        self.source_sampling = source_sampling
        self.target_sampling = target_sampling

    def remove_special_characters(self, batch):
        """Remove special characters (the one in chars_to_ignore_regex)."""
        batch["text"] = re.sub(self.chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
        batch["text"] = batch["text"].replace('`', '’')
        return batch

    # Audio Preprocessing
    def speech_file_to_array_fn(self, batch):
        if self.is_torch:
            speech_array, sampling_rate = torchaudio.load(batch["path"])
        else:
            speech_array, sampling_rate = sf.load(batch["path"])
        batch["speech"] = speech_array[0].numpy()
        batch["sampling_rate"] = sampling_rate
        batch["target_text"] = batch["text"]
        return batch

    def resample(self, batch):
        batch["speech"] = librosa.resample(np.asarray(batch["speech"]), self.source_sampling, self.target_sampling)
        batch["sampling_rate"] = self.target_sampling
        return batch


# Vocabulary
def extract_all_chars(batch):
    """Extract all unique characters from a text"""
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}
