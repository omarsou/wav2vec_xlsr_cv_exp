from phonemizer import phonemize
import re
import torchaudio
import soundfile as sf
import librosa
import numpy as np


# text to phoneme
def text2phoneme(batch):
    """Convert text to phoneme."""
    batch["sentence"] = phonemize(batch["sentence"], language='cs', backend="espeak")
    return batch


# Preprocessing
def remove_special_characters(batch, chars_to_ignore_regex):
    """Remove special characters (the one in chars_to_ignore_regex)."""
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    batch["text"] = batch["text"].replace('`', '’')
    return batch


# Vocabulary
def extract_all_chars(batch):
    """Extract all unique characters from a text"""
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


# Audio Preprocessing
def speech_file_to_array_fn(batch, is_torch=True):
    if is_torch:
        speech_array, sampling_rate = torchaudio.load(batch["path"])
    else:
        speech_array, sampling_rate = sf.load(batch["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["text"]
    return batch


def resample(batch, source_sampling=48_000, target_sampling=16_000):
    batch["speech"] = librosa.resample(np.asarray(batch["speech"]), source_sampling, target_sampling)
    batch["sampling_rate"] = target_sampling
    return batch
