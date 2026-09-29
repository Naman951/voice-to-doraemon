"""
Main voice converter module for transforming human voice to Doraemon voice.
"""

import numpy as np
import librosa
import soundfile as sf
import torch
from pathlib import Path
from typing import Union, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class DoraemonConverter:
    """
    Convert human voice to Doraemon cartoon character voice.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        sample_rate: int = 22050
    ):
        """
        Initialize the Doraemon voice converter.

        Args:
            model_path: Path to pre-trained model weights
            device: Device to run inference on ('cuda' or 'cpu')
            sample_rate: Audio sample rate in Hz
        """
        self.device = device
        self.sample_rate = sample_rate
        self.pitch_shift = 0  # Semitones
        self.speed_factor = 1.0
        self.model = None

        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            print("Warning: No model path provided. Initialize model before conversion.")

    def load_model(self, model_path: str) -> None:
        """Load pre-trained model from disk."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            print(f"Model loaded from {model_path}")
            # Model initialization logic here
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def convert_audio(
        self,
        input_path: str,
        output_path: str,
        preserve_length: bool = True
    ) -> bool:
        """
        Convert an audio file from human voice to Doraemon voice.

        Args:
            input_path: Path to input audio file
            output_path: Path to save converted audio
            preserve_length: Keep original audio length

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load audio
            audio, sr = librosa.load(input_path, sr=self.sample_rate)
            print(f"Loaded audio: {len(audio)/sr:.2f}s")

            # Process audio
            converted_audio = self.convert_array(audio)

            # Save output
            sf.write(output_path, converted_audio, self.sample_rate)
            print(f"Saved converted audio to {output_path}")
            return True

        except Exception as e:
            print(f"Error converting audio: {e}")
            return False

    def convert_array(
        self,
        audio_array: np.ndarray,
        sr: int = 22050
    ) -> np.ndarray:
        """
        Convert audio array to Doraemon voice.

        Args:
            audio_array: Input audio as numpy array
            sr: Sample rate of input audio

        Returns:
            Converted audio array
        """
        # Resample if necessary
        if sr != self.sample_rate:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=self.sample_rate)

        # Apply voice conversion pipeline
        audio_array = self._preprocess(audio_array)
        audio_array = self._voice_conversion(audio_array)
        audio_array = self._postprocess(audio_array)

        return audio_array

    def _preprocess(self, audio: np.ndarray) -> np.ndarray:
        """
        Preprocess audio before conversion.

        Args:
            audio: Input audio array

        Returns:
            Preprocessed audio
        """
        # Normalize audio
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

        # Remove DC offset
        audio = audio - np.mean(audio)

        # Apply high-pass filter to remove low frequency noise
        audio = librosa.effects.preemphasis(audio)

        return audio

    def _voice_conversion(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply voice conversion transformation.

        Args:
            audio: Preprocessed audio

        Returns:
            Voice-converted audio
        """
        # Extract features
        S = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)

        # Apply pitch shift
        if self.pitch_shift != 0:
            audio = librosa.effects.pitch_shift(
                audio,
                sr=self.sample_rate,
                n_steps=self.pitch_shift
            )

        # Apply time stretching
        if self.speed_factor != 1.0:
            audio = librosa.effects.time_stretch(audio, rate=self.speed_factor)

        return audio

    def _postprocess(self, audio: np.ndarray) -> np.ndarray:
        """
        Postprocess audio after conversion.

        Args:
            audio: Voice-converted audio

        Returns:
            Final audio array
        """
        # Normalize to [-1, 1]
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        # Apply gentle fade in/out
        fade_len = int(0.05 * self.sample_rate)  # 50ms
        if len(audio) > 2 * fade_len:
            audio[:fade_len] *= np.linspace(0, 1, fade_len)
            audio[-fade_len:] *= np.linspace(1, 0, fade_len)

        return audio

    def set_pitch_shift(self, semitones: float) -> None:
        """
        Set pitch shift in semitones (Doraemon typically ~2-3 semitones higher).

        Args:
            semitones: Number of semitones to shift (-12 to +12)
        """
        self.pitch_shift = np.clip(semitones, -12, 12)
        print(f"Pitch shift set to {self.pitch_shift} semitones")

    def set_speed(self, factor: float) -> None:
        """
        Set playback speed factor.

        Args:
            factor: Speed factor (1.0 = normal, >1.0 = faster, <1.0 = slower)
        """
        self.speed_factor = np.clip(factor, 0.5, 2.0)
        print(f"Speed factor set to {self.speed_factor}")

    def batch_convert(
        self,
        input_folder: str,
        output_folder: str,
        extension: str = '.wav'
    ) -> int:
        """
        Convert multiple audio files in a folder.

        Args:
            input_folder: Path to input folder
            output_folder: Path to output folder
            extension: File extension to process

        Returns:
            Number of files successfully converted
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        files = list(input_path.glob(f'*{extension}'))
        success_count = 0

        for file in files:
            output_file = output_path / f"{file.stem}_doraemon{extension}"
            if self.convert_audio(str(file), str(output_file)):
                success_count += 1

        print(f"Successfully converted {success_count}/{len(files)} files")
        return success_count

    def get_audio_info(self, audio_path: str) -> dict:
        """Get audio file information."""
        audio, sr = librosa.load(audio_path, sr=None)
        return {
            'duration': len(audio) / sr,
            'sample_rate': sr,
            'n_samples': len(audio),
            'n_channels': 1
        }
