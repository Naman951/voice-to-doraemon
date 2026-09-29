# Voice to Doraemon 🎙️➡️🤖

Convert human voice to Doraemon cartoon character voice using advanced AI voice conversion and synthesis techniques.

## Overview

This project uses cutting-edge audio processing, voice conversion, and text-to-speech technologies to transform any human voice into the distinctive voice of Doraemon, the famous robot cat from the anime series.

## Features

- 🎯 **Voice Conversion**: Transform any voice input to match Doraemon's voice characteristics
- 🎤 **Real-time Processing**: Process audio files or live microphone input
- 🎵 **High-Quality Output**: Crystal clear audio synthesis
- 📁 **Batch Processing**: Convert multiple audio files at once
- 🎛️ **Customizable Parameters**: Adjust pitch, speed, and tone
- 📊 **Audio Analysis**: Visualize waveforms and spectrograms

## Tech Stack

- **Audio Processing**: librosa, scipy
- **Voice Conversion**: PyTorch, VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)
- **Speech Recognition**: Google Speech-to-Text API / Whisper
- **Web Interface**: Flask/FastAPI
- **Deep Learning**: PyTorch

## Installation

### Prerequisites
- Python 3.8+
- FFmpeg
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/Naman951/voice-to-doraemon.git
cd voice-to-doraemon

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models
python scripts/download_models.py
```

## Usage

### Command Line

```bash
# Convert a single audio file
python convert.py --input input_audio.wav --output output_audio.wav

# Batch processing
python convert.py --input input_folder/ --output output_folder/

# Real-time voice conversion
python real_time_convert.py
```

### Python API

```python
from voice_converter import DoraemonConverter

# Initialize converter
converter = DoraemonConverter(model_path='models/doraemon_model.pth')

# Convert audio file
converter.convert_audio('input.wav', 'output.wav')

# Get audio array
output_audio = converter.convert_array(input_audio_array)
```

### Web Interface

```bash
python app.py
# Visit http://localhost:5000
```

## Project Structure

```
voice-to-doraemon/
├── README.md
├── requirements.txt
├── setup.py
├── convert.py                 # Main conversion script
├── real_time_convert.py       # Real-time processing
├── app.py                     # Web interface
├── scripts/
│   ├── download_models.py     # Download pre-trained models
│   └── train_model.py         # Model training script
├── src/
│   ├── __init__.py
│   ├── voice_converter.py     # Core conversion logic
│   ├── audio_processor.py     # Audio processing utilities
│   ├── models.py              # Model architectures
│   └── utils.py               # Helper functions
├── models/                    # Pre-trained model weights
├── tests/
│   ├── test_converter.py
│   └── test_audio.py
├── examples/
│   ├── sample_input.wav
│   └── sample_output.wav
└── templates/                 # Web UI templates
    └── index.html
```

## How It Works

1. **Audio Input**: Accept audio file or microphone input
2. **Speech Recognition**: Convert speech to text (optional)
3. **Feature Extraction**: Extract audio features (MFCC, spectrograms)
4. **Voice Conversion**: Use neural network to map voice characteristics
5. **Audio Synthesis**: Generate output audio with Doraemon voice
6. **Post-processing**: Apply filters and normalization
7. **Output**: Save or stream the converted audio

## Models Used

- **VITS**: For high-quality voice synthesis
- **WaveGlow/MelGAN**: For vocoding
- **Whisper**: For speech recognition
- **ResNet/Transformer**: For feature mapping

## Training

To train a custom model on Doraemon voice samples:

```bash
python scripts/train_model.py --config config/train.yaml --data path/to/doraemon_audio/
```

## API Reference

### DoraemonConverter Class

```python
class DoraemonConverter:
    def __init__(self, model_path, device='cuda')
    def convert_audio(self, input_path, output_path)
    def convert_array(self, audio_array, sr=22050)
    def set_pitch_shift(self, semitones)
    def set_speed(self, factor)
```

## Examples

See the `examples/` directory for sample usage and audio files.

## Performance

- Input processing: ~0.5s per second of audio
- Conversion: ~2-5s per second of audio (CPU)
- GPU acceleration: ~0.5-1s per second of audio

## Limitations

- Works best with clear, monophonic speech
- Background noise may affect quality
- Voice should be within vocal range of Doraemon

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This project is for educational and entertainment purposes only. Doraemon is a copyrighted character owned by Fujiko F. Fujio. This tool does not claim ownership or create any commercial product.

## References

- [VITS: A Single Stage Text-to-Speech with Conditional Adversarial Learning](https://arxiv.org/abs/2106.06103)
- [Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron](https://arxiv.org/abs/1906.09672)
- [WaveGlow: A Generative Flow for Raw Audio](https://arxiv.org/abs/1811.00002)

## Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

Made with ❤️ for Doraemon fans!
