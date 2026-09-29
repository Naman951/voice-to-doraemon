"""
Command-line script for converting human voice to Doraemon voice.
"""

import argparse
import sys
from pathlib import Path
from src.voice_converter import DoraemonConverter


def main():
    parser = argparse.ArgumentParser(
        description='Convert human voice to Doraemon cartoon character voice',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Convert single file
  python convert.py --input input.wav --output output.wav
  
  # Batch processing
  python convert.py --input input_folder/ --output output_folder/
  
  # With parameters
  python convert.py --input audio.wav --output out.wav --pitch 3 --speed 1.1
        '''
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input audio file or folder path'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output audio file or folder path'
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        default='models/doraemon_model.pth',
        help='Path to model weights'
    )

    parser.add_argument(
        '--pitch',
        type=float,
        default=2.5,
        help='Pitch shift in semitones (default: 2.5, Doraemon is higher pitched)'
    )

    parser.add_argument(
        '--speed',
        type=float,
        default=1.0,
        help='Playback speed factor (default: 1.0)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use (default: auto-detect)'
    )

    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    # Initialize converter
    converter = DoraemonConverter(model_path=args.model, device=device)
    converter.set_pitch_shift(args.pitch)
    converter.set_speed(args.speed)

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Process input
    if input_path.is_file():
        # Single file conversion
        print(f"Converting: {args.input}")
        success = converter.convert_audio(str(input_path), str(output_path))
        sys.exit(0 if success else 1)

    elif input_path.is_dir():
        # Batch conversion
        print(f"Batch converting from: {args.input}")
        print(f"Output folder: {args.output}")
        converter.batch_convert(str(input_path), str(output_path))
        sys.exit(0)

    else:
        print(f"Error: Input path not found: {args.input}")
        sys.exit(1)


if __name__ == '__main__':
    main()
