"""
Flask web application for voice-to-doraemon converter.
"""

import os
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
from pathlib import Path
import tempfile
import traceback
from src.voice_converter import DoraemonConverter

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Ensure upload folder exists
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

# Initialize converter
converter = DoraemonConverter(device='cuda')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/convert', methods=['POST'])
def convert_voice():
    """
    API endpoint for voice conversion.
    Expects multipart form data with audio file and parameters.
    """
    try:
        # Check if file is present
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Use: wav, mp3, ogg, m4a'}), 400

        # Get parameters
        pitch = float(request.form.get('pitch', 2.5))
        speed = float(request.form.get('speed', 1.0))

        # Save uploaded file
        filename = secure_filename(file.filename)
        temp_input = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_{filename}')
        file.save(temp_input)

        # Set converter parameters
        converter.set_pitch_shift(pitch)
        converter.set_speed(speed)

        # Convert audio
        temp_output = os.path.join(app.config['UPLOAD_FOLDER'], f'out_{filename}')
        success = converter.convert_audio(temp_input, temp_output)

        if not success:
            return jsonify({'error': 'Conversion failed'}), 500

        # Send file
        return send_file(
            temp_output,
            mimetype='audio/wav',
            as_attachment=True,
            download_name=f'doraemon_{filename}'
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

    finally:
        # Cleanup
        if os.path.exists(temp_input):
            os.remove(temp_input)


@app.route('/api/info', methods=['GET'])
def get_info():
    """Get converter information."""
    return jsonify({
        'name': 'Voice to Doraemon',
        'version': '1.0.0',
        'max_file_size': MAX_FILE_SIZE / (1024 * 1024),
        'supported_formats': list(ALLOWED_EXTENSIONS)
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'ok'})


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server error."""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
