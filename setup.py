"""
Setup script for voice-to-doraemon package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="voice-to-doraemon",
    version="1.0.0",
    author="Naman951",
    description="Convert human voice to Doraemon cartoon character voice",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Naman951/voice-to-doraemon",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "Flask>=2.3.0",
    ],
    entry_points={
        "console_scripts": [
            "voice-to-doraemon=convert:main",
        ],
    },
)
