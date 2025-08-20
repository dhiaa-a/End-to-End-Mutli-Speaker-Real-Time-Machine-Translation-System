# NLP-Enhanced Multi-Speaker ASR System

## Project Overview

This repository contains the complete implementation of an **NLP-Enhanced Multi-Speaker Automatic Speech Recognition System** - a comprehensive 7-stage audio processing pipeline for multi-speaker environments.

## 🎯 Project Goals

- **Real-time multi-speaker speech recognition** with high accuracy
- **Speaker separation and diarization** in noisy environments
- **NLP-based post-processing** for enhanced transcription quality
- **Multi-language translation** capabilities
- **End-to-end audio-to-speech synthesis** pipeline

## 🏗️ System Architecture

### Stage 1: Initial Mixer
**Location**: `stage 1 - initial mixer/`
- **Purpose**: Audio preprocessing and noise mixing
- **Input**: Clean audio files (male_1.flac, female_1.flac, etc.)
- **Output**: Mixed audio with controlled noise levels
- **Key Features**:
  - Multi-speaker audio mixing
  - Noise injection and control
  - Spectrogram generation for analysis
  - Audio quality assessment

### Stage 2: Smart Denoiser 
**Location**: `stage 2 - smart denoiser/`
- **Purpose**: Adaptive noise reduction while preserving overlapped speech
- **Input**: Noisy mixed audio from Stage 1
- **Output**: Cleaned audio with reduced noise
- **Key Features**:
  - Adaptive spectral subtraction
  - Overlap-aware noise reduction
  - Voice activity detection
  - Frequency response analysis

### Stage 3: Speaker Separation
**Location**: `stage 3 - speaker seperation/`
- **Purpose**: Speaker diarization and separation using SepFormer
- **Input**: Cleaned audio from Stage 2
- **Output**: Individual speaker tracks
- **Key Features**:
  - PyAnnote-based speaker diarization
  - SepFormer model for source separation
  - Speaker embedding and classification
  - Evaluation metrics and quality assessment

### Stage 4: Automatic Speech Recognition (ASR)
**Location**: `stage 4 - ASR/`
- **Purpose**: Speech-to-text conversion using Whisper
- **Input**: Separated speaker tracks from Stage 3
- **Output**: Raw transcriptions per speaker
- **Key Features**:
  - OpenAI Whisper integration
  - Multi-speaker transcription
  - Transcription enhancement
  - Quality metrics and evaluation

### Stage 5: NLP Correction
**Location**: `stage 5 - NLP correction/`
- **Purpose**: Post-processing text correction using NLP models
- **Input**: Raw transcriptions from Stage 4
- **Output**: Enhanced and corrected transcriptions
- **Key Features**:
  - Grammar and spelling correction
  - Context-aware text enhancement
  - Performance visualization
  - Correction confidence scoring

### Stage 6: Translation
**Location**: `stage 6 - translation/`
- **Purpose**: Multi-language translation to Arabic
- **Input**: Corrected transcriptions from Stage 5
- **Output**: Translated text in Arabic
- **Key Features**:
  - Neural machine translation
  - Multi-speaker translation preservation
  - Translation quality assessment
  - Language detection and processing

### Stage 7: Text-to-Speech (TTS)
**Location**: `stage 7 - tts/`
- **Purpose**: Speech synthesis from translated text
- **Input**: Translated text from Stage 6
- **Output**: Synthesized speech audio
- **Key Features**:
  - High-quality speech synthesis
  - Voice cloning and adaptation
  - Multi-language TTS support
  - Audio quality optimization

## 📊 Key Technologies

### Machine Learning & AI
- **SepFormer**: State-of-the-art source separation
- **PyAnnote Audio**: Speaker diarization and embedding
- **OpenAI Whisper**: Automatic speech recognition
- **Transformers**: NLP correction and enhancement
- **Neural Machine Translation**: Multi-language support

### Audio Processing
- **LibROSA**: Audio analysis and feature extraction
- **SoundFile**: Audio I/O operations
- **SciPy**: Signal processing and filtering
- **NumPy**: Numerical computations

### Data Science & Visualization
- **Matplotlib**: Visualization and plotting
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning utilities
- **Seaborn**: Statistical data visualization

## 🔧 Development Environment

### Python Environment
- **Python 3.12**: Core runtime
- **Virtual Environment**: Isolated dependencies
- **Requirements**: See individual stage requirements

### Key Dependencies
- `torch` - PyTorch for deep learning models
- `transformers` - Hugging Face transformer models
- `librosa` - Audio processing and analysis
- `soundfile` - Audio file I/O
- `pyannote-audio` - Speaker diarization
- `speechbrain` - Speech processing toolkit
- `whisper` - OpenAI speech recognition

## 📁 Data Flow

```
Input Audio → Mixing & Noise → Denoising → Speaker Separation 
     ↓              ↓             ↓              ↓
Raw Audio → Mixed Audio → Clean Audio → Individual Speakers
     ↓              ↓             ↓              ↓
ASR Processing → NLP Correction → Translation → TTS Output
     ↓              ↓             ↓              ↓
Transcriptions → Enhanced Text → Arabic Text → Synthesized Speech
```

## 📈 Performance Metrics

### Audio Quality
- **Signal-to-Noise Ratio (SNR)** improvements
- **Spectral similarity** measurements
- **Voice activity detection** accuracy

### Speaker Separation
- **Speaker diarization** error rates
- **Source separation** quality metrics
- **Speaker identification** accuracy

### ASR Performance
- **Word Error Rate (WER)** measurements
- **Transcription quality** assessments
- **Real-time factor** analysis

### NLP Enhancement
- **Correction accuracy** metrics
- **Grammar improvement** scores
- **Context preservation** evaluation

## 🎓 Thesis Contributions

1. **Adaptive Noise Reduction**: Novel approach preserving overlapped speech
2. **Multi-Stage Processing**: Comprehensive end-to-end pipeline
3. **Real-Time Performance**: Optimized for live audio processing
4. **Quality Enhancement**: NLP-based post-processing improvements
5. **Multi-Language Support**: Arabic translation and TTS integration

## 📝 Evaluation & Results

### Quantitative Results
- **Processing Time**: Real-time factor analysis
- **Accuracy Metrics**: WER, BLEU scores, quality assessments
- **Performance Benchmarks**: Comparison with baseline systems

### Qualitative Analysis
- **Audio Quality**: Subjective listening tests
- **Translation Quality**: Human evaluation scores
- **System Usability**: User experience assessment

## 🔬 Research Impact

This system demonstrates significant advances in:
- **Multi-speaker ASR** in noisy environments
- **Real-time audio processing** pipelines
- **Cross-language speech technology**
- **End-to-end speech processing** systems

## ⚠️ Important Notes

### Large Files Excluded
- Model checkpoints (*.ckpt) are excluded from the repository
- Pre-trained models must be downloaded separately
- Audio data files are included for demonstration purposes

### Environment Setup
1. Create virtual environment: `python -m venv venv`
2. Activate environment: `venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`
4. Download required models (see individual stage documentation)

### Hardware Requirements
- **GPU**: CUDA-compatible GPU recommended for model inference
- **RAM**: Minimum 16GB for processing large audio files
- **Storage**: SSD recommended for fast model loading

## 📞 Contact & Support

For questions regarding implementation details, model configurations, or research methodology, please refer to the thesis documentation or contact the research team.

---

**Project Type**: Master's Thesis Research  
**Domain**: Speech Processing, Machine Learning, NLP  
**Status**: Complete Implementation  
**Languages**: Python, English/Arabic Processing
