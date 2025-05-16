
# Emotion Detection Using Speech and Facial Recognition

## Overview

This project implements a multimodal emotion detection system that combines facial expression recognition using CNNs and speech emotion recognition using Whisper (LLM) with audio feature extraction.

---

## Features

- Detects 7 facial emotions using Mini-XCEPTION CNN
- Transcribes and classifies speech-based emotions using Whisper + MFCC + SVM
- Interactive Streamlit dashboard for audio/image input
- Works with pre-trained models, supports Google Colab

---

## Technologies Used

- **Facial Emotion**: OpenCV, Keras, Mini-XCEPTION, FER2013 Dataset
- **Speech Emotion**: Whisper LLM, Librosa, scikit-learn, MFCC & Pitch Features
- **Interface**: Streamlit
- **Language**: Python

---

## System Pipelines

### Facial Emotion Recognition:
```
[Image Input] → [Face Detection] → [Preprocessing] → [Mini-XCEPTION Model] → [Emotion Label]
```

### Speech Emotion Recognition:
```
[Audio Input] → [WAV Conversion] → [MFCC/Pitch Extraction] → [SVM Classifier] → [Emotion Label]
```

---

## Setup

```bash
pip install keras opencv-python openai-whisper librosa==0.10.0.post2 scikit-learn numpy==1.23.5 pydub ffmpeg-python streamlit
```

---

## Usage

Run the notebook or scripts from Google Colab:

- Upload an image for facial emotion prediction
- Upload an audio file (WAV preferred) for speech emotion and transcription
- View annotated output image or detected emotion + language

To launch Streamlit dashboard:
```bash
streamlit run app.py
```

---

## Limitations

- Facial model assumes frontal images
- Speech classifier is trained on synthetic data
- Separate handling of facial and audio inputs
- Not real-time capable in current version

---

## Future Enhancements

- Real-time webcam and mic integration
- Use wav2vec2.0 or SER datasets for improved audio accuracy
- Multimodal fusion of visual and audio features
- Context-aware emotion detection via LLMs

---

## Authors

- [Your Name]
- [Collaborators or Supervisors]a

## License

This project is licensed under the MIT License.
