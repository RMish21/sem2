# Medical Audio Transcription and Evaluation using Whisper

## Overview
This project implements an Automatic Speech Recognition (ASR) evaluation system using OpenAI's Whisper model (large-v3) specifically for Hindi language audio transcription. The system is designed to transcribe medical audio files and evaluate the transcription quality against ground truth data.

## Purpose
The code evaluates the accuracy of speech-to-text transcription for Hindi medical audio recordings by:
- Transcribing audio files using the Whisper large-v3 model
- Comparing transcriptions against known ground truth text
- Calculating Word Error Rate (WER) and Character Error Rate (CER) metrics
- Providing performance assessments

## Key Features

### 1. **Audio Processing**
- Loads audio files in various formats (FLAC, WAV, etc.)
- Converts stereo audio to mono
- Resamples audio to 16kHz (required for Whisper)
- Normalizes audio waveforms for consistent processing

### 2. **Ground Truth Management**
- Reads ground truth transcriptions from `grounds_truth.txt`
- Supports two formats:
  - `audio_file.flac | transcription text`
  - `transcription text` (defaults to "audio.flac")

### 3. **Hindi Transcription**
- Uses Whisper large-v3 model pre-trained on multilingual data
- Specifically configured for Hindi language transcription
- Employs beam search (5 beams) for improved accuracy
- Generates up to 440 new tokens per transcription

### 4. **Evaluation Metrics**
- **Word Error Rate (WER)**: Measures word-level transcription errors
- **Character Error Rate (CER)**: Measures character-level transcription errors
- Provides per-file and average metrics across all audio files

### 5. **Performance Assessment**
- **Good**: WER < 0.2 (less than 20% word errors)
- **Fair**: WER between 0.2 and 0.4
- **Needs Improvement**: WER > 0.4

## Requirements

```python
torch
torchaudio
transformers
jiwer
```

## File Structure

```
project_folder/
│
├── med_audi_whip.ipynb    # Main evaluation script
├── grounds_truth.txt       # Ground truth transcriptions
├── audio.flac              # Audio file(s) to transcribe
└── README.md               # This file
```

## Ground Truth File Format

Create a `grounds_truth.txt` file in the same directory with one of these formats:

**Format 1** (with filename):
```
audio1.flac | यह एक परीक्षण है
audio2.flac | मरीज की स्थिति स्थिर है
```

**Format 2** (single file, defaults to "audio.flac"):
```
यह एक परीक्षण है
```

## How It Works

### Step 1: Model Loading
```python
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
```
Loads the Whisper large-v3 model and processor for Hindi transcription.

### Step 2: Audio Preprocessing
- Loads audio file using `torchaudio`
- Converts to mono if stereo
- Resamples to 16kHz
- Normalizes amplitude

### Step 3: Transcription
- Processes audio through Whisper model
- Forces Hindi language output
- Uses beam search for optimal results
- Decodes predicted tokens to text

### Step 4: Evaluation
- Compares predicted transcription with ground truth
- Calculates WER and CER metrics
- Displays results for each file
- Computes average metrics

## Output Example

```
Loading Whisper model...
Processing 2 audio file(s)...

File: audio1.flac
Predicted: यह एक परीक्षण है
Actual: यह एक परीक्षण है
WER: 0.000 | CER: 0.000

File: audio2.flac
Predicted: मरीज की स्थिति अच्छी है
Actual: मरीज की स्थिति स्थिर है
WER: 0.200 | CER: 0.118

RESULTS
Average WER: 0.100
Average CER: 0.059
Performance: Good
```

## Use Cases

1. **Medical Documentation**: Transcribe doctor-patient conversations in Hindi
2. **Quality Assessment**: Evaluate ASR system performance on medical terminology
3. **Model Benchmarking**: Compare different Whisper models or configurations
4. **Data Annotation**: Generate initial transcriptions for manual correction

## Limitations

- Requires GPU for faster processing (CPU works but is slower)
- Model size is large (~3GB download)
- Best performance on clear audio with minimal background noise
- Medical terminology accuracy depends on model training data

## Performance Tips

- Use high-quality audio recordings (16kHz or higher)
- Minimize background noise
- Ensure clear pronunciation
- Keep audio segments under 30 seconds for best results

## Error Handling

The code includes robust error handling for:
- Missing audio files
- Corrupted audio data
- Transcription failures
- Missing ground truth files

## Metrics Interpretation

### Word Error Rate (WER)
- **0.0 - 0.2**: Excellent transcription quality
- **0.2 - 0.4**: Good transcription, minor corrections needed
- **0.4+**: Significant errors, review audio quality or model configuration

### Character Error Rate (CER)
- Generally lower than WER
- More granular measure of transcription accuracy
- Useful for evaluating spelling and diacritical marks

## Future Enhancements

- Support for batch processing multiple files
- Fine-tuning on medical Hindi corpus
- Real-time transcription capability
- Support for other Indian languages
- Integration with medical terminology databases

## License

This project uses OpenAI's Whisper model, which is released under the MIT License.

## Author

Created for medical audio transcription research and evaluation.
