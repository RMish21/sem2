"""
Multi-Model ASR Comparison for Hindi Medical Audio
Compares different models: Whisper variants and AI4Bharat IndicWhisper
"""

import os
import time
import torch
import soundfile as sf
import numpy as np
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    AutoModelForCausalLM
)
from jiwer import wer, cer
import warnings
from tabulate import tabulate

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded environment variables from .env file")
except ImportError:
    print("⚠ python-dotenv not installed. Environment variables must be set manually.")
    print("  Install with: pip install python-dotenv")

warnings.filterwarnings("ignore")

# Check for GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")


class ModelConfig:
    """Configuration for each ASR model"""
    def __init__(self, name, model_id, model_class, processor_class, model_type="whisper"):
        self.name = name
        self.model_id = model_id
        self.model_class = model_class
        self.processor_class = processor_class
        self.model_type = model_type  # "whisper", "qwen", "api"


# Model configurations to test
MODELS = [
    # ModelConfig(
    #     "Whisper Large-v3",
    #     "openai/whisper-large-v3",
    #     WhisperForConditionalGeneration,
    #     WhisperProcessor,
    #     "whisper"
    # ),
    # ModelConfig(
    #     "Whisper Medium",
    #     "openai/whisper-medium",
    #     WhisperForConditionalGeneration,
    #     WhisperProcessor,
    #     "whisper"
    # ),
    # ModelConfig(
    #     "Whisper Small",
    #     "openai/whisper-small",
    #     WhisperForConditionalGeneration,
    #     WhisperProcessor,
    #     "whisper"
    # ),
    ModelConfig(
        "AI4Bharat Whisper-Medium-Hi",
        "vasista22/whisper-hindi-medium",
        WhisperForConditionalGeneration,
        WhisperProcessor,
        "whisper"
    ),
    ModelConfig(
        "Qwen-Audio",
        "Qwen/Qwen-Audio",
        AutoModelForCausalLM,
        AutoProcessor,
        "qwen"
    ),
    # ModelConfig(
    #     "Gemini Audio",
    #     "gemini-1.5-flash",
    #     None,  # API-based, no local model
    #     None,  # API-based, no local processor
    #     "gemini"
    # ),
    # Note: SpeechGPT is research-only, no public model available
    # ModelConfig("SpeechGPT", "microsoft/speechgpt", ..., "api")
]


def count_parameters(model):
    """Count total and trainable parameters in model"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_grounds_truth(file_path):
    """Load ground truth transcriptions from file"""
    grounds_truths = {}
    
    if not os.path.exists(file_path):
        print(f"Ground truth file not found: {file_path}")
        return grounds_truths
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if "|" in line:
                audio_file, transcription = line.split("|", 1)
                audio_file = audio_file.strip() or "audio.flac"
            else:
                audio_file = "audio.flac"
                transcription = line
            
            grounds_truths[audio_file] = transcription.strip()
    
    return grounds_truths


def process_audio(file_path):
    """Load and process audio file"""
    try:
        # Use soundfile to load audio
        waveform, sr = sf.read(file_path)
        
        # Convert to torch tensor
        waveform = torch.from_numpy(waveform).float()
        
        # Handle stereo to mono conversion
        if len(waveform.shape) > 1:
            waveform = torch.mean(waveform, dim=1)
        
        # Add channel dimension
        waveform = waveform.unsqueeze(0)
        
        # Resample to 16kHz if needed
        if sr != 16000:
            import torchaudio
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        # Normalize
        waveform = waveform / torch.max(torch.abs(waveform))
        
        return waveform
    except Exception as e:
        print(f"Error processing audio {file_path}: {e}")
        return None


def transcribe_whisper(processor, model, audio, model_name=""):
    """Transcribe audio using Whisper-based models"""
    try:
        audio_array = audio.squeeze().numpy()
        
        inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
        
        # Move inputs to GPU if available
        if DEVICE == "cuda":
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # AI4Bharat models don't use forced_decoder_ids the same way
        generate_kwargs = {
            "max_new_tokens": 440,
            "num_beams": 5,
            "do_sample": False
        }
        
        # Only use forced_decoder_ids for OpenAI Whisper models
        if "openai" in model_name.lower() or "vasista22" not in model_name.lower():
            try:
                forced_decoder_ids = processor.get_decoder_prompt_ids(
                    language="hindi", 
                    task="transcribe"
                )
                generate_kwargs["forced_decoder_ids"] = forced_decoder_ids
            except:
                pass  # Model doesn't support forced_decoder_ids
        
        with torch.no_grad():
            predicted_ids = model.generate(
                inputs["input_features"],
                **generate_kwargs
            )
        
        transcription = processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        return transcription.strip()
    except Exception as e:
        print(f"Error in transcription: {e}")
        return ""


def transcribe_qwen(processor, model, audio_path):
    """Transcribe audio using Qwen-Audio model"""
    try:
        # Qwen-Audio uses conversation-based input format
        # Must provide file path, not audio array
        query = processor.from_list_format([
            {'audio': audio_path},
            {'text': 'कृपया इस ऑडियो को हिंदी में लिखें। Please transcribe this audio in Hindi language.'},
        ])
        
        inputs = processor(query, return_tensors='pt')
        
        # Move inputs to same device as model
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            pred = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )
        
        # Decode and extract transcription
        response = processor.decode(pred[0], skip_special_tokens=False)
        
        # Qwen includes the prompt in output, extract only the response
        # Format is usually: [prompt]<|im_end|>\n<|im_start|>assistant\n[response]
        if '<|im_start|>assistant' in response:
            transcription = response.split('<|im_start|>assistant')[-1].strip()
            transcription = transcription.replace('<|im_end|>', '').strip()
        else:
            transcription = response
        
        return transcription.strip()
    except Exception as e:
        print(f"  ⚠ Error in Qwen transcription: {e}")
        import traceback
        traceback.print_exc()
        return ""


def transcribe_gemini_audio(audio_path, api_key=None):
    """Transcribe audio using Gemini Audio API"""
    try:
        import google.generativeai as genai
        
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY")
        
        if not api_key:
            raise ValueError("Gemini API key not found. Set GOOGLE_API_KEY environment variable.")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Upload audio file
        print(f"    Uploading audio file: {audio_path}")
        audio_file = genai.upload_file(path=audio_path)
        
        # Generate transcription with Hindi instruction
        response = model.generate_content([
            "कृपया इस ऑडियो को हिंदी में लिखें। Please transcribe this audio in Hindi language accurately.",
            audio_file
        ])
        
        return response.text.strip()
    except ImportError:
        raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
    except Exception as e:
        print(f"  ⚠ Error in Gemini transcription: {e}")
        return ""


def evaluate_gemini_model(model_config, grounds_truths):
    """Special evaluation function for Gemini Audio (API-based)"""
    print(f"✓ Using Gemini Audio API")
    
    # Check for API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print(f"✗ GOOGLE_API_KEY not found in environment variables")
        print(f"  Please set it using: set GOOGLE_API_KEY=your-key-here (Windows)")
        print(f"  or: export GOOGLE_API_KEY=your-key-here (Linux/Mac)")
        return None
    
    print(f"✓ API Key found (ends with: ...{api_key[-4:]})")
    
    # Prepare results structure
    results = {
        'model_name': model_config.name,
        'model_id': model_config.model_id,
        'parameters_M': 0,  # API model, unknown parameters
        'load_time': 0,
        'predictions': {},
        'wer_scores': [],
        'cer_scores': [],
        'inference_times': [],
        'total_time': 0
    }
    
    print(f"\nProcessing {len(grounds_truths)} audio file(s)...")
    
    total_start = time.time()
    
    for audio_file, gt_text in grounds_truths.items():
        if not os.path.exists(audio_file):
            print(f"  ✗ File not found: {audio_file}")
            continue
        
        try:
            # Transcribe via API
            inference_start = time.time()
            predicted = transcribe_gemini_audio(audio_file, api_key)
            inference_time = time.time() - inference_start
            
            if not predicted:
                print(f"  ✗ Empty transcription for: {audio_file}")
                continue
            
            # Calculate metrics
            wer_score = wer(gt_text, predicted)
            cer_score = cer(gt_text, predicted)
            
            results['predictions'][audio_file] = {
                'predicted': predicted,
                'actual': gt_text,
                'wer': wer_score,
                'cer': cer_score,
                'time': inference_time
            }
            
            results['wer_scores'].append(wer_score)
            results['cer_scores'].append(cer_score)
            results['inference_times'].append(inference_time)
            
            print(f"  ✓ {audio_file}: WER={wer_score:.3f}, CER={cer_score:.3f}, Time={inference_time:.2f}s")
            
        except Exception as e:
            print(f"  ✗ Error transcribing {audio_file}: {e}")
            print(f"  ✗ Error type: {type(e).__name__}")
            import traceback
            print(f"  ✗ Full traceback:")
            traceback.print_exc()
    
    results['total_time'] = time.time() - total_start
    
    # Calculate averages
    if results['wer_scores']:
        results['avg_wer'] = np.mean(results['wer_scores'])
        results['avg_cer'] = np.mean(results['cer_scores'])
        results['avg_inference_time'] = np.mean(results['inference_times'])
        results['files_processed'] = len(results['wer_scores'])
    else:
        results['avg_wer'] = None
        results['avg_cer'] = None
        results['avg_inference_time'] = None
        results['files_processed'] = 0
    
    print(f"\n✓ Completed in {results['total_time']:.2f}s")
    
    return results


def evaluate_model(model_config, grounds_truths):
    """Evaluate a single model on all audio files"""
    print(f"\n{'='*80}")
    print(f"Loading model: {model_config.name}")
    print(f"Model ID: {model_config.model_id}")
    print(f"Model Type: {model_config.model_type}")
    print(f"{'='*80}")
    
    # Special handling for API-based models (Gemini)
    if model_config.model_type == "gemini":
        return evaluate_gemini_model(model_config, grounds_truths)
    
    # Load model
    start_load = time.time()
    processor = None
    model = None
    
    try:
        # Qwen-Audio requires special loading
        if model_config.model_type == "qwen":
            print(f"⚙ Loading Qwen-Audio with special configuration...")
            processor = model_config.processor_class.from_pretrained(
                model_config.model_id,
                trust_remote_code=True
            )
            print(f"✓ Processor loaded")
            
            if DEVICE == "cuda":
                print(f"⚙ Loading model to GPU with float16...")
                try:
                    model = model_config.model_class.from_pretrained(
                        model_config.model_id,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        trust_remote_code=True
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"⚠ GPU OOM - Falling back to CPU...")
                        model = model_config.model_class.from_pretrained(
                            model_config.model_id,
                            trust_remote_code=True
                        )
                    else:
                        raise
            else:
                print(f"⚙ Loading model to CPU...")
                model = model_config.model_class.from_pretrained(
                    model_config.model_id,
                    trust_remote_code=True
                )
        else:
            print(f"⚙ Loading standard Whisper model...")
            processor = model_config.processor_class.from_pretrained(model_config.model_id)
            print(f"✓ Processor loaded")
            
            model = model_config.model_class.from_pretrained(model_config.model_id)
            print(f"✓ Model loaded")
            
            # Move model to GPU if available (for non-Qwen models)
            if DEVICE == "cuda":
                try:
                    print(f"⚙ Moving model to GPU...")
                    model = model.to(DEVICE)
                    print(f"✓ Model on GPU")
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"⚠ GPU OOM - Model will stay on CPU")
                    else:
                        raise
        
        model.eval()
        
        # Count parameters
        total_params, trainable_params = count_parameters(model)
        params_millions = total_params / 1e6
        
        load_time = time.time() - start_load
        print(f"✓ Model loaded in {load_time:.2f}s")
        print(f"✓ Parameters: {params_millions:.2f}M ({total_params:,})")
        print(f"✓ Device: {next(model.parameters()).device if hasattr(model, 'parameters') else DEVICE}")
        if DEVICE == "cuda":
            print(f"✓ GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print(f"✗ Error type: {type(e).__name__}")
        import traceback
        print(f"✗ Full traceback:")
        traceback.print_exc()
        return None
    
    # Evaluate on all audio files
    results = {
        'model_name': model_config.name,
        'model_id': model_config.model_id,
        'parameters_M': params_millions,
        'load_time': load_time,
        'predictions': {},
        'wer_scores': [],
        'cer_scores': [],
        'inference_times': [],
        'total_time': 0
    }
    
    print(f"\nProcessing {len(grounds_truths)} audio file(s)...")
    
    total_start = time.time()
    
    for audio_file, gt_text in grounds_truths.items():
        if not os.path.exists(audio_file):
            print(f"  ✗ File not found: {audio_file}")
            continue
        
        audio = process_audio(audio_file)
        if audio is None:
            print(f"  ✗ Failed to load: {audio_file}")
            continue
        
        try:
            # Transcribe
            inference_start = time.time()
            
            # Route to appropriate transcription function
            if model_config.model_type == "qwen":
                predicted = transcribe_qwen(processor, model, audio_file)
            else:
                predicted = transcribe_whisper(processor, model, audio, model_config.model_id)
            
            inference_time = time.time() - inference_start
            
            if not predicted:
                print(f"  ⚠ Empty transcription for: {audio_file}")
                continue
            
            # Calculate metrics
            wer_score = wer(gt_text, predicted)
            cer_score = cer(gt_text, predicted)
            
            results['predictions'][audio_file] = {
                'predicted': predicted,
                'actual': gt_text,
                'wer': wer_score,
                'cer': cer_score,
                'time': inference_time
            }
            
            results['wer_scores'].append(wer_score)
            results['cer_scores'].append(cer_score)
            results['inference_times'].append(inference_time)
            
            print(f"  ✓ {audio_file}: WER={wer_score:.3f}, CER={cer_score:.3f}, Time={inference_time:.2f}s")
            
        except Exception as e:
            print(f"  ✗ Error transcribing {audio_file}: {e}")
            print(f"  ✗ Error type: {type(e).__name__}")
            import traceback
            print(f"  ✗ Full traceback:")
            traceback.print_exc()
    
    results['total_time'] = time.time() - total_start
    
    # Calculate averages
    if results['wer_scores']:
        results['avg_wer'] = np.mean(results['wer_scores'])
        results['avg_cer'] = np.mean(results['cer_scores'])
        results['avg_inference_time'] = np.mean(results['inference_times'])
        results['files_processed'] = len(results['wer_scores'])
    else:
        results['avg_wer'] = None
        results['avg_cer'] = None
        results['avg_inference_time'] = None
        results['files_processed'] = 0
    
    print(f"\n✓ Completed in {results['total_time']:.2f}s")
    
    # Clean up
    del model, processor
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        print(f"✓ GPU Memory Freed")
    
    return results


def print_comparison_table(all_results):
    """Print comparison table of all models"""
    print("\n\n" + "="*120)
    print("MODEL COMPARISON TABLE")
    print("="*120 + "\n")
    
    table_data = []
    headers = ["Model", "Parameters (M)", "Avg WER", "Avg CER", "Avg Time (s)", "Total Time (s)", "Files"]
    
    for result in all_results:
        if result and result['files_processed'] > 0:
            table_data.append([
                result['model_name'],
                f"{result['parameters_M']:.1f}",
                f"{result['avg_wer']:.4f}",
                f"{result['avg_cer']:.4f}",
                f"{result['avg_inference_time']:.2f}",
                f"{result['total_time']:.2f}",
                result['files_processed']
            ])
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Find best model for each metric
    if table_data:
        valid_results = [r for r in all_results if r and r['files_processed'] > 0]
        
        best_wer = min(valid_results, key=lambda x: x['avg_wer'])
        best_cer = min(valid_results, key=lambda x: x['avg_cer'])
        best_time = min(valid_results, key=lambda x: x['avg_inference_time'])
        
        print("\n🏆 BEST MODELS:")
        print(f"  • Best WER: {best_wer['model_name']} ({best_wer['avg_wer']:.4f})")
        print(f"  • Best CER: {best_cer['model_name']} ({best_cer['avg_cer']:.4f})")
        print(f"  • Fastest: {best_time['model_name']} ({best_time['avg_inference_time']:.2f}s)")


def print_detailed_predictions(all_results):
    """Print detailed predictions for each model"""
    print("\n\n" + "="*120)
    print("DETAILED PREDICTIONS BY MODEL")
    print("="*120)
    
    for result in all_results:
        if not result or result['files_processed'] == 0:
            continue
        
        print(f"\n\n{'='*120}")
        print(f"MODEL: {result['model_name']}")
        print(f"{'='*120}\n")
        
        for audio_file, pred_data in result['predictions'].items():
            print(f"📁 File: {audio_file}")
            print(f"   Predicted: {pred_data['predicted']}")
            print(f"   Actual:    {pred_data['actual']}")
            print(f"   WER: {pred_data['wer']:.4f} | CER: {pred_data['cer']:.4f} | Time: {pred_data['time']:.2f}s")
            print()


def save_results_to_file(all_results, output_file="model_comparison_results.txt"):
    """Save all results to a text file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*120 + "\n")
        f.write("HINDI MEDICAL AUDIO ASR MODEL COMPARISON\n")
        f.write("="*120 + "\n\n")
        
        # Summary table
        f.write("SUMMARY TABLE:\n")
        f.write("-"*120 + "\n")
        for result in all_results:
            if result and result['files_processed'] > 0:
                f.write(f"\nModel: {result['model_name']}\n")
                f.write(f"  Parameters: {result['parameters_M']:.2f}M\n")
                f.write(f"  Average WER: {result['avg_wer']:.4f}\n")
                f.write(f"  Average CER: {result['avg_cer']:.4f}\n")
                f.write(f"  Average Inference Time: {result['avg_inference_time']:.2f}s\n")
                f.write(f"  Total Time: {result['total_time']:.2f}s\n")
                f.write(f"  Files Processed: {result['files_processed']}\n")
        
        # Detailed predictions
        f.write("\n\n" + "="*120 + "\n")
        f.write("DETAILED PREDICTIONS:\n")
        f.write("="*120 + "\n")
        
        for result in all_results:
            if not result or result['files_processed'] == 0:
                continue
            
            f.write(f"\n\nMODEL: {result['model_name']}\n")
            f.write("-"*120 + "\n")
            
            for audio_file, pred_data in result['predictions'].items():
                f.write(f"\nFile: {audio_file}\n")
                f.write(f"  Predicted: {pred_data['predicted']}\n")
                f.write(f"  Actual:    {pred_data['actual']}\n")
                f.write(f"  WER: {pred_data['wer']:.4f} | CER: {pred_data['cer']:.4f} | Time: {pred_data['time']:.2f}s\n")
    print("\n📋 Models to test:")
    for i, model in enumerate(MODELS, 1):
        print(f"  {i}. {model.name} ({model.model_type})")
    
    print("\n⚠️  Notes:")
    print("  • SpeechGPT: Research-only, no public model available")
    print("  • Gemini Audio: Requires GOOGLE_API_KEY environment variable")
    print("  • Qwen-Audio: May require additional setup")
    
    print(f"\n✓ Results saved to: {output_file}")


def main():
    """Main execution function"""
    print("="*120)
    print("HINDI MEDICAL AUDIO ASR MODEL COMPARISON")
    print("="*120)
    
    # Load ground truth
    grounds_truth_file = "grounds_truth.txt"
    grounds_truths = load_grounds_truth(grounds_truth_file)
    
    if not grounds_truths:
        print(f"Error: No ground truth data found in {grounds_truth_file}")
        return
    
    print(f"\n✓ Loaded {len(grounds_truths)} audio files from ground truth")
    
    # Evaluate all models
    all_results = []
    
    for model_config in MODELS:
        try:
            result = evaluate_model(model_config, grounds_truths)
            all_results.append(result)
        except Exception as e:
            print(f"\n✗ Failed to evaluate {model_config.name}: {e}")
            all_results.append(None)
    
    # Print comparison table
    print_comparison_table(all_results)
    
    # Print detailed predictions
    print_detailed_predictions(all_results)
    
    # Save to file
    save_results_to_file(all_results)
    
    print("\n" + "="*120)
    print("COMPARISON COMPLETE!")
    print("="*120)


if __name__ == "__main__":
    main()
