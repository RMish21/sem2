"""
Sample Audio Generator for Hindi Medical Transcription Testing
Creates synthetic audio files using Google Text-to-Speech (gTTS)
"""

import os
from gtts import gTTS

# Medical phrases in Hindi for testing
MEDICAL_SAMPLES = [
    {
        "filename": "audio.wav",
        "text": "मरीज की स्थिति स्थिर है",
        "translation": "The patient's condition is stable"
    },
    {
        "filename": "audio 2.wav",
        "text": "रक्तचाप सामान्य है कृपया दवा जारी रखें",
        "translation": "Blood pressure is normal, please continue medication"
    },
    {
        "filename": "consultation.wav",
        "text": "अगले सप्ताह फॉलोअप के लिए आएं",
        "translation": "Come for follow-up next week"
    },
    {
        "filename": "prescription.wav",
        "text": "दवा की खुराक को दोगुना कर दें",
        "translation": "Double the dosage of the medicine"
    },
    {
        "filename": "diagnosis.wav",
        "text": "निदान पूर्ण है उपचार शुरू करें",
        "translation": "Diagnosis is complete, start treatment"
    },
    {
        "filename": "vitals.wav",
        "text": "शरीर का तापमान सामान्य है",
        "translation": "Body temperature is normal"
    },
    {
        "filename": "symptoms.wav",
        "text": "बुखार और खांसी की शिकायत है",
        "translation": "Complaining of fever and cough"
    },
    {
        "filename": "test_results.wav",
        "text": "रक्त परीक्षण की रिपोर्ट सामान्य आई है",
        "translation": "Blood test report has come normal"
    }
]

def create_audio_files(output_dir="assets", generate_ground_truth=True):
    """
    Generate sample audio files using gTTS
    
    Args:
        output_dir: Directory to save audio files (default: 'assets')
        generate_ground_truth: Whether to create grounds_truth.txt (default: True)
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ Created directory: {output_dir}")
    
    print("\n🎙️  Generating Hindi Medical Audio Samples...\n")
    print("=" * 70)
    
    ground_truth_entries = []
    
    for idx, sample in enumerate(MEDICAL_SAMPLES, 1):
        filename = sample["filename"]
        text = sample["text"]
        translation = sample["translation"]
        
        filepath = os.path.join(output_dir, filename)
        
        try:
            # Generate audio using gTTS
            tts = gTTS(text=text, lang='hi', slow=False)
            tts.save(filepath)
            
            # Store for ground truth file
            ground_truth_entries.append(f"{filename} | {text}")
            
            print(f"[{idx}/{len(MEDICAL_SAMPLES)}] ✅ {filename}")
            print(f"    Hindi: {text}")
            print(f"    English: {translation}")
            print(f"    Saved to: {filepath}\n")
            
        except Exception as e:
            print(f"[{idx}/{len(MEDICAL_SAMPLES)}] ❌ Failed to create {filename}")
            print(f"    Error: {e}\n")
    
    print("=" * 70)
    
    # Generate grounds_truth.txt file
    if generate_ground_truth:
        ground_truth_path = "grounds_truth.txt"
        
        try:
            with open(ground_truth_path, 'w', encoding='utf-8') as f:
                for entry in ground_truth_entries:
                    f.write(entry + "\n")
            
            print(f"\n✅ Ground truth file created: {ground_truth_path}")
            print(f"   Contains {len(ground_truth_entries)} entries\n")
            
        except Exception as e:
            print(f"\n❌ Failed to create ground truth file")
            print(f"   Error: {e}\n")
    
    return len(ground_truth_entries)

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import gtts
        return True
    except ImportError:
        print("❌ Error: gTTS package not found!")
        print("\n📦 Please install it using:")
        print("   pip install gtts")
        print("\nOr install all requirements:")
        print("   pip install gtts requests")
        return False

def main():
    """Main function"""
    print("\n" + "=" * 70)
    print("   Hindi Medical Audio Sample Generator")
    print("   Using Google Text-to-Speech (gTTS)")
    print("=" * 70 + "\n")
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Generate audio files
    try:
        count = create_audio_files(output_dir="assets", generate_ground_truth=True)
        
        print("=" * 70)
        print(f"\n🎉 SUCCESS! Generated {count} audio files")
        print("\n📋 Next Steps:")
        print("   1. Check the 'assets/' folder for audio files")
        print("   2. Review 'grounds_truth.txt' file")
        print("   3. Run your Jupyter notebook to test transcription")
        print("\n💡 To test immediately, run:")
        print("   python med_audi_whip.py")
        print("\n" + "=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during generation: {e}\n")

if __name__ == "__main__":
    main()
