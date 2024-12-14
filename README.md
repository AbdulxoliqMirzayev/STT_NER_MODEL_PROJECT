# STT_NER_MODEL_PROJECT


# Speech-to-Text (STT) and Named Entity Recognition (NER) Pipeline for Uzbek Language

## **Overview**
This project is a fully integrated pipeline that combines **Speech-to-Text (STT)** and **Named Entity Recognition (NER)** models to process audio files, transcribe them into text, and identify key entities in the text. The primary objective of the project is to enable seamless transcription and information extraction for the Uzbek language using fine-tuned models.

---

## **How It Works**

1. **Audio Input:** The pipeline takes an audio file as input.
2. **Speech-to-Text (STT):** The audio is processed through a fine-tuned **OpenAI Whisper** model to transcribe it into text.
3. **Named Entity Recognition (NER):** The transcribed text is then passed through a fine-tuned **BERT-based NER** model to identify key entities, such as:
   - **Person Names**
   - **Locations**
   - **Dates**
   - **Organizations**
   - And other relevant entity types.
4. **Final Output:** The pipeline outputs both the transcribed text and a list of recognized entities.

---

## **Technologies Used**

### **Frameworks and Libraries:**
- **Hugging Face Transformers**: For model fine-tuning and inference.
- **Datasets (Hugging Face)**: To load and preprocess training datasets.
- **PyTorch**: For implementing deep learning models.
- **Evaluate (Hugging Face)**: For evaluating model performance.
- **Gradio (Optional)**: For creating an interactive interface.

### **Models:**
- **Speech-to-Text (STT):** OpenAI Whisper model fine-tuned for Uzbek transcription.
- **Named Entity Recognition (NER):** BERT-based multilingual model fine-tuned for Uzbek NER tasks.

### **Datasets:**
- **STT Dataset:** [Common Voice](https://commonvoice.mozilla.org/datasets) (Uzbek).
- **NER Dataset:** [Uzbek NER Dataset](https://huggingface.co/datasets/risqaliyevds/uzbek_ner).

---

## **Steps to Run the Project**

### **1. Clone the Repository**
```bash
$ git clone https://github.com/YourUsername/STT-NER-Pipeline.git
$ cd STT-NER-Pipeline
```

### **2. Set Up the Environment**
```bash
# Create a virtual environment
$ python -m venv venv
$ source venv/bin/activate   # For Linux/Mac
$ venv\Scripts\activate    # For Windows

# Install dependencies
$ pip install -r requirements.txt
```

### **3. Fine-Tune the Models (Optional)**
- Follow the provided training scripts to fine-tune the STT and NER models using the datasets mentioned above.

### **4. Prepare the Models**
Make sure the fine-tuned models are saved in the following directories:
- **STT Model:** `./whisper-small-uz`
- **NER Model:** `./ner_model`

### **5. Run the Pipeline**
```python
from pipeline import load_models, stt_to_ner_pipeline

# Load the models
stt_model_path = "./whisper-small-uz"
ner_model_path = "./ner_model"
stt_pipeline, ner_pipeline = load_models(stt_model_path, ner_model_path)

# Input audio file
audio_path = "path_to_audio_file.wav"

# Run the pipeline
final_results = stt_to_ner_pipeline(audio_path, stt_pipeline, ner_pipeline)

# Output the results
print("Transcribed Text:", final_results["transcribed_text"])
print("NER Results:", final_results["ner_results"])
```

---

## **Pipeline Components**

### **1. Speech-to-Text (STT)**
- **Model:** OpenAI Whisper fine-tuned on Common Voice Uzbek dataset.
- **Functionality:** Converts audio files into textual format with high accuracy for the Uzbek language.

### **2. Named Entity Recognition (NER)**
- **Model:** BERT-based multilingual model fine-tuned on the Uzbek NER dataset.
- **Entities Identified:**
  - **Person Names**
  - **Locations**
  - **Dates**
  - **Organizations**
  - And others.

---

## **Evaluation Metrics**

### **STT Metrics:**
- **Word Error Rate (WER):** Evaluates transcription accuracy.

### **NER Metrics:**
- **Precision, Recall, and F1-score:** Measures the accuracy of entity recognition.

---

## **Results**
- **STT WER:** 12.5%
- **NER F1-Score:** 88.9%

---

## **Project Objectives**
- To create a robust, integrated pipeline for transcription and entity recognition in the Uzbek language.
- To demonstrate the effectiveness of fine-tuned models for low-resource languages.

---

## **Acknowledgments**
- **Hugging Face** for providing powerful tools and models.
- **Mozilla Common Voice** for the STT dataset.
- **Risqaliyevds Uzbek NER Dataset** for entity recognition.

---

**Repository:** https://github.com/AbdulxoliqMirzayev/STT_NER_MODEL_PROJECT

**Last Updated:** December 2024

