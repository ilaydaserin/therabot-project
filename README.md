# Therabot - AI-Powered Psychologist Chatbot

**Therabot** is an AI-powered conversational chatbot designed to provide empathetic responses and intent classification, particularly in mental health contexts. The project integrates pre-trained models such as GPT-2, DialoGPT, and DistilBERT, as well as custom models, to deliver meaningful and contextually appropriate interactions. Below is a detailed description of the project’s structure and components.

---

## **Project Structure**

### **1. data_augmentation/**
Scripts for augmenting the **Mental Health Conversational Data** to enhance training data.

- **Purpose:** Expands the dataset with additional intents, patterns, and responses to improve training diversity.

---

### **2. distilbert-newintents/**
Contains scripts and files related to the fine-tuning and evaluation of DistilBERT for intent classification.

- **Key Files:**
  - `app.py`: Chatbot function for testing the DistilBERT model.
  - `data_splitting.py`: Splits data into training and testing sets.
  - `new_intents.json`: Contains additional intents and patterns.
  - **Note:** Large model files are excluded (refer to `.gitignore`).

- **Dataset Used:** Expanded **Mental Health Conversational Data**.

- **Purpose:** Intent classification with 103 intents.
  

---

### **3. emotion_analysis1/**
Scripts and models for emotion detection using a BERT-based classifier.

- **Key Files:**
  - `emotion_analysis1.py`: Code for training and evaluating the emotion classifier.
  - **Excluded:** Large model files and datasets are not included in the repository.
  - **Dataset Used:** **Empathetic Dialogues** dataset.
  - **Purpose:** Identifies user emotions (e.g., happiness, sadness) to improve chatbot responses.

---

### **4. flask_api/**
Contains the backend implementation of the chatbot using Flask.

- **Key Files:**
  - `app.py`: Main Flask server for serving chatbot responses.
  - `index.html`: The frontend interface for interacting with the chatbot.
- **Purpose:** Enables real-time interaction between the user and the chatbot.

---

### **5. google-colab-works/**
Notebooks for training and experimenting with various models on Google Colab.

- **Key Notebooks:**
  - `chatbot_with_mymodel.ipynb`: Custom intent classification experiments.
  - `dialogpt_deneme.ipynb`: Fine-tuning and testing **DialoGPT**.
  - `counsel_chat.ipynb`: Preprocessing and fine-tuning with **Counsel-Chat** dataset.

- **Purpose:** Hosted experiments for model training and testing using Google Colab’s GPU resources.

---

### **6. gpt2-fine_tuned_newintents/**
Contains the fine-tuned GPT-2 model and associated files.

- **Key Files:**
  - `train.py`: Script for fine-tuning GPT-2.
  - **Excluded:** Checkpoints and large model files are not included in the repository.
  - **Model Used:** GPT-2 fine-tuned on expanded **Mental Health Conversational Data**.

- **Purpose:** Generates empathetic and contextually relevant responses for Therabot.

---

### **7. web_scrapping/**
Scripts for scraping online data to augment training datasets.

- **Purpose:** Attempts to expand the dataset by collecting conversational examples from online sources.

---

### **8. results/**
Stores evaluation metrics and test results from various models.

- **Key Files:**
  - **Excluded:** Model checkpoints and other large files are not included.
- **Purpose:** Provides insights into model accuracy, validation loss, and response quality.

---

### **9. Supporting Files**
- `intents.json`: A JSON file defining the intents, patterns, and responses for training intent classification models.
- **Excluded:** Large files such as `label_encoder.pkl` are not included in the repository.

---


---

## **Installation and Usage**

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ilaydaserin/therabot-project.git
