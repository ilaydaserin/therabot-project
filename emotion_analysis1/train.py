import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
import tensorflow as tf

train_data=pd.read_csv('emotion_analysis1\\train_data.csv')
test_data=pd.read_csv('emotion_analysis1\\test_data.csv')

# LabelEncoder'ı başlat
label_encoder = LabelEncoder()

# 'emotion' sütununu sayısal değerlere dönüştür
train_data['emotion'] = label_encoder.fit_transform(train_data['emotion'])
test_data['emotion'] = label_encoder.transform(test_data['emotion'])

num_classes = len(label_encoder.classes_)
print(f"Num classes: {num_classes}")

# Tokenizer'ı yükleyin
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Eğitim verisini tokenize et
train_encodings = tokenizer(list(train_data['response']), truncation=True, padding=True, max_length=128)

# Test verisini tokenize et
test_encodings = tokenizer(list(test_data['response']), truncation=True, padding=True, max_length=128)

class ChatbotDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # labels doğru türde
        return item


# Eğitim ve test datasetlerini oluşturun
train_dataset = ChatbotDataset(train_encodings, train_data['emotion'].values)  # Hedef değişkeni 'emotion'
test_dataset = ChatbotDataset(test_encodings, test_data['emotion'].values)

# BERT modelini yükleyin
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=44)  

# Eğitim ayarlarını yapın
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    report_to="none"  # wandb kullanmamak için
)

# Trainer'ı oluşturun
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Modeli eğitin
trainer.train()
model.save_pretrained('emotion_analysis1/emotion_analysis_model')
print("Model başarıyla emotion_analysis1 dizinine kaydedildi!")