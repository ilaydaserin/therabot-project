import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import joblib
import pandas as pd

# Verilerinizi yükleyin
train_data = pd.read_csv("emotion_analysis1\\train_data.csv")  # Eğitim verisini yükleyin
test_data = pd.read_csv("emotion_analysis1\\test_data.csv")    # Test verisini yükleyin

# LabelEncoder'ı başlat
label_encoder = LabelEncoder()

# 'emotion' sütununu sayısal değerlere dönüştür
label_encoder.fit(train_data['emotion'])  # Eğitim verisindeki 'emotion' etiketlerine fit et

# LabelEncoder'ı kaydedin
joblib.dump(label_encoder, 'label_encoder.pkl')

# Modeli ve Tokenizer'ı yükleyin
model = BertForSequenceClassification.from_pretrained("emotion_analysis1\emotion_analysis_model") 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Modeli CUDA (GPU) cihazına taşıyın (GPU varsa)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Chatbot cevap fonksiyonu
def chatbot_response(input_text):
    # Girdi metnini tokenize et
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)

    # Modelden çıktıyı al
    with torch.no_grad():
        logits = model(**inputs).logits

    # Logit'lerden tahmini etiket
    predicted_class = torch.argmax(logits, dim=-1).item()

    # Etiketi geri çevir (sayısal etiketleri orijinal metin etiketlerine dönüştür)
    predicted_emotion = label_encoder.inverse_transform([predicted_class])[0]
    
    return f"I detected the emotion as {predicted_emotion}"

while True:
    # Kullanıcıdan giriş alın
    user_input = input("You: ")
    response = chatbot_response(user_input)
    print(response)
