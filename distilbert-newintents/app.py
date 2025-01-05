import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import json
import random
from pathlib import Path

# Model ve tokenizer'ı yükleyin
MODEL_DIR = Path('distilbert-newintents/saved_model')

# Modeli yükleyin
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)

# Cihazı belirleyin (CUDA varsa GPU kullan, yoksa CPU kullan)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Label Encoder'ı yükleyin
with open(MODEL_DIR / 'label_classes.json', 'r', encoding='utf-8') as f:
    label_encoder_classes = json.load(f)

# Veriseti (intentler ve yanıtlar) yükleyin
with open('distilbert-newintents/new_intents.json', 'r', encoding='utf-8') as f:
    intents_data = json.load(f)

# Intentlere göre yanıtları bir sözlükte toplayın
intent_responses = {}
for intent in intents_data['intents']:
    intent_responses[intent['tag']] = intent['responses']

# Yanıt üretme fonksiyonu
# Yanıt üretme fonksiyonu
def get_response(user_input):
    # Giriş metnini tokenize et
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Cihaz seçimi (CUDA/CPU)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Modeli değerlendirme modunda çalıştır
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Sınıfı belirle
    predicted_class_idx = torch.argmax(logits, dim=-1).item()
    predicted_label = label_encoder_classes[predicted_class_idx]

    responses = intent_responses.get(predicted_label, ["Üzgünüm, anlamadım."])

    # Yanıt listesinden rastgele birini seç
    chosen_response = random.choice(responses)

    # Eğer seçilen yanıt bir listeyse, ilk öğesini al
    if isinstance(chosen_response, list):
        chosen_response = chosen_response[0]  # Liste ise, ilk öğeyi alıyoruz

    return chosen_response  # Yanıt olarak sadece string döndürüyoruz

# Chatbot fonksiyonu
def chatbot():
    print("Merhaba! Yardımcı olabilir miyim? (Çıkmak için 'exit' yazın)")
    while True:
        user_input = input("Siz: ")

        if user_input.lower() == 'exit':
            print("Görüşmek üzere!")
            break

        # Kullanıcı girdisini işleyin ve yanıt alın
        response = get_response(user_input)

        # Yanıtı yazdırın
        print(f"Bot: {response}")

# Chatbot'u başlatın
chatbot()
