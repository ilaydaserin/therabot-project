import json

# JSON verisini yükleyin
with open('distilbert-newintents/new_intents.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Intent sayısını öğrenme
num_intents = len(data['intents'])
print(f"Toplam intent sayısı: {num_intents}")
