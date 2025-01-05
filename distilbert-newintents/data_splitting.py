import json
from sklearn.model_selection import train_test_split

# Veri setini yükleme
with open('distilbert-newintents\\new_intents.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

train_data = []
test_data = []

# Her intent için examples ve responses'ları ayır
for intent_data in data['intents']:
    examples = intent_data['patterns']
    responses = intent_data['responses']
    
    # Eğer sadece bir örnek varsa, train_test_split yapma
    if len(examples) > 1:
        # Veriyi train ve test olarak ayır (%80 train, %20 test)
        train_examples, test_examples = train_test_split(examples, test_size=0.2, random_state=42)
    else:
        # Sadece eğitim verisi olacak
        train_examples = examples
        test_examples = []

    # Eğitim verisine ekle
    train_data.append({
        "intent": intent_data['tag'],
        "examples": train_examples,
        "responses": responses  # Tüm yanıtları hem eğitim hem testte kullanabiliriz
    })
    
    # Test verisine ekle
    test_data.append({
        "intent": intent_data['tag'],
        "examples": test_examples,
        "responses": responses  # Aynı şekilde yanıtlar testte de bulunabilir
    })

# Sonuçları iki ayrı dosyaya kaydet
with open('distilbert-newintents/train_data.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

with open('distilbert-newintents/test_data.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)

print("Veri başarıyla train ve test olarak ayrıldı.")
