import pandas as pd
import json

# Veri seti yolunu belirtin
file_path = 'emotion_analysis1/emotion-emotion_69k.csv'

# Veri setini yükle
data = pd.read_csv(file_path)

# Konuşmaları gruplamak için bir yapı
conversations = []

# Veri setini satır satır işleme
for _, row in data.iterrows():
    situation = row['Situation']  # Sohbetin bağlamı
    emotion = row['emotion']  # Duygusal durum
    customer_dialogue = row['empathetic_dialogues']  # Müşteri ifadeleri
    agent_response = row['labels']  # Agent yanıtları

    # Müşteri ve Agent ifadelerini temizle
    customer_dialogue_cleaned = (
        customer_dialogue.replace("Customer :", "")
        .replace("Agent :", "")
        .strip()
        if isinstance(customer_dialogue, str)
        else ""
    )
    agent_response_cleaned = (
        agent_response.replace("Customer :", "")
        .replace("Agent :", "")
        .strip()
        if isinstance(agent_response, str)
        else ""
    )

    # JSON formatında bir yapı ekle
    conversations.append({
        'situation': situation,
        'emotion': emotion,
        'customer_lines': customer_dialogue_cleaned,
        'agent_lines': agent_response_cleaned
    })

# JSON dosyasını kaydet (utf-8 kodlamasıyla)
with open('formatted_conversations.json', 'w', encoding='utf-8') as f:
    json.dump(conversations, f, ensure_ascii=False, indent=4)

print("JSON dosyası başarıyla oluşturuldu!")

# Input ve Output için veri çerçevesi oluşturun
prepared_data = pd.DataFrame({
    'input': [conv['customer_lines'] for conv in conversations],
    'response': [conv['agent_lines'] for conv in conversations],
    'emotion': [conv['emotion'] for conv in conversations]
})

# Boş cevapları filtrele
prepared_data = prepared_data[prepared_data['response'].str.strip() != ""].reset_index(drop=True)

# Veriyi CSV olarak kaydet (utf-8 kodlamasıyla)
prepared_data.to_csv('prepared_dataset.csv', index=False, encoding='utf-8')

print("Veri başarıyla hazırlandı ve 'prepared_dataset.csv' olarak kaydedildi!")
