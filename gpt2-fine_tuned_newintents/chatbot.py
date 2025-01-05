from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Model ve Tokenizer yükleme
model_name = "gpt2-fine_tuned_newintents/gpt2-fine_tuned_newintents_model"  # Eğitilmiş model dosyanızın yolu
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Tokenizer için padding token ayarı
tokenizer.pad_token = tokenizer.eos_token

# Chatbot fonksiyonu
def chatbot(user_input):
    # Girdi metnini formatla
    input_text = f"User: {user_input}\nBot:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)  # GPU'ya taşı

    # Modelle yanıt oluştur
    output = model.generate(
        input_ids=input_ids,
        max_length=100,              # Maksimum yanıt uzunluğu
        num_return_sequences=1,      # Tek yanıt döndür
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,              # Örnekleme kullan
        temperature=0.7,             # Yaratıcılığı kontrol eder
        top_k=40,                    # İlk 40 olası kelime arasından seçim
        top_p=0.9,                   # Nucleus sampling
        repetition_penalty=1.2       # Tekrar eden ifadeleri azalt
    )

    # Yanıtı decode ederek temizle
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    response = response.split("Bot:")[-1].strip()  # "Bot:" kısmından sonrası
    response = response.split("\n")[0]  # Sadece ilk satırı al
    return response


# Sohbet testi
if __name__ == "__main__":
    print("Therabot: Merhaba! Yardım etmek için buradayım. Çıkmak için 'exit' yazabilirsiniz.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Therabot: Görüşmek üzere! Kendinize iyi bakın. 😊")
            break
        try:
            response = chatbot(user_input)
            print(f"Bot: {response}")
        except Exception as e:
            print(f"Therabot: Bir hata oluştu. {str(e)}")
