from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Flask uygulaması oluştur
app = Flask(__name__)

# Model ve Tokenizer yükleme
model_name = "gpt2-fined_tuned_again\gpt2-finetuned"  
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Tokenizer için padding token ayarı
tokenizer.pad_token = tokenizer.eos_token

def generate_response(user_input):
    # Girdi metnini formatla
    input_text = f"User: {user_input}\\nBot:"
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
    response = response.split(".")[0] + "."  # İlk anlamlı cümleyi al
    response = response.replace("User:", "").strip()  # "User:"'ı kaldır

    return response

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# Chat endpoint'i
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")

    if not user_input:
        return jsonify({"error": "Message cannot be empty."}), 400

    response = generate_response(user_input)

    return jsonify({"response": response})

if __name__ == "__main__":
    print("Starting Therabot API...\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
