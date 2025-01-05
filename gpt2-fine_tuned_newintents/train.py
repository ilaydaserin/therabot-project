from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

# Model ve Tokenizer'ı yükle
model_name = "gpt2"  
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Veri setini yükle
def load_dataset(file_path, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )
    return dataset

dataset = load_dataset("gpt2-fine_tuned_newintents\gpt2_ready_dataset.txt", tokenizer)
# Data collator (maskesiz metin modeli için)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Eğitim parametrelerini ayarla
training_args = TrainingArguments(
    output_dir="gpt2-fine_tuned_newintents/gpt2-fine_tuned_newintents_model",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=2,
    save_steps=1000,
    save_total_limit=2,
    prediction_loss_only=True,
    report_to="none"
)

# Trainer'ı başlat
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# Eğitimi başlat
trainer.train()

# Modeli kaydet
trainer.save_model("gpt2-fine_tuned_newintents/gpt2-fine_tuned_newintents_model")
tokenizer.save_pretrained("gpt2-fine_tuned_newintents/gpt2-fine_tuned_newintents_model")
