from sklearn.model_selection import train_test_split
import pandas as pd

prepared_data=pd.read_csv("emotion_analysis1\prepared_dataset.csv")
# Eğitim ve test setlerine böl
train_data, test_data = train_test_split(prepared_data, test_size=0.2, random_state=42)

# Kaydet
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)
