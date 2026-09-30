import pandas as pd

print("=" * 70)
print("SMS DATASET")
print("=" * 70)

sms = pd.read_csv("data/raw/spam_sms.csv", encoding="latin-1")

print("\nShape:", sms.shape)
print("\nColumns:")
print(sms.columns.tolist())

print("\nFirst 5 rows:")
print(sms.head())

print("\n")

print("=" * 70)
print("EMAIL DATASET")
print("=" * 70)

email = pd.read_csv("data/raw/spam_email.csv")

print("\nShape:", email.shape)
print("\nColumns:")
print(email.columns.tolist())

print("\nFirst 5 rows:")
print(email.head())