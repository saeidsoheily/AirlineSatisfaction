import pickle

with open("model_xgboost.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)

print("Model loaded successfully!")
print(type(model))
print(type(dv))