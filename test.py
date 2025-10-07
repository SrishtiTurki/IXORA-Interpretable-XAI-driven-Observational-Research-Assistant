<<<<<<< HEAD
"""import os
print("Google API Key:", os.getenv("GOOGLE_API_KEY"))

from transformers import pipeline


from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "microsoft/BioGPT-Large"

print(f"⏳ Downloading {model_id} ...")

# download tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# download model
model = AutoModelForCausalLM.from_pretrained(model_id)

print("✅ Download complete and cached locally.")



try:
    pipe = pipeline("text-generation", model="microsoft/BioGPT-Large")
    print("✅ BioGPT is available (cached or downloaded).")
except Exception as e:
    print("❌ BioGPT not found / will be downloaded on first use.", e)


from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_name = "microsoft/biogpt"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Example prompt
prompt = "What are the symptoms of Parkinson's disease?"

# Encode and generate
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)

# Decode response
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
import sys
sys.stdout.flush()

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Pick one sample to explain
sample = X_test.iloc[0]

import lime
import lime.lime_tabular

# Create explainer
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=data.target_names,
    mode="classification"
)

# Explain one prediction
lime_exp = lime_explainer.explain_instance(
    data_row=sample,
    predict_fn=model.predict_proba,
    num_features=10
)

# Show explanation
print(lime_exp.as_list())

import shap

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

print(type(shap_values))
if isinstance(shap_values, list):
    print([sv.shape for sv in shap_values])
else:
    print(shap_values.shape)

print("X_test shape:", X_test.shape)


# pick one instance and one class
i = 0  # sample index
class_idx = 1  # choose which class you want to explain

shap.force_plot(
    explainer.expected_value[class_idx],   # expected value for that class
    shap_values[i, :, class_idx],          # shap values for sample i and class
    features=X_test.iloc[i],               # actual feature values
    feature_names=X_test.columns"""

import spacy

# Load different models
sm = spacy.load("en_core_sci_sm")
md = spacy.load("en_core_sci_md")
lg = spacy.load("en_core_sci_lg")
scibert = spacy.load("en_core_sci_scibert")

text = "The patient was treated with ibuprofen for inflammation."

# Run through all models
docs = [model(text) for model in [sm, md, lg, scibert]]

# Collect entities
all_entities = []
for i, doc in enumerate(docs):
    all_entities.extend([(ent.text, ent.label_, i) for ent in doc.ents])

print(all_entities)





=======
from redis import Redis
r = Redis(host='localhost', port=6379)
print(r.ping())
>>>>>>> 526eba8 (Initial commit on feature branch)
