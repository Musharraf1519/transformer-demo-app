# 🧠 Transformer Demo App  

A **Streamlit web app** showcasing the power of **Transformer-based models** for **NLP** and **Computer Vision** using 🤗 Hugging Face.  

## 🚀 Features  
- 📊 **Sentiment Analysis** → Detect emotions (Positive / Negative / Neutral) in text.  
- 📝 **Text Generation** → Generate creative text with GPT-2.  
- 📖 **Summarizer** → Convert long text into concise summaries.  
- 🌍 **Machine Translation** → Translate English text into French.  
- 🖼️ **Image Classification** → Identify what’s inside an uploaded image.  

---

## 🛠️ Tech Stack  
- [Streamlit](https://streamlit.io/) → UI framework  
- [Hugging Face Transformers](https://huggingface.co/transformers/) → Pre-trained NLP & CV models  
- [Pillow (PIL)](https://pillow.readthedocs.io/en/stable/) → Image handling  

---

## 📦 Installation  

1️⃣ Clone the repository  
```bash
git clone https://github.com/your-username/transformer-demo-app.git
cd transformer-demo-app

2️⃣ Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

3️⃣ Install dependencies
pip install -r requirements.txt

▶️ Run the App
streamlit run app.py
