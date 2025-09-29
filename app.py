#Installations :
# pip install transformers
# pip install streamlit

import streamlit as st
from transformers import pipeline
from PIL import Image
from transformers import AutoTokenizer

# Sidebar dropdown
choice = st.sidebar.selectbox(
    "Select an option:",
    ("Home","Sentiment Analysis", "Text Generation", "Summarizer", "Machine Translation","Image Classification","About")
)

# Main Content
#Home
if choice == "Home":
    st.write("""
    ## 👋 Welcome to the Transformer Demo App!

    This app demonstrates the power of Transformer-based models for various **NLP & CV tasks** using 🤗 Hugging Face and 🎨 Streamlit.

    ### 🚀 Available Features:
    - 📊 **Sentiment Analysis** → Detect emotions (Positive / Negative / Neutral) in text.  
    - 📝 **Text Generation** → Generate creative text with GPT-2.  
    - 📖 **Summarizer** → Convert long text into concise summaries.  
    - 🌍 **Machine Translation** → Translate English text into French.  
    - 🖼️ **Image Classification** → Identify what’s inside an uploaded image.  

    ---
    🔗 Select a feature from the **sidebar** to get started!
    """)

#Sentiment Analysis
elif choice == "Sentiment Analysis":
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"  # a common sentiment model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name
    )
    st.title("📊 Sentiment Analysis App")
    st.write("Type some text below and I’ll tell you if it’s Positive or Negative.")
    user_input = st.text_area("Enter your text:")
    if st.button("Analyze Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter some text first!")
        else:
            result = sentiment_pipeline(user_input)[0]
            label = result['label']
            score = result['score']
            st.success(f"**Sentiment:** {label}")
            st.write(f"**Confidence:** {score:.2f}")
#Text Generation
elif choice == "Text Generation":
    # Load the KAT-Dev model for text generation
    generation_pipeline = pipeline("text-generation", model="gpt2")
    st.title("📝 AI Text Generator")
    st.write(
        """
        ✨ This app utilizes the **GPT-2** model to generate text based on your prompts.  
        Perfect for creative writing, brainstorming, or exploring AI capabilities!
        """
    )
    # User input
    user_input = st.text_area("Enter your text prompt:")
        # Generate text
    if st.button("Generate Text"):
        if user_input.strip():
            results = generation_pipeline(user_input, max_length=100, num_return_sequences=1,
                        do_sample=True,
                    repetition_penalty=1.2)
            st.write(results[0]['generated_text'])
        else:
            st.warning("Please enter some text to generate.")
#Text Summarizer
elif choice == "Summarizer":
    model_name = "facebook/bart-large-cnn"
    summarizer_pipeline = pipeline(
        "summarization",
        model=model_name,
        tokenizer=model_name
    )
    st.title("📖 AI Text Summarizer")
    st.write(
        """
        ✨ This app uses **state-of-the-art NLP models** to turn long text into concise, meaningful summaries.  
        Perfect for students, researchers, or anyone who wants to save time reading!
        """
    )
    user_input = st.text_area("📝 Paste your text here:")
    if st.button("✨ Generate Summary"):
        if user_input.strip() == "":
            st.warning("Please enter some text first!")
        else:
            result = summarizer_pipeline(user_input,max_length=130, min_length=30, do_sample=False)
            st.success(result[0]['summary_text'])
#Translator        
elif choice == "Machine Translation":
    model_name = "Helsinki-NLP/opus-mt-en-fr"  # simple and light model
    translator_pipeline = pipeline("translation", model=model_name)

    st.title("🌍 Simple AI Translator")
    st.write("""
    ✨ This is a lightweight English → French translator using Hugging Face MarianMT.  
    Runs comfortably on CPU without extra complexity.
    """)

    # Input text area
    user_input = st.text_area("📝 Enter English text:", 
                            height=150, 
                            placeholder="Type something in English...")

    # Translate button
    if st.button("🚀 Translate"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text first!")
        else:
            result = translator_pipeline(user_input)
            translated_text = result[0]['translation_text']
            st.success(translated_text)

            
#Image Classifier
elif choice == "Image Classification":
    # Load pipeline once (cached)
    classification_pipeline = pipeline("image-classification", model="WinKawaks/vit-small-patch16-224")
    st.title("🖼️ AI Image Classifier")
    st.write(
        """
        ✨ This app uses **state-of-the-art Vision Transformer (ViT) models** to classify images into meaningful categories.  
        Perfect for students, researchers, or anyone curious about computer vision!
        """
    )
    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image")
        # Predict
        if st.button("Classify"):
            with st.spinner("Classifying..."):
                results = classification_pipeline(image, top_k=3)  # get top 3 predictions
                print(results[0])
            st.subheader("Predictions:")
            label = results[0]['label']
            score = results[0]['score']
            st.success(f"{label}")
            st.write(f"**Confidence:** {score:.2f}")
#About
elif choice == "About":
    st.write("""
    ## 📌 About This Project  

    This project showcases different **Transformer-based applications** using 🤗 Hugging Face and 🎨 Streamlit.  
    It is designed as a hands-on demo for **NLP and Computer Vision** tasks.

    ### 👨‍💻 Author  
    **Musharraf Hussain Khan**  

    ### 🛠️ Tech Stack  
    - Python  
    - Streamlit  
    - Hugging Face Transformers  
    - PIL (for image handling)  

    ---
    💡 Feel free to explore, fork, and contribute on GitHub!
    """)
