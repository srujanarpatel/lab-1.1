import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import re

# Load GPT2
@st.cache_resource
def load_model():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to("cpu")
    return tokenizer, model

tokenizer, model = load_model()

# UI
st.title("📖 AI Story Generator")
st.markdown("Generate a ~200-word story based on your prompt!")

prompt = st.text_input("Enter your story idea:", value="a lonely robot on Mars")

if st.button("Generate Story"):
    if prompt.strip() == "":
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating story..."):
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            outputs = model.generate(
                inputs,
                max_length=300,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                early_stopping=True,
                temperature=0.9,
                top_k=50,
                top_p=0.95
            )
            story = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Clean output: remove URLs, citations, brackets, etc.
            story = re.sub(r"http\S+|www\S+|ftp\S+|\[.*?\]|\(.*?access.*?\)", "", story)
            story_words = story.split()
            story = ' '.join(story_words[:200])

        st.subheader("📝 Generated Story:")
        st.write(story)
