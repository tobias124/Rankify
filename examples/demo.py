import streamlit as st
from rankify.dataset.dataset import Document, Question, Answer, Context
from rankify.retrievers.retriever import Retriever
from rankify.models.reranking import Reranking
from rankify.utils.pre_defind_models import HF_PRE_DEFIND_MODELS
import base64

# Set custom page config with your logo as favicon
st.set_page_config(
    page_title="Rankify Demo",
    page_icon="../images/rankify-crop.png",  # ✅ your custom logo here
    layout="centered"  # Optional: gives more horizontal space
)
# ============== Custom CSS =====================
st.markdown("""
<style>
    /* Logo + title layout */
    .logo-container {
        text-align: center;
        margin-bottom: 10px;
    }

    .logo-img {
        width: 150px;
        border-radius: 15px;
    }

    h1 {
        text-align: center;
        color: #1e90ff;
        font-weight: bold;
    }

    .stTextArea, .stSelectbox, .stSlider {
        background-color: #f0f8ff !important;
        border-radius: 10px;
        padding: 5px;
    }

    .stButton>button {
        width: 100%;
        background-color: #1e90ff;
        color: white;
        border-radius: 10px;
        font-size: 1.1rem;
        padding: 10px;
        margin-top: 20px;
    }

    .stButton>button:hover {
        background-color: #63b3ed;
    }

    .stExpander {
        border: 1px solid #cce7ff !important;
        border-radius: 10px;
    }

    .block-container {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============== Embed and Display Logo ==========
with open("../images/rankify-crop.png", "rb") as img_file:
    b64_logo = base64.b64encode(img_file.read()).decode()

st.markdown(f"""
<div class="logo-container">
    <img class="logo-img" src="data:image/png;base64,{b64_logo}" />
</div>
<h1>Rankify Demo</h1>
""", unsafe_allow_html=True)

# ============== User Input: Question =============
query = st.text_area("🧠 Enter your question:")

# ============== Configuration ====================
st.markdown("### 🔧 Configuration")
retriever_method = st.selectbox("Select Retriever:", ["dpr", "bm25", "contriever", "ance", "colbert"])
index_type = st.selectbox("Select Index Type:", ["wiki", "msmarco"])
top_k = st.slider("Number of Retrieved Documents:", 1, 20, 5)

apply_reranking = st.selectbox("Apply Reranking:", ["No Reranking"] + list(HF_PRE_DEFIND_MODELS.keys()))
reranking_model = None
if apply_reranking != "No Reranking":
    reranking_model = st.selectbox("Select Reranking Model:", list(HF_PRE_DEFIND_MODELS[apply_reranking].keys()))

# ============== Button: Retrieve ================
if st.button("🚀 Retrieve Documents"):
    if not query.strip():
        st.warning("⚠️ Please enter a question to proceed.")
    else:
        with st.spinner("🔍 Retrieving documents..."):
            documents = [Document(question=Question(query), answers=Answer([]), contexts=[])]
            retriever = Retriever(method=retriever_method.lower(), n_docs=top_k, index_type=index_type.lower())
            retrieved_documents = retriever.retrieve(documents)

            if apply_reranking != "No Reranking" and reranking_model:
                reranker = Reranking(method=apply_reranking, model_name=reranking_model)
                with st.spinner("✨ Applying reranking..."):
                    reranked_documents = reranker.rank(retrieved_documents)

        # ========== Display Retrieved ==============
        st.subheader("📄 Retrieved Documents")
        for doc in retrieved_documents:
            for context in doc.contexts[:top_k]:
                with st.expander(f"📘 {context.title}"):
                    st.write(context.text)

        # ========== Display Reranked ===============
        if apply_reranking != "No Reranking" and reranking_model:
            st.subheader("🔁 Reranked Documents")
            for doc in reranked_documents:
                for context in doc.reorder_contexts[:top_k]:
                    with st.expander(f"📗 {context.title}"):
                        st.write(context.text)