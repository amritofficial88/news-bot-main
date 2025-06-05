from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

st.set_page_config(page_title="NewsBot")
st.title("NewsBot: AI-Powered News Q&A")
st.sidebar.title("Drop Your Source")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", placeholder="Paste your link here")
    urls.append(url)

uploaded_file = st.sidebar.file_uploader("Or upload a PDF article", type=["pdf"])

process_url_clicked = st.sidebar.button("Load Source")
file_path = "faiss_store_openai.pkl"

if uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    data = [Document(page_content=text)]


main_placeholder = st.empty()
# llm = OpenAI(temperature=0.9, max_tokens=500)
llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.9, max_tokens=500)

if process_url_clicked:
    
    # load data
    if uploaded_file is not None:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        # Add required metadata with 'source'
        data = [Document(page_content=text, metadata={"source": uploaded_file.name})]

    else:
        loader = UnstructuredURLLoader(urls=[u for u in urls if u])
        main_placeholder.text("Loading your source data...")
        data = loader.load()
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
           
           # vectorstore = pickle.load(f)
            faiss_index, docstore, index_to_docstore_id = pickle.load(f)
            
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS(
                index=faiss_index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id,
                embedding_function=embeddings
            )
            print("Embedding type:", type(vectorstore.embedding_function))

            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)




