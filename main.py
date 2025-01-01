from langchain_community.document_loaders import SeleniumURLLoader
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from openai import OpenAI
from flask import Flask, request,jsonify,render_template
from flask_restful import Api,Resource
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from langchain.docstore import InMemoryDocstore
import numpy as np
import os

os.environ["OPENAI_API_KEY"] = "api_key"
 
urls = ["https://brainlox.com/courses/category/technical"]

loader = SeleniumURLLoader(urls=urls)

data = loader.load()

#test if the data is correct
#print(data) 
def load_and_prepare_data(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    split_documents = text_splitter.split_documents(data)

    split_texts = [doc.page_content for doc in split_documents]

    return split_texts

data = load_and_prepare_data(data)

embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
doc_texts = [doc for doc in data]
embeddings = embedding.embed_documents(doc_texts)

embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])  
index.add(embeddings)

docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(data)})
index_to_docstore_id = {i: str(i) for i in range(len(data))}

vector_store = FAISS(
    index=index, 
    docstore=docstore, 
    index_to_docstore_id=index_to_docstore_id,
    embedding_function=embedding.embed_query 
)

vector_store.save_local("faiss_index")

llm = ChatOpenAI(
    model_name="gpt-4", 
    temperature=0.7,
    openai_api_key="your-api-key-here" 
)

chatbot_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(),
    return_source_documents=True,
)

conversation_history = []

class ChatBot(Resource):
    def post(self):
        global conversation_history

        user_input = request.json.get("message", "")
        if not user_input:
            return jsonify({"error": "Message is required."}), 400

        result = chatbot_chain({"question": user_input, "chat_history": conversation_history})

        conversation_history.append((user_input, result["answer"]))

        return jsonify({
            "response": result["answer"],
            "source_documents": [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in result["source_documents"]
            ]
        })



app = Flask(__name__)
api = Api(app)

api.add_resource(ChatBot, "/chat")

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
