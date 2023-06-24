from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from googletrans import Translator
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain


# Load the PDF document
loader = PyPDFLoader("/tmp/Siri Guru Granth - English Translation (matching pages).pdf")
data = loader.load()

# Split the document into texts
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
texts = text_splitter.split_documents(data)

# Load OpenAI embeddings
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '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')
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Initialize Pinecone
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '6b2f9b50-de08-4a4c-900f-0c8413fef00b')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'us-west4-gcp-free')
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
index_name = "chatbot"

# Create Pinecone index and load texts
docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

# Initialize Google Translate
translator = Translator()

# Initialize OpenAI LLM
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")

def translate_text(text, dest):
    translation = translator.translate(text, dest=dest)
    return translation.text

def process_query(query):
    # Translate the query
    query_trans = translate_text(query, 'en')

    # Perform document search
    docs = docsearch.similarity_search(query_trans)

    # Run question-answering chain
    ans = chain.run(input_documents=docs, question=query_trans)

    # Translate the answer back to the original language
    ans_trans = translate_text(ans, 'pa')

    return ans_trans

def main(request):
    if request.method == 'POST':
        # Get the query from the request data
        data = request.get_json()
        query = data.get('query')

        # Process the query
        answer = process_query(query)

        # Return the answer as a JSON response
        return jsonify({'answer': answer})

    # Return an error for other HTTP methods
    return 'Method not allowed.', 405
