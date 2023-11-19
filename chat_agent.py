from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# from langchain.llms import Ollama
# from langchain.embeddings import OllamaEmbeddings
from custom_llm import OllamaCustomEmbedding, OllamaCustomLLM
from langchain.vectorstores import Chroma
# from langchain.docstore.document import Document

import fastapi
from fastapi.middleware.cors import CORSMiddleware


# python -m uvicorn chat_agent:app --reload
# curl -X POST http://127.0.0.1:8000/chatbot/init/
# curl -X POST http://127.0.0.1:8000/chatbot/load_data/


VECTOR_STORE = None
QA_CHAIN = None
PROMPT = PromptTemplate(input_variables=['question', 'context'], 
                        template="[INST]<<SYS>> Você é um assistente para tarefas de perguntas e respostas. Semore eesponda a pergunta (question) em português brasileiro, nunca em inglês, e Utilize os trechos de contexto (context) recuperados a seguir para responder à pergunta. Se você não souber a resposta, apenas diga que não sabe. Responda de forma sucinta e objetiva <</SYS>> \nQuestion: {question} \nContext: {context} \nAnswer: [/INST]")


app = fastapi.FastAPI(title="Chat bot") 

API_ORIGINS = ["*"] #os.getenv("API_ORIGINS", "http://localhost").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def main():


    init_vectorstore()

    init_qa()

    load_data()

    answer_question()



def search_webpage(url: str) -> str:

    loader = WebBaseLoader(url)
    data = loader.load()    

    # text = 'As raízes etimológicas do termo "Brasil" são de difícil reconstrução. O filólogo Adelino José da Silva Azevedo postulou que se trata de uma palavra de procedência celta (uma lenda que fala de uma "terra de delícias", vista entre nuvens), mas advertiu também que as origens mais remotas do termo poderiam ser encontradas na língua dos antigos fenícios. Na época colonial, cronistas da importância de João de Barros, frei Vicente do Salvador e Pero de Magalhães Gândavo apresentaram explicações concordantes acerca da origem do nome "Brasil". De acordo com eles, o nome "Brasil" é derivado de "pau-brasil", designação dada a um tipo de madeira empregada na tinturaria de tecidos. Na época dos descobrimentos, era comum aos exploradores guardar cuidadosamente o segredo de tudo quanto achavam ou conquistavam, a fim de explorá-lo vantajosamente, mas não tardou em se espalhar na Europa que haviam descoberto certa "ilha Brasil" no meio do oceano Atlântico, de onde extraíam o pau-brasil (madeira cor de brasa).[27] Antes de ficar com a designação atual, "Brasil", as novas terras descobertas foram designadas de: Monte Pascoal (quando os portugueses avistaram terras pela primeira vez), Ilha de Vera Cruz, Terra de Santa Cruz, Nova Lusitânia, Cabrália, Império do Brasil e Estados Unidos do Brasil.[28] Os habitantes naturais do Brasil são denominados brasileiros, cujo gentílico é registrado em português a partir de 1706[29] e referia-se inicialmente apenas aos que comercializavam pau-brasil.'
    # doc =  Document(page_content=text, metadata={"source": "local"})

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    all_splits = text_splitter.split_documents(data)

    VECTOR_STORE.add_documents(documents=all_splits)

# @app.post("/chatbot/init/")
# def init():
#     init_vectorstore()
#     init_qa()

def init_qa():

    llm = OllamaCustomLLM(
    model="mistral"
    )

    global QA_CHAIN
    QA_CHAIN = RetrievalQA.from_chain_type(
        llm,
        retriever=VECTOR_STORE.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT},
    )

def init_vectorstore():
    
    global VECTOR_STORE
    if not VECTOR_STORE:
        embeddings = OllamaCustomEmbedding()
        VECTOR_STORE = Chroma("langchain_store", embeddings, persist_directory="./chroma_db")

@app.post("/chatbot/load_data/")
def load_data():

    if not VECTOR_STORE:
        init_vectorstore()

    url_list = ['https://pt.wikipedia.org/wiki/Brasil']

    for url in url_list:
        search_webpage(url)

@app.get("/chatbot/answer/{question}")
def answer_question(question: str):

    qa_chain = get_qachain()

    result = qa_chain({"query": question})

    print(result)

    return result
    
def get_qachain():

    if not VECTOR_STORE:
        init_vectorstore()
    if not QA_CHAIN:
        init_qa()

    return QA_CHAIN

    

# if __name__ == '__main__':
#     main()
