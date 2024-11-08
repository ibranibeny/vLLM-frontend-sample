
import argparse
import gradio as gr
import time
from openai import OpenAI
from PyPDF2 import PdfReader
from fastapi import FastAPI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings

parser = argparse.ArgumentParser(
    description='Chatbot Interface with Customizable Parameters')
parser.add_argument('--url',
                    type=str,
                    default='http://localhost:8000/v1',
                    help='Model URL')
parser.add_argument('--filepath',
                    type=str,
                    required=False,
                    help='file path for pdf')
parser.add_argument('-m',
                    '--model',
                    type=str,
                    required=True,
                    help='Model name for the chatbot')
parser.add_argument('--temp',
                    type=float,
                    default=0.8,
                    help='Temperature for text generation')
parser.add_argument('--max_tokens',
                    type=int,
                    default=512,
                    help='Maximum number of tokens for text generation')

args = parser.parse_args()



client = OpenAI(
    api_key="EMPTY",
    base_url=args.url,
)

def user(user_prompt, history):
    return "", history + [[user_prompt, None]]

def generate_response(rag_selection, history):
    user_message = history[-1][0]

    if rag_selection == "Direct Output with LLM":
      response = [{"role": "user", "content": user_message}]
    else:
      retrieved_documents = retriever.invoke(user_message)
      context = ' '.join([retrieved_documents[i].page_content for i in range(len(retrieved_documents))])
    prompt = """You are travel assistant that specialise on Indonesia. Your taks is to answer the user's question to the best of your ability.\
                  Use the following pieces of retrieved context to answer the question only if the retrieved context is highly relevant. \
                  If the retrieved context is not relevant, answer the question to your best ability. \

                  {context}"""

    response = [{"role": "system", "content": prompt},
                  {"role": "user", "content": user_message}]

    completion = client.chat.completions.create(
        model=args.model,  # Model name to use
        messages=response,  # Chat history
        stream=True,
        stream_options={"include_usage": True},
        max_tokens=args.max_tokens,  # Maximum number of tokens to generate
        temperature=args.temp,  # Temperature for text generation
    )

    bot_response = ""
    tokens_num = 0
    st = time.time()

    for chunk in completion:
      if hasattr(chunk.choices[0], 'finish_reason') and chunk.choices[0].finish_reason is not None:
        next_chunk = next(completion, None)
        if next_chunk:
          tokens_num = next_chunk.usage.completion_tokens
          et = time.time() - st
          yield history, f'{round(tokens_num/et,2)}'
        else:
          return None # Handle the case where there's no next chunk
      else:
        et = time.time() - st
        chunk_message = chunk.choices[0].delta.content or ""
        bot_response += chunk_message
        history[-1][1] = bot_response
        yield history, f'{round(tokens_num/et,2)}'

with gr.Blocks(theme=gr.Theme.from_hub('HaleyCH/HaleyCH_Theme'), css=".column-form .wrap {flex-direction: column;}") as demo:
    with gr.Row():
      gr.Markdown("""
            <h1><center>Intel Chatbot with vLLM Model Serving
            <center><img src="https://upload.wikimedia.org/wikipedia/commons/6/64/Intel-logo-2022.png" width=200px>
            <h2>Inferenced instance powered by Intel 4th Gen Xeon with AMX Acceleration</h2></n>Using Meta Llama 3.1 8B</center>
            """)
    with gr.Row():
        with gr.Column(visible=True, min_width=250, scale=0) as sidebar:
          with gr.Row():
            gr.Markdown("""
              This is a simple Chatbot<br>""")
          with gr.Row():
            dropdown = gr.Dropdown(["Direct Output with LLM", "PDF-Augmented Output"], label="Llama3.1 8B")
        with gr.Column() as main:
            with gr.Row():
                chatbot = gr.Chatbot(label="Conversation")
            with gr.Row():
                prompt_input = gr.Textbox(label="Enter your message here", placeholder="Type your message...",scale=7)
                token_output = gr.Textbox(label="Tokens/sec", placeholder="0.00",scale=3)

                prompt_input.submit(user, [prompt_input, chatbot], [prompt_input, chatbot], queue=False).then(
                generate_response, inputs=[dropdown,chatbot], outputs=[chatbot,token_output], queue=True)

        with gr.Column(visible=True, min_width=250, scale=0) as examplelist:
            gr.Examples(
            examples=[
            ["Tell me about Universitas Indonesia"],
            ["When Universitas Indonesia founded"],
        ],
        inputs=prompt_input,
    )



    with gr.Row():
            gr.Markdown("""
            <center><br><h3>An Intel Collaboration with Beny Ibrani // Malcolm Chan</h3>
            """)

demo.queue(default_concurrency_limit=100).launch(debug=True, share=True)

#CUSTOM_PATH = "/vllm-ipex"
#app = FastAPI()

#@app.get("/")
#def read_main():
#    return {"message": "This is your main app"}

#app = gr.mount_gradio_app(app, demo, path=CUSTOM_PATH)

