import os
import discord
import json
import numpy as np
import faiss
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
INSTRUCTIONS = os.getenv("INSTRUCTIONS")
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID")
DEFAULT_USER_NICKNAME = os.getenv("DEFAULT_USER_NICKNAME")

client_ai = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=OPENROUTER_KEY,
)

model = SentenceTransformer('modelo_podrao')
index = faiss.read_index("faiss_index.index")

intents = discord.Intents.default()
intents.message_content = True  # Necessário para ler o conteúdo das mensagens
client = discord.Client(intents=intents)

model = SentenceTransformer('modelo_podrao')
index = faiss.read_index("faiss_index.index")

with open('conversas_podroes_filtrado.json', 'r', encoding='utf-8') as f:
    conversas = json.load(f)

def recuperar_contexto(query, k=3):
    query_embedding = model.encode([query], convert_to_tensor=False)
    query_embedding = np.array(query_embedding).astype("float32")
    
    distances, indices = index.search(query_embedding, k)
    context_msgs = [conversas[i]["message"] for i in indices[0]]

    return "\n".join(context_msgs)

def chat_response(mensagem):
    contexto = recuperar_contexto(mensagem)

    prompt = f"""
    {INSTRUCTIONS}
    Aqui estão algumas mensagens anteriores para basear em sua resposta:
    {contexto}
    Agora responda a esta nova mensagem: "{mensagem}"
    """
    
    resposta = client_ai.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    error_api = "Lembrei que minha planta de estimação precisa de terapia e tenho que cuidar dela."
    error_choices = "So um momento. Minha geladeira deu um erro 404 e preciso reiniciá-la manualmente."
    error_none = "Já volto! Um esquilo invadiu minha casa e preciso negociar a rendição dele."

    if resposta is None:
        return error_api
    
    if not resposta.choices:   
        return error_choices
    
    if resposta.choices[0].message is None:
        return error_none

    return resposta.choices[0].message.content.strip()


@client.event
async def on_ready():
    print(f'Bot conectado como {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return  # Ignorar mensagens do próprio bot
    
    if message.author.id == DEFAULT_USER_ID or message.author.display_name == DEFAULT_USER_NICKNAME:
        resposta = "ZeroDivisionError"
    else:
        resposta = chat_response(message.content)

    channel = discord.utils.get(message.guild.text_channels, name='meiogpt')
    await channel.send(resposta)

client.run(TOKEN)
