import os
import discord
import json
import random
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.message_content = True  # Necessário para ler o conteúdo das mensagens
client = discord.Client(intents=intents)

model = SentenceTransformer('modelo_podrao')
index = faiss.read_index("faiss_index.index")

with open('conversas_podroes_filtrado.json', 'r', encoding='utf-8') as f:
    conversas = json.load(f)

def recuperar_contexto(query, k=3):
    # Gerar embedding para a query
    query_embedding = model.encode([query], convert_to_tensor=False)
    query_embedding = np.array(query_embedding).astype("float32")
    
    # Buscar os k vetores mais próximos no índice FAISS
    distances, indices = index.search(query_embedding, k)
    
    # Recuperar as mensagens de contexto com base nos índices
    context_msgs = [conversas[i]["message"] for i in indices[0]]
    return context_msgs


def gerar_resposta_aleatoria(context_msgs):
    if context_msgs:
        return random.choice(context_msgs)
    else:
        return "Vou na padaria e volto já."

@client.event
async def on_ready():
    print(f'Bot conectado como {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return  # Ignorar mensagens do próprio bot

    contexto = recuperar_contexto(message.content)
    resposta = random.choice(contexto)

    channel = discord.utils.get(message.guild.text_channels, name='meiogpt')
    if channel:
        await channel.send(resposta)
    else:
        print("Canal 'meioGPT' não encontrado.")

client.run(TOKEN)
