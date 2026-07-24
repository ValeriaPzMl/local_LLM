# importar librerías
import discord

from config import DISCORD_TOKEN
# cargar el token



# Configuramos qué eventos recibirá el bot
intents = discord.Intents.default()
intents.message_content = True


# Creamos el cliente
client = discord.Client(intents=intents)


# Evento: el bot logró conectarse
@client.event
async def on_ready():
    print(f"✅ ComputahMind conectado como {client.user}")

@client.event
async def on_message(message):

    # Evitamos que el bot se responda a sí mismo
    if message.author == client.user:
        return

    if message.content.lower() == "hola":
        await message.channel.send("¡Hola! Soy ComputahMind 🤖")


# Iniciamos el bot
client.run(DISCORD_TOKEN)
# configurar intents

# crear el cliente

# evento al conectarse

# evento cuando llega un mensaje

# iniciar el bot