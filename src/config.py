from dotenv import load_dotenv
import os

# Carga las variables del archivo .env
load_dotenv()

# Token del bot
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")