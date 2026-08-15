import os
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from google import genai
from google.genai import types

app = FastAPI()

# Manejo de CORS para Producción
# Lee los dominios permitidos desde el entorno. Si no hay, usa localhost por defecto.
# Cuando lo subas a producción, reemplazá "http://localhost:8000" por tu dominio real.
origenes_permitidos = os.getenv(
    "ALLOWED_ORIGINS", 
    "http://localhost:5500,http://127.0.0.1:5500,http://localhost:8000,null"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origenes_permitidos,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializamos el cliente (la clave la podés mover a un .env más adelante usando os.getenv)
client = genai.Client(api_key="key de prueba")

@app.post("/api/extraer-cotizacion")
async def extraer_cotizacion(imagen: UploadFile = File(...)):
    try:
        image_bytes = await imagen.read()
        img = Image.open(io.BytesIO(image_bytes))

        prompt = """
        Analiza la imagen adjunta, que es una planilla de cotización de insumos médicos.
        Extrae los datos de la tabla principal y devuelve EXCLUSIVAMENTE un array JSON con este formato exacto para cada fila:
        [
          {
            "renglon": 1,
            "cantidad": 3000,
            "descripcion": "LOPERAMIDA COMPRIMIDOS...",
            "marca": "VANNIER",
            "precio_unitario": 60.00,
            "importe_total": 180000.00
          }
        ]
        Si la columna 'marca' no está presente o un dato está vacío, asigna el valor null.
        No incluyas texto adicional ni formato markdown, solo el JSON puro.
        """
        
        configuracion = types.GenerateContentConfig(response_mime_type="application/json")

        # Implementación de Fallback y llamada Asíncrona (AIO) para no bloquear FastAPI
        try:
            # 1. Intentamos con el modelo pinneado más actual
            modelo_fijo = 'gemini-2.5-flash'
            response = await client.aio.models.generate_content(
                model=modelo_fijo,
                contents=[img, prompt],
                config=configuracion
            )
        except Exception as e_fijo:
            print(f"⚠️ Falló el modelo {modelo_fijo} ({e_fijo}). Usando fallback a flash-latest...")
            # 2. Si falla (por cuota, deprecación, etc), caemos en el fallback seguro
            response = await client.aio.models.generate_content(
                model='gemini-flash-latest',
                contents=[img, prompt],
                config=configuracion
            )

        # Limpiamos las posibles etiquetas de markdown por si las devuelve
        texto_crudo = response.text.strip()
        if texto_crudo.startswith("```json"):
            texto_crudo = texto_crudo[7:]
        if texto_crudo.endswith("```"):
            texto_crudo = texto_crudo[:-3]
            
        return json.loads(texto_crudo.strip())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))