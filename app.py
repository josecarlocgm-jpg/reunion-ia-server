"""
Servidor para procesar reuniones: Graba audio → Whisper (transcribe) → Claude (resume)
Despliega gratis en Railway o Render.

Requiere variables de entorno:
  OPENAI_API_KEY=sk-proj-xxx
  ANTHROPIC_API_KEY=sk-ant-api03-xxx
"""

import os
import tempfile
import math
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import requests

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Whisper tiene limite de 25MB por archivo
WHISPER_MAX_SIZE = 25 * 1024 * 1024  # 25MB

PROMPT_RESUMEN = """Eres un asistente ejecutivo experto. Analiza la siguiente transcripcion de una reunion y genera un reporte estructurado.

FORMATO DE SALIDA:

1. RESUMEN EJECUTIVO
3-5 lineas que capturen la esencia de la reunion.

2. DECISIONES TOMADAS
Lista de cada decision con quien la tomo.

3. ACTION ITEMS
Formato: Tarea | Responsable | Fecha limite
Si no se menciono responsable o fecha, pon "Por definir".

4. RIESGOS Y BLOQUEOS
Problemas mencionados que podrian afectar el progreso.

5. PROXIMOS PASOS
Que viene despues segun lo acordado.

6. DATOS IMPORTANTES
Numeros, fechas, montos o datos especificos mencionados.

Reglas: Se directo. No inventes informacion que no este en la transcripcion. Si algo no se discutio, omite esa seccion.

TRANSCRIPCION:
{transcripcion}"""


def transcribir_audio(filepath):
    """
    Transcribe audio usando Whisper API.
    Si el archivo es mayor a 25MB, lo divide en partes.
    """
    file_size = os.path.getsize(filepath)
    
    if file_size <= WHISPER_MAX_SIZE:
        # Archivo pequeno, transcribir directo
        return _transcribir_chunk(filepath)
    else:
        # Archivo grande, dividir y transcribir por partes
        return _transcribir_archivo_grande(filepath)


def _transcribir_chunk(filepath):
    """Transcribe un archivo de audio con Whisper API."""
    with open(filepath, "rb") as audio_file:
        response = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            files={"file": audio_file},
            data={"model": "whisper-1", "language": "es"}
        )
    
    if response.status_code != 200:
        raise Exception(f"Error Whisper: {response.text}")
    
    return response.json().get("text", "")


def _transcribir_archivo_grande(filepath):
    """
    Divide un archivo grande en chunks de ~24MB y transcribe cada uno.
    Usa ffmpeg para dividir por tiempo.
    """
    import subprocess
    
    # Obtener duracion del audio
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", filepath],
        capture_output=True, text=True
    )
    
    try:
        import json
        duration = float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, json.JSONDecodeError):
        duration = 3600  # Asumir 1 hora si no puede leer
    
    file_size = os.path.getsize(filepath)
    num_chunks = math.ceil(file_size / WHISPER_MAX_SIZE)
    chunk_duration = duration / num_chunks
    
    transcripcion_completa = []
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(num_chunks):
            start_time = i * chunk_duration
            chunk_path = os.path.join(tmpdir, f"chunk_{i}.m4a")
            
            # Dividir con ffmpeg
            subprocess.run([
                "ffmpeg", "-y", "-i", filepath,
                "-ss", str(start_time),
                "-t", str(chunk_duration),
                "-c", "copy",
                chunk_path
            ], capture_output=True)
            
            if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 0:
                texto = _transcribir_chunk(chunk_path)
                transcripcion_completa.append(texto)
    
    return " ".join(transcripcion_completa)


def resumir_con_claude(transcripcion):
    """Envia la transcripcion a Claude y obtiene el resumen."""
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        },
        json={
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 4000,
            "messages": [{
                "role": "user",
                "content": PROMPT_RESUMEN.format(transcripcion=transcripcion)
            }]
        }
    )
    
    if response.status_code != 200:
        raise Exception(f"Error Claude: {response.text}")
    
    data = response.json()
    return data["content"][0]["text"]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "mensaje": "Servidor de Reunion IA activo",
        "uso": "POST /procesar con un archivo de audio"
    })


@app.route("/procesar", methods=["POST"])
def procesar_reunion():
    """
    Endpoint principal.
    Recibe un archivo de audio, lo transcribe y lo resume.
    """
    # Verificar que se envio un archivo
    if "audio" not in request.files:
        return jsonify({"error": "No se envio archivo de audio. Usa el campo 'audio'"}), 400
    
    archivo = request.files["audio"]
    if archivo.filename == "":
        return jsonify({"error": "Archivo vacio"}), 400
    
    try:
        # Guardar archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp:
            archivo.save(tmp.name)
            tmp_path = tmp.name
        
        # Paso 1: Transcribir
        print(f"Transcribiendo: {archivo.filename} ({os.path.getsize(tmp_path)} bytes)")
        transcripcion = transcribir_audio(tmp_path)
        
        if not transcripcion.strip():
            return jsonify({"error": "No se pudo transcribir el audio"}), 400
        
        # Paso 2: Resumir con Claude
        print(f"Resumiendo transcripcion ({len(transcripcion)} caracteres)")
        resumen = resumir_con_claude(transcripcion)
        
        # Limpiar archivo temporal
        os.unlink(tmp_path)
        
        return jsonify({
            "status": "ok",
            "transcripcion": transcripcion,
            "resumen": resumen
        })
    
    except Exception as e:
        # Limpiar archivo temporal si existe
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        return jsonify({"error": str(e)}), 500


@app.route("/solo-transcribir", methods=["POST"])
def solo_transcribir():
    """Solo transcribe, sin resumir. Util para debug."""
    if "audio" not in request.files:
        return jsonify({"error": "No se envio archivo de audio"}), 400
    
    archivo = request.files["audio"]
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp:
            archivo.save(tmp.name)
            tmp_path = tmp.name
        
        transcripcion = transcribir_audio(tmp_path)
        os.unlink(tmp_path)
        
        return jsonify({"status": "ok", "transcripcion": transcripcion})
    
    except Exception as e:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
