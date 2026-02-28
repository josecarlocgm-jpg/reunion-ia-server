"""
Servidor para procesar reuniones: Graba audio -> Whisper (transcribe) -> Claude (resume)
Version 2: Procesamiento en segundo plano + pagina web con contrasena.

Requiere variables de entorno:
  OPENAI_API_KEY=sk-proj-xxx
  ANTHROPIC_API_KEY=sk-ant-api03-xxx
  APP_PASSWORD=tu_contrasena_secreta
"""

import os
import tempfile
import math
import threading
import uuid
import hashlib
from datetime import datetime
from functools import wraps
from flask import Flask, request, jsonify, Response, session, redirect, url_for
from werkzeug.utils import secure_filename
import requests

app = Flask(__name__)
app.secret_key = os.environ.get("APP_SECRET", uuid.uuid4().hex)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
APP_PASSWORD = os.environ.get("APP_PASSWORD", "reunion2026")

WHISPER_MAX_SIZE = 25 * 1024 * 1024

resultados = {}

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


def requiere_login(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('autenticado'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def transcribir_audio(filepath):
    file_size = os.path.getsize(filepath)
    if file_size <= WHISPER_MAX_SIZE:
        return _transcribir_chunk(filepath)
    else:
        return _transcribir_archivo_grande(filepath)


def _transcribir_chunk(filepath):
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
    import subprocess
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", filepath],
        capture_output=True, text=True
    )
    try:
        import json
        duration = float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, json.JSONDecodeError):
        duration = 3600

    file_size = os.path.getsize(filepath)
    num_chunks = math.ceil(file_size / WHISPER_MAX_SIZE)
    chunk_duration = duration / num_chunks
    transcripcion_completa = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(num_chunks):
            start_time = i * chunk_duration
            chunk_path = os.path.join(tmpdir, f"chunk_{i}.m4a")
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


def procesar_en_segundo_plano(job_id, tmp_path, filename):
    try:
        resultados[job_id]["estado"] = "transcribiendo"
        print(f"[{job_id}] Transcribiendo: {filename}")
        transcripcion = transcribir_audio(tmp_path)

        if not transcripcion.strip():
            resultados[job_id]["estado"] = "error"
            resultados[job_id]["error"] = "No se pudo transcribir el audio"
            os.unlink(tmp_path)
            return

        resultados[job_id]["estado"] = "resumiendo"
        print(f"[{job_id}] Resumiendo ({len(transcripcion)} caracteres)")
        resumen = resumir_con_claude(transcripcion)

        resultados[job_id]["estado"] = "listo"
        resultados[job_id]["resumen"] = resumen
        resultados[job_id]["transcripcion"] = transcripcion
        print(f"[{job_id}] Completado!")

    except Exception as e:
        resultados[job_id]["estado"] = "error"
        resultados[job_id]["error"] = str(e)
        print(f"[{job_id}] Error: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.route("/login", methods=["GET", "POST"])
def login():
    error = ""
    if request.method == "POST":
        password = request.form.get("password", "")
        if password == APP_PASSWORD:
            session['autenticado'] = True
            return redirect(url_for('home'))
        else:
            error = "Contrasena incorrecta"

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reunion IA - Login</title>
<style>
body {{ font-family: -apple-system, Arial, sans-serif; background: #1a1a2e; color: #eee;
       display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }}
.login-box {{ background: #16213e; padding: 40px; border-radius: 16px; width: 320px; text-align: center; }}
h1 {{ color: #e94560; margin-bottom: 10px; font-size: 24px; }}
p {{ color: #888; font-size: 14px; }}
input[type=password] {{ width: 100%; padding: 14px; border: 2px solid #0f3460; border-radius: 10px;
       background: #1a1a2e; color: #eee; font-size: 16px; margin: 15px 0; box-sizing: border-box; }}
input[type=password]:focus {{ border-color: #e94560; outline: none; }}
button {{ width: 100%; padding: 14px; background: #e94560; color: white; border: none;
         border-radius: 10px; font-size: 16px; cursor: pointer; font-weight: bold; }}
button:hover {{ background: #c13350; }}
.error {{ color: #e74c3c; font-size: 14px; margin-top: 10px; }}
</style></head><body>
<div class="login-box">
<h1>Reunion IA</h1>
<p>Ingresa tu contrasena para ver los resumenes</p>
<form method="POST">
<input type="password" name="password" placeholder="Contrasena" autofocus>
<button type="submit">Entrar</button>
</form>
{"<p class='error'>" + error + "</p>" if error else ""}
</div></body></html>"""


@app.route("/logout")
def logout():
    session.pop('autenticado', None)
    return redirect(url_for('login'))


@app.route("/", methods=["GET"])
@requiere_login
def home():
    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reunion IA</title>
<style>
body { font-family: -apple-system, Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }
.header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
h1 { color: #e94560; margin: 0; }
.logout { color: #888; text-decoration: none; font-size: 14px; padding: 8px 16px; border: 1px solid #444; border-radius: 8px; }
.logout:hover { color: #e94560; border-color: #e94560; }
.job { background: #16213e; border-radius: 12px; padding: 20px; margin: 15px 0; }
.job h3 { margin-top: 0; color: #e94560; }
.estado { padding: 4px 12px; border-radius: 20px; font-size: 14px; display: inline-block; }
.listo { background: #2ecc71; color: #000; }
.procesando { background: #f39c12; color: #000; }
.error { background: #e74c3c; color: #fff; }
pre { background: #0f3460; padding: 15px; border-radius: 8px; white-space: pre-wrap; word-wrap: break-word; font-size: 14px; max-height: 500px; overflow-y: auto; }
.empty { text-align: center; color: #666; padding: 40px; }
.copy-btn { background: #e94560; color: white; border: none; padding: 8px 16px; border-radius: 8px; cursor: pointer; font-size: 14px; margin-top: 10px; }
.copy-btn:hover { background: #c13350; }
.transcripcion-btn { background: #0f3460; color: #aaa; border: 1px solid #444; padding: 6px 12px; border-radius: 8px; cursor: pointer; font-size: 13px; margin-top: 8px; margin-left: 8px; }
</style></head><body>
<div class="header">
<h1>Reunion IA</h1>
<a href="/logout" class="logout">Cerrar sesion</a>
</div>
"""

    if not resultados:
        html += '<div class="empty">No hay reuniones procesadas aun.<br><br>Envia un audio desde Notas de Voz en tu iPhone.</div>'
    else:
        for job_id, data in sorted(resultados.items(), key=lambda x: x[1].get("fecha", ""), reverse=True):
            estado = data.get("estado", "desconocido")
            fecha = data.get("fecha", "")

            if estado == "listo":
                badge = '<span class="estado listo">Listo</span>'
            elif estado == "error":
                badge = '<span class="estado error">Error</span>'
            else:
                badge = f'<span class="estado procesando">{estado.capitalize()}...</span>'

            html += f'<div class="job"><h3>Reunion - {fecha} {badge}</h3>'

            if estado == "listo":
                resumen = data.get("resumen", "").replace("<", "&lt;").replace(">", "&gt;")
                html += f'<pre id="r-{job_id}">{resumen}</pre>'
                html += f'<button class="copy-btn" onclick="copyText(\'r-{job_id}\')">Copiar resumen</button>'
                html += f'<button class="transcripcion-btn" onclick="toggleTranscripcion(\'t-{job_id}\')">Ver transcripcion</button>'
                transcripcion = data.get("transcripcion", "").replace("<", "&lt;").replace(">", "&gt;")
                html += f'<pre id="t-{job_id}" style="display:none; background:#0a1628;">{transcripcion}</pre>'
            elif estado == "error":
                error_msg = data.get("error", "").replace("<", "&lt;").replace(">", "&gt;")
                html += f'<pre>{error_msg}</pre>'
            else:
                html += '<p>Procesando... la pagina se actualiza automaticamente.</p>'
                html += '<script>setTimeout(function(){location.reload()}, 15000);</script>'

            html += '</div>'

    html += """
<script>
function copyText(id) {
    var text = document.getElementById(id).innerText;
    navigator.clipboard.writeText(text).then(function() {
        alert('Resumen copiado!');
    }).catch(function() {
        var range = document.createRange();
        range.selectNode(document.getElementById(id));
        window.getSelection().removeAllRanges();
        window.getSelection().addRange(range);
        document.execCommand('copy');
        alert('Resumen copiado!');
    });
}
function toggleTranscripcion(id) {
    var el = document.getElementById(id);
    el.style.display = el.style.display === 'none' ? 'block' : 'none';
}
</script>
</body></html>"""

    return html


@app.route("/procesar", methods=["POST"])
def procesar_reunion():
    """Recibe audio, inicia procesamiento en segundo plano, responde inmediatamente."""
    if "audio" not in request.files:
        return "Error: No se envio archivo de audio. Usa el campo 'audio'", 400

    archivo = request.files["audio"]
    if archivo.filename == "":
        return "Error: Archivo vacio", 400

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp:
            archivo.save(tmp.name)
            tmp_path = tmp.name

        job_id = str(uuid.uuid4())[:8]
        fecha = datetime.now().strftime("%d/%m/%Y %H:%M")
        resultados[job_id] = {
            "estado": "recibido",
            "fecha": fecha,
            "archivo": archivo.filename
        }

        hilo = threading.Thread(
            target=procesar_en_segundo_plano,
            args=(job_id, tmp_path, archivo.filename)
        )
        hilo.start()

        dominio = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "reunion-ia-server-production.up.railway.app")

        return f"Audio recibido! Tu reunion se esta procesando.\n\nRevisa el resumen en:\nhttps://{dominio}\n\nID: {job_id}\nFecha: {fecha}", 200, {'Content-Type': 'text/plain; charset=utf-8'}

    except Exception as e:
        return f"Error: {str(e)}", 500


@app.route("/resultado/<job_id>", methods=["GET"])
def ver_resultado(job_id):
    if job_id not in resultados:
        return "Job no encontrado", 404

    data = resultados[job_id]
    estado = data.get("estado", "")

    if estado == "listo":
        return data.get("resumen", ""), 200, {'Content-Type': 'text/plain; charset=utf-8'}
    elif estado == "error":
        return f"Error: {data.get('error', '')}", 500, {'Content-Type': 'text/plain; charset=utf-8'}
    else:
        return f"Procesando... Estado: {estado}. Actualiza en unos minutos.", 200, {'Content-Type': 'text/plain; charset=utf-8'}


@app.route("/ultimo", methods=["GET"])
def ultimo_resultado():
    completados = {k: v for k, v in resultados.items() if v.get("estado") == "listo"}
    if not completados:
        return "No hay resumenes listos aun.", 200, {'Content-Type': 'text/plain; charset=utf-8'}
    ultimo = max(completados.items(), key=lambda x: x[1].get("fecha", ""))
    return ultimo[1].get("resumen", ""), 200, {'Content-Type': 'text/plain; charset=utf-8'}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
