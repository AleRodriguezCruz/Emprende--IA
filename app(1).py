from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import os
import sqlite3
from math import radians, cos, sin, asin, sqrt
from datetime import datetime

# ==============================================================================
# 1. CONFIGURACIÓN
# ==============================================================================
BASE_DIR = '/home/alejandraRodriguez'

app = Flask(__name__, template_folder=BASE_DIR, static_folder=BASE_DIR)
CORS(app)

# ==============================================================================
# 2. RUTAS Y CARGA DE RECURSOS
# ==============================================================================
ruta_modelo = os.path.join(BASE_DIR, 'modelo_oportunidades_smote.pkl')
ruta_clases = os.path.join(BASE_DIR, 'clases_oportunidades_smote.pkl')
ruta_db = os.path.join(BASE_DIR, 'lugares.db')

modelo = None
clases_dict = {}

print("\n" + "="*60)
print("INICIANDO SISTEMA (MODO PURO - SIN REGLAS)")
print("="*60)

if os.path.exists(ruta_modelo) and os.path.exists(ruta_clases):
    try:
        modelo = joblib.load(ruta_modelo)
        clases_df = joblib.load(ruta_clases)

        # Mapeo de clases
        if 'target' in clases_df.columns and 'oportunidad_negocio' in clases_df.columns:
            clases_dict = dict(zip(clases_df['target'], clases_df['oportunidad_negocio']))
        elif 'codigo' in clases_df.columns and 'oportunidad_negocio' in clases_df.columns:
            clases_dict = dict(zip(clases_df['codigo'], clases_df['oportunidad_negocio']))

        print(f"✅ Modelo cargado. Clases disponibles: {len(clases_dict)}")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
else:
    print("❌ No se encontraron los archivos del modelo.")

# ==============================================================================
# 3. BASE DE DATOS (SQLITE)
# ==============================================================================
def get_db_connection():
    if not os.path.exists(ruta_db): return None
    conn = sqlite3.connect(ruta_db)
    conn.row_factory = sqlite3.Row
    return conn

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1) * cos(lat2) * sin((lon2-lon1)/2)**2
    return 2 * asin(sqrt(a)) * 6371

def contar_cercanos_sql(lat, lng):
    """Obtiene características del entorno usando SQL."""
    features = [0, 0, 0, 0, 0, 0] # Escuelas, Hospitales, Gyms, Papelerias, Farmacias, Comercios
    lugares_detalle = []

    conn = get_db_connection()
    if not conn: return features, lugares_detalle

    try:
        cursor = conn.cursor()
        # Filtro rápido 1km
        cursor.execute("""
            SELECT nom_estab, latitud, longitud, codigo_act FROM lugares
            WHERE latitud BETWEEN ? AND ? AND longitud BETWEEN ? AND ?
        """, (lat-0.01, lat+0.01, lng-0.01, lng+0.01))

        # Filtro exacto 500m
        for row in cursor.fetchall():
            if haversine(lng, lat, row['longitud'], row['latitud']) <= 0.5:
                c = str(row['codigo_act'])
                tipo = "otro"
                if c.startswith('61'): features[0]+=1; tipo="escuela"
                elif c.startswith('62'): features[1]+=1; tipo="hospital"
                elif c.startswith('7139'): features[2]+=1; tipo="gym"
                elif c.startswith('46531'): features[3]+=1; tipo="papeleria"
                elif c.startswith('46411'): features[4]+=1; tipo="farmacia"
                elif c.startswith('46') and not c.startswith(('46531','46411')): features[5]+=1; tipo="comercio"

                if tipo != "otro":
                    lugares_detalle.append({'nombre': row['nom_estab'], 'lat': row['latitud'], 'lng': row['longitud'], 'tipo': tipo})

        return features, lugares_detalle
    except Exception as e:
        print(f"❌ Error DB: {e}")
        return features, lugares_detalle

# ==============================================================================
# 4. PREDICCIÓN
# ==============================================================================
def predecir_puro(features):
    """
    Ejecuta el modelo. Si la predicción ganadora es 'OTRO' (genérico),
    intenta recomendar la segunda mejor opción específica para dar valor al usuario.
    """
    if modelo is None: return None

    # 1. Obtener todas las probabilidades
    features_array = np.array([features], dtype=float)
    probs = modelo.predict_proba(features_array)[0]

    # 2. Ordenar de mayor a menor probabilidad (Top de ganadores)
    # indices_ordenados será algo como [2, 0, 5, 1...] (índices del mejor al peor)
    indices_ordenados = np.argsort(probs)[::-1]

    # 3. SELECCIÓN DEL GANADOR
    idx_ganador = indices_ordenados[0]
    nombre_ganador = clases_dict.get(idx_ganador, "OTRO")
    confianza = probs[idx_ganador]

    # SI EL GANADOR ES "OTRO" O "COMERCIO", MIRAMOS AL SEGUNDO LUGAR
    # (Solo si el segundo lugar tiene una probabilidad decente, ej. > 10%)
    if nombre_ganador.lower() in ['otro', 'comercio', 'otros']:
        print(f"⚠️ La IA predijo '{nombre_ganador}' ({confianza:.2%}). Buscando alternativa...")

        # Revisar el segundo mejor
        idx_segundo = indices_ordenados[1]
        prob_segundo = probs[idx_segundo]

        # Si el segundo tiene al menos 10% de probabilidad, lo promovemos a ganador
        if prob_segundo > 0.10:
            idx_ganador = idx_segundo
            nombre_ganador = clases_dict.get(idx_ganador, "OTRO")
            confianza = prob_segundo
            print(f"✅ Alternativa encontrada: {nombre_ganador} ({confianza:.2%})")
        else:
            print("❌ No hay alternativa fuerte. Se mantiene 'OTRO'.")

    # 4. Generar Top 3 para mostrar
    top3 = []
    for idx in indices_ordenados[:3]: # Solo los 3 mejores
        if idx in clases_dict:
            top3.append({
                'oportunidad': clases_dict[idx],
                'probabilidad': float(probs[idx])
            })

    return {
        'nombre': nombre_ganador,
        'codigo': int(idx_ganador),
        'confianza': float(confianza),
        'top3': top3
    }

# ==============================================================================
# 5. ENDPOINTS
# ==============================================================================
@app.route('/')
def home(): return render_template('index.html')

@app.route('/health')
def health_check():
    return jsonify({'status': 'online', 'db': os.path.exists(ruta_db), 'ai': modelo is not None})

@app.route('/predecir_ubicacion', methods=['POST'])
def predict_location():
    try:
        data = request.get_json()
        lat, lng = float(data.get('lat', 0)), float(data.get('lng', 0))
        print(f"\n📍 Analizando: {lat}, {lng}")

        # 1. Obtener features reales
        features, lugares = contar_cercanos_sql(lat, lng)

        # 2. Inferencia Pura
        resultado = predecir_puro(features)

        if not resultado:
            return jsonify({'status': 'error', 'mensaje': 'Modelo no disponible'}), 500

        # 3. Respuesta
        nombre = resultado['nombre']
        confianza = resultado['confianza']

        # Mensajes neutros (Objetivos)
        mensaje = f"✅ Análisis completado. Oportunidad detectada: {nombre.upper()}"
        icono = "✅"

        if confianza < 0.50:
            mensaje = f"⚠️ Predicción con baja confianza ({confianza:.0%}). Revisa el Top 3."
            icono = "📉"

        return jsonify({
            'status': 'success',
            'oportunidad': nombre.upper(),
            'mensaje': mensaje,
            'icono': icono,
            'confianza': confianza,
            'codigo': resultado['codigo'],
            'caracteristicas': {
                'escuelas': features[0], 'hospitales': features[1],
                'gimnasios': features[2], 'papelerias': features[3],
                'farmacias': features[4], 'comercios': features[5]
            },
            'top3_oportunidades': resultado['top3'],
            'lugares_encontrados': lugares
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'status': 'error', 'mensaje': str(e)}), 400

if __name__ == '__main__':
    print("🚀 SERVIDOR INICIADO (Modo: Pura IA)")
    app.run(debug=True, host='0.0.0.0', port=5000)