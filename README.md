# 🚀 Emprende IA

> **Transformamos datos geográficos en oportunidades de negocio con Inteligencia Artificial.**

![Python](https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgrey?style=for-the-badge&logo=flask)
![Leaflet](https://img.shields.io/badge/Leaflet-Maps-green?style=for-the-badge&logo=leaflet)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

## 📋 Descripción

**Emprende IA** es un sistema inteligente de geolocalización diseñado para emprendedores en Ensenada y alrededores. Utilizando algoritmos de análisis espacial, la aplicación escanea un radio de 500 metros alrededor de una ubicación seleccionada para detectar la densidad de servicios existentes (escuelas, hospitales, gimnasios, etc.).

Basándose en estos datos, el sistema **predice y recomienda** las mejores oportunidades de negocio para esa zona específica (ej. "Aquí hace falta una papelería"), ayudando a reducir el riesgo de inversión.

## 📱 Demo en Vivo

🌐 **Prueba la aplicación aquí:** [https://alejandrarodriguez.pythonanywhere.com/](https://alejandrarodriguez.pythonanywhere.com/)

O escanea el QR para ver la versión móvil optimizada:

<p align="center">
  <img src="qr_emprende_slogan.png" alt="QR del Proyecto" width="250"/>
</p>

*(Nota: Este QR te llevará directamente a la aplicación)*

## ✨ Características Principales

* **📍 Análisis Geoespacial en Tiempo Real:** Escaneo automático de un radio de 500m.
* **🧠 Motor de Recomendación (IA):** Algoritmo que identifica nichos de mercado desatendidos.
* **📱 Diseño 100% Responsive:** Interfaz adaptada a móviles con panel inferior deslizable (estilo Google Maps).
* **📊 Top 3 Oportunidades:** Ranking de probabilidad de éxito para diferentes tipos de negocios.
* **🗺️ Mapa Interactivo:** Visualización clara con iconos personalizados para cada tipo de establecimiento.

## 🛠️ Tecnologías Utilizadas

* **Frontend:** HTML5, CSS3 (Diseño Adaptativo Móvil/PC), JavaScript.
* **Mapas:** Leaflet.js + OpenStreetMap.
* **Backend:** Python (Flask).
* **Despliegue:** PythonAnywhere.
* **Librerías Python:** `Flask`, `flask-cors`, `numpy` (para cálculos de densidad), `qrcode`.

## 🚀 Instalación Local

Si deseas correr este proyecto en tu computadora para desarrollo:

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/AleRodriguezCruz/Emprende--IA.git](https://github.com/AleRodriguezCruz/Emprende--IA.git)
    cd Emprende--IA
    ```

2.  **Crear un entorno virtual (Opcional pero recomendado):**
    ```bash
    python -m venv venv
    # En Windows:
    venv\Scripts\activate
    # En Mac/Linux:
    source venv/bin/activate
    ```

3.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ejecutar la aplicación:**
    ```bash
    python app.py
    ```
    *Abre tu navegador en `http://127.0.0.1:5000`*

## 📸 Capturas de Pantalla

| Vista Móvil (Panel Deslizable) | Vista Escritorio (Panel Lateral) |
|:------------------------------:|:--------------------------------:|
| ![Movil](screenshot_movil.png) | ![PC](screenshot_pc.png)         |

*(Recuerda subir tus capturas de pantalla a la carpeta del proyecto con estos nombres)*

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la Licencia MIT.

---
⌨️ Desarrollado con ❤️ por [Alejandra Rodríguez](https://github.com/AleRodriguezCruz)