# VIXπ-Fusion · Dashboard App

## Instalación local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue en Streamlit Cloud (gratis)

1. Sube esta carpeta a un repositorio de GitHub
2. Ve a https://share.streamlit.io
3. Conecta tu cuenta de GitHub
4. Selecciona el repositorio y el archivo app.py
5. ¡Listo! Obtienes una URL pública accesible desde cualquier dispositivo

## Despliegue en Railway / Render (gratis)

```bash
# Procfile
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

## Uso

1. Sube tu archivo Excel de cotizaciones 1min (columnas: datetime, open, high, low, close)
2. Ajusta los parámetros en el sidebar (SL, TP, contratos, filtro)
3. Explora las 4 pestañas:
   - 📊 Resumen: métricas generales y curva de equity
   - 🗓️ Tabla Día×Hora: combinaciones óptimas TP/SL
   - 📈 Backtest: análisis detallado por hora y operación
   - 🔔 Señales: últimos cruces y estado actual del indicador
