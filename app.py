import streamlit as st
import geopandas as gpd
import libpysal
from esda.moran import Moran, Moran_Local
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit.components.v1 as components

# Diccionario de descripciones educativas para cada dataset
DATASET_DESCRIPTIONS = {
    "Columbus Crime": {
        "nombre": "Criminalidad en Columbus (1980)",
        "desc": "Este dataset contiene datos sobre delincuencia y factores socioeconómicos en 49 vecindarios de Columbus, Ohio, compilados en 1980 por Anselin.",
        "moran_meaning": "Al analizar la tasa de delitos residenciales (**CRIME**), el Índice de Moran evalúa si la criminalidad tiende a formar 'focos' o 'zonas rojas' geográficamente agrupadas (autocorrelación positiva) o si los delitos ocurren de forma dispersa y aislada por la ciudad."
    },
    "Guerry Moral Statistics": {
        "nombre": "Estadísticas Morales de Guerry (Francia, 1830s)",
        "desc": "Presenta 85 departamentos franceses con datos recolectados por André-Michel Guerry en la década de 1830 sobre alfabetización, suicidios, criminalidad y donaciones.",
        "moran_meaning": "Si se analiza la alfabetización (**Litercy**), el Índice de Moran cuantifica y hace evidente la histórica brecha de educación norte-sur en Francia, demostrando si los departamentos con niveles similares de educación se concentran contiguos en el espacio."
    },
    "US Income": {
        "nombre": "Ingreso de los Estados Unidos (1929-2009)",
        "desc": "Recopila la evolución del ingreso personal per cápita anual en los 48 estados contiguos de EE.UU. a lo largo de 80 años.",
        "moran_meaning": "Al elegir un año (ej. **2008**), el Índice de Moran permite medir las disparidades económicas regionales (ej. el noreste industrializado frente al sur agrícola), verificando si la riqueza nacional se organiza en clústeres geográficos definidos o si es homogénea."
    },
    "SIDS North Carolina": {
        "nombre": "SIDS en Carolina del Norte (1974-1984)",
        "desc": "Un dataset de epidemiología espacial clásico con la incidencia del Síndrome de Muerte Súbita del Lactante (SIDS) en 100 condados de Carolina del Norte.",
        "moran_meaning": "Calcular Moran sobre la tasa de muertes por SIDS (ej. **SID79**) ayuda a los epidemiólogos a determinar si las muertes infantiles están agrupadas espacialmente (sugiriendo la presencia de riesgos ambientales, genéticos o de acceso a salud comunes entre condados colindantes) o si son aleatorias."
    },
    "Mexico Demographics": {
        "nombre": "PIB y Desigualdad en México (1940-2000)",
        "desc": "Estadísticas de desarrollo y Producto Interno Bruto (PIB) per cápita de los 32 estados de la República Mexicana.",
        "moran_meaning": "Analizar la variable del PIB (ej. **GD1990**) evidencia el nivel de polarización económica territorial de México, midiendo si los polos de desarrollo industrial (norte/centro) se extienden a estados vecinos o si quedan aislados en un entorno de bajos ingresos."
    },
    "Boston Housing": {
        "nombre": "Mercado de Vivienda en Boston",
        "desc": "Contiene datos de 506 áreas censales sobre el valor de la propiedad, tasas de criminalidad y calidad del aire en la zona metropolitana de Boston.",
        "moran_meaning": "Si analizas el valor medio de las viviendas (**MEDV**), el Índice de Moran hace visible la segregación residencial del mercado inmobiliario (áreas caras agrupadas juntas). Si analizas la concentración de óxidos de nitrógeno (**NOX**), cuantifica la concentración espacial del esmog e industria."
    }
}

# Configuración de la página
st.set_page_config(page_title="Análisis Espacial: Índice de Moran", layout="wide")

st.title("📊 Análisis de Autocorrelación Espacial: Índice de Moran")
st.markdown("""
Esta aplicación interactiva permite evidenciar el **Índice de Moran** (Global y Local) en múltiples conjuntos de datos espaciales reales.
La autocorrelación espacial mide hasta qué punto los valores cercanos en el espacio tienden a ser similares (autocorrelación positiva) o disímiles (autocorrelación negativa).
""")

# --- Funciones de Carga de Datos ---
@st.cache_data
def load_pysal_data(dataset_name):
    """
    Carga y procesa datasets clásicos integrados en libpysal.
    """
    if dataset_name == "Columbus Crime":
        libpysal.examples.load_example("columbus")
        path = libpysal.examples.get_path('columbus.shp')
        return gpd.read_file(path)
        
    elif dataset_name == "Guerry Moral Statistics":
        libpysal.examples.load_example("Guerry")
        path = libpysal.examples.get_path('guerry.shp')
        return gpd.read_file(path)
        
    elif dataset_name == "US Income":
        libpysal.examples.load_example("us_income")
        path = libpysal.examples.get_path('us48.shp')
        gdf = gpd.read_file(path)
        csv = pd.read_csv(libpysal.examples.get_path('usjoin.csv'))
        # Unir por nombre del estado
        gdf = gdf.merge(csv, left_on='STATE_NAME', right_on='Name')
        return gdf
        
    elif dataset_name == "SIDS North Carolina":
        libpysal.examples.load_example("sids2")
        path = libpysal.examples.get_path('sids2.shp')
        return gpd.read_file(path)
        
    elif dataset_name == "Mexico Demographics":
        libpysal.examples.load_example("mexico")
        path = libpysal.examples.get_path('mexicojoin.shp')
        return gpd.read_file(path)
        
    elif dataset_name == "Boston Housing":
        libpysal.examples.load_example("Bostonhsg")
        path = libpysal.examples.get_path('boston.shp')
        return gpd.read_file(path)
        
    return None

@st.cache_data
def load_custom_url(url):
    """
    Intenta cargar un dataset desde una URL pública (ej. GeoJSON).
    """
    try:
        gdf = gpd.read_file(url)
        return gdf
    except Exception as e:
        st.error(f"Error al cargar la URL: {e}")
        return None

# --- Barra Lateral (Sidebar) ---
st.sidebar.header("🛠️ Configuración de Datos")

dataset_type = st.sidebar.radio(
    "Selecciona el origen del dataset:",
    ("Datasets de PySAL (Integrados)", "Cargar desde URL (Internet)")
)

gdf = None

if dataset_type == "Datasets de PySAL (Integrados)":
    dataset_name = st.sidebar.selectbox(
        "Elige un dataset espacial real:",
        (
            "Columbus Crime", 
            "Guerry Moral Statistics", 
            "US Income", 
            "SIDS North Carolina", 
            "Mexico Demographics", 
            "Boston Housing"
        )
    )
    gdf = load_pysal_data(dataset_name)
    st.sidebar.info(f"Cargado dataset: {dataset_name} con {len(gdf)} regiones.")
else:
    url_input = st.sidebar.text_input(
        "Ingresa la URL del archivo (GeoJSON/Shapefile):", 
        value="https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/mexico.geojson"
    )
    if url_input:
        gdf = load_custom_url(url_input)
        if gdf is not None:
            st.sidebar.success(f"Cargado dataset externo con {len(gdf)} regiones.")

# Validar que los datos tengan geometría válida
if gdf is not None:
    # Asegurarnos de que no hay geometrías vacías o nulas que rompan el cálculo de pesos
    gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid]
    
    st.sidebar.header("🎛️ Variables y Pesos")
    
    # Seleccionar solo columnas numéricas para el análisis
    numeric_cols = gdf.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filtrar columnas que no sean útiles (ej. IDs automáticos)
    numeric_cols = [c for c in numeric_cols if c.lower() not in ['id', 'objectid', 'index', 'poly_id', 'state_fips', 'state_id', 'state_', 'townno', 'tract']]
    
    if not numeric_cols:
        st.sidebar.error("El dataset cargado no contiene columnas numéricas aptas para el análisis.")
    else:
        # Preseleccionar una variable interesante según el dataset
        default_index = 0
        if dataset_type == "Datasets de PySAL (Integrados)":
            if dataset_name == "Columbus Crime" and 'CRIME' in numeric_cols:
                default_index = numeric_cols.index('CRIME')
            elif dataset_name == "Guerry Moral Statistics" and 'Litercy' in numeric_cols:
                default_index = numeric_cols.index('Litercy')
            elif dataset_name == "US Income" and '2008' in numeric_cols:
                default_index = numeric_cols.index('2008')
            elif dataset_name == "SIDS North Carolina" and 'SID79' in numeric_cols:
                default_index = numeric_cols.index('SID79')
            elif dataset_name == "Mexico Demographics" and 'GD1990' in numeric_cols:
                default_index = numeric_cols.index('GD1990')
            elif dataset_name == "Boston Housing" and 'MEDV' in numeric_cols:
                default_index = numeric_cols.index('MEDV')
                
        variable = st.sidebar.selectbox("Selecciona la variable a analizar:", numeric_cols, index=default_index)
        
        weight_type = st.sidebar.selectbox(
            "Tipo de Contigüidad (Pesos Espaciales):",
            ("Queen (Reina - Aristas y Vértices compartidos)", "Rook (Torre - Solo Aristas compartidas)")
        )

        st.markdown("---")
        
        # Mostrar información contextual del dataset y el significado del Índice de Moran
        if dataset_type == "Datasets de PySAL (Integrados)":
            info_data = DATASET_DESCRIPTIONS[dataset_name]
            st.info(f"💡 **Contexto del Dataset:** *{info_data['nombre']}*\n\n"
                    f"📄 **Descripción:** {info_data['desc']}\n\n"
                    f"🔍 **¿Qué mide el Índice de Moran en este caso?:** {info_data['moran_meaning']}")
        else:
            st.info(f"💡 **Contexto del Dataset:** *Dataset Personalizado desde URL*\n\n"
                    f"🔍 **¿Qué mide el Índice de Moran en este caso?:** El análisis sobre la variable **{variable}** cuantificará si las regiones con valores similares tienden a agruparse en el espacio (autocorrelación positiva), dispersarse (autocorrelación negativa) o si su distribución es puramente aleatoria.")

        st.markdown("---")
        
        # Eliminar NaNs en la variable para evitar errores en PySAL
        gdf_clean = gdf.dropna(subset=[variable]).copy()
        
        # Intentar buscar una columna de nombre para usar en hover de mapas y gráficos
        hover_col = None
        for col in ['STATE_NAME', 'STATE', 'Department', 'dept', 'TOWN', 'CNTY_ID', 'POLYID', 'name', 'NOM_ENT']:
            if col in gdf_clean.columns:
                hover_col = col
                break

        # Renderizar la información del dataset
        st.subheader("🌐 Paso 1: Mapa Interactivo de la Variable")
        st.write(f"Explora la distribución espacial de **{variable}**. Pasa el cursor sobre el mapa para ver los detalles de cada región.")
        
        # Crear Mapa Interactivo usando explore()
        with st.spinner("Generando mapa interactivo..."):
            m_orig = gdf_clean.explore(
                column=variable, 
                cmap='viridis', 
                scheme='quantiles', 
                k=5, 
                legend=True,
                tooltip=[hover_col, variable] if hover_col else [variable],
                popup=True,
                style_kwds=dict(fillOpacity=0.75, weight=0.5, color="white")
            )
            # Renderizar en streamlit usando un iframe de folium para no recargar la app al hacer zoom/pan
            components.html(m_orig._repr_html_(), height=500)

        st.markdown("---")
        st.subheader("🧮 Paso 2: Cálculo del Índice de Moran Global")
        st.markdown("""
        El **Índice de Moran Global** evalúa la autocorrelación espacial en todo el conjunto de datos:
        *   Valores cercanos a **+1**: Agrupamiento (clústeres de valores similares).
        *   Valores cercanos a **-1**: Dispersión (vecinos muy disímiles).
        *   Valores cercanos a **0**: Patrón espacial aleatorio.
        """)
        
        # Cálculo de Pesos Espaciales
        with st.spinner("Calculando matriz de pesos espaciales..."):
            if weight_type.startswith("Queen"):
                w = libpysal.weights.Queen.from_dataframe(gdf_clean)
            else:
                w = libpysal.weights.Rook.from_dataframe(gdf_clean)
            
            # Estandarización por filas (esencial para Moran)
            w.transform = 'r'
            
        # Cálculo del Índice
        with st.spinner("Calculando el Índice de Moran..."):
            y = gdf_clean[variable]
            moran = Moran(y, w)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Moran's I", f"{moran.I:.4f}")
            col2.metric("P-Valor (Simulación)", f"{moran.p_sim:.4f}")
            col3.metric("Z-Score", f"{moran.z_sim:.4f}")
            
            if moran.p_sim < 0.05:
                if moran.I > 0:
                    st.success(f"Hay evidencia estadísticamente significativa (p < 0.05) de **autocorrelación espacial positiva** (Clustering).")
                else:
                    st.info(f"Hay evidencia estadísticamente significativa (p < 0.05) de **autocorrelación espacial negativa** (Dispersión).")
            else:
                st.warning("No hay evidencia estadística suficiente (p >= 0.05) para rechazar la hipótesis nula de aleatoriedad. El patrón es aleatorio.")

        # Moran Scatterplot usando Plotly Express
        st.markdown("#### 📈 Diagrama de Dispersión de Moran (Interactivo)")
        st.write("""
        Compara la variable estandarizada frente a su **rezago espacial** (el promedio de sus vecinos). 
        *   **Cuadrante I (HH)**: Valores altos rodeados de altos.
        *   **Cuadrante II (LH)**: Valores bajos rodeados de altos.
        *   **Cuadrante III (LL)**: Valores bajos rodeados de bajos.
        *   **Cuadrante IV (HL)**: Valores altos rodeados de bajos.
        """)
        
        # Calcular rezago espacial y estandarizar
        lag_y = libpysal.weights.lag_spatial(w, y)
        y_std = (y - y.mean()) / y.std()
        lag_y_std = (lag_y - lag_y.mean()) / lag_y.std()

        # Añadir al df para el hover de Plotly
        gdf_clean['y_std'] = y_std
        gdf_clean['lag_y_std'] = lag_y_std

        # Generar gráfico Plotly
        fig_scatter = px.scatter(
            gdf_clean,
            x='y_std',
            y='lag_y_std',
            hover_name=hover_col if hover_col else gdf_clean.index,
            hover_data={variable: True, 'y_std': ':.2f', 'lag_y_std': ':.2f'},
            labels={
                'y_std': f'{variable} (Estandarizado)',
                'lag_y_std': 'Rezago Espacial de Vecinos (Estandarizado)'
            },
            opacity=0.75
        )
        
        # Agregar líneas de cuadrantes
        fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Agregar línea de regresión (cuya pendiente es el Índice de Moran)
        m_slope, b_intercept = np.polyfit(y_std, lag_y_std, 1)
        x_trend = np.array([y_std.min(), y_std.max()])
        fig_scatter.add_trace(go.Scatter(
            x=x_trend,
            y=m_slope * x_trend + b_intercept,
            mode='lines',
            name=f"Pendiente (Moran's I = {moran.I:.4f})",
            line=dict(color='red', width=2)
        ))
        
        fig_scatter.update_layout(
            height=500,
            xaxis_title=f"{variable} (Estandarizado)",
            yaxis_title="Rezago Espacial (Vecinos Estandarizados)",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("---")
        st.subheader("📍 Paso 3: Índice de Moran Local (LISA)")
        st.markdown("""
        El Indicador Local de Asociación Espacial (LISA) nos permite ubicar en qué partes del mapa se encuentran los clústeres calientes y fríos de manera estadísticamente significativa.
        *   🔴 **High-High (HH)**: Zonas de valores altos rodeados de vecinos altos (Clúster caliente).
        *   🔵 **Low-Low (LL)**: Zonas de valores bajos rodeados de vecinos bajos (Clúster frío).
        *   lightblue **Low-High (LH)**: Valor bajo rodeado de valores altos (Outlier).
        *   lightred **High-Low (HL)**: Valor alto rodeado de valores bajos (Outlier).
        *   ⚪ **No Significativo**: Zonas sin patrones de agrupamiento espacial sistemático (p >= 0.05).
        """)
        
        # Cálculo de Moran Local (LISA)
        with st.spinner("Calculando autocorrelación local..."):
            moran_loc = Moran_Local(y, w)
            
            # Clasificación de categorías
            sig_threshold = 0.05
            categories = ['No Significativo', 'High-High (HH)', 'Low-High (LH)', 'Low-Low (LL)', 'High-Low (HL)']
            
            labels = [categories[0]] * len(moran_loc.q)
            for i in range(len(moran_loc.q)):
                if moran_loc.p_sim[i] < sig_threshold:
                    # moran_loc.q toma valores {1, 2, 3, 4} correspondientes a HH, LH, LL, HL respectivamente
                    labels[i] = categories[moran_loc.q[i]]
            
            gdf_clean['LISA_Cluster'] = labels
            
            # Truco para que el mapa muestre siempre los 5 colores en la leyenda sin desfasarse
            # Añadimos filas dummy (con geometría nula) para categorías ausentes
            dummies = []
            for cat in categories:
                if cat not in gdf_clean['LISA_Cluster'].values:
                    dummies.append({
                        'geometry': None,
                        'LISA_Cluster': cat,
                        hover_col: 'Dummy',
                        variable: 0
                    })
            if dummies:
                gdf_lisa = pd.concat([gdf_clean, gpd.GeoDataFrame(dummies)], ignore_index=True)
            else:
                gdf_lisa = gdf_clean.copy()
            
            # Forzar Categorical con orden específico
            gdf_lisa['LISA_Cluster'] = pd.Categorical(gdf_lisa['LISA_Cluster'], categories=categories, ordered=True)
            # Ordenar el df para que la leyenda y mapeo de colores sea consistente
            gdf_lisa = gdf_lisa.sort_values('LISA_Cluster')
            
            # Colores asignados a cada categoría en el orden de las categorías
            # Gris claro, Rojo, Celeste, Azul, Rojo claro
            lisa_colors = ['#eeeeee', '#d9534f', '#85c1e9', '#0275d8', '#f1948a']
            
            m_lisa = gdf_lisa.explore(
                column='LISA_Cluster',
                cmap=lisa_colors,
                tooltip=[hover_col, 'LISA_Cluster'] if hover_col else ['LISA_Cluster'],
                popup=True,
                legend=True,
                style_kwds=dict(fillOpacity=0.8, weight=0.5, color="white")
            )
            components.html(m_lisa._repr_html_(), height=500)
