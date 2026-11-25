from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kmeans_core import (
    ETIQUETAS,
    FEATURES,
    construir_perfiles,
    entrenar,
    escalar,
    evaluar_k,
    generar_datos,
)

st.set_page_config(
    page_title="Segmentación de Flotillas | K-Means",
    layout="wide",
    initial_sidebar_state="auto",
)

css = (Path(__file__).parent / "assets" / "styles.css").read_text(encoding="utf-8")
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

AZUL = "#0284C7"
AZUL_CLARO = "#BAE6FD"
AMBAR = "#F59E0B"
INDIGO = "#6366F1"
GRIS = "#64748B"

PASOS = [
    "01 · Exploración de telemetría vehicular",
    "02 · Escalado de variables",
    "03 · Número óptimo de grupos",
    "04 · Entrenamiento del modelo",
    "05 · Perfilamiento logístico",
    "06 · Visualización y descarga final",
]

CONFIG_GRAFICA = {"displayModeBar": False, "displaylogo": False}
CONFIG_FINAL = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}


@st.cache_data(show_spinner=False)
def cargar_telemetria_flotilla(semilla: float) -> pd.DataFrame:
    return generar_datos(semilla)


def step_head(numero: int, titulo: str, descripcion: str) -> None:
    st.markdown(
        f'<div class="step-head"><span class="step-num">PASO {numero:02d}</span>'
        f"<div><h2>{titulo}</h2><p>{descripcion}</p></div></div>",
        unsafe_allow_html=True,
    )


def callout(tipo: str, etiqueta: str, texto: str) -> None:
    st.markdown(
        f'<div class="callout {tipo}">'
        f'<strong class="callout-label">{etiqueta}</strong>'
        f"<p>{texto}</p></div>",
        unsafe_allow_html=True,
    )


def panel_head(titulo: str, subtitulo: str) -> None:
    st.markdown(
        f'<div class="panel-head"><h3>{titulo}</h3><p>{subtitulo}</p></div>',
        unsafe_allow_html=True,
    )


def estilo_fig(fig: go.Figure, altura: int = 430, leyenda: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=altura,
        margin=dict(l=10, r=10, t=14, b=10),
        font=dict(family="Fira Sans, sans-serif", size=13, color="#334155"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor="#0B1220",
            bordercolor="#0B1220",
            font=dict(family="Fira Sans", size=12, color="#F8FAFC"),
        ),
        showlegend=leyenda,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            font=dict(size=12, color="#475569"),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(
        gridcolor="#E2E8F0",
        zeroline=False,
        linecolor="#CBD5E1",
        tickfont=dict(size=12, color="#64748B"),
        title_font=dict(size=12, color="#475569"),
    )
    fig.update_yaxes(
        gridcolor="#E2E8F0",
        zeroline=False,
        linecolor="#CBD5E1",
        tickfont=dict(size=12, color="#64748B"),
        title_font=dict(size=12, color="#475569"),
    )
    return fig


def scatter_vehiculos(
    x: np.ndarray,
    y: np.ndarray,
    vehiculos: np.ndarray,
    x_titulo: str,
    y_titulo: str,
) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=9,
                color=AZUL,
                opacity=0.60,
                line=dict(width=1, color="#FFFFFF"),
            ),
            customdata=vehiculos,
            hovertemplate=(
                "<b>Vehículo %{customdata[0]}</b><br>"
                + x_titulo
                + ": %{x:.1f}<br>"
                + y_titulo
                + ": %{y:.2f}<extra></extra>"
            ),
            name="Unidades de Transporte",
        )
    )
    fig.update_xaxes(title_text=x_titulo)
    fig.update_yaxes(title_text=y_titulo)
    return estilo_fig(fig, altura=460)


def csv_flotillas(asignaciones: pd.DataFrame) -> bytes:
    export = asignaciones.drop(columns=["color", "categoria"], errors="ignore")
    export = export.rename(
        columns={
            "vehiculo_id": "Vehículo",
            "km_recorridos_dia": "Distancia diaria (km)",
            "horas_ralenti_dia": "Tiempo en ralentí (horas)",
            "cluster": "Clúster",
            "perfil": "Perfil Logístico",
        }
    )
    return export.to_csv(index=False).encode("utf-8-sig")


st.markdown(
    """
<div class="hero">
  <div class="badge-row">
    <span class="badge">Unidad 5 · Subtema 5.1</span>
    <span class="badge accent">K-Means Clustering</span>
    <span class="badge">Telemetría IoT & Logística</span>
  </div>
  <h1>Segmentación Inteligente de Flotillas de Transporte con K-Means</h1>
  <p>
    Análisis no supervisado de patrones de movilidad y uso de motor (distancia diaria vs. horas en ralentí)
    para la optimización operativa, reducción de huella de carbono y mantenimiento preventivo de unidades.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
<div class="side-kicker">Panel de Control</div>
<div class="side-title">Gestión de Flota</div>
<div class="side-copy">
Recorre cada fase del pipeline de agrupamiento no supervisado aplicado a telemetría vehicular.
</div>
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown("")

semilla_int = st.sidebar.number_input(
    "Semilla de simulación (DuckDB)",
    min_value=1,
    max_value=999,
    value=42,
    step=1,
    help="Controla la generación determinista de los 150 registros de telemetría vehicular.",
)
semilla = float(semilla_int) / 100.0

st.sidebar.caption(f"Semilla activa: {semilla:.2f} · 150 vehículos monitoreados")

paso = st.sidebar.radio("Ruta de abordaje", PASOS, index=0)
indice_paso = PASOS.index(paso)
st.sidebar.progress(
    (indice_paso + 1) / len(PASOS),
    text=f"Paso {indice_paso + 1} de {len(PASOS)}",
)

st.sidebar.markdown(
    """
<div class="side-meta">
  <strong>Materia:</strong> Análisis de Datos<br>
  <strong>Técnica:</strong> K-Means · StandardScaler<br>
  <strong>Infraestructura:</strong> DuckDB in-memory & Plotly
</div>
""",
    unsafe_allow_html=True,
)

df = cargar_telemetria_flotilla(semilla)
if not all(col in df.columns for col in FEATURES):
    st.cache_data.clear()
    df = generar_datos(semilla)
X, scaler = escalar(df)
resultados_k = evaluar_k(X)

# ==============================================================================
# PASO 1 · EXPLORACIÓN DE LA TELEMETRÍA
# ==============================================================================
if paso == PASOS[0]:
    step_head(
        1,
        "Exploración de la telemetría vehicular",
        "Inspección preliminar del comportamiento operacional de 150 unidades de la flotilla.",
    )

    col_izq, col_der = st.columns([1.7, 1.3], gap="large")

    with col_izq:
        panel_head(
            "Distribución en el plano de operación",
            "Relación entre kilómetros recorridos al día y horas con el motor encendido en ralentí.",
        )
        fig = scatter_vehiculos(
            df[FEATURES[0]].to_numpy(),
            df[FEATURES[1]].to_numpy(),
            df[["vehiculo_id"]].to_numpy(),
            ETIQUETAS[FEATURES[0]],
            ETIQUETAS[FEATURES[1]],
        )
        st.plotly_chart(fig, theme=None, config=CONFIG_GRAFICA)

    with col_der:
        panel_head("Métricas clave de la flota", "Resumen estadístico de las 150 unidades monitoreadas.")
        km_medio = df[FEATURES[0]].mean()
        ralenti_medio = df[FEATURES[1]].mean()
        km_max = df[FEATURES[0]].max()
        ralenti_max = df[FEATURES[1]].max()

        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Distancia media", f"{km_medio:.1f} km", help="Promedio diario por unidad")
        col_m2.metric("Ralentí medio", f"{ralenti_medio:.2f} hrs", help="Tiempo medio con motor encendido detenido")

        col_m3, col_m4 = st.columns(2)
        col_m3.metric("Rango distancia", f"{df[FEATURES[0]].min():.0f} - {km_max:.0f} km")
        col_m4.metric("Rango ralentí", f"{df[FEATURES[1]].min():.1f} - {ralenti_max:.1f} hrs")

        callout(
            "info",
            "¿Por qué es crítico medir el ralentí?",
            "El ralentí excesivo (motor encendido sin desplazamiento) representa un desperdicio del 20% al 35% de combustible, "
            "acelera la acumulación de carbonilla en inyectores y eleva las emisiones de CO₂ sin generar entregas efectivas.",
        )

        callout(
            "insight",
            "Patrones visuales detectados",
            "Se observan tres concentraciones naturales: unidades con bajo kilometraje y bajo ralentí (reparto ágil), "
            "unidades de alto kilometraje (rutas foráneas) y unidades con kilometraje moderado pero ralentí severo (atascos urbanos).",
        )

    with st.expander("Ver muestra de telemetría tabular (primeros 10 registros)"):
        st.dataframe(
            df.head(10).rename(
                columns={
                    "vehiculo_id": "Vehículo ID",
                    "km_recorridos_dia": "Distancia diaria (km)",
                    "horas_ralenti_dia": "Tiempo ralentí (hrs)",
                }
            ),
            use_container_width=True,
        )

# ==============================================================================
# PASO 2 · ESCALADO DE VARIABLES
# ==============================================================================
elif paso == PASOS[1]:
    step_head(
        2,
        "Escalado y normalización de variables",
        "Estandarización z-score para equilibrar el peso matemático de ambas magnitudes operativas.",
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        panel_head("Magnitudes originales", "Distancia en kilómetros (50 - 350) vs Ralentí en horas (0.8 - 6.0).")
        fig_orig = scatter_vehiculos(
            df[FEATURES[0]].to_numpy(),
            df[FEATURES[1]].to_numpy(),
            df[["vehiculo_id"]].to_numpy(),
            ETIQUETAS[FEATURES[0]],
            ETIQUETAS[FEATURES[1]],
        )
        st.plotly_chart(fig_orig, theme=None, config=CONFIG_GRAFICA)

    with col2:
        panel_head("Espacio estandarizado (Z)", "Variables con media μ = 0 y desviación estándar σ = 1.")
        fig_z = go.Figure(
            go.Scatter(
                x=X[:, 0],
                y=X[:, 1],
                mode="markers",
                marker=dict(
                    size=9,
                    color=INDIGO,
                    opacity=0.60,
                    line=dict(width=1, color="#FFFFFF"),
                ),
                customdata=df[["vehiculo_id"]].to_numpy(),
                hovertemplate=(
                    "<b>Vehículo %{customdata[0]}</b><br>"
                    "Z-Distancia: %{x:.2f}<br>"
                    "Z-Ralentí: %{y:.2f}<extra></extra>"
                ),
                name="Unidades (Z)",
            )
        )
        fig_z.update_xaxes(title_text="Distancia estandarizada (Z)")
        fig_z.update_yaxes(title_text="Ralentí estandarizado (Z)")
        estilo_fig(fig_z, altura=460)
        st.plotly_chart(fig_z, theme=None, config=CONFIG_GRAFICA)

    callout(
        "warn",
        "Riesgo de omitir el escalado en algoritmos basados en distancia",
        "K-Means calcula distancias euclidianas: d = √[(Δkm)² + (Δhoras)²]. Como la variación en kilómetros "
        "(amplitud de ~300 km) es 50 veces superior a la variación de horas de ralentí (amplitud de ~5 hrs), "
        "sin estandarización el algoritmo ignoraría casi por completo el tiempo de motor detenido.",
    )

# ==============================================================================
# PASO 3 · NÚMERO ÓPTIMO DE GRUPOS
# ==============================================================================
elif paso == PASOS[2]:
    step_head(
        3,
        "Determinación del número óptimo de grupos (k)",
        "Evaluación combinada del Método del Codo (Inercia) y Coeficiente de Silueta.",
    )

    col_izq, col_der = st.columns(2, gap="large")

    with col_izq:
        panel_head("Método del Codo (Inercia)", "Suma de distancias al cuadrado de cada vehículo a su centroide.")
        fig_codo = go.Figure()
        fig_codo.add_trace(
            go.Scatter(
                x=resultados_k["k"],
                y=resultados_k["inercia"],
                mode="lines+markers",
                line=dict(color=AZUL, width=3),
                marker=dict(size=10, color=AZUL),
                name="Inercia",
            )
        )
        fig_codo.update_xaxes(title_text="Número de clústers (k)", dtick=1)
        fig_codo.update_yaxes(title_text="Inercia intraclúster")
        estilo_fig(fig_codo, altura=400, leyenda=False)
        st.plotly_chart(fig_codo, theme=None, config=CONFIG_GRAFICA)

    with col_der:
        panel_head("Coeficiente de Silueta", "Métrica de cohesión interna vs separación respecto a otros grupos.")
        colores_sil = [
            AMBAR if k == 3 else AZUL_CLARO for k in resultados_k["k"]
        ]
        fig_sil = go.Figure(
            go.Bar(
                x=resultados_k["k"],
                y=resultados_k["silueta"],
                marker_color=colores_sil,
                text=resultados_k["silueta"].apply(lambda v: f"{v:.3f}"),
                textposition="outside",
            )
        )
        fig_sil.update_xaxes(title_text="Número de clústers (k)", dtick=1)
        fig_sil.update_yaxes(title_text="Puntaje de silueta (máx = 1)", range=[0, 0.9])
        estilo_fig(fig_sil, altura=400, leyenda=False)
        st.plotly_chart(fig_sil, theme=None, config=CONFIG_GRAFICA)

    k_max_sil = int(resultados_k.loc[resultados_k["silueta"].idxmax(), "k"])
    sil_max = float(resultados_k["silueta"].max())

    callout(
        "ok",
        f"Diagnóstico cuantitativo: k = {k_max_sil} es la solución matemáticamente óptima",
        f"El valor k = {k_max_sil} presenta el quiebre de codo más pronunciado y alcanza una silueta sobresaliente "
        f"de {sil_max:.3f}, garantizando grupos compactos, homogéneos y con límites nítidos.",
    )

# ==============================================================================
# PASO 4 · ENTRENAMIENTO DEL MODELO
# ==============================================================================
elif paso == PASOS[3]:
    step_head(
        4,
        "Entrenamiento y localización de centroides",
        "Ajuste iterativo de K-Means y convergencia de las posiciones centrales de cada grupo.",
    )

    k_seleccionado = st.slider(
        "Selecciona el número de clústers para entrenar (k):",
        min_value=2,
        max_value=6,
        value=3,
        help="Recomendado: k = 3 por máxima silueta y coherencia operativa.",
    )

    modelo = entrenar(X, k=k_seleccionado)
    centroides_z = modelo.cluster_centers_
    centroides_km_hrs = scaler.inverse_transform(centroides_z)

    col1, col2 = st.columns([1.6, 1.4], gap="large")

    with col1:
        panel_head(
            f"Agrupamiento con k = {k_seleccionado}",
            "Visualización de unidades y ubicación de los centroides calculados.",
        )
        fig_ent = go.Figure()

        # Scatter de puntos
        fig_ent.add_trace(
            go.Scatter(
                x=df[FEATURES[0]],
                y=df[FEATURES[1]],
                mode="markers",
                marker=dict(
                    size=8,
                    color=modelo.labels_,
                    colorscale="Viridis",
                    opacity=0.65,
                    line=dict(width=1, color="#FFFFFF"),
                ),
                customdata=df[["vehiculo_id"]].to_numpy(),
                hovertemplate="<b>Vehículo %{customdata[0]}</b><br>Distancia: %{x:.1f} km<br>Ralentí: %{y:.2f} hrs<extra></extra>",
                name="Vehículos",
            )
        )

        # Centroides
        fig_ent.add_trace(
            go.Scatter(
                x=centroides_km_hrs[:, 0],
                y=centroides_km_hrs[:, 1],
                mode="markers",
                marker=dict(
                    symbol="star",
                    size=18,
                    color="#EF4444",
                    line=dict(width=1.5, color="#FFFFFF"),
                ),
                name="Centroides",
                hovertemplate="<b>Centroide Clúster</b><br>Distancia: %{x:.1f} km<br>Ralentí: %{y:.2f} hrs<extra></extra>",
            )
        )

        fig_ent.update_xaxes(title_text=ETIQUETAS[FEATURES[0]])
        fig_ent.update_yaxes(title_text=ETIQUETAS[FEATURES[1]])
        estilo_fig(fig_ent, altura=480)
        st.plotly_chart(fig_ent, theme=None, config=CONFIG_GRAFICA)

    with col2:
        panel_head("Coordenadas de los Centroides", "Valores en escala física real de operación.")
        df_cent = pd.DataFrame(
            centroides_km_hrs,
            columns=["Distancia media (km)", "Ralentí medio (hrs)"],
        )
        df_cent.index.name = "Clúster"
        st.dataframe(df_cent.round(2), use_container_width=True)

        callout(
            "insight",
            "Convergencia del algoritmo",
            f"El algoritmo convergió en {modelo.n_iter_} iteraciones con una inercia residual de "
            f"{modelo.inertia_:.1f}. Cada centroide sintetiza el centro de gravedad logístico de su clúster.",
        )

# ==============================================================================
# PASO 5 · PERFILAMIENTO LOGÍSTICO
# ==============================================================================
elif paso == PASOS[4]:
    step_head(
        5,
        "Perfilamiento logístico y toma de decisiones",
        "Traducción de clústers matemáticos a estrategias ejecutivas de flota y mantenimiento.",
    )

    k_perfil = st.selectbox(
        "Número de clústers a perfilar:",
        options=[2, 3, 4, 5, 6],
        index=1,  # k = 3
    )

    modelo = entrenar(X, k=k_perfil)
    asignaciones, perfiles = construir_perfiles(df, modelo, scaler)

    panel_head(
        "Perfiles Logísticos Identificados",
        "Clasificación automática basada en z-scores y asignación de políticas operativas.",
    )

    cols = st.columns(len(perfiles))
    for idx, (_, row) in enumerate(perfiles.iterrows()):
        with cols[idx]:
            st.markdown(
                f"""
                <div class="profile-card" style="border-top: 4px solid {row['color']};">
                  <div class="profile-top">
                    <span class="profile-dot" style="background: {row['color']};"></span>
                    <span class="profile-cluster">CLÚSTER {row['cluster']}</span>
                    <span class="profile-share">{row['porcentaje']:.0f}% flota</span>
                  </div>
                  <h3>{row['nombre']}</h3>
                  <div class="profile-metrics">
                    <div>
                      <span>UNIDADES</span>
                      <strong>{int(row['n'])}</strong>
                    </div>
                    <div>
                      <span>DISTANCIA</span>
                      <strong>{row['media_km']:.0f} km</strong>
                    </div>
                    <div>
                      <span>RALENTÍ</span>
                      <strong>{row['media_ralenti']:.1f} hrs</strong>
                    </div>
                  </div>
                  <div class="profile-decision">
                    <b>DECISIÓN ESTRATÉGICA:</b>
                    {row['decision']}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ==============================================================================
# PASO 6 · VISUALIZACIÓN FINAL Y DESCARGA
# ==============================================================================
elif paso == PASOS[5]:
    step_head(
        6,
        "Visualización final y cierre ejecutivo",
        "Mapa integral de la flota con centroides, filtros interactivos y descarga de resultados.",
    )

    k_elegido = st.slider("Clústers activos (k):", 2, 6, 3)
    modelo = entrenar(X, k=k_elegido)
    asignaciones, perfiles = construir_perfiles(df, modelo, scaler)

    col_toggle, col_dummy = st.columns([1, 2])
    with col_toggle:
        escala_z = st.toggle(
            "Ver en escala estandarizada (Z)",
            value=False,
            help="Alterna entre magnitudes reales (km / hrs) y coordenadas z usadas por K-Means.",
        )

    if escala_z:
        coordenadas = X
        centroides = modelo.cluster_centers_
        titulo_x = "Distancia estandarizada (Z)"
        titulo_y = "Ralentí estandarizado (Z)"
        formato = ".2f"
    else:
        coordenadas = df[FEATURES].to_numpy()
        centroides = scaler.inverse_transform(modelo.cluster_centers_)
        titulo_x = ETIQUETAS[FEATURES[0]]
        titulo_y = ETIQUETAS[FEATURES[1]]
        formato = ".1f"

    fig = go.Figure()
    for _, perfil in perfiles.iterrows():
        mascara = asignaciones["cluster"] == perfil["cluster"]
        fig.add_trace(
            go.Scatter(
                x=coordenadas[mascara, 0],
                y=coordenadas[mascara, 1],
                mode="markers",
                name=perfil["nombre"],
                marker=dict(
                    size=9,
                    color=perfil["color"],
                    opacity=0.82,
                    line=dict(width=1, color="#FFFFFF"),
                ),
                customdata=asignaciones.loc[mascara, ["vehiculo_id"]].to_numpy(),
                hovertemplate=(
                    f"<b>Vehículo %{{customdata[0]}}</b><br>"
                    f"{titulo_x}: %{{x:{formato}}}<br>"
                    f"{titulo_y}: %{{y:{formato}}}<extra></extra>"
                ),
            )
        )

    for _, perfil in perfiles.iterrows():
        centro = centroides[int(perfil["cluster"])]
        fig.add_trace(
            go.Scatter(
                x=[centro[0]],
                y=[centro[1]],
                mode="markers",
                name=perfil["nombre"],
                legendgroup=perfil["nombre"],
                showlegend=False,
                marker=dict(
                    symbol="star",
                    size=20,
                    color=perfil["color"],
                    line=dict(width=1.2, color="#FFFFFF"),
                ),
                hovertemplate=(
                    f"<b>Centroide · {perfil['nombre']}</b><br>"
                    f"%{{x:{formato}}}<br>%{{y:{formato}}}<extra></extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="Centroides",
            marker=dict(
                symbol="star",
                size=14,
                color="#0F172A",
                line=dict(width=1, color="#FFFFFF"),
            ),
            hoverinfo="skip",
        )
    )

    fig.update_xaxes(title_text=titulo_x)
    fig.update_yaxes(title_text=titulo_y)
    estilo_fig(fig, altura=540)
    st.plotly_chart(fig, theme=None, config=CONFIG_FINAL)

    resumen = "".join(
        f"<li><strong>{perfil['nombre']}</strong> ({int(perfil['n'])} unidades, "
        f"{perfil['porcentaje']:.0f}% de la flota): {perfil['decision']}</li>"
        for _, perfil in perfiles.iterrows()
    )
    st.markdown(
        f'<div class="callout ok"><strong class="callout-label">Cierre Ejecutivo de Logística</strong>'
        f'<ul class="summary-list">{resumen}</ul></div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Perfiles Identificados", f"{len(perfiles)}")
    col2.metric("Silueta del Modelo", f"{float(resultados_k.loc[resultados_k['k'] == k_elegido, 'silueta'].iloc[0]):.3f}")
    col3.metric("Flota Analizada", "150 vehículos")

    st.write("")

    col_a, col_b = st.columns([1, 1.4])
    with col_a:
        st.download_button(
            "Descargar flota clasificada (CSV)",
            data=csv_flotillas(asignaciones),
            file_name="flotilla_segmentada_kmeans.csv",
            mime="text/csv",
            key="descarga_paso_6",
        )
    with col_b:
        st.caption(
            "El archivo CSV incluye ID del vehículo, distancia diaria, horas de ralentí, clúster asignado y perfil logístico recomendado."
        )

st.markdown(
    """
<div class="footer">
  <span>Unidad 5 · Subtema 5.1 · Segmentación y Optimización de Flotillas de Transporte</span>
  <span>K-Means · DuckDB · scikit-learn · Streamlit</span>
</div>
""",
    unsafe_allow_html=True,
)
