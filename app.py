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
    page_title="Segmentación de Consumo Eléctrico | K-Means",
    layout="wide",
    initial_sidebar_state="auto",
)

css = (Path(__file__).parent / "assets" / "styles.css").read_text(encoding="utf-8")
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

AZUL = "#1E40AF"
AZUL_CLARO = "#BFDBFE"
AMBAR = "#F59E0B"
INDIGO = "#4F46E5"
GRIS = "#64748B"

PASOS = [
    "01 · Exploración de la telemetría",
    "02 · Escalado de variables",
    "03 · Número óptimo de grupos",
    "04 · Entrenamiento del modelo",
    "05 · Perfilamiento de negocio",
    "06 · Visualización final",
]

CONFIG_GRAFICA = {"displayModeBar": False, "displaylogo": False}
CONFIG_FINAL = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}


@st.cache_data(show_spinner=False)
def cargar_datos(semilla: float) -> pd.DataFrame:
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


def scatter_hogares(
    x: np.ndarray,
    y: np.ndarray,
    hogares: np.ndarray,
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
                opacity=0.55,
                line=dict(width=1, color="#FFFFFF"),
            ),
            customdata=hogares,
            hovertemplate=(
                "<b>Hogar %{customdata[0]}</b><br>"
                + x_titulo
                + ": %{x:.1f}<br>"
                + y_titulo
                + ": %{y:.1f}<extra></extra>"
            ),
            name="Hogares",
        )
    )
    fig.update_xaxes(title_text=x_titulo)
    fig.update_yaxes(title_text=y_titulo)
    return estilo_fig(fig, altura=460)


def csv_hogares(asignaciones: pd.DataFrame) -> bytes:
    export = asignaciones.drop(columns=["color", "categoria"], errors="ignore")
    export = export.rename(
        columns={
            "hogar_id": "Hogar",
            "consumo_diurno_kwh": "Consumo diurno (kWh)",
            "consumo_nocturno_kwh": "Consumo nocturno (kWh)",
            "cluster": "Clúster",
            "perfil": "Perfil",
        }
    )
    return export.to_csv(index=False).encode("utf-8-sig")


st.markdown(
    """
<div class="hero">
  <div class="badge-row">
    <span class="badge">Unidad 5 · Subtema 5.1</span>
    <span class="badge accent">K-Means · No supervisado</span>
    <span class="badge">150 hogares</span>
  </div>
  <h1>Segmentación Inteligente de Consumo Eléctrico Residencial</h1>
  <p>Planes de ahorro energético a partir de telemetría diurna y nocturna. Descubre los perfiles reales de consumo, valida cuántos grupos existen y traduce cada clúster en una decisión de negocio.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
<div class="side-kicker">Energía · Ciencia de datos</div>
<div class="side-title">Segmentación de consumo eléctrico</div>
<div class="side-copy">La comisión reguladora y el ayuntamiento deben repartir un presupuesto limitado de subsidios entre 150 hogares.</div>
""",
    unsafe_allow_html=True,
)
st.sidebar.markdown("")

semilla_int = st.sidebar.number_input(
    "Semilla de simulación",
    min_value=0,
    max_value=99,
    value=42,
    step=1,
    help="Genera los mismos 150 hogares en cada ejecución. La semilla 42 equivale a setseed(0.42).",
)
semilla = semilla_int / 100
st.sidebar.caption(f"Semilla activa: {semilla:.2f} · 150 hogares")

paso = st.sidebar.radio("Ruta de abordaje", PASOS, index=0)
indice_paso = PASOS.index(paso)
st.sidebar.progress(
    (indice_paso + 1) / len(PASOS),
    text=f"Paso {indice_paso + 1} de {len(PASOS)}",
)
st.sidebar.markdown(
    '<div class="side-meta">Datos sintéticos reproducibles · variables clave: consumo diurno (08:00–18:00) y nocturno (18:01–07:59).</div>',
    unsafe_allow_html=True,
)

df = cargar_datos(semilla)
X, scaler = escalar(df)
resultados_k = evaluar_k(X, random_state=semilla_int)
mejor_k = int(resultados_k.loc[resultados_k["silueta"].idxmax(), "k"])

if st.session_state.get("_semilla_actual") != semilla_int:
    st.session_state["_semilla_actual"] = semilla_int
    st.session_state["k_manual"] = mejor_k

k_elegido = int(st.session_state.get("k_manual", mejor_k))

if paso == PASOS[0]:
    step_head(
        1,
        "Generación y exploración inicial de la telemetría",
        "Se cargan los 150 hogares con semilla fija y se observa si los puntos forman islas naturales.",
    )

    callout(
        "info",
        "Datos sin etiquetas",
        "Cada hogar aporta únicamente su consumo diurno y nocturno en kWh. No sabemos de antemano quién ahorra y quién derrocha: ese es el reto del aprendizaje no supervisado.",
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Hogares monitoreados", f"{len(df)}")
    col2.metric(
        "Consumo diurno promedio",
        f"{df[FEATURES[0]].mean():,.1f} kWh",
    )
    col3.metric(
        "Consumo nocturno promedio",
        f"{df[FEATURES[1]].mean():,.1f} kWh",
    )
    correlacion = df[FEATURES].corr().iloc[0, 1]
    col4.metric("Correlación diurno–nocturno", f"{correlacion:.2f}")

    st.write("")

    with st.container(border=True):
        panel_head(
            "Diagrama de dispersión inicial",
            "Cada punto representa un hogar: consumo diurno en X, consumo nocturno en Y.",
        )
        fig = scatter_hogares(
            df[FEATURES[0]].to_numpy(),
            df[FEATURES[1]].to_numpy(),
            df[["hogar_id"]].to_numpy(),
            ETIQUETAS[FEATURES[0]],
            ETIQUETAS[FEATURES[1]],
        )
        st.plotly_chart(fig, theme=None, config=CONFIG_GRAFICA)

    with st.expander("Ver registros crudos de telemetría"):
        st.dataframe(
            df.head(20),
            hide_index=True,
            column_config={
                "hogar_id": st.column_config.NumberColumn("Hogar", format="%d"),
                "consumo_diurno_kwh": st.column_config.NumberColumn(
                    "Diurno (kWh)", format="%.2f"
                ),
                "consumo_nocturno_kwh": st.column_config.NumberColumn(
                    "Nocturno (kWh)", format="%.2f"
                ),
            },
        )

    callout(
        "insight",
        "Lectura visual",
        "A simple vista se distinguen zonas con densidad propia: hogares de bajo consumo, hogares con pico diurno y hogares con pico nocturno. La correlación lineal es débil, así que una recta no basta para separarlos; hay que medir distancias en el plano.",
    )

elif paso == PASOS[1]:
    step_head(
        2,
        "Preprocesamiento y escalamiento de variables",
        "K-Means calcula distancias euclidianas: todas las variables deben competir en igualdad de condiciones.",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        callout(
            "warn",
            "El problema técnico",
            "Si una variable tuviera magnitudes mucho mayores que la otra, dominaría el cálculo de distancias y sesgaría los grupos. Aunque aquí ambas se miden en kWh, sus medias y dispersiones no son idénticas.",
        )
    with col_b:
        callout(
            "ok",
            "La solución: Z-score",
            "Estandarizamos cada variable para que quede con media 0 y desviación estándar 1. Así el espacio geométrico trata el consumo diurno y el nocturno con el mismo peso.",
        )

    desv_diurna = df[FEATURES[0]].std()
    desv_nocturna = df[FEATURES[1]].std()
    col1, col2, col3 = st.columns(3)
    col1.metric("Desv. estándar diurna", f"{desv_diurna:,.1f} kWh")
    col2.metric("Desv. estándar nocturna", f"{desv_nocturna:,.1f} kWh")
    col3.metric("Relación de escalas", f"{desv_diurna / desv_nocturna:.2f}×")

    st.write("")

    col_a, col_b = st.columns(2)
    with col_a:
        with st.container(border=True):
            panel_head(
                "Antes del escalado",
                "Coordenadas originales en kWh: las distancias dependen de la escala física.",
            )
            fig = scatter_hogares(
                df[FEATURES[0]].to_numpy(),
                df[FEATURES[1]].to_numpy(),
                df[["hogar_id"]].to_numpy(),
                ETIQUETAS[FEATURES[0]],
                ETIQUETAS[FEATURES[1]],
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, theme=None, config=CONFIG_GRAFICA)

    with col_b:
        with st.container(border=True):
            panel_head(
                "Después del escalado",
                "Coordenadas Z: media 0 y desviación estándar 1 en ambas variables.",
            )
            fig = scatter_hogares(
                X[:, 0],
                X[:, 1],
                df[["hogar_id"]].to_numpy(),
                "Consumo diurno (z)",
                "Consumo nocturno (z)",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, theme=None, config=CONFIG_GRAFICA)

    with st.container(border=True):
        panel_head(
            "Verificación del escalado",
            "La media debe quedar en 0 y la desviación estándar en 1.",
        )
        comparativa = pd.DataFrame(
            {
                "Variable": [ETIQUETAS[f] for f in FEATURES],
                "Media original (kWh)": [df[f].mean() for f in FEATURES],
                "Desv. estándar original (kWh)": [df[f].std() for f in FEATURES],
                "Media estandarizada": [
                    0.0 if abs(X[:, i].mean()) < 1e-9 else X[:, i].mean()
                    for i in range(2)
                ],
                "Desv. estándar estandarizada": [X[:, i].std() for i in range(2)],
            }
        )
        st.dataframe(
            comparativa,
            hide_index=True,
            column_config={
                "Media original (kWh)": st.column_config.NumberColumn(format="%.2f"),
                "Desv. estándar original (kWh)": st.column_config.NumberColumn(
                    format="%.2f"
                ),
                "Media estandarizada": st.column_config.NumberColumn(format="%.4f"),
                "Desv. estándar estandarizada": st.column_config.NumberColumn(
                    format="%.4f"
                ),
            },
        )

    callout(
        "info",
        "A partir de aquí",
        "Todas las distancias y centroides se calculan en el espacio estandarizado. Al interpretar resultados, volveremos a convertir los centroides a kWh reales.",
    )

elif paso == PASOS[2]:
    step_head(
        3,
        "Búsqueda del número óptimo de grupos mediante el coeficiente de silueta",
        "No sabemos si existen 2, 3, 4 o más perfiles: probamos candidatos de k = 2 a k = 6.",
    )

    mejor_silueta = float(
        resultados_k.loc[resultados_k["k"] == mejor_k, "silueta"].iloc[0]
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Mejor valor de k", f"{mejor_k}")
    col2.metric("Silueta del mejor k", f"{mejor_silueta:.3f}")
    col3.metric("Candidatos evaluados", f"{len(resultados_k)}")

    st.write("")

    col_a, col_b = st.columns([1.35, 1])
    with col_a:
        with st.container(border=True):
            panel_head(
                "Coeficiente de silueta por valor de k",
                "Se selecciona el k con el puntaje promedio más alto.",
            )
            colores = [
                AMBAR if int(k) == mejor_k else AZUL_CLARO
                for k in resultados_k["k"]
            ]
            fig = go.Figure(
                go.Bar(
                    x=resultados_k["k"],
                    y=resultados_k["silueta"],
                    marker_color=colores,
                    marker_line=dict(width=1, color="#FFFFFF"),
                    text=[f"{v:.3f}" for v in resultados_k["silueta"]],
                    textposition="outside",
                    textfont=dict(family="Fira Code", size=12, color="#334155"),
                    hovertemplate="k = %{x}<br>Silueta = %{y:.3f}<extra></extra>",
                )
            )
            fig.add_hline(
                y=0.5,
                line_dash="dot",
                line_color="#94A3B8",
                annotation_text="0.5 · estructura razonable",
                annotation_position="bottom right",
                annotation_font=dict(size=10, color="#64748B"),
            )
            fig.update_xaxes(title_text="Número de grupos (k)", dtick=1)
            fig.update_yaxes(title_text="Coeficiente de silueta", range=[0, 1])
            estilo_fig(fig, altura=400, leyenda=False)
            st.plotly_chart(fig, theme=None, config=CONFIG_GRAFICA)

    with col_b:
        with st.container(border=True):
            panel_head(
                "Inercia por valor de k",
                "Apoyo visual: la inercia siempre baja al agregar grupos.",
            )
            fig = go.Figure(
                go.Scatter(
                    x=resultados_k["k"],
                    y=resultados_k["inercia"],
                    mode="lines+markers",
                    line=dict(color=GRIS, width=2),
                    marker=dict(size=8, color=GRIS),
                    hovertemplate="k = %{x}<br>Inercia = %{y:.2f}<extra></extra>",
                )
            )
            fig.update_xaxes(title_text="Número de grupos (k)", dtick=1)
            fig.update_yaxes(title_text="Inercia")
            estilo_fig(fig, altura=400, leyenda=False)
            st.plotly_chart(fig, theme=None, config=CONFIG_GRAFICA)

    st.markdown(
        """
<div class="mini-grid">
  <div class="mini-card">
    <span class="tag">Cercano a +1</span>
    <h4>Bien clasificado</h4>
    <p>El hogar es muy parecido a su grupo y está lejos de los grupos vecinos.</p>
  </div>
  <div class="mini-card">
    <span class="tag">Cercano a 0</span>
    <h4>Frontera difusa</h4>
    <p>El hogar está justo entre dos grupos: su asignación es ambigua.</p>
  </div>
  <div class="mini-card">
    <span class="tag">Negativo</span>
    <h4>Asignación dudosa</h4>
    <p>El hogar probablemente quedó en el grupo equivocado.</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    if mejor_silueta >= 0.7:
        tono, lectura = "ok", "estructura fuerte"
    elif mejor_silueta >= 0.5:
        tono, lectura = "insight", "estructura razonable"
    else:
        tono, lectura = "warn", "estructura débil"

    callout(
        tono,
        "Resultado",
        f"El coeficiente de silueta más alto aparece con k = {mejor_k} ({mejor_silueta:.3f}), lo que indica una {lectura}. Se usará este valor como referencia para entrenar el modelo definitivo.",
    )

elif paso == PASOS[3]:
    step_head(
        4,
        "Entrenamiento del modelo definitivo y asignación de etiquetas",
        "Se ejecuta K-Means con el k seleccionado y se calculan los centroides de cada grupo.",
    )

    with st.container(border=True):
        panel_head(
            "Configuración del modelo",
            "El valor sugerido es el óptimo por silueta, pero puedes explorar otros valores.",
        )
        k_entrenamiento = int(
            st.select_slider(
                "Número de grupos (k)",
                options=list(range(2, 7)),
                key="k_manual",
                help="Cambiar k vuelve a entrenar el modelo en el siguiente paso.",
            )
        )

    if k_entrenamiento != mejor_k:
        callout(
            "warn",
            "Exploración manual",
            f"El óptimo por silueta es k = {mejor_k}. Estás usando k = {k_entrenamiento}; compara la silueta obtenida antes de tomar decisiones.",
        )
    else:
        callout(
            "ok",
            "Coincidencia",
            f"k = {k_entrenamiento} coincide con el valor óptimo encontrado en el paso anterior.",
        )

    modelo = entrenar(X, k_entrenamiento, random_state=semilla_int)
    silueta_modelo = float(
        resultados_k.loc[resultados_k["k"] == k_entrenamiento, "silueta"].iloc[0]
    )
    etiquetas = modelo.labels_

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Grupos (k)", f"{k_entrenamiento}")
    col2.metric("Coeficiente de silueta", f"{silueta_modelo:.3f}")
    col3.metric("Inercia final", f"{modelo.inertia_:,.2f}")
    col4.metric("Iteraciones", f"{modelo.n_iter_}")

    st.write("")

    centroides = pd.DataFrame(
        scaler.inverse_transform(modelo.cluster_centers_),
        columns=["Diurno real (kWh)", "Nocturno real (kWh)"],
    )
    centroides.insert(0, "Clúster", [f"Clúster {c}" for c in range(k_entrenamiento)])
    centroides["Hogares"] = [
        int((etiquetas == c).sum()) for c in range(k_entrenamiento)
    ]
    centroides["Participación (%)"] = centroides["Hogares"] / len(df) * 100
    centroides["Diurno (z)"] = modelo.cluster_centers_[:, 0]
    centroides["Nocturno (z)"] = modelo.cluster_centers_[:, 1]
    centroides_z = centroides[["Clúster", "Diurno (z)", "Nocturno (z)"]]
    centroides_tabla = centroides.drop(columns=["Diurno (z)", "Nocturno (z)"])

    with st.container(border=True):
        panel_head(
            "Centroides del modelo",
            "El centroide es el 'hogar promedio' de cada grupo, expresado en kWh reales.",
        )
        st.dataframe(
            centroides_tabla,
            hide_index=True,
            column_config={
                "Clúster": st.column_config.TextColumn(width="small"),
                "Diurno real (kWh)": st.column_config.NumberColumn(
                    width="medium", format="%.1f"
                ),
                "Nocturno real (kWh)": st.column_config.NumberColumn(
                    width="medium", format="%.1f"
                ),
                "Hogares": st.column_config.NumberColumn(
                    width="small", format="%d"
                ),
                "Participación (%)": st.column_config.NumberColumn(
                    width="small", format="%.1f%%"
                ),
            },
        )
        with st.expander("Ver centroides en escala z"):
            st.dataframe(
                centroides_z,
                hide_index=True,
                column_config={
                    "Clúster": st.column_config.TextColumn(width="small"),
                    "Diurno (z)": st.column_config.NumberColumn(format="%.2f"),
                    "Nocturno (z)": st.column_config.NumberColumn(format="%.2f"),
                },
            )

    with st.container(border=True):
        panel_head(
            "Tamaño de cada grupo",
            "Distribución de los 150 hogares entre las etiquetas asignadas.",
        )
        conteos = pd.Series(etiquetas).value_counts().sort_index()
        fig = go.Figure(
            go.Bar(
                x=conteos.values,
                y=[f"Clúster {c}" for c in conteos.index],
                orientation="h",
                marker_color=AZUL,
                marker_line=dict(width=1, color="#FFFFFF"),
                text=conteos.values,
                textposition="outside",
                textfont=dict(family="Fira Code", size=12, color="#334155"),
                hovertemplate="Clúster %{y}<br>%{x} hogares<extra></extra>",
            )
        )
        fig.update_xaxes(title_text="Hogares")
        fig.update_yaxes(autorange="reversed")
        estilo_fig(fig, altura=210, leyenda=False)
        st.plotly_chart(fig, theme=None, config=CONFIG_GRAFICA)

    callout(
        "info",
        "Etiquetas arbitrarias",
        "Los números 0, 1, 2… solo identifican grupos: no implican orden ni gravedad. El significado aparece en el siguiente paso, al traducir cada centroide a un perfil de consumo.",
    )

elif paso == PASOS[4]:
    step_head(
        5,
        "Perfilamiento de negocio y toma de decisiones",
        "El número de clúster no le dice nada al gerente de energía: hay que bautizar los grupos y calcular su consumo real.",
    )

    modelo = entrenar(X, k_elegido, random_state=semilla_int)
    asignaciones, perfiles = construir_perfiles(df, modelo, scaler)

    col1, col2, col3 = st.columns(3)
    col1.metric("Perfiles identificados", f"{len(perfiles)}")
    col2.metric(
        "Hogar de mayor consumo",
        f"{asignaciones[FEATURES].sum(axis=1).max():,.0f} kWh",
    )
    col3.metric(
        "Perfil más numeroso",
        f"{int(perfiles['n'].max())} hogares",
    )

    st.write("")

    numero_columnas = 3 if len(perfiles) == 3 else 2
    columnas = st.columns(numero_columnas)
    for posicion, (_, perfil) in enumerate(perfiles.iterrows()):
        with columnas[posicion % numero_columnas]:
            st.markdown(
                f"""
<div class="profile-card" style="--profile-color: {perfil['color']}">
  <div class="profile-top">
    <span class="profile-dot"></span>
    <span class="profile-cluster">Clúster {perfil['cluster']}</span>
    <span class="profile-share">{perfil['porcentaje']:.0f}% de los hogares</span>
  </div>
  <h3>{perfil['nombre']}</h3>
  <div class="profile-metrics">
    <div><span>Diurno</span><strong>{perfil['media_diurno']:,.1f} kWh</strong></div>
    <div><span>Nocturno</span><strong>{perfil['media_nocturno']:,.1f} kWh</strong></div>
    <div><span>Hogares</span><strong>{int(perfil['n'])}</strong></div>
  </div>
  <div class="profile-decision">
    <b>Decisión sugerida</b>
    {perfil['decision']}
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

    with st.container(border=True):
        panel_head(
            "Promedio real de consumo por perfil",
            "Valores en kWh, calculados sobre los hogares originales (no sobre los datos escalados).",
        )
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=perfiles["nombre"],
                x=perfiles["media_diurno"],
                name="Consumo diurno",
                orientation="h",
                marker_color=AMBAR,
                hovertemplate="%{y}<br>Diurno: %{x:.1f} kWh<extra></extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                y=perfiles["nombre"],
                x=perfiles["media_nocturno"],
                name="Consumo nocturno",
                orientation="h",
                marker_color=INDIGO,
                hovertemplate="%{y}<br>Nocturno: %{x:.1f} kWh<extra></extra>",
            )
        )
        fig.update_layout(barmode="group")
        fig.update_xaxes(title_text="Consumo promedio (kWh)")
        fig.update_yaxes(autorange="reversed", tickfont=dict(size=11))
        estilo_fig(fig, altura=max(320, 90 + 70 * len(perfiles)))
        st.plotly_chart(fig, theme=None, config=CONFIG_GRAFICA)

    with st.expander("Ver y descargar la clasificación de los 150 hogares"):
        tabla = asignaciones.drop(columns=["color", "categoria"]).rename(
            columns={
                "hogar_id": "Hogar",
                "consumo_diurno_kwh": "Consumo diurno (kWh)",
                "consumo_nocturno_kwh": "Consumo nocturno (kWh)",
                "cluster": "Clúster",
                "perfil": "Perfil",
            }
        )
        st.dataframe(
            tabla,
            hide_index=True,
            height=340,
            column_config={
                "Consumo diurno (kWh)": st.column_config.NumberColumn(format="%.1f"),
                "Consumo nocturno (kWh)": st.column_config.NumberColumn(
                    format="%.1f"
                ),
            },
        )
        st.download_button(
            "Descargar hogares clasificados (CSV)",
            data=csv_hogares(asignaciones),
            file_name="hogares_segmentados.csv",
            mime="text/csv",
            key="descarga_paso_5",
        )

    callout(
        "insight",
        "Traducción a política pública",
        "Cada perfil habilita una decisión distinta: reconocimiento para los eco-eficientes, paneles solares para el pico diurno, aislamiento e inversores para el pico nocturno y una auditoría prioritaria para el consumo alto general.",
    )

else:
    step_head(
        6,
        "Visualización final de clústeres y centroides",
        "Se colorea cada hogar según su perfil y se marca con una estrella la posición exacta de cada centroide.",
    )

    modelo = entrenar(X, k_elegido, random_state=semilla_int)
    asignaciones, perfiles = construir_perfiles(df, modelo, scaler)

    with st.container(border=True):
        panel_head(
            "Mapa final de segmentación",
            "Cada color representa un perfil de consumo; las estrellas son los centroides.",
        )
        escala_z = st.toggle(
            "Ver en escala estandarizada (z)",
            value=False,
            help="Alterna entre kWh reales y coordenadas Z usadas por el algoritmo.",
        )

        if escala_z:
            coordenadas = X
            centroides = modelo.cluster_centers_
            titulo_x = "Consumo diurno (z)"
            titulo_y = "Consumo nocturno (z)"
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
                    customdata=asignaciones.loc[mascara, ["hogar_id"]].to_numpy(),
                    hovertemplate=(
                        f"<b>Hogar %{{customdata[0]}}</b><br>"
                        f"%{{x:{formato}}}<br>%{{y:{formato}}}<extra></extra>"
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
        f"<li><strong>{perfil['nombre']}</strong> ({int(perfil['n'])} hogares, "
        f"{perfil['porcentaje']:.0f}%): {perfil['decision']}</li>"
        for _, perfil in perfiles.iterrows()
    )
    st.markdown(
        f'<div class="callout ok"><strong class="callout-label">Cierre ejecutivo</strong>'
        f'<ul class="summary-list">{resumen}</ul></div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    perfil_lider = perfiles.loc[perfiles["total"].idxmax()]
    col1.metric("Perfiles finales", f"{len(perfiles)}")
    col2.metric("Silueta del modelo", f"{float(resultados_k.loc[resultados_k['k'] == k_elegido, 'silueta'].iloc[0]):.3f}")
    col3.metric(
        "Consumo medio más alto",
        f"{perfil_lider['total']:,.0f} kWh",
    )
    st.caption(
        f"El perfil con mayor consumo promedio es {perfil_lider['nombre']} "
        "(diurno + nocturno)."
    )

    st.write("")

    col_a, col_b = st.columns([1, 1.4])
    with col_a:
        st.download_button(
            "Descargar hogares clasificados (CSV)",
            data=csv_hogares(asignaciones),
            file_name="hogares_segmentados.csv",
            mime="text/csv",
            key="descarga_paso_6",
        )
    with col_b:
        st.caption(
            "El CSV incluye hogar, consumos originales, etiqueta numérica de clúster y perfil de negocio asignado."
        )

st.markdown(
    """
<div class="footer">
  <span>Unidad 5 · Subtema 5.1 · Segmentación Inteligente de Consumo Eléctrico Residencial</span>
  <span>K-Means · DuckDB · scikit-learn · Streamlit</span>
</div>
""",
    unsafe_allow_html=True,
)
