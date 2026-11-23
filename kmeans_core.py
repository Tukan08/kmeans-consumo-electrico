from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["consumo_diurno_kwh", "consumo_nocturno_kwh"]

ETIQUETAS = {
    "consumo_diurno_kwh": "Consumo diurno (kWh)",
    "consumo_nocturno_kwh": "Consumo nocturno (kWh)",
}

CATEGORIAS = {
    "eco": {
        "nombre": "Hogares Eco-Eficientes",
        "decision": "No requieren subsidio. Enviar reconocimiento y aplicar la tarifa preferencial base.",
    },
    "moderado": {
        "nombre": "Hogares de Consumo Moderado",
        "decision": "Tarifa preferencial y campaña de eficiencia energética. Reevaluar si su consumo crece.",
    },
    "diurno": {
        "nombre": "Consumidores Diurnos Intensivos",
        "decision": "Candidatos ideales para el subsidio de paneles solares: su pico de demanda coincide con la radiación solar.",
    },
    "nocturno": {
        "nombre": "Vampiros Nocturnos / Climatización",
        "decision": "Vales para renovación de aires acondicionados, inversores y aislamiento térmico. Los paneles sin baterías no cubren su demanda nocturna.",
    },
    "alto": {
        "nombre": "Consumo Alto General",
        "decision": "Auditoría energética prioritaria y paquete combinado de paneles con baterías y aislamiento térmico.",
    },
}

ORDEN_CATEGORIAS = ["eco", "moderado", "diurno", "nocturno", "alto"]

COLOR_VARIANTES = {
    "eco": ["#10B981", "#059669", "#34D399"],
    "moderado": ["#0EA5E9", "#0284C7", "#38BDF8"],
    "diurno": ["#F59E0B", "#D97706", "#FBBF24"],
    "nocturno": ["#4F46E5", "#4338CA", "#818CF8"],
    "alto": ["#EF4444", "#DC2626", "#F87171"],
}

NIVELES = [
    "nivel bajo",
    "nivel medio",
    "nivel alto",
    "nivel muy alto",
    "nivel extremo",
    "nivel máximo",
]


def generar_datos(seed: float = 0.42) -> pd.DataFrame:
    con = duckdb.connect(database=":memory:")
    try:
        con.execute("SELECT setseed(?);", [float(seed)])
        datos = con.execute(
            """
            CREATE TABLE telemetria_hogares AS
            WITH base AS (
                SELECT
                    hogar_id,
                    CASE
                        WHEN hogar_id <= 50 THEN 1
                        WHEN hogar_id <= 100 THEN 2
                        ELSE 3
                    END AS grupo_oculto
                FROM generate_series(1, 150) AS t(hogar_id)
            )
            SELECT
                hogar_id,
                ROUND(CASE
                    WHEN grupo_oculto = 1 THEN 80 + (random() * 60)
                    WHEN grupo_oculto = 2 THEN 260 + (random() * 90)
                    ELSE 110 + (random() * 70)
                END, 2) AS consumo_diurno_kwh,
                ROUND(CASE
                    WHEN grupo_oculto = 1 THEN 90 + (random() * 60)
                    WHEN grupo_oculto = 2 THEN 120 + (random() * 70)
                    ELSE 290 + (random() * 110)
                END, 2) AS consumo_nocturno_kwh
            FROM base
            """
        )
        datos = con.execute("SELECT * FROM telemetria_hogares ORDER BY hogar_id").df()
    finally:
        con.close()
    return datos


def escalar(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES].to_numpy())
    return X, scaler


def evaluar_k(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 6,
    random_state: int = 42,
) -> pd.DataFrame:
    filas = []
    for k in range(k_min, k_max + 1):
        modelo = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        etiquetas = modelo.fit_predict(X)
        filas.append(
            {
                "k": k,
                "silueta": float(silhouette_score(X, etiquetas)),
                "inercia": float(modelo.inertia_),
            }
        )
    return pd.DataFrame(filas)


def entrenar(X: np.ndarray, k: int, random_state: int = 42) -> KMeans:
    modelo = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    modelo.fit(X)
    return modelo


def centroides_reales(modelo: KMeans, scaler: StandardScaler) -> pd.DataFrame:
    return pd.DataFrame(
        scaler.inverse_transform(modelo.cluster_centers_),
        columns=FEATURES,
    )


def _categoria(z_diurno: float, z_nocturno: float, umbral: float = 0.25) -> str:
    if abs(z_diurno) <= umbral and abs(z_nocturno) <= umbral:
        return "moderado"
    if z_diurno > 0 and z_nocturno > 0:
        return "alto"
    if z_diurno > 0:
        return "diurno"
    if z_nocturno > 0:
        return "nocturno"
    return "eco"


def construir_perfiles(
    df: pd.DataFrame,
    modelo: KMeans,
    scaler: StandardScaler,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    asignaciones = df.copy()
    asignaciones["cluster"] = modelo.labels_.astype(int)

    medias = asignaciones.groupby("cluster")[FEATURES].mean()
    conteos = asignaciones["cluster"].value_counts().sort_index()
    centroides_z = pd.DataFrame(modelo.cluster_centers_, columns=FEATURES)

    filas = []
    for cluster in medias.index:
        z_diurno = float(centroides_z.loc[cluster, FEATURES[0]])
        z_nocturno = float(centroides_z.loc[cluster, FEATURES[1]])
        categoria = _categoria(z_diurno, z_nocturno)
        filas.append(
            {
                "cluster": int(cluster),
                "categoria": categoria,
                "n": int(conteos.loc[cluster]),
                "porcentaje": float(conteos.loc[cluster] / len(asignaciones) * 100),
                "media_diurno": float(medias.loc[cluster, FEATURES[0]]),
                "media_nocturno": float(medias.loc[cluster, FEATURES[1]]),
                "z_diurno": z_diurno,
                "z_nocturno": z_nocturno,
            }
        )

    perfiles = pd.DataFrame(filas)
    perfiles["total"] = perfiles["media_diurno"] + perfiles["media_nocturno"]
    orden = {categoria: i for i, categoria in enumerate(ORDEN_CATEGORIAS)}
    perfiles["orden_categoria"] = perfiles["categoria"].map(orden)
    perfiles = perfiles.sort_values(
        ["orden_categoria", "total"], ascending=[True, True]
    ).reset_index(drop=True)

    for categoria, grupo in perfiles.groupby("categoria", sort=False):
        if len(grupo) == 1:
            sufijos = [""]
        elif len(grupo) == 2:
            sufijos = [" - nivel bajo", " - nivel alto"]
        else:
            sufijos = [f" - {NIVELES[pos]}" for pos in range(len(grupo))]

        for sufijo, indice in zip(sufijos, grupo.index):
            perfiles.loc[indice, "sufijo"] = sufijo
            perfiles.loc[indice, "nombre"] = (
                CATEGORIAS[categoria]["nombre"] + sufijo
            )
            perfiles.loc[indice, "decision"] = CATEGORIAS[categoria]["decision"]
            perfiles.loc[indice, "color"] = COLOR_VARIANTES[categoria][
                list(grupo.index).index(indice) % len(COLOR_VARIANTES[categoria])
            ]

    mapa_nombres = dict(zip(perfiles["cluster"], perfiles["nombre"]))
    mapa_colores = dict(zip(perfiles["cluster"], perfiles["color"]))
    asignaciones["perfil"] = asignaciones["cluster"].map(mapa_nombres)
    asignaciones["categoria"] = asignaciones["cluster"].map(
        dict(zip(perfiles["cluster"], perfiles["categoria"]))
    )
    asignaciones["color"] = asignaciones["cluster"].map(mapa_colores)

    return asignaciones, perfiles
