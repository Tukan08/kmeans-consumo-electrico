from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["km_recorridos_dia", "horas_ralenti_dia"]

ETIQUETAS = {
    "km_recorridos_dia": "Distancia diaria (km)",
    "horas_ralenti_dia": "Tiempo en ralentí (horas)",
}

CATEGORIAS = {
    "eficiente": {
        "nombre": "Flotilla Urbana Eficiente",
        "decision": "Bajo desgaste y bajo consumo innecesario. Mantener calendario regular de mantenimiento preventivo y certificar a operadores.",
    },
    "moderado": {
        "nombre": "Operación Logística Estándar",
        "decision": "Rendimiento balanceado. Capacitación continua en eco-driving y monitoreo bimestral de consumo de combustible.",
    },
    "foraneo": {
        "nombre": "Transporte Foráneo / Carretera",
        "decision": "Monitoreo riguroso de neumáticos, frenos y aerodinámica. Implementar telepeaje automatizado y control de velocidad crucero.",
    },
    "congestion": {
        "nombre": "Reparto en Congestión Crítica",
        "decision": "Candidatos prioritarios para electrificación / hibridación de flota o reprogramación de ventanas horarias nocturnas para evadir tráfico.",
    },
    "critico": {
        "nombre": "Operación de Alta Severidad / Sobrecarga",
        "decision": "Auditoría inmediata de rutas GPS, rotación obligatoria de operadores por fatiga y revisión mecánica exhaustiva por desgaste prematuro.",
    },
}

ORDEN_CATEGORIAS = ["eficiente", "moderado", "foraneo", "congestion", "critico"]

COLOR_VARIANTES = {
    "eficiente": ["#10B981", "#059669", "#34D399"],
    "moderado": ["#0EA5E9", "#0284C7", "#38BDF8"],
    "foraneo": ["#F59E0B", "#D97706", "#FBBF24"],
    "congestion": ["#6366F1", "#4F46E5", "#818CF8"],
    "critico": ["#EF4444", "#DC2626", "#F87171"],
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
        con.execute(
            """
            CREATE TABLE telemetria_flotillas AS
            WITH base AS (
                SELECT
                    vehiculo_id,
                    CASE
                        WHEN vehiculo_id <= 50 THEN 1  -- Perfil latente urbano eficiente
                        WHEN vehiculo_id <= 100 THEN 2 -- Perfil latente foráneo larga distancia
                        ELSE 3                         -- Perfil latente reparto urbano con alto tráfico
                    END AS grupo_oculto
                FROM generate_series(1, 150) AS t(vehiculo_id)
            )
            SELECT
                vehiculo_id,
                ROUND(CASE
                    WHEN grupo_oculto = 1 THEN 50 + (random() * 60)    -- 50 a 110 km
                    WHEN grupo_oculto = 2 THEN 240 + (random() * 110)  -- 240 a 350 km
                    ELSE 75 + (random() * 70)                          -- 75 a 145 km
                END, 1) AS km_recorridos_dia,
                ROUND(CASE
                    WHEN grupo_oculto = 1 THEN 0.8 + (random() * 1.2)  -- 0.8 a 2.0 hrs ralentí
                    WHEN grupo_oculto = 2 THEN 1.0 + (random() * 1.5)  -- 1.0 a 2.5 hrs ralentí
                    ELSE 3.8 + (random() * 2.2)                        -- 3.8 a 6.0 hrs ralentí
                END, 2) AS horas_ralenti_dia
            FROM base
            """
        )
        datos = con.execute("SELECT * FROM telemetria_flotillas ORDER BY vehiculo_id").df()
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


def _categoria(z_km: float, z_ralenti: float, umbral: float = 0.25) -> str:
    if abs(z_km) <= umbral and abs(z_ralenti) <= umbral:
        return "moderado"
    if z_km > 0 and z_ralenti > 0:
        return "critico"
    if z_km > 0:
        return "foraneo"
    if z_ralenti > 0:
        return "congestion"
    return "eficiente"


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
        z_km = float(centroides_z.loc[cluster, FEATURES[0]])
        z_ralenti = float(centroides_z.loc[cluster, FEATURES[1]])
        categoria = _categoria(z_km, z_ralenti)
        filas.append(
            {
                "cluster": int(cluster),
                "categoria": categoria,
                "n": int(conteos.loc[cluster]),
                "porcentaje": float(conteos.loc[cluster] / len(asignaciones) * 100),
                "media_km": float(medias.loc[cluster, FEATURES[0]]),
                "media_ralenti": float(medias.loc[cluster, FEATURES[1]]),
                "z_km": z_km,
                "z_ralenti": z_ralenti,
            }
        )

    perfiles = pd.DataFrame(filas)
    perfiles["total_impacto"] = perfiles["media_km"] + (perfiles["media_ralenti"] * 25.0)
    orden = {categoria: i for i, categoria in enumerate(ORDEN_CATEGORIAS)}
    perfiles["orden_categoria"] = perfiles["categoria"].map(orden)
    perfiles = perfiles.sort_values(
        ["orden_categoria", "total_impacto"], ascending=[True, True]
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
