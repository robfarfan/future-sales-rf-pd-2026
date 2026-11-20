"""Carga y preparación de los datos de Predict Future Sales.

Implementa las decisiones de limpieza del EDA
(``notebooks/01_exploratory_data_analysis.ipynb``, sección 4.5) y la
construcción de la malla mensual tienda × producto que se usa para
entrenar y para predecir.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from future_sales.config import EXTERNAL_DATA_DIR, RAW_DATA_DIR

# Decisiones del EDA (sección 4.5).
MAPA_TIENDAS: dict[int, int] = {0: 57, 1: 58, 11: 10}
PRECIO_MAX: float = 100_000
CANTIDAD_MAX: float = 1_000

# Target: unidades del mes recortadas como en Kaggle.
LIMITE_INFERIOR: float = 0.0
LIMITE_SUPERIOR: float = 20.0
MES_TEST: int = 34
CLAVES: list[str] = ["date_block_num", "shop_id", "item_id"]


def cargar_ventas(ruta: Path = RAW_DATA_DIR / "sales_train.csv"
                  ) -> pd.DataFrame:
    """Lee las ventas diarias crudas.

    Args:
        ruta: Ruta de ``sales_train.csv``.

    Returns:
        Ventas diarias sin modificar.
    """
    return pd.read_csv(ruta)


def cargar_catalogos(
    ruta_raw: Path = RAW_DATA_DIR,
    ruta_externa: Path = EXTERNAL_DATA_DIR,
) -> dict[str, pd.DataFrame]:
    """Lee el catálogo de productos y las traducciones manuales.

    Args:
        ruta_raw: Carpeta con ``items.csv``.
        ruta_externa: Carpeta con las traducciones ``*_traducido.csv``.

    Returns:
        Diccionario con ``items``, ``categorias`` y ``tiendas``.
    """
    return {
        "items": pd.read_csv(ruta_raw / "items.csv"),
        "categorias": pd.read_csv(
            ruta_externa / "item_categories_traducido.csv"
        ).fillna({"subcategoria": ""}),
        "tiendas": pd.read_csv(ruta_externa / "shops_traducido.csv"),
    }


def unificar_tiendas(shop_id: pd.Series) -> pd.Series:
    """Reemplaza los IDs de tiendas duplicadas por el ID que usa test.

    Args:
        shop_id: Columna de IDs de tienda.

    Returns:
        IDs con 0→57, 1→58 y 11→10.
    """
    return shop_id.replace(MAPA_TIENDAS)


def limpiar_ventas(ventas: pd.DataFrame) -> pd.DataFrame:
    """Aplica la limpieza decidida en el EDA.

    Elimina duplicados exactos, precios no positivos o mayores a
    ``PRECIO_MAX`` y cantidades mayores a ``CANTIDAD_MAX``; unifica las
    tiendas repetidas. Las devoluciones se conservan (se netean al
    agregar por mes).

    Args:
        ventas: Ventas diarias crudas.

    Returns:
        Copia limpia de las ventas.
    """
    limpias = ventas.drop_duplicates()
    limpias = limpias[
        (limpias["item_price"] > 0)
        & (limpias["item_price"] <= PRECIO_MAX)
        & (limpias["item_cnt_day"] <= CANTIDAD_MAX)
    ].copy()
    limpias["shop_id"] = unificar_tiendas(limpias["shop_id"])
    return limpias.reset_index(drop=True)


def ventas_mensuales(ventas: pd.DataFrame,
                     items: pd.DataFrame) -> pd.DataFrame:
    """Agrega las ventas diarias limpias por mes, tienda y producto.

    Args:
        ventas: Ventas diarias limpias.
        items: Catálogo con ``item_id`` e ``item_category_id``.

    Returns:
        Una fila por (mes, tienda, producto) con venta registrada, con
        ``item_cnt_month`` (unidades netas, sin recortar),
        ``precio_medio`` e ``item_category_id``.
    """
    mensual = (
        ventas.groupby(CLAVES, as_index=False)
        .agg(item_cnt_month=("item_cnt_day", "sum"),
             precio_medio=("item_price", "mean"))
        .merge(items[["item_id", "item_category_id"]], on="item_id",
               how="left")
    )
    tipos = {"date_block_num": "int16", "shop_id": "int16",
             "item_id": "int32", "item_category_id": "int16",
             "item_cnt_month": "float32", "precio_medio": "float32"}
    return mensual.astype(tipos)


def recortar_target(valores: pd.Series | np.ndarray
                    ) -> pd.Series | np.ndarray:
    """Recorta unidades mensuales al rango [0, 20] de Kaggle.

    Args:
        valores: Unidades mensuales o predicciones.

    Returns:
        Valores recortados, del mismo tipo que la entrada.
    """
    return np.clip(valores, LIMITE_INFERIOR, LIMITE_SUPERIOR)


def construir_malla(mensual: pd.DataFrame,
                    meses: range | list[int]) -> pd.DataFrame:
    """Construye la malla tienda × producto de cada mes, como en test.

    Para cada mes, combina todas las tiendas con venta en el mes con
    todos los productos vendidos en el mes; los pares sin venta quedan
    en 0. El target se recorta a [0, 20].

    Args:
        mensual: Salida de :func:`ventas_mensuales`.
        meses: Meses (``date_block_num``) a incluir.

    Returns:
        Malla con ``date_block_num``, ``shop_id``, ``item_id`` y
        ``item_cnt_month`` (recortado), ordenada por mes, tienda y
        producto.
    """
    bloques = []
    for mes in meses:
        del_mes = mensual[mensual["date_block_num"] == mes]
        indice = pd.MultiIndex.from_product(
            [[mes], np.sort(del_mes["shop_id"].unique()),
             np.sort(del_mes["item_id"].unique())],
            names=CLAVES,
        )
        bloques.append(indice.to_frame(index=False))
    malla = pd.concat(bloques, ignore_index=True).astype(
        {"date_block_num": "int16", "shop_id": "int16", "item_id": "int32"}
    )
    malla = malla.merge(mensual[CLAVES + ["item_cnt_month"]], on=CLAVES,
                        how="left")
    malla["item_cnt_month"] = recortar_target(
        malla["item_cnt_month"].fillna(0)).astype("float32")
    return malla


def calendario(n_meses: int = MES_TEST + 1) -> pd.DataFrame:
    """Atributos de calendario de cada ``date_block_num``.

    Args:
        n_meses: Número de meses desde enero 2013 (0 a ``n_meses - 1``).

    Returns:
        DataFrame indexado por ``date_block_num`` con ``mes``,
        ``dias_mes`` y ``fines_semana`` (sábados y domingos del mes).
    """
    inicio = pd.date_range("2013-01-01", periods=n_meses, freq="MS")
    filas = []
    for bloque, primer_dia in enumerate(inicio):
        dias = pd.date_range(primer_dia, primer_dia + pd.offsets.MonthEnd(0),
                             freq="D")
        filas.append({"date_block_num": bloque, "mes": primer_dia.month,
                      "dias_mes": len(dias),
                      "fines_semana": int((dias.day_of_week >= 5).sum())})
    return pd.DataFrame(filas).set_index("date_block_num")
