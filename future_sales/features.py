"""Operadores de ingeniería de características compatibles con sklearn.

- :class:`Mapper`: agrega columnas mapeando otra columna con un
  diccionario (por ejemplo ``shop_id`` → ``ciudad``).
- :class:`GeneradorLags`: agrega las ventas de meses anteriores (lags)
  a partir de un historial mensual, sin fuga de información.

Más funciones auxiliares para el pipeline completo.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from future_sales.dataset import recortar_target

Mapeos = dict[str, tuple[str, dict[Any, Any]]]

# Multiplicadores para combinar IDs en una sola llave entera.
_M_TIENDA = 100        # shop_id < 100
_M_ITEM = 100_000      # item_id < 100,000
_M_CATEGORIA = 1_000   # item_category_id < 1,000

COLUMNAS_HISTORIAL = ["date_block_num", "shop_id", "item_id",
                      "item_category_id", "item_cnt_month",
                      "precio_medio"]


class Mapper(BaseEstimator, TransformerMixin):
    """Agrega columnas nuevas mapeando valores con diccionarios.

    Cada entrada de ``mapeos`` es ``nueva: (origen, diccionario)``. Los
    mapeos se aplican en orden, así que una columna creada puede servir
    de origen para la siguiente. Los valores que no están en el
    diccionario quedan como NaN, para que un imputador posterior los
    trate (por ejemplo, una tienda nueva sin ciudad conocida).

    Args:
        mapeos: Diccionario ``{nueva: (origen, {valor: mapeado})}``.
    """

    def __init__(self, mapeos: Mapeos | None = None) -> None:
        self.mapeos = mapeos

    def fit(self, X: pd.DataFrame, y: Any = None) -> "Mapper":
        """Verifica que existan las columnas de origen.

        Args:
            X: Datos de entrada.
            y: Ignorado.

        Returns:
            El propio transformador.

        Raises:
            KeyError: Si falta una columna de origen.
        """
        disponibles = set(X.columns)
        for nueva, (origen, _) in (self.mapeos or {}).items():
            if origen not in disponibles:
                raise KeyError(f"Mapper: falta la columna '{origen}'")
            disponibles.add(nueva)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Agrega las columnas mapeadas.

        Args:
            X: Datos de entrada.

        Returns:
            Copia de ``X`` con las columnas nuevas al final.
        """
        X = X.copy()
        for nueva, (origen, diccionario) in (self.mapeos or {}).items():
            X[nueva] = X[origen].map(diccionario)
        return X

    def get_feature_names_out(self, input_features: Any = None
                              ) -> np.ndarray:
        """Nombres de las columnas de salida."""
        nuevas = [c for c in (self.mapeos or {})
                  if c not in self.feature_names_in_]
        return np.asarray(list(self.feature_names_in_) + nuevas,
                          dtype=object)


class GeneradorLags(BaseEstimator, TransformerMixin):
    """Agrega ventas de meses anteriores (lags) desde un historial.

    El historial es la tabla mensual de ventas (salida de
    ``future_sales.dataset.ventas_mensuales``). Para una fila del mes
    ``m``, el lag ``k`` usa solo el mes ``m - k``; nunca el mes de la
    fila ni posteriores, así que no hay fuga de información:

    - si ``m - k`` está dentro del historial y no hubo venta, vale 0;
    - si ``m - k`` está fuera del historial (antes del mes 0 o después
      del último mes conocido), vale NaN y se imputa después.

    Además agrega el último precio medio conocido del producto y los
    meses transcurridos desde la primera venta del producto y del par
    (NaN si todavía no se había vendido).

    Requiere en ``X`` las columnas ``date_block_num``, ``shop_id``,
    ``item_id`` y, si hay lags por categoría, ``item_category_id``.

    Args:
        historial: Ventas mensuales con ``COLUMNAS_HISTORIAL``.
        lags_par: Lags de las unidades del par tienda-producto.
        lags_item: Lags de las unidades del producto en todas las
            tiendas.
        lags_tienda: Lags de las unidades totales de la tienda.
        lags_categoria: Lags de las unidades de la categoría.
        lags_tienda_categoria: Lags de las unidades de la categoría en
            la tienda.
    """

    def __init__(
        self,
        historial: pd.DataFrame | None = None,
        lags_par: Sequence[int] = (1, 2, 3),
        lags_item: Sequence[int] = (1,),
        lags_tienda: Sequence[int] = (1,),
        lags_categoria: Sequence[int] = (1,),
        lags_tienda_categoria: Sequence[int] = (1,),
    ) -> None:
        self.historial = historial
        self.lags_par = lags_par
        self.lags_item = lags_item
        self.lags_tienda = lags_tienda
        self.lags_categoria = lags_categoria
        self.lags_tienda_categoria = lags_tienda_categoria

    # ------------------------------------------------------------ llaves
    @staticmethod
    def _llave(nivel: str, mes: np.ndarray, tienda: np.ndarray,
               item: np.ndarray, categoria: np.ndarray) -> np.ndarray:
        """Combina mes e IDs en una llave entera por nivel."""
        if nivel == "par":
            return (mes * _M_TIENDA + tienda) * _M_ITEM + item
        if nivel == "item":
            return mes * _M_ITEM + item
        if nivel == "tienda":
            return mes * _M_TIENDA + tienda
        if nivel == "categoria":
            return mes * _M_CATEGORIA + categoria
        return (mes * _M_TIENDA + tienda) * _M_CATEGORIA + categoria

    def _niveles(self) -> dict[str, Sequence[int]]:
        return {"par": self.lags_par, "item": self.lags_item,
                "tienda": self.lags_tienda,
                "categoria": self.lags_categoria,
                "tienda_categoria": self.lags_tienda_categoria}

    # --------------------------------------------------------------- fit
    def fit(self, X: pd.DataFrame, y: Any = None) -> "GeneradorLags":
        """Precalcula las tablas de búsqueda a partir del historial.

        Args:
            X: Datos de entrada (solo se registran sus columnas).
            y: Ignorado.

        Returns:
            El propio transformador.

        Raises:
            ValueError: Si no hay historial o le faltan columnas.
        """
        if self.historial is None:
            raise ValueError("GeneradorLags necesita un historial")
        faltan = set(COLUMNAS_HISTORIAL) - set(self.historial.columns)
        if faltan:
            raise ValueError(f"Al historial le faltan: {sorted(faltan)}")

        h = self.historial
        mes = h["date_block_num"].to_numpy(np.int64)
        tienda = h["shop_id"].to_numpy(np.int64)
        item = h["item_id"].to_numpy(np.int64)
        categoria = h["item_category_id"].to_numpy(np.int64)
        unidades = h["item_cnt_month"].to_numpy(np.float64)

        self.ultimo_mes_ = int(mes.max())
        self.tablas_ = {}
        for nivel, lags in self._niveles().items():
            if lags:
                llave = self._llave(nivel, mes, tienda, item, categoria)
                self.tablas_[nivel] = (pd.Series(unidades).groupby(llave)
                                       .sum().astype("float32"))

        # Último precio medio conocido del producto antes de cada mes.
        precios = (h.groupby(["item_id", "date_block_num"])["precio_medio"]
                   .mean().unstack()
                   .reindex(columns=range(self.ultimo_mes_ + 1)))
        previos = precios.ffill(axis=1).shift(1, axis=1)
        previos[self.ultimo_mes_ + 1] = precios.ffill(axis=1)[
            self.ultimo_mes_]
        previos = previos.stack().astype("float32")
        self.precio_previo_ = pd.Series(
            previos.to_numpy(),
            index=(previos.index.get_level_values(1).to_numpy(np.int64)
                   * _M_ITEM
                   + previos.index.get_level_values(0).to_numpy(np.int64)),
        )

        self.primer_mes_item_ = h.groupby("item_id")["date_block_num"].min()
        primer_par = h.groupby(["shop_id", "item_id"])["date_block_num"].min()
        self.primer_mes_par_ = pd.Series(
            primer_par.to_numpy(),
            index=(primer_par.index.get_level_values(0).to_numpy(np.int64)
                   * _M_ITEM
                   + primer_par.index.get_level_values(1).to_numpy(np.int64)),
        )
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    # --------------------------------------------------------- transform
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Agrega las columnas de lags, precio y antigüedad.

        Args:
            X: Filas con ``date_block_num``, ``shop_id``, ``item_id`` y
                (si aplica) ``item_category_id``.

        Returns:
            Copia de ``X`` con las columnas nuevas al final.
        """
        X = X.copy()
        mes = X["date_block_num"].to_numpy(np.int64)
        tienda = X["shop_id"].to_numpy(np.int64)
        item = X["item_id"].to_numpy(np.int64)
        if "item_category_id" in X:
            cat_float = X["item_category_id"].to_numpy(np.float64)
        else:
            cat_float = np.full(len(X), np.nan)
        sin_categoria = np.isnan(cat_float)
        categoria = np.where(sin_categoria, 0, cat_float).astype(np.int64)

        for nivel, lags in self._niveles().items():
            for k in lags:
                mes_k = mes - k
                llave = self._llave(nivel, mes_k, tienda, item, categoria)
                valores = (self.tablas_[nivel].reindex(llave)
                           .to_numpy(np.float64))
                disponible = (mes_k >= 0) & (mes_k <= self.ultimo_mes_)
                valores = np.where(np.isnan(valores) & disponible, 0.0,
                                   valores)
                valores[~disponible] = np.nan
                if nivel in ("categoria", "tienda_categoria"):
                    valores[sin_categoria] = np.nan
                X[f"ventas_{nivel}_lag_{k}"] = valores.astype("float32")

        X["precio_item_previo"] = (self.precio_previo_
                                   .reindex(mes * _M_ITEM + item)
                                   .to_numpy(np.float32))
        primero_item = (self.primer_mes_item_.reindex(item)
                        .to_numpy(np.float64))
        primero_par = (self.primer_mes_par_
                       .reindex(tienda * _M_ITEM + item)
                       .to_numpy(np.float64))
        for nombre, primero in [("meses_desde_lanzamiento_item",
                                 primero_item),
                                ("meses_desde_primera_venta_par",
                                 primero_par)]:
            # Solo cuenta si la primera venta fue antes del mes de la fila.
            meses = np.where(primero < mes, mes - primero, np.nan)
            X[nombre] = meses.astype("float32")
        return X

    def get_feature_names_out(self, input_features: Any = None
                              ) -> np.ndarray:
        """Nombres de las columnas de salida."""
        nuevas = [f"ventas_{nivel}_lag_{k}"
                  for nivel, lags in self._niveles().items() for k in lags]
        nuevas += ["precio_item_previo", "meses_desde_lanzamiento_item",
                   "meses_desde_primera_venta_par"]
        return np.asarray(list(self.feature_names_in_) + nuevas,
                          dtype=object)


def construir_mapeos(catalogos: dict[str, pd.DataFrame],
                     calendario: pd.DataFrame) -> Mapeos:
    """Arma los diccionarios del :class:`Mapper` del proyecto.

    Args:
        catalogos: Salida de ``future_sales.dataset.cargar_catalogos``.
        calendario: Salida de ``future_sales.dataset.calendario``.

    Returns:
        Mapeos de producto → categoría, categoría → grupo y formato
        digital, tienda → ciudad y tipo, y mes → atributos de
        calendario.
    """
    items = catalogos["items"].set_index("item_id")
    categorias = catalogos["categorias"].set_index("item_category_id")
    tiendas = catalogos["tiendas"].set_index("shop_id")
    return {
        "item_category_id": ("item_id",
                             items["item_category_id"].to_dict()),
        "categoria_principal": ("item_category_id",
                                categorias["categoria_principal"]
                                .to_dict()),
        "es_digital": ("item_category_id",
                       categorias["es_digital"].to_dict()),
        "ciudad": ("shop_id", tiendas["ciudad"].to_dict()),
        "tipo_tienda": ("shop_id", tiendas["tipo_tienda"].to_dict()),
        "mes": ("date_block_num", calendario["mes"].to_dict()),
        "dias_mes": ("date_block_num", calendario["dias_mes"].to_dict()),
        "fines_semana": ("date_block_num",
                         calendario["fines_semana"].to_dict()),
    }


def identidad(y: np.ndarray) -> np.ndarray:
    """Función identidad para ``TransformedTargetRegressor``."""
    return y


def recortar_prediccion(y: np.ndarray) -> np.ndarray:
    """Recorta las predicciones al rango [0, 20] del target."""
    return recortar_target(y)
