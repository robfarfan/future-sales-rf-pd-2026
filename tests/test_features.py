"""Pruebas de los operadores Mapper y GeneradorLags."""

import numpy as np
import pandas as pd
import pytest

from future_sales.dataset import calendario, construir_malla, recortar_target
from future_sales.features import GeneradorLags, Mapper, recortar_prediccion


@pytest.fixture
def historial() -> pd.DataFrame:
    """Ventas mensuales sintéticas de los meses 0 a 3."""
    return pd.DataFrame({
        "date_block_num": [0, 1, 1, 2, 3, 3],
        "shop_id": [5, 5, 6, 5, 5, 6],
        "item_id": [10, 10, 10, 20, 10, 20],
        "item_category_id": [1, 1, 1, 2, 1, 2],
        "item_cnt_month": [3.0, 4.0, 1.0, -1.0, 7.0, 2.0],
        "precio_medio": [100.0, 110.0, 90.0, 50.0, 120.0, 55.0],
    })


def generar(historial: pd.DataFrame, filas: pd.DataFrame,
            **lags: tuple[int, ...]) -> pd.DataFrame:
    """Ajusta un GeneradorLags y transforma las filas."""
    parametros = {"lags_par": (1, 2), "lags_item": (1,),
                  "lags_tienda": (1,), "lags_categoria": (1,),
                  "lags_tienda_categoria": (1,)}
    parametros.update(lags)
    return GeneradorLags(historial, **parametros).fit(filas).transform(
        filas)


def test_mapper_agrega_columnas_y_deja_nan_en_desconocidos() -> None:
    """Los valores fuera del diccionario quedan como NaN."""
    datos = pd.DataFrame({"shop_id": [1, 2, 99]})
    mapper = Mapper({"ciudad": ("shop_id", {1: "Moscú", 2: "Omsk"}),
                     "es_moscu": ("ciudad", {"Moscú": 1, "Omsk": 0})})
    salida = mapper.fit_transform(datos)
    assert salida["ciudad"].tolist()[:2] == ["Moscú", "Omsk"]
    assert pd.isna(salida.loc[2, "ciudad"])
    assert salida["es_moscu"].tolist()[:2] == [1, 0]
    assert list(mapper.get_feature_names_out()) == [
        "shop_id", "ciudad", "es_moscu"]


def test_mapper_falla_si_falta_el_origen() -> None:
    """fit avisa cuando no existe la columna de origen."""
    with pytest.raises(KeyError):
        Mapper({"x": ("no_existe", {})}).fit(pd.DataFrame({"a": [1]}))


def test_lags_del_par_y_agregados(historial: pd.DataFrame) -> None:
    """Los lags toman el valor del mes m - k, con 0 si no hubo venta."""
    filas = pd.DataFrame({"date_block_num": [2, 4], "shop_id": [5, 5],
                          "item_id": [10, 10], "item_category_id": [1, 1]})
    salida = generar(historial, filas)
    # Mes 2: lag 1 = mes 1 (4), lag 2 = mes 0 (3).
    assert salida.loc[0, "ventas_par_lag_1"] == 4.0
    assert salida.loc[0, "ventas_par_lag_2"] == 3.0
    # Producto 10 en el mes 1: 4 (tienda 5) + 1 (tienda 6).
    assert salida.loc[0, "ventas_item_lag_1"] == 5.0
    # Mes 4 (como el mes de test): lag 1 = mes 3 (7); lag 2 = mes 2, sin
    # venta del par → 0.
    assert salida.loc[1, "ventas_par_lag_1"] == 7.0
    assert salida.loc[1, "ventas_par_lag_2"] == 0.0
    # Tienda 5 en el mes 3: 7 unidades.
    assert salida.loc[1, "ventas_tienda_lag_1"] == 7.0


def test_lags_fuera_del_historial_son_nan(historial: pd.DataFrame) -> None:
    """Antes del mes 0 o después del último mes, el lag es NaN."""
    filas = pd.DataFrame({"date_block_num": [1, 6], "shop_id": [5, 5],
                          "item_id": [10, 10], "item_category_id": [1, 1]})
    salida = generar(historial, filas)
    assert np.isnan(salida.loc[0, "ventas_par_lag_2"])   # mes -1
    assert salida.loc[0, "ventas_par_lag_1"] == 3.0       # mes 0
    assert np.isnan(salida.loc[1, "ventas_par_lag_1"])   # mes 5 > 3


def test_sin_fuga_de_informacion(historial: pd.DataFrame) -> None:
    """Quitar del historial el mes de la fila y los siguientes no cambia
    las features."""
    filas = pd.DataFrame({"date_block_num": [3, 3], "shop_id": [5, 6],
                          "item_id": [10, 20], "item_category_id": [1, 2]})
    completo = generar(historial, filas)
    recortado = generar(historial[historial["date_block_num"] < 3], filas)
    pd.testing.assert_frame_equal(completo, recortado)


def test_precio_y_antiguedad(historial: pd.DataFrame) -> None:
    """Precio previo y antigüedad usan solo meses anteriores."""
    filas = pd.DataFrame({"date_block_num": [0, 2, 3],
                          "shop_id": [5, 6, 6],
                          "item_id": [10, 10, 20],
                          "item_category_id": [1, 1, 2]})
    salida = generar(historial, filas)
    # Mes 0: el producto no se había vendido → NaN.
    assert np.isnan(salida.loc[0, "precio_item_previo"])
    assert np.isnan(salida.loc[0, "meses_desde_lanzamiento_item"])
    # Mes 2, producto 10: último precio medio conocido = mes 1 (100).
    assert salida.loc[1, "precio_item_previo"] == pytest.approx(100.0)
    assert salida.loc[1, "meses_desde_lanzamiento_item"] == 2.0
    assert salida.loc[1, "meses_desde_primera_venta_par"] == 1.0
    # Mes 3, par (6, 20): su primera venta es el mes 3 → NaN.
    assert np.isnan(salida.loc[2, "meses_desde_primera_venta_par"])


def test_generador_sin_historial_falla() -> None:
    """fit avisa si no recibe historial."""
    with pytest.raises(ValueError):
        GeneradorLags().fit(pd.DataFrame({"date_block_num": [1]}))


def test_malla_y_recorte(historial: pd.DataFrame) -> None:
    """La malla combina tiendas y productos del mes y recorta a [0, 20]."""
    malla = construir_malla(historial, [3])
    assert len(malla) == 4                       # 2 tiendas × 2 productos
    assert malla["item_cnt_month"].between(0, 20).all()
    assert recortar_target(np.array([-3.0, 25.0])).tolist() == [0.0, 20.0]
    assert recortar_prediccion(np.array([21.0]))[0] == 20.0


def test_calendario() -> None:
    """Noviembre 2015 (mes 34): 30 días y 9 días de fin de semana."""
    cal = calendario()
    assert cal.loc[34, "mes"] == 11
    assert cal.loc[34, "dias_mes"] == 30
    assert cal.loc[34, "fines_semana"] == 9
