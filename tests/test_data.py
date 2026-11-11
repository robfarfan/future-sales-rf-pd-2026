"""Pruebas de disponibilidad y estructura de los datos crudos."""

import pandas as pd
import pytest

from future_sales.config import RAW_DATA_DIR

COLUMNAS_ESPERADAS: dict[str, list[str]] = {
    "sales_train.csv": [
        "date", "date_block_num", "shop_id", "item_id",
        "item_price", "item_cnt_day",
    ],
    "items.csv": ["item_name", "item_id", "item_category_id"],
    "item_categories.csv": ["item_category_name", "item_category_id"],
    "shops.csv": ["shop_name", "shop_id"],
    "test.csv": ["ID", "shop_id", "item_id"],
}


@pytest.mark.parametrize("archivo", sorted(COLUMNAS_ESPERADAS))
def test_columnas_datos_crudos(archivo: str) -> None:
    """Cada CSV crudo existe y trae las columnas esperadas.

    Args:
        archivo: nombre del CSV dentro de ``data/raw``.
    """
    ruta = RAW_DATA_DIR / archivo
    assert ruta.exists(), f"Falta {ruta}; descárgalo de Kaggle."
    columnas = pd.read_csv(ruta, nrows=0).columns.tolist()
    assert columnas == COLUMNAS_ESPERADAS[archivo]
