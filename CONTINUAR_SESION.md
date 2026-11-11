# Predict Future Sales: estado del proyecto y próximos pasos

> Documento de traspaso entre sesiones (plantilla de papd-product-development).
> Léelo al inicio de cada sesión y actualízalo al cerrar cada fase.

## Objetivo

Predecir las unidades vendidas por cada par tienda-producto en noviembre de 2015 (`date_block_num = 34`), con las ventas diarias de 1C Company (Kaggle, *Predict Future Sales*). El problema es de regresión (forecasting mensual). El target `item_cnt_month` se construye sumando `item_cnt_day` por mes, tienda y producto. Kaggle lo recorta a [0, 20] y evalúa con RMSE; **el recorte y la métrica final se deciden en la fase 2** (decisión del usuario).

La consigna de cada fase está en los `.docx` de la carpeta superior (`../Prompt Proyecto Fase 1 Exploratory Data Analysis.docx`).

## Estado actual

- ✅ Estructura del proyecto con Cookiecutter Data Science v2 (`ccds`, paquete `future_sales`, licencia MIT).
- ✅ Entorno conda `env_future_sales` (Python 3.11, OpenBLAS), con su kernel registrado.
- ✅ Análisis exploratorio en `notebooks/01_exploratory_data_analysis.ipynb`, ejecutado en VS Code y versionado con sus salidas.
- ✅ Traducciones manuales de tiendas y categorías en `data/external/`.
- ✅ Repositorio git (`main`) publicado en GitHub (público): https://github.com/robfarfan/future-sales-rf-pd-2026
- ⬜ Feature engineering (fase 2).
- ⬜ Entrenamiento, inferencia, MLflow y API.

## Estructura relevante

```
future-sales/
├── data/raw/                  <- CSV originales de Kaggle (ignorados en git)
├── data/external/             <- shops_traducido.csv, item_categories_traducido.csv (versionados)
├── future_sales/config.py     <- rutas (RAW_DATA_DIR, EXTERNAL_DATA_DIR, ...)
├── notebooks/01_exploratory_data_analysis.ipynb
├── tests/test_data.py         <- verifica que los CSV crudos existan y tengan sus columnas
├── scripts/                   <- (vacío) scripts de inferencia, después
├── requirements.txt           <- versiones fijadas, en orden alfabético
├── pyproject.toml             <- requires-python >=3.11, pytest, ruff a 79
└── CONTINUAR_SESION.md
```

Los CSV originales también siguen en `../1C Company/Dataset/` como respaldo.

## Datos

| Archivo | Filas | Contenido |
|---|---|---|
| sales_train.csv | 2,935,849 | Ventas diarias ene 2013 a oct 2015 (34 meses, 1,034 días) |
| items.csv | 22,170 | Productos (`item_name` en ruso, `item_category_id`) |
| item_categories.csv | 84 | Categorías (en ruso) |
| shops.csv | 60 | Tiendas (en ruso) |
| test.csv | 214,200 | 42 tiendas × 5,100 productos; índice `ID` |

## Entorno

- **Nombre:** `env_future_sales`, Python 3.11.18.
- **Python del entorno:** `C:\Users\rfarfan\AppData\Local\anaconda3\envs\env_future_sales\python.exe`.
- **Kernel de Jupyter:** "Python (env_future_sales)". En VS Code, selecciónalo también como intérprete del workspace.
- **Instalar una librería:** `conda install -n env_future_sales -c conda-forge <paquete>`, agregarla a `requirements.txt` con `==<versión>` y verificar que `libblas` siga en OpenBLAS.
- En conda-forge, feature-engine se llama `feature_engine`.

## Convenciones y forma de trabajo

- Nombre del notebook del EDA: el que pide la consigna (`01_exploratory_data_analysis.ipynb`). **Para los siguientes notebooks, pregunta si se sigue `02_...` o el convenio CCDS.**
- Cada paso del notebook lleva su título y una descripción, y su código va en su propia celda. Los notebooks se generan por script fuera de VS Code; después, el usuario hace "File: Revert File".
- Preguntar las decisiones metodológicas, no asumir.
- Commits solo cuando el usuario lo pida, sin línea de atribución.

## Notebook 01: análisis exploratorio

Sigue los 10 pasos de la consigna.

**Decisiones del usuario:**

| Tema | Decisión |
|---|---|
| Texto cirílico | Traducción manual de 60 tiendas y 84 categorías (con ciudad, tipo de tienda, categoría principal, subcategoría y `es_digital`); `item_name` queda en ruso, como fuente de features de texto |
| Duplicados | Eliminar los 6 duplicados exactos; conservar los 22 parciales (misma fecha-tienda-producto, otro precio) |
| Tiendas repetidas | Unificar 0→57, 1→58, 11→10 (IDs de test); 39 y 40 quedan separadas (venden en paralelo) |
| Inválidos | Eliminar el precio ≤ 0 (1 fila), el precio > 100,000 (1) y la cantidad > 1,000 (1) |
| Devoluciones | Conservar y netear al agregar al mes (7,356 filas) |
| Target | Recorte [0, 20] y métrica: se deciden en la fase 2 |

En el EDA la limpieza se aplica **en memoria**; hay que implementarla en la fase 2.

**Resultados principales:**

- No hay nulos ni días faltantes, y `date_block_num` es consistente. Frecuencia elegida: **mensual** (densidad tienda-producto: 0.67 % diaria, 11.2 % mensual).
- Tendencia descendente: -17 % de unidades en 2014 y -26 % en 2015 (ene-oct). La serie mensual no es estacionaria (ADF p = 0.15; en primera diferencia, p < 0.001).
- Estacionalidad: diciembre con índice 1.64; noviembre vendió 2 % y 10 % más que octubre en 2013 y 2014; el sábado tiene índice 1.41.
- ACF/PACF: el total mensual se comporta como un AR(1). Tiendas, productos y pares, también con el rezago 1. El rezago 12 sube (0.26) pero no es significativo con 34 meses. En la serie diaria hay un ciclo semanal fuerte.
- Cobertura de test: 52 % de los pares con historial, 41 % con el producto vendido en otra tienda, 7 % de productos nuevos (363). Solo el 35 % de los pares vendió algo en los últimos 6 meses.
- Malla de los meses 22 a 33: 85 % de ceros.

Relación con el target (malla de los meses 22-33):

| Variable | Métrica | Target recortado | Sin recortar |
|---|---|---|---|
| ventas_lag_1 | r | 0.34 | 0.55 |
| item_category_id | η | 0.32 | 0.35 |
| ventas_lag_2 | r | 0.29 | 0.45 |
| ventas_lag_3 | r | 0.26 | 0.43 |
| ventas_lag_12 | r | 0.20 | 0.51 |
| shop_id | η | 0.18 | 0.07 |
| categoria_principal | η | 0.16 | 0.08 |
| ciudad | η | 0.13 | 0.05 |
| meses_desde_lanzamiento | r | -0.09 | -0.03 |
| log_precio_mediano | r | 0.06 | 0.01 |
| es_digital | r | -0.06 | -0.02 |
| mes_del_anio | η | 0.05 | 0.02 |
| tipo_tienda | η | 0.02 | 0.01 |

**Nota sobre `pipeline_config.yaml` / `MODEL_FEATURES`:** en este proyecto las features no son columnas crudas que se seleccionan, sino variables construidas (rezagos, promedios con rezago, calendario). Por eso el EDA no generó `MODEL_FEATURES`. Hay que decidir con el usuario cómo adaptar esa parte de la arquitectura en la fase 2.

## Git

- Remoto `origin`: https://github.com/robfarfan/future-sales-rf-pd-2026.git (público), rama `main`.
- El `.gitignore` ignora `/data/**` (salvo los `.gitkeep` y `data/external/*_traducido.csv`) y `/models/*`.
- Las salidas de los notebooks muestran rutas locales (el log de `PROJ_ROOT`); el usuario está al tanto.
- Las siguientes fases van en ramas `feature/<tema>`; el usuario decide cuándo se integran a `main`.

## Pendientes y decisiones abiertas

- Fase 2: recorte del target, construcción de la malla mensual, rezagos y promedios con rezago, features de texto de `item_name`, banderas de tiendas y productos especiales, tratamiento de los productos nuevos, esquema de validación temporal (por ejemplo, validar con el mes 33) y cómo encaja en el pipeline de sklearn.
