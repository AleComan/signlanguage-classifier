# signlanguage-classifier

**URL del proyecto:** 

[![GitHub](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/AleComan/signlanguage-classifier)

Proyecto base para clasificacion y generacion de imagenes de lenguaje de signos con cuatro pipelines:

1. Baseline de ML clasico (features profundas + clasificador de scikit-learn).
2. CNN entrenada desde cero (sin pesos preentrenados).
3. Fine-tuning de modelo preentrenado (congelacion parcial configurable).
4. Generacion condicional ASL (`Clase -> Imagen`) con cGAN y secuenciacion de frases a GIF.

## Índice

- [Estructura del proyecto](#estructura-del-proyecto)
- [Dataset ASL Alphabet](#dataset-asl-alphabet)
- [Requisitos](#requisitos)
- [Entorno (Conda)](#entorno-conda)
- [Variables de entorno](#variables-de-entorno)
- [Ejecutar pipelines (notebooks)](#ejecutar-pipelines-notebooks)
- [Pipeline generativo ASL](#pipeline-generativo-asl)
- [Resultados experimentales](#resultados-experimentales)
  - [Tabla comparativa de modelos](#tabla-comparativa-de-modelos)
  - [Resultado del baseline ML](#resultado-del-baseline-ml)
- [Conclusiones (Accuracy, Overfitting y Despliegue)](#conclusiones-accuracy-overfitting-y-despliegue)
  - [1) ¿Cuál es el mejor modelo por accuracy?](#1-cuál-es-el-mejor-modelo-por-accuracy)
  - [2) ¿Generaliza bien o está sobreajustado?](#2-generaliza-bien-o-está-sobreajustado)
  - [3) ¿Lo desplegaríamos ya a producción?](#3-lo-desplegaríamos-ya-a-producción)
- [Interfaz Streamlit](#interfaz-streamlit)
  - [Ejecutar app Streamlit](#ejecutar-app-streamlit)
  - [Flujo recomendado](#flujo-recomendado)
- [Autores del proyecto](#-autores-del-proyecto)

## Estructura del proyecto

```text
signlanguage-classifier/
|-- app/
|   `-- streamlit_app.py
|-- configs/
|   |-- baseline_ml.yaml
|   |-- dataset_asl.yaml
|   |-- finetune.yaml
|   |-- generation.yaml
|   `-- scratch_cnn.yaml
|-- notebooks/
|   |-- 01_eda.ipynb
|   |-- 02_baseline_ml.ipynb
|   |-- 03_scratch_cnn.ipynb
|   |-- 04_finetune.ipynb
|   `-- 05_generation_train.ipynb
|-- scripts/
|   `-- prepare_dataset.py
|-- src/
|   |-- data/
|   |-- eda/
|   |-- evaluation/
|   |-- features/
|   |-- inference/
|   |-- models/
|   |-- training/
|   `-- utils/
|-- tests/
|-- README.md
|-- TASKS.md
`-- requirements.txt
```

## Dataset ASL Alphabet

El dataset utilizado es el **ASL Alphabet** ([American Sign Language](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)), una colección de imágenes diseñada para la clasificación de los gestos manuales del alfabeto de la lengua de signos americana. 

El conjunto consta de 29 clases en total:
- **26 letras** (A-Z).
- **3 clases especiales:** *SPACE* (espacio), *DELETE* (borrar) y *NOTHING* (ningún gesto, fondo vacío).

A continuación se muestra un ejemplo de las imágenes que componen el dataset:

Ejemplo: 

![Muestra del dataset ASL](./docs/asl_sample.jpg)

Estructura esperada del dataset raw principal:

```text
data/asl_alphabet_v1/raw/
`-- asl_alphabet_train/
    |-- A/
    |-- B/
    |-- ...
    |-- del/
    |-- nothing/
    `-- space/
```

Preprocesado a formato `ImageFolder` con split estratificado por clase:

```bash
conda activate DL; python scripts/prepare_dataset.py --config configs/dataset_asl.yaml
```

Salida generada:
- `data/asl_alphabet_v1/processed/train|val/<label>/...`
- `data/asl_alphabet_v1/processed/metadata.csv`
- `data/asl_alphabet_v1/processed/summary.json`

Notas:
- Se renombra automaticamente `del` -> `delete` (configurable en `configs/dataset_asl.yaml`).
- El split por defecto es 85/15 desde `asl_alphabet_train`.
- La evaluación real se ha llevado a cabo con fotos propias o webcam, fuera del split train/val.

## Requisitos

- Python 3.10+ recomendado.
- Dataset procesado en estructura tipo `ImageFolder` de `torchvision`:

```text
data/
|-- train/
|   |-- class_a/
|   `-- class_b/
`-- val/
    |-- class_a/
    `-- class_b/
```

## Entorno (Conda)

```bash
conda activate DL
```

Este repositorio asume que ya tienes las dependencias instaladas en tu entorno `DL`.
`requirements.txt` se mantiene como referencia para reproducibilidad.

## Variables de entorno

1. Crea un archivo `.env` en la raíz del proyecto.
2. Ajusta las variables necesarias.

Ejemplo:

```env
WANDB_API_KEY=tu_api_key
WANDB_ENTITY=tu_usuario_o_equipo
WANDB_PROJECT=signlanguage-classifier
# Opcional: modelo discriminativo por defecto en Streamlit
MODEL_PATH=artifacts/baseline_ml/baseline_model.joblib
# Opcional: generador por defecto para la pestana Generacion
GENERATOR_MODEL_PATH=artifacts/generation/conditional_gan.pt
```

## Ejecutar pipelines (notebooks)

Lanza Jupyter desde la raíz del proyecto para que los notebooks puedan importar `src/` automaticamente:

```bash
conda activate DL; jupyter lab
```

Notebooks disponibles en `notebooks/`:

- `01_eda.ipynb` - exploración del dataset (clases, tamanos, color, muestras).
- `02_baseline_ml.ipynb` - features de ResNet18 + clasificadores sklearn (LogReg, SVM, RF) y comparativa.
- `03_scratch_cnn.ipynb` - `SimpleCNN` entrenada desde cero con curvas y evaluación final.
- `04_finetune.ipynb` - fine-tuning de ResNet18 con freezing parcial y LRs discriminativos.
- `05_generation_train.ipynb` - cGAN condicional, visualizacion de muestras y prueba `frase -> GIF`.

Cada notebook lee su YAML de `configs/`, guarda los artefactos en `artifacts/<pipeline>/` y, si `tracking.use_wandb` esta activo y hay `WANDB_API_KEY`, registra metricas en Weights & Biases.

Artefactos habituales:
- `artifacts/baseline_ml/baseline_model.joblib` (baseline ML con scaler + class_names).
- checkpoints Torch en `artifacts/...` (`.pt`, `.pth`, `.ckpt`) para los pipelines deep learning.
- `artifacts/generation/conditional_gan.pt` (generador condicional ASL).
- `artifacts/generation/samples/epoch_XXXX.png` (muestras con fixed noise para inspeccion visual).

## Pipeline generativo ASL

El proyecto incluye una inversion generica del flujo discriminativo:

```text
Clase ASL -> cGAN condicional -> Imagen sintetica
Frase -> tokens ASL -> frames -> GIF
```

Clases soportadas:
- `A` a `Z`
- `delete`
- `nothing`
- `space`

Estas clases coinciden con la salida de `scripts/prepare_dataset.py`, incluyendo el renombrado `del` -> `delete`.

Entrenamiento por CLI:

```bash
conda activate DL; python -m src.training.gan_trainer --config configs/generation.yaml
```

Tambien puedes usar `notebooks/05_generation_train.ipynb`, que documenta el pipeline paso a paso:
- lectura de `configs/generation.yaml`,
- comprobacion del dataset procesado,
- visualizacion de un lote real,
- overrides opcionales para pruebas rapidas,
- entrenamiento de la cGAN,
- revision de muestras generadas,
- prueba del motor `generate_phrase_sequence()`.

Tracking:
- W&B registra `generator_loss`, `discriminator_loss` y muestras generadas con fixed noise.
- `configs/generation.yaml` incluye un bloque `metrics` para activar Inception Score y FID de forma opcional.
- Las metricas generativas estan desactivadas por defecto porque son mas lentas y pueden requerir pesos InceptionV3 de `torchvision`.

Inferencia generativa:
- `src/inference/generator.py` expone `generate_phrase_sequence(phrase, frame_duration)`.
- La funcion tokeniza por caracteres, ignora simbolos no soportados y convierte espacios en la clase `space`.
- Si existe un checkpoint generativo, genera imagenes con la cGAN.
- Si no existe checkpoint, usa imagenes del dataset como fallback y, en ultimo caso, frames placeholder deterministas.

## Resultados experimentales

### Tabla comparativa de modelos

[![Weights & Biases](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white)](https://wandb.ai/adne-image-classification/image-classification)

> **Nota Metodológica:** Para maximizar el volumen de datos de entrenamiento, se decidió estratégicamente prescindir de un *test split* clásico del dataset original. En su lugar, la capacidad de generalización ha sido validada cualitativa y empíricamente mediante pruebas directas en la aplicación con imágenes propias, evaluando el modelo desde cero frente a entornos reales (variaciones de iluminación, distintas resoluciones, etc.). 
> 
> La siguiente tabla muestra los mejores modelos considerando todas las dimensiones. Además del *accuracy*, la selección está respaldada por métricas avanzadas (F1-score, Recall, Precision) monitorizadas en Weights & Biases para garantizar el rendimiento equitativo en todas las clases. Los tiempos y épocas varían para asegurar la correcta convergencia de cada arquitectura.

| Modelo / Run | Familia | Val Accuracy | Train Accuracy | Gap (train-val) | Val F1-Score | Runtime |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `finetune-resnet18` (10 epochs) | Transfer learning (ResNet18) | **0.9986** | 0.9987 | +0.0001 | 0.9985 | 10152 s (~169 min) |
| `scratch-cnn-trial-03-10-epochs` | CNN desde cero | 0.9955 | 0.9968 | +0.0014 | 0.9955 | 4698 s (~78 min) |
| `baseline-ml` (RF sobre features profundas) | ML clasico + deep features | 0.9955 | N/D | N/D | 0.9956 | 959 s (~16 min) |
| `scratch-cnn` (config base 10 epochs) | CNN desde cero | 0.9847 | 0.9346 | -0.0501 | 0.9881 | 3040 s (~51 min) |

### Resultado del baseline ML

En `baseline-ml`, el mejor candidato fue `random_forest`:

- `random_forest`: val_accuracy = 0.9955 (val_f1_macro = 0.9956)
- `logistic_regression`: val_accuracy = 0.9728
- `linear_svc`: val_accuracy = 0.9629

## Conclusiones (Accuracy, Overfitting y Despliegue)

### 1) ¿Cuál es el mejor modelo por accuracy?
El mejor modelo es **`finetune-resnet18`** con un **`0.9986`** de *accuracy* en validación. 
Se consolida como el mejor candidato ya que es el que ha sido sometido a un mayor testeo empírico a nivel de aplicación, demostrando un rendimiento superior frente a imágenes reales.

### 2) ¿Generaliza bien o está sobreajustado?
En la ejecución de *fine-tuning*, la diferencia `train-val` es de apenas `~0.0001`, lo cual es mínimo. Esto sugiere una **excelente asimilación dentro del split disponible** y no muestra señales numéricas de *overfitting*. 

Aunque no se ha utilizado un conjunto de *test* externo formal, esta decisión se tomó para maximizar el uso de los datos en el entrenamiento. La capacidad de generalización real ha sido validada cualitativamente pasando imágenes propias a través de la aplicación, confirmando que el modelo se comporta sorprendentemente bien ante cambios de iluminación, fondos y resolución.

### 3) ¿Lo desplegaríamos ya a producción?
**Sí, pero únicamente dentro de un entorno controlado y como candidato inicial.**

Poner un modelo en producción no es solo cuestión de *accuracy*, sino de viabilidad técnica y requisitos de sistema. 

Motivos para proceder con cautela:
- **Restricciones de Hardware y Consumo:** `finetune-resnet18` es una arquitectura pesada. Si la aplicación final requiere ejecución en dispositivos de bajo consumo (Edge AI) o latencias mínimas en tiempo real, habría que evaluar si el dispositivo puede soportarlo sin penalizar la experiencia del usuario (batería, calentamiento).
- **Alternativas ligeras:** Si los requisitos técnicos de hardware son estrictos, el `baseline-ml` (como un Random Forest) podría ser la alternativa necesaria a costa de sacrificar algo de precisión.

Recomendación práctica: Iniciar el despliegue de `finetune-resnet18` de forma controlada (fase beta o entorno cerrado) mientras se monitoriza su impacto en el rendimiento de la aplicación y en los recursos del sistema.

#### **Siguientes pasos recomendados para el entorno productivo:**
1. **Despliegue Principal:** Integrar la versión final de `finetune-resnet18` como el motor de clasificación principal de la aplicación.
2. **Alternativa de Baja Latencia:** Mantener documentado el `baseline-ml` (Random Forest) como modelo de respaldo en caso de que futuras actualizaciones requieran una inferencia mucho más rápida o de menor coste computacional en dispositivos menos potentes.
3. **Bucle de Mejora Continua (Data Flywheel):** Ahora que el modelo está en la app, hemos pensado en habilitar una funcionalidad para que los usuarios puedan reportar cuando el modelo se equivoque. Estas "imágenes difíciles" reales podrán ser capturadas y añadidas al dataset para seguir reentrenando y mejorando el modelo en el futuro.

## Interfaz Streamlit

La aplicacion (`app/streamlit_app.py`) tiene dos modos:

1. **Clasificacion por imagenes** (batch)
2. **Video en tiempo real** (webcam, si `streamlit-webrtc` esta disponible)

![Aplicación Streamlit](./docs/StreamlitApp.png)

## Ejecutar app Streamlit

```bash
conda activate DL; streamlit run app/streamlit_app.py
```

La app permite:
- seleccionar modelo desde `artifacts/` (`.joblib`, `.pt`, `.pth`, `.ckpt`),
- ver caracteristicas del modelo en la barra lateral,
- subir una o varias imagenes,
- obtener top-k predicciones (hasta top-5) por imagen,
- limpiar imagenes subidas con un boton dedicado,
- generar una secuencia GIF desde texto en el modo `Generacion`,
- descargar el GIF generado.

Notas:
- Si existe `MODEL_PATH` en `.env`, se usa como opcion discriminativa por defecto.
- Si existe `GENERATOR_MODEL_PATH` en `.env`, la pestana `Generacion` usa ese checkpoint.
- Si no hay checkpoint generativo, la pestana `Generacion` usa fallback de dataset/placeholders para seguir funcionando.
- Si no hay modelos discriminativos detectados en `artifacts/`, la app lo indica con un mensaje explicito.

### Flujo recomendado

1. Arranca la app:
   ```bash
   conda activate DL; streamlit run app/streamlit_app.py
   ```
2. En la barra lateral, selecciona el modelo (`.joblib`, `.pt`, `.pth`, `.ckpt`).
3. En **"Clasificacion por imagenes"**, sube una o varias imagenes (drag & drop o selector de archivos).
4. La app muestra:
   - tabla top-k por imagen,
   - miniaturas con top-5,
   - confianza media del lote,
   - accuracy/error del lote **solo** cuando la etiqueta real se puede inferir del nombre del archivo.

## 👥 Autores del Proyecto

Este proyecto ha sido desarrollado por:

* **Alejandro Torres Martínez** - [GitHub](https://github.com/alejandrotm22) | [LinkedIn](https://www.linkedin.com/in/alejandro-torres-martinez-b65490301/)
* **Alejandro Coman Venceslá** -  [GitHub](https://github.com/AleComan) | [LinkedIn](https://www.linkedin.com/in/aleecv/)
