# signlanguage-classifier

Proyecto base para clasificacion de imagenes de lenguaje de signos con tres pipelines:

1. Baseline de ML clasico (con opcion de features profundas + clasificador de scikit-learn).
2. CNN entrenada desde cero (sin pesos preentrenados).
3. Fine-tuning de modelo preentrenado (congelacion parcial configurable).

## Estructura del proyecto

```text
signlanguage-classifier/
├── app/
│   └── streamlit_app.py
├── configs/
│   ├── baseline_ml.yaml
│   ├── dataset_asl.yaml
│   ├── scratch_cnn.yaml
│   └── finetune.yaml
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_ml.ipynb
│   ├── 03_scratch_cnn.ipynb
│   └── 04_finetune.ipynb
├── scripts/
│   └── prepare_dataset.py
├── src/
│   ├── data/
│   ├── eda/
│   ├── evaluation/
│   ├── features/
│   ├── inference/
│   ├── models/
│   ├── training/
│   └── utils/
├── tests/
├── .cursorignore
├── .env.example
├── .gitignore
├── README.md
├── TASKS.md
└── requirements.txt
```

Los pipelines de EDA, baseline, scratch y fine-tuning viven ahora como notebooks de Jupyter en `notebooks/`. La logica reusable sigue en `src/`, y `scripts/prepare_dataset.py` se usa una sola vez para preparar el dataset.

## Dataset ASL Alphabet (recomendado)

Estructura esperada del dataset descargado (raw):

```text
data/asl_alphabet_v1/raw/
├── asl_alphabet_train/
│   ├── A/
│   ├── B/
│   ├── ...
│   ├── del/
│   ├── nothing/
│   └── space/
└── asl_alphabet_test/
    ├── A_test.jpg
    ├── B_test.jpg
    ├── ...
    ├── nothing_test.jpg
    └── space_test.jpg
```

Preprocesado a formato `ImageFolder` con split estratificado por clase:

```bash
conda activate DL; python scripts/prepare_dataset.py --config configs/dataset_asl.yaml
```

Salida generada:
- `data/asl_alphabet_v1/processed/train|val|test/<label>/...`
- `data/asl_alphabet_v1/processed/metadata.csv`
- `data/asl_alphabet_v1/processed/summary.json`

Notas:
- Se renombra automaticamente `del` -> `delete` (configurable en `configs/dataset_asl.yaml`).
- El split por defecto es 70/15/15 desde `asl_alphabet_train`.
- Opcionalmente se agregan las imagenes de `asl_alphabet_test` al split `test`.

## Requisitos

- Python 3.10+ recomendado.
- Dataset procesado en estructura tipo `ImageFolder` de `torchvision`:

```text
data/
├── train/
│   ├── class_a/
│   └── class_b/
└── val/
    ├── class_a/
    └── class_b/
```

## Entorno (Conda)

```bash
conda activate DL
```

Este repositorio asume que ya tienes las dependencias instaladas en tu entorno `DL`.
`requirements.txt` se mantiene como referencia para reproducibilidad o para otros entornos.

## Variables de entorno

1. Copia `.env.example` a `.env`.
2. Ajusta las variables necesarias (no subas `.env` al repo).

Ejemplo:

```env
WANDB_API_KEY=tu_api_key
WANDB_ENTITY=tu_usuario_o_equipo
WANDB_PROJECT=signlanguage-classifier
```

## Ejecutar pipelines (notebooks)

Lanza Jupyter desde la raiz del proyecto para que los notebooks puedan importar `src/` automaticamente:

```bash
conda activate DL; jupyter lab
```

Notebooks disponibles en `notebooks/`:

- `01_eda.ipynb` - exploracion del dataset (clases, tamanos, color, muestras).
- `02_baseline_ml.ipynb` - features de ResNet18 + clasificadores sklearn (LogReg, SVM, RF) y comparativa.
- `03_scratch_cnn.ipynb` - `SimpleCNN` entrenada desde cero con curvas y test final.
- `04_finetune.ipynb` - fine-tuning de ResNet18 con freezing parcial y LRs discriminativos.

Cada notebook lee su YAML de `configs/` (`baseline_ml.yaml`, `scratch_cnn.yaml`, `finetune.yaml`), guarda los artefactos en `artifacts/<pipeline>/` y, si `tracking.use_wandb` esta activo y hay `WANDB_API_KEY`, registra metricas en Weights & Biases.

## Ejecutar app Streamlit

```bash
conda activate DL; streamlit run app/streamlit_app.py
```

La app permite subir una imagen, intentar cargar un checkpoint y mostrar top-k predicciones.
Si no existe checkpoint, mostrara un mensaje explicito para guiar el siguiente paso.

## Notas de seguridad y reproducibilidad

- No guardar claves reales en archivos versionados.
- `wandb/`, checkpoints, datasets y archivos temporales se ignoran en `.gitignore`.
- Se incluye utilitario de seeds reproducibles para Python, NumPy y PyTorch.