# TASKS

## Fase 1 - Scaffolding (hecho)
- [x] Estructura base de carpetas para data/features/models/training/evaluation/inference/utils.
- [x] Configuracion YAML para baseline, scratch y finetuning.
- [x] Configuracion y script de preprocesado para ASL Alphabet (raw -> train/val).
- [x] Utilidades de reproducibilidad, carga de config y setup seguro de W&B.
- [x] App Streamlit minima para inferencia con mensaje de checkpoint ausente.
- [x] Refactor a notebooks: `notebooks/01_eda.ipynb`, `02_baseline_ml.ipynb`, `03_scratch_cnn.ipynb`, `04_finetune.ipynb` con logica reusable en `src/`.

## Fase 2 - Baseline ML
- [ ] Implementar comparativa de clasificadores (SVM, RandomForest, LogisticRegression) con busqueda de hiperparametros.
- [ ] Añadir pipeline de features clasicas (HOG/ORB) ademas de features profundas.
- [ ] Guardar metricas por clase y matriz de confusion.

## Fase 3 - CNN from scratch
- [ ] Definir arquitectura CNN configurable por YAML.
- [ ] Añadir scheduler, early stopping y checkpoints por mejor validacion.
- [ ] Incluir augmentations configurables.

## Fase 4 - Fine-tuning
- [ ] Permitir desbloqueo progresivo de capas por bloques.
- [ ] Añadir discriminative learning rates (backbone vs classifier head).
- [ ] Comparar varios backbones preentrenados.

## Fase 5 - App e inferencia
- [ ] Cargar multiples tipos de modelos desde un registro local.
- [ ] Mejorar UX con visualizacion de probabilidades top-k.
- [ ] Agregar webcam y prediccion en tiempo real.

## Fase 6 - Calidad y MLOps
- [ ] Pruebas unitarias y de integracion para utilidades y carga de datos.
- [ ] Integrar Ruff + hooks pre-commit.
- [ ] Documentar proceso de experimentacion reproducible.
