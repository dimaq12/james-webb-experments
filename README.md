# sft_torch Universal Spectral Analysis Pipeline

Анализ любого изображения через спектральный оператор sft_torch:
κ_W, Tail α, Ω-вектор, итерированные остатки.

## Установка

```bash
pip install torch numpy scipy matplotlib pillow astropy
# sft_torch из соседней директории:
export PYTHONPATH=/home/dima/FA/sft_torch:$PYTHONPATH
```

## Быстрый старт

```bash
# Анализ фото
python sft_pipeline.py photo.jpg

# Анализ FITS (JWST/HST)
python sft_pipeline.py jwst_image.fits --residual-depth 3

# Скачать JWST данные
python download_jwst.py smacs --filter f090w --outdir data/
python sft_pipeline.py data/jw02736-o001_t001_nircam_clear-f090w_i2d.fits
```

## Параметры

| Флаг | По умолчанию | Описание |
|---|---|---|
| `--residual-depth, -d` | 2 | Глубина итерированной регрессии (0–3) |
| `--max-regions, -n` | 300 | Максимум сканируемых блоков |
| `--stride, -s` | 80 | Шаг сканирования (пиксели) |
| `--region-size, -r` | 80 | Размер блока (пиксели) |
| `--n-grid` | 10 | Размер операторной сетки |
| `--m-params` | 5 | Число базисных параметров M |
| `--output-dir, -o` | `pipeline_out/` | Директория вывода |

## Выходные файлы

```
pipeline_out/
├── image_metrics.json    # Все метрики (κ_W, α, Ω, R², ...)
└── image_reveal.png      # Визуализация: оригинал + κ_W + остатки
```

## Формат metrics.json

```json
{
  "n_regions": 250,
  "kappa_W": {"min": 126, "max": 284, "mean": 257, "std": 22, "cv": 0.084},
  "alpha":    {"min": 0.734, "max": 2.528, "mean": 0.866, "std": 0.172, "cv": 0.199},
  "omega_norm": {"min": 0.068, "max": 0.477, "mean": 0.097, "std": 0.047, "cv": 0.489},
  "cumulative_R2": 0.928,
  "compression_ratio": 0.268,
  "regression": {
    "L0: κ_W ~ яркость+контраст": {"R2": 0.798, "delta": 0.798},
    "L1: resid0 ~ α+Ω+спектр+позиция": {"R2": 0.489, "delta": 0.099},
    "L2: resid1 ~ границы спектра+gap": {"R2": 0.305, "delta": 0.031}
  }
}
```

## Метрики

| Метрика | Описание |
|---|---|
| **κ_W** | Спектральная кривизна оператора W |
| **Tail α** | Показатель степенного хвоста спектра |
| **Ω vector** | Омега-вектор хвостовой диагностики (4 компоненты) |
| **\|\|Ω\|\|** | Норма омега-вектора |
| **cv(α)** | Коэффициент вариации α — спектральная гетерогенность |
| **cv(\|\|Ω\|\|)** | Пространственная вариация Ω-нормы |
| **Cumulative R²** | Доля объяснённой дисперсии κ_W |
| **Compression ratio** | σ(residual_final) / σ(κ_W) |

## Download JWST

```bash
python download_jwst.py list                    # список программ
python download_jwst.py smacs --list            # файлы SMACS 0723
python download_jwst.py ceers --filter f277w    # скачать CEERS F277W
python download_jwst.py nebula --outdir data/   # Southern Ring Nebula
```

Поддерживаемые программы: `smacs`, `ceers`, `nebula`. HUDF — внешний файл, скачивается отдельно.
