# fin-PAD

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Update the sensor paths in `./configs/dual_model.yaml`:

```yaml
TRAIN_SENSOR_PATH:
TEST_SENSOR_PATH:
```

2. Run the training:

```bash
python binary_classification.py -c ./configs/dual_model.yaml
```

## Note

Use the foreground dataset to train and test.

## Dataset Structure

```
├── 2011
├── 2013
├── 2015
│   ├── CrossMatch
│   │   ├── Test
│   │   │   ├── Live
│   │   │   └── Spoof
│   │   │       ├── Body Double
│   │   │       ├── Ecoflex
│   │   │       ├── Gelatin
│   │   │       ├── OOMOO
│   │   │       └── Playdoh
│   │   └── Train
│   │       ├── Live
│   │       └── Spoof
│   │           ├── Body Double
│   │           ├── Ecoflex
│   │           └── Playdoh
│   ├── DigitalPersona
│   │   ├── Test
│   │   │   ├── Live
│   │   │   └── Spoof
│   │   │       ├── Ecoflex 00-50
│   │   │       ├── Gelatine
│   │   │       ├── Latex
│   │   │       ├── Liquid Ecoflex
│   │   │       ├── RTV
│   │   │       └── WoodGlue
│   │   └── Train
│   │       ├── Live
│   │       └── Spoof
│   │           ├── Ecoflex 00-50
│   │           ├── Gelatine
│   │           ├── Latex
│   │           └── WoodGlue
```