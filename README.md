# NEBULA_SIMPLE_PHYSICS_NUMBER

# RSNA Intracranial Aneurysm Detection - Optical Neural Network

Este proyecto implementa una red neuronal óptica para la detección de aneurismas intracraneales en el concurso RSNA, utilizando interferómetros Mach-Zehnder en lugar de tecnologías convencionales como CNNs o transformers.

## Características Principales

- **Arquitectura Óptica**: Red neuronal basada en interferómetros Mach-Zehnder
- **Aprendizaje Físico**: Plasticidad Hebbiana + clustering físico
- **Multi-Experto**: 4 redes especializadas para diferentes ángulos de vista
- **Procesamiento MIP**: Maximum Intensity Projection en 4 direcciones
- **Clasificación Binaria**: Detección de aneurisma presente/ausente

## Estructura del Proyecto

```
SIMPLE_PHYSICS_RSNA2/
├── main01.cu              # Código principal CUDA
├── main01.exe             # Ejecutable compilado
├── create_train_mips.py   # Script para generar train_mips.csv
├── setup_rsna.bat         # Configuración inicial
├── run_rsna_train.bat     # Entrenamiento
├── run_rsna_infer.bat     # Inferencia
├── mips/                  # Imágenes MIP generadas
├── ckpt_front/           # Checkpoints red front
├── ckpt_back/            # Checkpoints red back
├── ckpt_left/            # Checkpoints red left
├── ckpt_right/           # Checkpoints red right
└── train_mips.csv        # Dataset procesado
```

## Configuración Inicial

1. **Ejecutar configuración inicial**:
   ```bash
   setup_rsna.bat
   ```

2. **Descargar dataset RSNA**:
   - Visite: https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/data
   - Descargue `train.csv` y las imágenes DICOM

3. **Generar train_mips.csv**:
   ```bash
   python create_train_mips.py --train_csv train.csv --images_dir images --output train_mips.csv
   ```

4. **Compilar código CUDA** (si es necesario):
   ```bash
   nvcc -o main01.exe main01.cu -lcublas -lcurand
   ```

## Entrenamiento

Entrenamiento completo (único flujo soportado):

```bash
process_large_dataset.bat rsna-intracranial-aneurysm-detection\train.csv rsna-intracranial-aneurysm-detection\series train_mips_full.csv 1.0
run_rsna_train_full.bat
```

Nota: Se valida `train_mips_full.csv` y MIPs antes de entrenar. No se soportan flujos “rápidos/medios”.

## Inferencia

Para realizar inferencia en nuevas imágenes:

```bash
run_rsna_infer.bat <front.pgm> <back.pgm> <left.pgm> <right.pgm>
```

### Ejemplo de Inferencia

```bash
run_rsna_infer.bat mips/series001_front.pgm mips/series001_back.pgm mips/series001_left.pgm mips/series001_right.pgm
```

### Salida de Inferencia

El programa genera un JSON con 15 probabilidades:

```json
{"probs":[0.123456,0.234567,0.345678,0.456789,0.567890,0.678901,0.789012,0.890123,0.901234,0.012345,0.123456,0.234567,0.345678,0.456789,0.567890]}
```

- Primeras 14 probabilidades: Detección en cada arteria
- Última probabilidad: Detección global

## Formato del Dataset

### train_mips.csv

El archivo debe tener las siguientes columnas:

```csv
SeriesInstanceUID,label,front,back,left,right
1.2.3.4.5.6.7.8.9.10,1,mips/series001_front.pgm,mips/series001_back.pgm,mips/series001_left.pgm,mips/series001_right.pgm
```

- `SeriesInstanceUID`: Identificador único de la serie
- `label`: Etiqueta binaria (0=sin aneurisma, 1=con aneurisma)
- `front/back/left/right`: Rutas a las imágenes MIP en formato PGM

### Imágenes MIP

Las imágenes deben estar en formato PGM (Portable Graymap):
- Resolución: 28x28 píxeles
- Profundidad: 8 bits
- Escala de grises

## Solución de Problemas

### Error: "RSNA no seleccionado"

Este error indica que no se encuentra `train_mips.csv` o las variables de entorno no están configuradas.

**Solución**: Ejecute `run_rsna_train.bat` o configure manualmente:
```bash
set RSNA_TRAIN_MIPS=1
set RSNA_MIPS_CSV=train_mips.csv
```

### Error: "No se encontraron archivos DICOM"

El script `create_train_mips.py` no puede encontrar las imágenes DICOM.

**Solución**: Verifique que:
- El directorio `images/` contiene las imágenes DICOM
- Los archivos tienen extensiones `.dcm`, `.DCM`, `.dicom`, o `.DICOM`
- Las imágenes están organizadas por SeriesInstanceUID

### Error: "CUDA devices not found"

No se detecta una GPU compatible con CUDA.

**Solución**: 
- Instale drivers CUDA actualizados
- Verifique que la GPU es compatible con CUDA
- Para CPU, modifique el código para usar implementación CPU

### Rendimiento Lento

Si el entrenamiento es muy lento:

1. Reduzca el tamaño del dataset:
   ```bash
   python create_train_mips.py --sample_rate 0.1 --max_samples 1000
   ```

2. Reduzca el número de épocas:
   ```bash
   set MAX_EPOCHS=10
   ```

3. Use lotes más pequeños:
   ```bash
   set MAX_BATCHES=100
   ```

## Arquitectura Técnica

### Red Neuronal Óptica

- **Entrada**: 784 neuronas (28x28 píxeles)
- **Capas ocultas**: 512 → 256 → 128 neuronas
- **Salida**: 2 neuronas (clasificación binaria)
- **Función de activación**: Tanh con normalización

### Interferómetros Mach-Zehnder

- **Fase superior/inferior**: Parámetros ajustables
- **Ratio de acoplamiento**: Controla la interferencia
- **Factor de pérdida**: Modela pérdidas ópticas realistas

### Aprendizaje Hebbiano

- **Plasticidad**: Modificación de pesos basada en actividad
- **STDP**: Spike-Timing Dependent Plasticity
- **Clustering físico**: Agrupación espacial de características

## Contribuciones

Este proyecto implementa conceptos de:
- Shen et al. (2017): Mach-Zehnder interferometer networks
- Feldmann et al. (2019): Optical matrix multiplication  
- Lin et al. (2018): Photonic neural computing

## Licencia


Proyecto desarrollado para el concurso RSNA Intracranial Aneurysm Detection.




## Actualización de rutas del dataset

- Dataset real dentro del repo: `rsna-intracranial-aneurysm-detection/`
- Archivos clave: `train.csv`, `train_localizers.csv`, carpeta `series/` con DICOM.
- Para generar MIPs con el dataset completo:
  ```bash
  python create_train_mips.py \
    --train_csv rsna-intracranial-aneurysm-detection\train.csv \
    --images_dir rsna-intracranial-aneurysm-detection\series \
    --output train_mips_full.csv --sample_rate 1.0
  ```
- Estructura de rutas en `train_mips*.csv`: `.\\mips\\<SeriesInstanceUID>\\{front,back,left,right}.pgm`
- La columna de etiqueta se toma de `Aneurysm Present` de `train.csv`.
