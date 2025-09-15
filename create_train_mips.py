#!/usr/bin/env python3
"""
Script para generar train_mips.csv a partir del dataset RSNA
Convierte las imágenes DICOM del dataset RSNA en imágenes PGM y crea el CSV
necesario para el entrenamiento del modelo óptico.

Uso:
    python create_train_mips.py --train_csv train.csv --images_dir images --output train_mips.csv

El script:
1. Lee el train.csv del RSNA con las etiquetas
2. Busca las imágenes DICOM correspondientes
3. Genera las 4 vistas MIP (front, back, left, right)
4. Convierte a formato PGM
5. Crea el train_mips.csv con las rutas
"""

import os
import csv
import argparse
import numpy as np
from pathlib import Path
import pydicom
import imageio
import json
import gc
import psutil
import time
import pickle
from datetime import datetime

def check_memory_usage():
    """Verifica el uso actual de memoria"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024**3)  # GB

def cleanup_memory():
    """Limpia la memoria forzando garbage collection"""
    gc.collect()

def save_progress(processed_count, output_file, progress_file="processing_progress.pkl"):
    """Guarda el progreso actual para poder reanudar"""
    progress_data = {
        'processed_count': processed_count,
        'output_file': output_file,
        'timestamp': datetime.now().isoformat()
    }
    with open(progress_file, 'wb') as f:
        pickle.dump(progress_data, f)

def load_progress(progress_file="processing_progress.pkl"):
    """Carga el progreso anterior"""
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'rb') as f:
                return pickle.load(f)
        except:
            return None
    return None

def load_dicom_volume(dicom_files, max_memory_gb=12.0):
    """Carga un volumen 3D desde archivos DICOM con control de memoria"""
    slices = []
    positions = []
    
    initial_memory = check_memory_usage()
    
    for i, dicom_file in enumerate(dicom_files):
        # Verificar memoria cada 5 archivos para mejor control
        if i % 5 == 0:
            current_memory = check_memory_usage()
            if current_memory - initial_memory > max_memory_gb:
                print(f"Límite de memoria alcanzado ({max_memory_gb}GB), procesando solo {i} archivos")
                break
        
        try:
            ds = pydicom.dcmread(dicom_file)
            pixel_array = ds.pixel_array.astype(np.float32)
            
            # Verificar tamaño de la imagen - aumentar límite y usar compresión
            image_size_mb = pixel_array.nbytes / (1024 * 1024)
            if image_size_mb > 500:  # Aumentar límite a 500MB
                print(f"Imagen muy grande ({image_size_mb:.1f}MB), saltando: {dicom_file}")
                continue
            
            # Comprimir imagen si es muy grande
            if image_size_mb > 200:  # Comprimir si > 200MB
                # Reducir resolución a la mitad
                from scipy.ndimage import zoom
                pixel_array = zoom(pixel_array, 0.5, order=1)
                print(f"Comprimiendo imagen de {image_size_mb:.1f}MB a {(pixel_array.nbytes / (1024 * 1024)):.1f}MB")
            
            slices.append(pixel_array)
            
            # Usar ImagePositionPatient para ordenar
            if hasattr(ds, 'ImagePositionPatient'):
                positions.append(float(ds.ImagePositionPatient[2]))  # Z coordinate
            else:
                positions.append(len(slices) - 1)
                
            # Limpiar memoria del objeto DICOM
            del ds
            del pixel_array
            
        except Exception as e:
            print(f"Error cargando {dicom_file}: {e}")
            continue
    
    if not slices:
        return None
    
    # Ordenar por posición Z
    sorted_indices = np.argsort(positions)
    volume = np.stack([slices[i] for i in sorted_indices], axis=0)
    
    # Limpiar listas temporales
    del slices
    del positions
    cleanup_memory()
    
    return volume

def create_mips(volume):
    """Crea las 4 vistas MIP (Maximum Intensity Projection)"""
    if volume is None:
        return None
    
    # Normalizar el volumen
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    
    # Crear MIPs en las 4 direcciones
    mips = {}
    
    # Front view (coronal) - proyección máxima en Y
    mips['front'] = np.max(volume, axis=1)
    
    # Back view (coronal invertida)
    mips['back'] = np.max(volume, axis=1)[::-1, :]
    
    # Left view (sagital) - proyección máxima en X
    mips['left'] = np.max(volume, axis=2)
    
    # Right view (sagital invertida)
    mips['right'] = np.max(volume, axis=2)[:, ::-1]
    
    return mips

def save_as_pgm(image_array, filename):
    """Guarda una imagen como archivo PGM usando imageio para robustez"""
    # Normalizar y convertir a uint8
    arr = np.asarray(image_array)
    arr_min = arr.min()
    arr_max = arr.max()
    denom = (arr_max - arr_min) if (arr_max - arr_min) != 0 else 1.0
    img_uint8 = ((arr - arr_min) / denom * 255.0).astype(np.uint8)

    # Crear directorio si no existe
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Guardar como PGM usando imageio, especificando el formato binario P5
    imageio.imwrite(filename, img_uint8, format='pgm', flags='-P5')

def find_dicom_files(series_dir):
    """Encuentra todos los archivos DICOM en un directorio"""
    dicom_files = []
    for ext in ['*.dcm', '*.DCM', '*.dicom', '*.DICOM']:
        dicom_files.extend(Path(series_dir).glob(ext))
    
    # También buscar archivos sin extensión que sean DICOM
    for file_path in Path(series_dir).iterdir():
        if file_path.is_file() and not file_path.suffix.lower() in ['.dcm', '.dicom']:
            try:
                # Intentar leer como DICOM
                pydicom.dcmread(file_path)
                dicom_files.append(file_path)
            except:
                pass
    
    return sorted(dicom_files)

def process_series(series_id, images_dir, output_dir, memory_limit=12.0):
    """Procesa una serie DICOM y genera las 4 vistas MIP"""
    series_dir = Path(images_dir) / series_id
    
    if not series_dir.exists():
        print(f"Directorio no encontrado: {series_dir}")
        return None
    
    # Encontrar archivos DICOM
    dicom_files = find_dicom_files(series_dir)
    
    if not dicom_files:
        print(f"No se encontraron archivos DICOM en: {series_dir}")
        return None
    
    print(f"Procesando {len(dicom_files)} archivos DICOM para {series_id}")
    
    # Cargar volumen 3D
    volume = load_dicom_volume(dicom_files, memory_limit)
    
    if volume is None:
        print(f"No se pudo cargar el volumen para {series_id}")
        return None
    
    # Crear MIPs
    mips = create_mips(volume)
    
    if mips is None:
        print(f"No se pudieron crear MIPs para {series_id}")
        return None
    
    # Guardar MIPs como PGM usando esquema: mips/<SeriesInstanceUID>/<view>.pgm
    output_paths = {}
    series_out_dir = os.path.join(output_dir, series_id)
    os.makedirs(series_out_dir, exist_ok=True)
    for view, mip in mips.items():
        filename = os.path.join(series_out_dir, f"{view}.pgm")
        save_as_pgm(mip, filename)
        output_paths[view] = filename
    
    return output_paths

def main():
    parser = argparse.ArgumentParser(description='Generar train_mips.csv para RSNA (Dataset Grande 300GB)')
    parser.add_argument('--train_csv', required=True, help='Archivo train.csv del RSNA')
    parser.add_argument('--images_dir', required=True, help='Directorio con las imágenes DICOM')
    parser.add_argument('--output', default='train_mips.csv', help='Archivo de salida CSV')
    parser.add_argument('--mips_dir', default='mips', help='Directorio para guardar MIPs')
    parser.add_argument('--max_samples', type=int, help='Máximo número de muestras a procesar')
    parser.add_argument('--sample_rate', type=float, default=1.0, help='Fracción de muestras a procesar (0.0-1.0) - DEFAULT 100% para dataset completo')
    parser.add_argument('--chunk_size', type=int, default=100, help='Procesar en chunks de N muestras')
    parser.add_argument('--resume', action='store_true', help='Reanudar procesamiento desde última posición')
    parser.add_argument('--memory_limit', type=float, default=12.0, help='Límite de memoria en GB')
    parser.add_argument('--skip_existing', action='store_true', help='Saltar MIPs que ya existen')
    
    args = parser.parse_args()
    
    # Crear directorio de salida
    os.makedirs(args.mips_dir, exist_ok=True)
    
    # Verificar si hay progreso anterior
    start_index = 0
    if args.resume:
        progress = load_progress()
        if progress:
            start_index = progress['processed_count']
            print(f"Reanudando desde muestra {start_index}")
    
    # Leer train.csv
    print(f"Leyendo {args.train_csv}...")
    samples = []
    
    try:
        with open(args.train_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append(row)
    except Exception as e:
        print(f"Error leyendo {args.train_csv}: {e}")
        return 1
    
    print(f"Encontradas {len(samples)} muestras")
    
    # Aplicar filtros si se especifican
    if args.sample_rate < 1.0:
        num_samples = int(len(samples) * args.sample_rate)
        samples = samples[:num_samples]
        print(f"Procesando {len(samples)} muestras (sample_rate={args.sample_rate})")
    
    if args.max_samples:
        samples = samples[:args.max_samples]
        print(f"Procesando máximo {len(samples)} muestras")
    
    # Configurar archivo de salida
    mode = 'a' if args.resume and start_index > 0 else 'w'
    write_header = not (args.resume and start_index > 0)
    
    # Procesar por chunks
    processed_samples = []
    total_processed = start_index
    start_time = time.time()
    
    for chunk_start in range(start_index, len(samples), args.chunk_size):
        chunk_end = min(chunk_start + args.chunk_size, len(samples))
        chunk_samples = samples[chunk_start:chunk_end]
        
        print(f"\n--- Procesando chunk {chunk_start}-{chunk_end} de {len(samples)} ---")
        print(f"Memoria actual: {check_memory_usage():.2f}GB")
        
        chunk_processed = []
        
        for i, sample in enumerate(chunk_samples):
            global_index = chunk_start + i
            series_id = sample.get('SeriesInstanceUID', '')
            
            if not series_id:
                print(f"Muestra {global_index} sin SeriesInstanceUID, saltando...")
                continue
            
            print(f"Procesando {global_index+1}/{len(samples)}: {series_id}")
            
            # Verificar si ya existen los MIPs
            if args.skip_existing:
                expected_files = [
                    f"{args.mips_dir}/{series_id}/front.pgm",
                    f"{args.mips_dir}/{series_id}/back.pgm",
                    f"{args.mips_dir}/{series_id}/left.pgm",
                    f"{args.mips_dir}/{series_id}/right.pgm"
                ]
                if all(os.path.exists(f) for f in expected_files):
                    print(f"  MIPs ya existen, saltando...")
                    # Añadir a procesados aunque se salte
                    label = 0
                    for label_col in ['label', 'Label', 'target', 'Target', 'Aneurysm Present']:
                        if label_col in sample:
                            try:
                                label = int(float(sample[label_col]))
                                break
                            except:
                                pass
                    
                    chunk_processed.append({
                        'SeriesInstanceUID': series_id,
                        'label': label,
                        'front': expected_files[0],
                        'back': expected_files[1],
                        'left': expected_files[2],
                        'right': expected_files[3]
                    })
                    continue
            
            # Procesar la serie
            try:
                mip_paths = process_series(series_id, args.images_dir, args.mips_dir, args.memory_limit)
                
                if mip_paths:
                    # Obtener la etiqueta (puede estar en diferentes columnas)
                    label = 0
                    for label_col in ['label', 'Label', 'target', 'Target', 'Aneurysm Present']:
                        if label_col in sample:
                            try:
                                label = int(float(sample[label_col]))
                                break
                            except:
                                pass
                    
                    chunk_processed.append({
                        'SeriesInstanceUID': series_id,
                        'label': label,
                        'front': mip_paths['front'],
                        'back': mip_paths['back'],
                        'left': mip_paths['left'],
                        'right': mip_paths['right']
                    })
                    total_processed += 1
                else:
                    print(f"Error procesando {series_id}, saltando...")
                    
            except Exception as e:
                print(f"Error procesando {series_id}: {e}")
                continue
        
        # Guardar chunk procesado
        if chunk_processed:
            with open(args.output, mode, newline='') as f:
                fieldnames = ['SeriesInstanceUID', 'label', 'front', 'back', 'left', 'right']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                if write_header:
                    writer.writeheader()
                    write_header = False
                
                writer.writerows(chunk_processed)
        
        # Guardar progreso
        save_progress(total_processed, args.output)
        
        # Limpiar memoria después de cada chunk
        cleanup_memory()
        
        # Mostrar estadísticas
        elapsed_time = time.time() - start_time
        samples_per_sec = total_processed / elapsed_time if elapsed_time > 0 else 0
        remaining_samples = len(samples) - total_processed
        eta_seconds = remaining_samples / samples_per_sec if samples_per_sec > 0 else 0
        eta_hours = eta_seconds / 3600
        
        print(f"Chunk completado. Total procesados: {total_processed}/{len(samples)}")
        print(f"Velocidad: {samples_per_sec:.2f} muestras/seg")
        print(f"Tiempo estimado restante: {eta_hours:.1f} horas")
        print(f"Memoria después de limpieza: {check_memory_usage():.2f}GB")
    
    print(f"\n¡Completado! Archivo guardado: {args.output}")
    print(f"MIPs guardados en: {args.mips_dir}/")
    print(f"Total muestras procesadas: {total_processed}")
    print(f"Tiempo total: {(time.time() - start_time)/3600:.1f} horas")
    
    # Limpiar archivo de progreso
    if os.path.exists("processing_progress.pkl"):
        os.remove("processing_progress.pkl")
    
    return 0

if __name__ == '__main__':
    exit(main())
