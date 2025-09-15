#!/usr/bin/env python3
"""
Monitor de procesamiento para dataset RSNA grande
Muestra estadísticas en tiempo real del procesamiento
"""

import os
import time
import pickle
from datetime import datetime, timedelta

def format_bytes(bytes_value):
    """Convierte bytes a formato legible"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"

def get_directory_size(path):
    """Calcula el tamaño total de un directorio"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except:
        pass
    return total_size

def count_files_in_directory(path, extension='.pgm'):
    """Cuenta archivos con una extensión específica"""
    count = 0
    try:
        for filename in os.listdir(path):
            if filename.endswith(extension):
                count += 1
    except:
        pass
    return count

def load_progress(progress_file="processing_progress.pkl"):
    """Carga el progreso del procesamiento"""
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'rb') as f:
                return pickle.load(f)
        except:
            return None
    return None

def main():
    print("==================================================")
    print("   MONITOR DE PROCESAMIENTO RSNA")
    print("==================================================")
    print()
    
    progress_file = "processing_progress.pkl"
    mips_dir = "mips"
    
    print("Iniciando monitor... (Ctrl+C para salir)")
    print()
    
    try:
        while True:
            # Limpiar pantalla (Windows)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("==================================================")
            print("   MONITOR DE PROCESAMIENTO RSNA")
            print("==================================================")
            print(f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # Cargar progreso
            progress = load_progress(progress_file)
            
            if progress:
                print("📊 PROGRESO ACTUAL:")
                print(f"   Muestras procesadas: {progress['processed_count']}")
                print(f"   Archivo de salida: {progress['output_file']}")
                print(f"   Última actualización: {progress['timestamp']}")
                print()
            else:
                print("📊 PROGRESO ACTUAL:")
                print("   No hay progreso disponible")
                print("   (El procesamiento puede no haber comenzado)")
                print()
            
            # Estadísticas del directorio MIPs
            if os.path.exists(mips_dir):
                mip_files = count_files_in_directory(mips_dir, '.pgm')
                mip_size = get_directory_size(mips_dir)
                
                print("📁 ESTADÍSTICAS MIPs:")
                print(f"   Archivos PGM generados: {mip_files}")
                print(f"   Tamaño total: {format_bytes(mip_size)}")
                print(f"   Series completas: {mip_files // 4} (4 vistas por serie)")
                print()
            else:
                print("📁 ESTADÍSTICAS MIPs:")
                print("   Directorio mips/ no encontrado")
                print()
            
            # Estadísticas del archivo CSV de salida
            output_files = ['train_mips.csv', 'test_mips.csv']
            for output_file in output_files:
                if os.path.exists(output_file):
                    file_size = os.path.getsize(output_file)
                    print(f"📄 ARCHIVO {output_file.upper()}:")
                    print(f"   Tamaño: {format_bytes(file_size)}")
                    
                    # Contar líneas (aproximado)
                    try:
                        with open(output_file, 'r') as f:
                            line_count = sum(1 for line in f)
                        print(f"   Líneas: {line_count} (incluye header)")
                        print(f"   Muestras: {line_count - 1}")
                    except:
                        print("   No se pudo leer el archivo")
                    print()
            
            # Estimaciones de tiempo
            if progress and progress['processed_count'] > 0:
                print("⏱️  ESTIMACIONES:")
                
                # Calcular velocidad aproximada basada en archivos MIPs
                if os.path.exists(mips_dir):
                    current_time = datetime.now()
                    progress_time = datetime.fromisoformat(progress['timestamp'])
                    time_diff = (current_time - progress_time).total_seconds()
                    
                    if time_diff > 0:
                        samples_per_sec = progress['processed_count'] / time_diff
                        print(f"   Velocidad aproximada: {samples_per_sec:.3f} muestras/seg")
                        
                        # Estimación para 1000 muestras más
                        eta_1000 = 1000 / samples_per_sec if samples_per_sec > 0 else 0
                        eta_hours = eta_1000 / 3600
                        print(f"   Tiempo para 1000 muestras más: {eta_hours:.1f} horas")
                print()
            
            print("💡 COMANDOS ÚTILES:")
            print("   - Para procesamiento rápido: quick_test_dataset.bat")
            print("   - Para procesamiento completo: process_large_dataset.bat")
            print("   - Para reanudar: process_large_dataset.bat --resume")
            print()
            print("🔄 Actualizando cada 30 segundos... (Ctrl+C para salir)")
            
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n\nMonitor detenido por el usuario.")
        print("El procesamiento continúa en segundo plano.")
        print("Puede reiniciar este monitor en cualquier momento.")

if __name__ == '__main__':
    main()



