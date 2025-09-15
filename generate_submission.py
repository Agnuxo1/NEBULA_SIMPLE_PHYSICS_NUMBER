#!/usr/bin/env python3
import os
import csv
import json
import subprocess
import argparse

def run_infer(front, back, left, right, exe_path='main01.exe'):
    env = os.environ.copy()
    env['RSNA_INFER'] = '1'
    env['RSNA_FRONT'] = front
    env['RSNA_BACK'] = back
    env['RSNA_LEFT'] = left
    env['RSNA_RIGHT'] = right
    # Do not force checkpoints; allow random weights or pre-set CKPT_* in env
    proc = subprocess.Popen([exe_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    out = proc.communicate()[0].decode('utf-8', errors='ignore')
    # Find last JSON line in output
    last_brace = out.rfind('}')
    first_brace = out.rfind('{', 0, last_brace+1)
    if first_brace >= 0 and last_brace > first_brace:
        try:
            data = json.loads(out[first_brace:last_brace+1])
            return data.get('probs', [])
        except Exception:
            pass
    raise RuntimeError('Could not parse JSON from inference output')

def main():
    ap = argparse.ArgumentParser(description='Generate Kaggle submission CSV from MIPs and main01.exe')
    ap.add_argument('--test_csv', required=True, help='CSV con SeriesInstanceUID (p.ej., kaggle_evaluation/test.csv)')
    ap.add_argument('--mips_dir', default='mips', help='Directorio con MIPs por serie')
    ap.add_argument('--output', default='submission.csv', help='Archivo CSV de salida')
    ap.add_argument('--train_csv', default='rsna-intracranial-aneurysm-detection/train.csv', help='Para obtener nombres de arterias (header)')
    args = ap.parse_args()

    # Obtener nombres de columnas de arterias del train.csv
    with open(args.train_csv, 'r', encoding='utf-8', errors='ignore') as f:
        header = next(csv.reader(f))
    # Columnas de arterias suelen estar desde la 5ta hasta la anteúltima, y última es Aneurysm Present
    # Encontrar índice de 'Aneurysm Present'
    try:
        ai = header.index('Aneurysm Present')
    except ValueError:
        ai = len(header)
    artery_cols = header[4:ai]

    # Preparar salida
    with open(args.output, 'w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['SeriesInstanceUID'] + artery_cols + ['Aneurysm Present'])

        with open(args.test_csv, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row['SeriesInstanceUID']
                base = os.path.join(args.mips_dir, sid)
                front = os.path.join(base, 'front.pgm')
                back = os.path.join(base, 'back.pgm')
                left = os.path.join(base, 'left.pgm')
                right = os.path.join(base, 'right.pgm')
                if not (os.path.exists(front) and os.path.exists(back) and os.path.exists(left) and os.path.exists(right)):
                    # saltar series sin MIPs
                    continue
                probs = run_infer(front, back, left, right)
                if not probs or len(probs) < len(artery_cols)+1:
                    # rellenar con 0.5 si no hay suficientes
                    vals = [0.5]*len(artery_cols)
                    ap = 0.5
                else:
                    vals = [float(x) for x in probs[:len(artery_cols)]]
                    ap = float(probs[len(artery_cols)])
                writer.writerow([sid] + vals + [ap])
    print(f"Submission escrito en {args.output}")

if __name__ == '__main__':
    main()

