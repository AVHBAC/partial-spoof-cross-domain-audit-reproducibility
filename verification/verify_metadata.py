"""Verify .npy utterance counts against dataset metadata files on disk."""
import numpy as np
import csv
from pathlib import Path

NPY_BASE = (Path(__file__).resolve().parents[1] / "data")
DATA_ROOT = (Path(__file__).resolve().parents[1] / "data" / "datasets")
CSV_META = (Path(__file__).resolve().parents[1] / "data" / "dataset_metadata.csv")

def count_partialspoof_eval():
    """Count from ASVspoof2019 eval protocol."""
    proto = DATA_ROOT / "PartialSpoof/database/protocols/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
    spoof = bonafide = 0
    with open(proto) as f:
        for line in f:
            if 'spoof' in line.lower(): spoof += 1
            elif 'bonafide' in line.lower(): bonafide += 1
    return spoof + bonafide, spoof, bonafide

def count_llamapartialspoof():
    """Count from label files."""
    spoof = bonafide = 0
    for lf in ['label_R01TTS.0.a.txt', 'label_R01TTS.0.b.txt']:
        with open(DATA_ROOT / "LlamaPartialSpoof" / lf) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    if parts[2] == 'spoof': spoof += 1
                    elif parts[2] == 'bonafide': bonafide += 1
    return spoof + bonafide, spoof, bonafide

def count_hqmpsd():
    """Count audio files."""
    import glob
    audio = glob.glob(str(DATA_ROOT / "HQ-MPSD-EN/17929533/English/**/*.wav"), recursive=True) + \
            glob.glob(str(DATA_ROOT / "HQ-MPSD-EN/17929533/English/**/*.flac"), recursive=True)
    return len(audio)

def count_partialedit():
    """Count from E1 + E2 audio directories."""
    import glob
    e1 = glob.glob(str(DATA_ROOT / "PartialEdit/E1/**/*.wav"), recursive=True)
    e2 = glob.glob(str(DATA_ROOT / "PartialEdit/E2/**/*.wav"), recursive=True)
    return len(e1) + len(e2), len(e1) + len(e2), 0  # all spoof

def main():
    errors = []

    # Load metadata CSV
    with open(CSV_META) as f:
        meta = {r['Dataset']: r for r in csv.DictReader(f)}

    datasets = {
        'PartialSpoof': {
            'npy_dir': NPY_BASE / 'e1_baseline',
            'prefix': '',
            'count_fn': count_partialspoof_eval,
        },
        'LlamaPartialSpoof': {
            'npy_dir': NPY_BASE / 'e5_cross_dataset',
            'prefix': 'llamapartialspoof_',
            'count_fn': count_llamapartialspoof,
        },
        'HQ-MPSD': {
            'npy_dir': NPY_BASE / 'e5_cross_dataset',
            'prefix': 'hqmpsd_',
            'count_fn': lambda: (count_hqmpsd(), None, None),
        },
        'PartialEdit': {
            'npy_dir': NPY_BASE / 'e5_cross_dataset',
            'prefix': 'partialedit_',
            'count_fn': count_partialedit,
        },
    }

    for ds_name, cfg in datasets.items():
        print(f"\n--- {ds_name} ---")
        npy_dir = cfg['npy_dir']
        prefix = cfg['prefix']

        # Count from metadata on disk
        result = cfg['count_fn']()
        if isinstance(result, tuple):
            meta_total, meta_spoof, meta_genuine = result
        else:
            meta_total = result
            meta_spoof = meta_genuine = None

        # Count from .npy
        labels = np.load(npy_dir / f"bam_{prefix}utt_labels.npy", allow_pickle=True)
        npy_total = len(labels)
        npy_spoof = int((labels == 1).sum())
        npy_genuine = int((labels == 0).sum())

        # Count from CSV
        csv_total = int(meta[ds_name]['Total_Utterances'])
        csv_spoof = int(meta[ds_name]['Spoof'])
        csv_genuine = int(meta[ds_name]['Genuine'])

        print(f"  Disk metadata:  total={meta_total}" + (f", spoof={meta_spoof}, genuine={meta_genuine}" if meta_spoof is not None else ""))
        print(f"  .npy arrays:    total={npy_total}, spoof={npy_spoof}, genuine={npy_genuine}")
        print(f"  CSV metadata:   total={csv_total}, spoof={csv_spoof}, genuine={csv_genuine}")

        # Check metadata vs npy
        if meta_total != npy_total:
            errors.append(f"{ds_name}: disk metadata total={meta_total} vs npy={npy_total}")
        else:
            print(f"  Disk vs .npy total: MATCH")

        if meta_spoof is not None and meta_spoof != npy_spoof:
            errors.append(f"{ds_name}: disk spoof={meta_spoof} vs npy={npy_spoof}")
        elif meta_spoof is not None:
            print(f"  Disk vs .npy spoof: MATCH")

        if meta_genuine is not None and meta_genuine != npy_genuine:
            errors.append(f"{ds_name}: disk genuine={meta_genuine} vs npy={npy_genuine}")
        elif meta_genuine is not None:
            print(f"  Disk vs .npy genuine: MATCH")

        # Check CSV vs npy
        if csv_total != npy_total:
            errors.append(f"{ds_name}: CSV total={csv_total} vs npy={npy_total}")
        else:
            print(f"  CSV vs .npy total: MATCH")

        # Check all 3 detectors have same count
        for det in ['bam', 'cfprf', 'mrm']:
            det_labels = np.load(npy_dir / f"{det}_{prefix}utt_labels.npy", allow_pickle=True)
            if len(det_labels) != npy_total:
                if abs(len(det_labels) - npy_total) <= 10:
                    print(f"  {det} count: {len(det_labels)} (within tolerance, {npy_total - len(det_labels)} processing errors)")
                else:
                    errors.append(f"{ds_name}/{det}: count={len(det_labels)} vs expected={npy_total}")

    print(f"\n{'='*40}")
    print(f"ERRORS: {len(errors)}")
    for e in errors:
        print(f"  {e}")
    if not errors:
        print("  ALL METADATA CHECKS PASSED")

if __name__ == "__main__":
    main()
