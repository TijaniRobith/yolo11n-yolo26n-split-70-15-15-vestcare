"""Buat laporan txt + CSV perbandingan YOLO11n-seg vs YOLO26n-seg
dari run yang SUDAH ADA (tanpa training ulang).

Simpan di folder scripts, lalu jalankan (venv aktif):
  python laporan_perbandingan.py --device 0
"""
import argparse
import csv
import statistics as st
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

MODELS = ["yolo11n", "yolo26n"]
RAIZ = Path(__file__).resolve().parent.parent


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) else 0.0


def baca_results(run_dir):
    """Ambil epoch efektif, epoch terbaik, dan waktu latih dari results.csv."""
    path = run_dir / "results.csv"
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        baris = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    if not baris:
        return {}

    def fit(r):
        return sum(0.1 * float(r[f"metrics/mAP50({t})"]) + 0.9 * float(r[f"metrics/mAP50-95({t})"])
                   for t in ("B", "M"))
    terbaik = max(baris, key=fit)
    return {"epoch_efektif": int(float(baris[-1]["epoch"])),
            "epoch_terbaik": int(float(terbaik["epoch"])),
            "waktu_latih_jam": round(float(baris[-1]["time"]) / 3600, 3)}


def evaluasi(bobot, a, split, nama):
    model = YOLO(str(bobot))
    params, gflops = None, None
    try:
        model.fuse()
        params = round(sum(p.numel() for p in model.model.parameters()) / 1e6, 3)
        from ultralytics.utils.torch_utils import get_flops
        gflops = round(float(get_flops(model.model, a.imgsz)), 2)
    except Exception as e:  # jangan hentikan laporan hanya karena info ukuran model
        print(f"[laporan] info ukuran model tidak tersedia: {e}")
    r = model.val(data=a.data, split=split, imgsz=a.imgsz, batch=a.batch,
                  device=a.device or None, plots=(split == "test"), verbose=False,
                  project=str(a.root / "_eval"), name=f"{nama}_{split}", exist_ok=True)
    out = {"params_juta": params, "gflops": gflops,
           "ms_pre": r.speed["preprocess"], "ms_inf": r.speed["inference"],
           "ms_post": r.speed["postprocess"], "kelas": {}}
    for tipe, m in (("box", r.box), ("mask", r.seg)):
        out[tipe] = {"P": float(m.mp), "R": float(m.mr), "mAP50": float(m.map50), "mAP50-95": float(m.map)}
        for i, c in enumerate(m.ap_class_index):
            p, rc, ap50, ap = m.class_result(i)
            out["kelas"].setdefault(r.names[int(c)], {})[tipe] = {
                "P": float(p), "R": float(rc), "mAP50": float(ap50), "mAP50-95": float(ap)}
    return out


def rata(daftar, *kunci):
    nilai = []
    for d in daftar:
        for k in kunci:
            d = d[k]
        nilai.append(d)
    return st.mean(nilai), (st.stdev(nilai) if len(nilai) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=RAIZ / "hasil/07_perbandingan_model/70_15_15/seg")
    ap.add_argument("--data", default=str(RAIZ / "hasil/06_dataset_yolo26_70_15_15/seg/data.yaml"))
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    L, csv_rows = [], []
    w = L.append
    w("=" * 70)
    w("LAPORAN PERBANDINGAN YOLO11n-seg vs YOLO26n-seg (split 70:15:15)")
    w(f"Dibuat      : {datetime.now():%Y-%m-%d %H:%M}")
    w(f"Folder run  : {a.root}")
    w(f"Dataset     : {a.data}")
    w(f"imgsz/batch : {a.imgsz} / {a.batch}")
    w("=" * 70)

    hasil = {}  # hasil[model][split] = [evaluasi per seed]
    info_latih = {}
    for m in MODELS:
        runs = sorted(a.root.glob(f"{m}_seed*"))
        if not runs:
            w(f"[PERINGATAN] run {m} tidak ditemukan di {a.root}")
            continue
        hasil[m], info_latih[m] = {"val": [], "test": []}, []
        for run in runs:
            bobot = run / "weights" / "best.pt"
            if not bobot.exists():
                w(f"[PERINGATAN] {bobot} tidak ada")
                continue
            info_latih[m].append({"run": run.name, **baca_results(run)})
            for split in ("val", "test"):
                print(f"[laporan] evaluasi {run.name} ({split}) ...")
                hasil[m][split].append(evaluasi(bobot, a, split, run.name))

    for m in hasil:
        w("")
        w("-" * 70)
        w(f"MODEL {m}-seg  | jumlah run (seed): {len(info_latih[m])}")
        for i in info_latih[m]:
            w(f"  {i['run']}: epoch efektif {i.get('epoch_efektif')}, epoch terbaik "
              f"{i.get('epoch_terbaik')}, waktu latih {i.get('waktu_latih_jam')} jam")
        for split in ("test", "val"):
            ev = hasil[m][split]
            if not ev:
                continue
            w("")
            w(f"[{split.upper()}] rata-rata {len(ev)} run (± simpangan baku)")
            for tipe in ("mask", "box"):
                s = {k: rata(ev, tipe, k) for k in ("P", "R", "mAP50", "mAP50-95")}
                f = st.mean(f1(e[tipe]["P"], e[tipe]["R"]) for e in ev)
                w(f"  {tipe:4s} P {s['P'][0]:.4f} | R {s['R'][0]:.4f} | F1 {f:.4f} | "
                  f"mAP50 {s['mAP50'][0]:.4f}±{s['mAP50'][1]:.4f} | "
                  f"mAP50-95 {s['mAP50-95'][0]:.4f}±{s['mAP50-95'][1]:.4f}")
            w(f"  Kecepatan: {rata(ev, 'ms_inf')[0]:.2f} ms inferensi "
              f"(+{rata(ev, 'ms_pre')[0]:.2f} pre, +{rata(ev, 'ms_post')[0]:.2f} post) per gambar")
            w(f"  Parameter: {ev[0]['params_juta']} juta | {ev[0]['gflops']} GFLOPs (fused)")
            w("  Per kelas (mask | box):  P / R / mAP50 / mAP50-95")
            for kelas in ev[0]["kelas"]:
                baris = []
                for tipe in ("mask", "box"):
                    v = {k: rata([{"x": e["kelas"][kelas][tipe][k]} for e in ev], "x")[0]
                         for k in ("P", "R", "mAP50", "mAP50-95")}
                    baris.append(f"{v['P']:.3f}/{v['R']:.3f}/{v['mAP50']:.3f}/{v['mAP50-95']:.3f}")
                    csv_rows.append({"model": m, "split": split, "kelas": kelas, "tipe": tipe,
                                     **{k: round(x, 4) for k, x in v.items()}})
                w(f"    {kelas:7s} {baris[0]}  |  {baris[1]}")

    if len(hasil) == 2:
        w("")
        w("=" * 70)
        w("SELISIH (yolo26n - yolo11n), nilai + berarti yolo26n lebih tinggi")
        for split in ("test", "val"):
            for tipe in ("mask", "box"):
                for k in ("mAP50", "mAP50-95"):
                    d = rata(hasil["yolo26n"][split], tipe, k)[0] - rata(hasil["yolo11n"][split], tipe, k)[0]
                    w(f"  {split:4s} {tipe:4s} {k:9s}: {d:+.4f}")
        w("Catatan: selisih kecil dengan sedikit seed belum cukup untuk klaim tegas.")

    teks = "\n".join(L)
    print("\n" + teks)
    (a.root.parent / "laporan_perbandingan.txt").write_text(teks, encoding="utf-8")
    with open(a.root.parent / "perbandingan_per_kelas_lengkap.csv", "w", newline="", encoding="utf-8") as f:
        if csv_rows:
            wr = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            wr.writeheader()
            wr.writerows(csv_rows)
    print(f"\nTersimpan di: {a.root.parent}")


if __name__ == "__main__":
    main()
