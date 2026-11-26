# Menyalin hasil asli dari laptop Anda ke folder repo ini, lalu commit dan push ke GitHub.
#
# Sebelum menjalankan:
#   1. Buat repo KOSONG di GitHub (tanpa README) bernama
#      yolo11n-yolo26n-split-70-15-15-vestcare  di akun TijaniRobith.
#   2. Pastikan git sudah terpasang dan Anda login (jendela login GitHub akan muncul saat push).
#
# Cara pakai (PowerShell, dari folder ini):
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\siapkan_dan_push.ps1
#   (tambahkan -TanpaPush jika hanya ingin menyalin dan commit)

param(
    [string]$Sumber  = "D:\laragon\www\perbandingan_YOLOV11N&V26N_Split70-10-10",
    [string]$RepoUrl = "https://github.com/TijaniRobith/yolo11n-yolo26n-split-70-15-15-vestcare.git",
    [switch]$TanpaPush
)

$ErrorActionPreference = "Stop"
$Tujuan = $PSScriptRoot
$Hasil  = Join-Path $Sumber "hasil\07_perbandingan_model\70_15_15"
if (-not (Test-Path $Hasil)) { throw "Folder hasil tidak ditemukan: $Hasil" }

function Salin($dari, $ke) {
    if (Test-Path $dari) {
        New-Item -ItemType Directory -Force -Path (Split-Path $ke) | Out-Null
        Copy-Item $dari $ke -Force
        Write-Host "OK    $ke"
    } else {
        Write-Host "SKIP  $dari (tidak ada)" -ForegroundColor Yellow
    }
}

$T = Join-Path $Tujuan "hasil\07_perbandingan_model\70_15_15"

foreach ($f in "perbandingan_detail.csv", "perbandingan_ringkasan.csv",
               "laporan_perbandingan.txt", "perbandingan_per_kelas_lengkap.csv") {
    Salin (Join-Path $Hasil $f) (Join-Path $T $f)
}

foreach ($m in "yolo11n_seed0", "yolo26n_seed0") {
    foreach ($f in "results.csv", "args.yaml", "results.png", "confusion_matrix.png",
                   "confusion_matrix_normalized.png", "MaskPR_curve.png", "MaskF1_curve.png",
                   "BoxPR_curve.png", "weights\best.pt") {
        Salin (Join-Path $Hasil "seg\$m\$f") (Join-Path $T "seg\$m\$f")
    }
}

# Grafik split test (dibuat oleh laporan_perbandingan.py di seg\_eval\<run>_test)
foreach ($m in "yolo11n_seed0", "yolo26n_seed0") {
    foreach ($f in "confusion_matrix.png", "confusion_matrix_normalized.png",
                   "MaskPR_curve.png", "MaskF1_curve.png", "BoxPR_curve.png", "BoxF1_curve.png") {
        Salin (Join-Path $Hasil "seg\_eval\${m}_test\$f") (Join-Path $Tujuan "grafik\test\$m\$f")
    }
}

Set-Location $Tujuan
if (-not (Test-Path ".git")) { git init | Out-Null }
git add .
git commit -m "Perbandingan yolo11n-seg vs yolo26n-seg pada split 70:15:15"
git branch -M main

$remote = git remote
if ($remote) { git remote set-url origin $RepoUrl } else { git remote add origin $RepoUrl }

if ($TanpaPush) {
    Write-Host "Selesai (tanpa push). Jalankan 'git push -u origin main' untuk mengunggah."
} else {
    git push -u origin main
    Write-Host "Selesai. Buka: $($RepoUrl -replace '\.git$','')"
}
