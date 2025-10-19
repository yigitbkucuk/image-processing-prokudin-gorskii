# main.py
import argparse
import time
import os
from pathlib import Path
import cv2
import numpy as np
import traceback

# Yeni import'lar
from enhancement import apply_enhancements
from utils import create_comparison_image
from alignment import align_channels


def load_image(image_path):
    """Görüntüyü yükle"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Görüntü yüklenemedi: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def split_image(img):
    """Görüntüyü 3 eşit parçaya böl (B, G, R)"""
    height = img.shape[0] // 3
    blue = img[0:height, :]
    green = img[height:2 * height, :]
    red = img[2 * height:3 * height, :]
    return blue, green, red


def apply_alignment(img, dx, dy):
    """Kaydırma uygula"""
    return np.roll(img, shift=(dy, dx), axis=(0, 1))


def create_color_image(b, g, r):
    """RGB görüntü oluştur - BASİT VERSİYON"""
    # Tüm kanalları 2 boyutlu yap (grayscale)
    b_gray = b if len(b.shape) == 2 else b[:, :, 0]
    g_gray = g if len(g.shape) == 2 else g[:, :, 0]
    r_gray = r if len(r.shape) == 2 else r[:, :, 0]

    # 3 kanallı RGB oluştur
    rgb_image = np.stack([r_gray, g_gray, b_gray], axis=2)
    return rgb_image

def enhance_image(img):
    """Görüntü iyileştirme uygula"""
    print(f"   DEBUG: enhance_image input shape: {img.shape}")
    enhanced = apply_enhancements(img, techniques=['histogram', 'gamma'])
    print(f"   DEBUG: enhance_image output shape: {enhanced.shape}")
    return enhanced

def auto_crop(img):
    """Otomatik kırpma"""
    return img, (0, 0, img.shape[1], img.shape[0])  # Basit versiyon


def process_image(input_path, output_dir, metric='ncc'):
    print(f"\n{'=' * 60}")
    print(f"İşleniyor: {input_path}")
    print(f"{'=' * 60}")

    start_time = time.time()

    try:
        print("1. Görsel yükleniyor...")
        img = load_image(input_path)
        print(f"   Görsel yüklendi: {img.shape}")

        print("2. Kanallara bölünüyor...")
        b, g, r = split_image(img)
        print(f"   Bölündü: B={b.shape}, G={g.shape}, R={r.shape}")

        print(f"3. Hizalama başlıyor ({metric} metriği)...")

        # DEBUG: alignment fonksiyonunu kontrol et
        print("   DEBUG: align_channels fonksiyonu çağrılıyor...")
        b_aligned, g_aligned, r_aligned, (dx_g, dy_g), (dx_r, dy_r) = align_channels(
            b, g, r, metric=metric, search_range=15
        )
        print("   DEBUG: align_channels başarılı!")
        print(f"   Green: dx={dx_g}, dy={dy_g}")
        print(f"   Red: dx={dx_r}, dy={dy_r}")

        print("4. Renkli görüntüler oluşturuluyor...")
        img_unaligned = create_color_image(b, g, r)
        print(f"   DEBUG: img_unaligned shape: {img_unaligned.shape}")

        img_aligned = create_color_image(b_aligned, g_aligned, r_aligned)
        print(f"   DEBUG: img_aligned shape: {img_aligned.shape}")

        print("5. Görüntü iyileştirme...")
        img_enhanced = enhance_image(img_aligned)
        print(f"   DEBUG: img_enhanced shape: {img_enhanced.shape}")

        print("6. Otomatik kırpma...")
        img_final, crop_coords = auto_crop(img_enhanced)
        print(f"   Kırpma koordinatları: {crop_coords}")

        # Üçlü karşılaştırma görseli oluştur
        base_name = Path(input_path).stem
        comparison_path = f"{output_dir}/comparison_{base_name}.jpg"
        print(f"7. Karşılaştırma görseli oluşturuluyor: {comparison_path}")
        create_comparison_image(
            img_unaligned,
            img_aligned,
            img_enhanced,
            comparison_path
        )
        print("   DEBUG: create_comparison_image başarılı!")

        print("8. Sonuçlar kaydediliyor...")
        cv2.imwrite(f"{output_dir}/{base_name}_unaligned.jpg", cv2.cvtColor(img_unaligned, cv2.COLOR_RGB2BGR))
        print(f"   ✓ {base_name}_unaligned.jpg kaydedildi")

        cv2.imwrite(f"{output_dir}/{base_name}_aligned.jpg", cv2.cvtColor(img_aligned, cv2.COLOR_RGB2BGR))
        print(f"   ✓ {base_name}_aligned.jpg kaydedildi")

        cv2.imwrite(f"{output_dir}/{base_name}_enhanced.jpg", cv2.cvtColor(img_enhanced, cv2.COLOR_RGB2BGR))
        print(f"   ✓ {base_name}_enhanced.jpg kaydedildi")

        cv2.imwrite(f"{output_dir}/{base_name}_final.jpg", cv2.cvtColor(img_final, cv2.COLOR_RGB2BGR))
        print(f"   ✓ {base_name}_final.jpg kaydedildi")

        elapsed_time = time.time() - start_time
        print(f"✓ TAMAMLANDI! Süre: {elapsed_time:.2f} saniye")

        return {
            'image': base_name,
            'g_shift': (dx_g, dy_g),
            'r_shift': (dx_r, dy_r),
            'time': elapsed_time
        }

    except Exception as e:
        print(f"❌ CRITICAL HATA: {str(e)}")
        print("DETAYLI HATA:")
        traceback.print_exc()

        # Ek debug info
        print("\n🔍 DEBUG BİLGİSİ:")
        print(f"   Input: {input_path}")
        print(f"   Image shape: {img.shape if 'img' in locals() else 'Yüklenemedi'}")
        if 'b' in locals():
            print(f"   Blue shape: {b.shape}")
            print(f"   Green shape: {g.shape}")
            print(f"   Red shape: {r.shape}")
        if 'b_aligned' in locals():
            print(f"   Blue aligned shape: {b_aligned.shape}")
            print(f"   Green aligned shape: {g_aligned.shape}")
            print(f"   Red aligned shape: {r_aligned.shape}")

        return None


def main():
    print("=== PROKUDIN-GORSKII GÖRÜNTÜ İŞLEME ===")

    parser = argparse.ArgumentParser(description='Prokudin-Gorskii Görüntü İşleme')
    parser.add_argument('--input', required=True, help='Girdi görüntüsü veya klasörü')
    parser.add_argument('--output', default='results', help='Çıktı klasörü')
    parser.add_argument('--metric', default='ncc', choices=['ssd', 'ncc'], help='Hizalama metriği')

    args = parser.parse_args()

    print(f"Girdi: {args.input}")
    print(f"Çıktı: {args.output}")
    print(f"Metrik: {args.metric}")

    os.makedirs(args.output, exist_ok=True)
    input_path = Path(args.input)

    # DOSYA BULMA KISMINI DÜZELTTİM
    if input_path.is_file():
        files = [input_path]
        print(f"Tek dosya işlenecek: {files[0]}")
    else:
        files = []
        # TÜM dosyaları al, sonra filtrele
        all_files = list(input_path.glob('*'))
        files = [f for f in all_files if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']]

        print(f"Klasörde {len(files)} dosya bulundu:")
        for file in files:
            print(f"  - {file.name}")

    if not files:
        print("❌ Hiçbir görüntü dosyası bulunamadı!")
        return

    results = []
    for file in files:
        result = process_image(str(file), args.output, args.metric)
        if result is not None:
            results.append(result)
        else:
            print(f"❌ {file.name} işlenirken hata oluştu!")

    if results:
        print(f"\n{'=' * 60}")
        print("ÖZET SONUÇLAR")
        print(f"{'=' * 60}")
        for r in results:
            print(f"{r['image']:<15} G{r['g_shift']} R{r['r_shift']} {r['time']:>6.2f}s")
    else:
        print("\n❌ Hiçbir görsel başarıyla işlenemedi!")


if __name__ == '__main__':
    main()