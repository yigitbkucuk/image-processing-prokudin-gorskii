import numpy as np
import cv2
import os


def crop_edges(image, crop_percent=0.1):
    """
    Kenarlardan % crop_percent oranında kırpma
    """
    height, width = image.shape[:2]
    crop_h = int(height * crop_percent)
    crop_w = int(width * crop_percent)
    return image[crop_h:height - crop_h, crop_w:width - crop_w]


def load_image(image_path):
    """
    Görüntüyü yükle ve numpy array'e dönüştür
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Görüntü bulunamadı: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    return image


def save_image(image, save_path):
    """
    Görüntüyü kaydet
    """
    cv2.imwrite(save_path, image)
    print(f"Görüntü kaydedildi: {save_path}")


def calculate_metrics(original, aligned):
    """
    Hizalama kalitesi metriklerini hesapla
    """
    mse = np.mean((original - aligned) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    return mse, psnr


def create_comparison_image(original, aligned, enhanced, save_path):
    """
    Üçlü karşılaştırma görseli oluştur - BOYUT KONTROLÜ EKLENDİ
    """
    print(
        f"   DEBUG: Comparison shapes - original: {original.shape}, aligned: {aligned.shape}, enhanced: {enhanced.shape}")

    # Boyutları kontrol et ve eşleştir
    h, w = original.shape[:2]

    # Eğer enhanced 2 boyutluysa, 3 boyutlu yap
    if len(enhanced.shape) == 2:
        enhanced = np.stack([enhanced, enhanced, enhanced], axis=-1)

    # Boyutları eşleştir
    aligned_resized = cv2.resize(aligned, (w, h))
    enhanced_resized = cv2.resize(enhanced, (w, h))

    comparison = np.hstack([original, aligned_resized, enhanced_resized])
    save_image(comparison, save_path)
    return comparison