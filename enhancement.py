import numpy as np
import cv2


def histogram_equalization(image):
    """
    Histogram eşitleme ile kontrast artırma
    """
    # Eğer görüntü zaten RGB ise
    if len(image.shape) == 3 and image.shape[2] == 3:
        # RGB'den YUV'ye dönüşüm
        img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        # Y kanalında histogram eşitleme
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        # YUV'den RGB'ye dönüşüm
        equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    else:
        # Grayscale görüntü için direkt histogram eşitleme
        equalized = cv2.equalizeHist(image)

    return equalized


def gamma_correction(image, gamma=1.2):
    """
    Gama düzeltme ile parlaklık ayarlama
    """
    # Gama düzeltme formülü
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def laplacian_sharpening(image):
    """
    Laplasyen filtresi ile kenar keskinleştirme
    """
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened


def adjust_saturation(image, factor=1.3):
    """
    HSV uzayında renk doygunluğu ayarlama (sadece RGB için)
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        saturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return saturated
    else:
        return image  # Grayscale için orijinali döndür


def auto_contrast(image):
    """
    Otomatik kontrast ayarlama
    """
    # Piksel değerlerini normalize etme
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized


def apply_enhancements(image, techniques=['histogram', 'gamma']):
    """
    Tüm iyileştirme tekniklerini uygula
    """
    enhanced = image.copy()

    for technique in techniques:
        if technique == 'histogram':
            enhanced = histogram_equalization(enhanced)
        elif technique == 'gamma':
            enhanced = gamma_correction(enhanced, gamma=1.2)
        elif technique == 'sharpening':
            enhanced = laplacian_sharpening(enhanced)
        elif technique == 'saturation':
            enhanced = adjust_saturation(enhanced, factor=1.3)
        elif technique == 'contrast':
            enhanced = auto_contrast(enhanced)

    return enhanced