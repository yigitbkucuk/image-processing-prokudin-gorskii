import numpy as np
import cv2
from utils import crop_edges


def compute_ssd(img1, img2):
    """SSD (Sum of Squared Differences) metrik"""
    return np.sum((img1 - img2) ** 2)


def compute_ncc(img1, img2):
    """NCC (Normalized Cross Correlation) metrik"""
    img1_normalized = (img1 - np.mean(img1)) / (np.std(img1) + 1e-8)
    img2_normalized = (img2 - np.mean(img2)) / (np.std(img2) + 1e-8)
    return np.sum(img1_normalized * img2_normalized)


def align_channels(blue, green, red, metric='ncc', search_range=15, crop_percent=0.1):
    """
    RGB kanallarını hizala
    """
    # Kenarları kırp
    blue_cropped = crop_edges(blue, crop_percent)
    green_cropped = crop_edges(green, crop_percent)
    red_cropped = crop_edges(red, crop_percent)

    # Referans kanal (mavi)
    ref_channel = blue_cropped

    # Yeşil kanalı hizala
    best_dx_g, best_dy_g = find_best_offset(ref_channel, green_cropped, metric, search_range)
    green_aligned = np.roll(green, shift=(best_dx_g, best_dy_g), axis=(0, 1))

    # Kırmızı kanalı hizala
    best_dx_r, best_dy_r = find_best_offset(ref_channel, red_cropped, metric, search_range)
    red_aligned = np.roll(red, shift=(best_dx_r, best_dy_r), axis=(0, 1))

    return blue, green_aligned, red_aligned, (best_dx_g, best_dy_g), (best_dx_r, best_dy_r)


def find_best_offset(ref_img, mov_img, metric, search_range):
    """
    En iyi kaydırma değerlerini bul
    """
    best_score = -float('inf') if metric == 'ncc' else float('inf')
    best_dx, best_dy = 0, 0

    for dx in range(-search_range, search_range + 1):
        for dy in range(-search_range, search_range + 1):
            moved = np.roll(mov_img, shift=(dx, dy), axis=(0, 1))

            if metric == 'ncc':
                score = compute_ncc(ref_img, moved)
                if score > best_score:
                    best_score = score
                    best_dx, best_dy = dx, dy
            else:  # ssd
                score = compute_ssd(ref_img, moved)
                if score < best_score:
                    best_score = score
                    best_dx, best_dy = dx, dy

    return best_dx, best_dy