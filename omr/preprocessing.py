from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pdf2image import convert_from_path

from omr.utils import imread, is_pdf_file


def pdf_page_to_bgr(
    pdf_path: str | Path,
    dpi: int = 300,
    page: int = 1,
    poppler_path: str | None = None,
) -> np.ndarray:
    pages = convert_from_path(
        str(pdf_path),
        dpi=dpi,
        first_page=page,
        last_page=page,
        poppler_path=poppler_path,
    )
    if not pages:
        raise ValueError(f"PDF no readable pages: {pdf_path}")
    rgb = np.array(pages[0])
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def load_input_as_bgr(
    path: str | Path,
    dpi: int = 300,
    page: int = 1,
    poppler_path: str | None = None,
) -> np.ndarray:
    if is_pdf_file(path):
        return pdf_page_to_bgr(path, dpi=dpi, page=page, poppler_path=poppler_path)
    return imread(path, cv2.IMREAD_COLOR)


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def adaptive_binarize(gray: np.ndarray, block_size: int = 31, c: int = 15) -> np.ndarray:
    if block_size % 2 == 0:
        block_size += 1
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c,
    )


def estimate_skew_angle(gray: np.ndarray) -> float:
    bin_img = adaptive_binarize(gray)
    coords = cv2.findNonZero(bin_img)
    if coords is None or len(coords) < 50:
        return 0.0
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    return float(angle)


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(m[0, 0])
    sin = abs(m[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    m[0, 2] += (new_w / 2) - center[0]
    m[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(image, m, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))


def deskew(image: np.ndarray, max_abs_angle: float = 8.0) -> np.ndarray:
    gray = to_gray(image)
    angle = estimate_skew_angle(gray)
    if abs(angle) < 0.15 or abs(angle) > max_abs_angle:
        return image
    return rotate_image(image, angle)


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_warp(image: np.ndarray, pts: np.ndarray, out_size: tuple[int, int] | None = None) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_w = int(max(width_a, width_b))
    max_h = int(max(height_a, height_b))
    if out_size is not None:
        max_w, max_h = out_size
    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype=np.float32,
    )
    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, m, (max_w, max_h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))


def detect_page_corners(image: np.ndarray) -> np.ndarray | None:
    gray = to_gray(image)
    bin_img = adaptive_binarize(gray, block_size=41, c=12)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_area = image.shape[0] * image.shape[1]
    for contour in contours[:10]:
        area = cv2.contourArea(contour)
        if area < img_area * 0.35:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None


def align_to_template_orb(
    scan: np.ndarray,
    template: np.ndarray,
    nfeatures: int = 5000,
    min_matches: int = 20,
    min_inliers: int = 30,
    keep_ratio: float = 0.25,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    gray_scan = to_gray(scan)
    gray_tpl = to_gray(template)
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(gray_scan, None)
    kp2, des2 = orb.detectAndCompute(gray_tpl, None)
    debug: dict[str, Any] = {"num_kp_scan": len(kp1), "num_kp_template": len(kp2)}
    if des1 is None or des2 is None or len(kp1) < min_matches or len(kp2) < min_matches:
        return None, debug
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)
    keep = max(min_matches, int(len(matches) * keep_ratio))
    matches = matches[:keep]
    debug["num_matches"] = len(matches)
    if len(matches) < min_matches:
        return None, debug
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    h_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    debug["inliers"] = inliers
    if h_matrix is None or inliers < min_inliers:
        return None, debug
    h, w = template.shape[:2]
    aligned = cv2.warpPerspective(scan, h_matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return aligned, debug


def rectify_or_align(
    scan: np.ndarray,
    template: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    debug_images: dict[str, np.ndarray] = {}
    debug_meta: dict[str, Any] = {}

    working = deskew(scan, max_abs_angle=cfg.get("max_abs_deskew_angle", 8.0)) if cfg.get("deskew", True) else scan
    debug_images["deskewed"] = working

    aligned, orb_meta = align_to_template_orb(
        working,
        template,
        nfeatures=cfg.get("orb_features", 5000),
        min_matches=cfg.get("min_match_count", 20),
        min_inliers=cfg.get("orb_min_inliers", 30),
        keep_ratio=cfg.get("orb_keep_ratio", 0.25),
    )
    debug_meta["orb"] = orb_meta
    if aligned is not None:
        debug_images["aligned"] = aligned
        return aligned, debug_images, debug_meta

    corners = detect_page_corners(working)
    if corners is not None:
        h, w = template.shape[:2]
        aligned = four_point_warp(working, corners, out_size=(w, h))
        debug_images["aligned"] = aligned
        debug_meta["fallback"] = "page_corners"
        return aligned, debug_images, debug_meta

    resized = cv2.resize(working, (template.shape[1], template.shape[0]), interpolation=cv2.INTER_LINEAR)
    debug_images["aligned"] = resized
    debug_meta["fallback"] = "resize_only"
    return resized, debug_images, debug_meta


def preprocess_scan_and_template(
    scan_path: str | Path,
    template_path: str | Path,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    dpi = int(cfg.get("dpi", 300))
    page = int(cfg.get("page", 1))
    poppler_path = cfg.get("poppler_path")
    template = load_input_as_bgr(template_path, dpi=dpi, page=page, poppler_path=poppler_path)
    scan = load_input_as_bgr(scan_path, dpi=dpi, page=page, poppler_path=poppler_path)
    aligned, debug_images, debug_meta = rectify_or_align(scan, template, cfg)
    gray = to_gray(aligned)
    binary = adaptive_binarize(
        gray,
        block_size=int(cfg.get("adaptive_block_size", 31)),
        c=int(cfg.get("adaptive_c", 15)),
    )
    debug_images["gray"] = gray
    debug_images["binary"] = binary
    return aligned, binary, debug_images, debug_meta


__all__ = [
    "pdf_page_to_bgr",
    "load_input_as_bgr",
    "to_gray",
    "adaptive_binarize",
    "deskew",
    "order_points",
    "four_point_warp",
    "detect_page_corners",
    "align_to_template_orb",
    "rectify_or_align",
    "preprocess_scan_and_template",
]
