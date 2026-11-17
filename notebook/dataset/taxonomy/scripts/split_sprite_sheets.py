"""Lossless sprite crops, with explicit pending states for ambiguous layouts."""
from collections import Counter

import numpy as np
from PIL import Image


def foreground(image):
    pixels = np.asarray(image.convert('RGBA'))
    if np.any(pixels[:, :, 3] == 0):
        return pixels[:, :, 3] > 0, None
    border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
    color, count = Counter(map(tuple, border)).most_common(1)[0]
    if count / len(border) < 0.8:
        return None, None
    return np.any(pixels != np.asarray(color), axis=2), list(map(int, color))


def occupied_bands(occupied, min_gap=2):
    """Separate only on at least min_gap entirely empty lines."""
    indices = np.flatnonzero(occupied)
    if len(indices) == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > min_gap)
    starts = [int(indices[0])] + [int(indices[i + 1]) for i in breaks]
    ends = [int(indices[i]) + 1 for i in breaks] + [int(indices[-1]) + 1]
    return list(zip(starts, ends))


def propose_crops(image, grid=None):
    mask, background = foreground(image)
    if mask is None:
        return [], {'status': 'pending_geometry', 'reason': 'No reliable transparent/solid background'}
    height, width = mask.shape
    boxes = []
    if grid:
        cell_w, cell_h = grid['cell_width'], grid['cell_height']
        margin_x, margin_y = grid.get('margin_x', 0), grid.get('margin_y', 0)
        spacing_x, spacing_y = grid.get('spacing_x', 0), grid.get('spacing_y', 0)
        if min(cell_w, cell_h) <= 0 or min(margin_x, margin_y, spacing_x, spacing_y) < 0:
            raise ValueError('Invalid sprite grid')
        if (width - margin_x + spacing_x) % (cell_w + spacing_x) or (height - margin_y + spacing_y) % (cell_h + spacing_y):
            raise ValueError('Grid does not fit the full source image')
        covered = np.zeros_like(mask)
        for y in range(margin_y, height, cell_h + spacing_y):
            for x in range(margin_x, width, cell_w + spacing_x):
                right, bottom = x + cell_w, y + cell_h
                tile = mask[y:bottom, x:right]
                if tile.any():
                    boxes.append([x, y, right, bottom])
                covered[y:bottom, x:right] = True
        if np.any(mask & ~covered):
            raise ValueError('Grid would discard visible pixels in margins/gutters')
        method = 'explicit_grid'
    else:
        for top, bottom in occupied_bands(mask.any(axis=1)):
            band = mask[top:bottom]
            for left, right in occupied_bands(band.any(axis=0)):
                ys, xs = np.nonzero(mask[top:bottom, left:right])
                if len(xs):
                    boxes.append([left + int(xs.min()), top + int(ys.min()),
                                  left + int(xs.max()) + 1, top + int(ys.max()) + 1])
        method = 'empty_row_column_separators'
    if len(boxes) < 2:
        return [], {'status': 'pending_geometry', 'reason': 'No unambiguous multi-sprite partition'}
    # Coverage is mandatory: never silently discard small pieces, text or effects.
    covered = np.zeros_like(mask)
    for left, top, right, bottom in boxes:
        if covered[top:bottom, left:right].any():
            raise ValueError('Overlapping crop boxes')
        covered[top:bottom, left:right] = True
    if np.any(mask & ~covered):
        raise ValueError('Crop partition would discard foreground')
    return boxes, {'status': 'auto_split_unreviewed', 'method': method,
                   'key_background_rgba': background, 'foreground_coverage': 1.0,
                   'crop_count': len(boxes), 'needs_visual_review': True}


def save_crop(image, bbox, destination, background=None):
    crop = image.convert('RGBA').crop(bbox)
    # Preserve pixels and alpha exactly. Background keying is recorded, not applied.
    crop.save(destination, format='PNG')
    return crop
