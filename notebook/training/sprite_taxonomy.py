"""Taxonomie descriptive des sprites ; les annotations restent à vérifier."""
import re

ROLE_WORDS = re.compile(r'\b(?:enemy|enemies|playable|player|npc|boss|pet|ally|allies|allied|opponent|opponents)\b', re.I)


def clean_caption(text):
    text = re.sub(r'\b(?:enemy|playable|player|npc|boss|pet|ally|allied|opponent)\s+(?=(?:character|sprite|creature|monster|figure|unit)\b)', '', text or '', flags=re.I)
    text = re.sub(r'\b(?:enemy|enemies|player|npc|boss|pet|ally|allies|opponent|opponents)\b', 'character', text, flags=re.I)
    text = re.sub(r'\b(?:playable|allied)\b', '', text, flags=re.I)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def visual_hierarchy(view, pose):
    # Valeurs de position explicitement reconnues ; les champs absents ne sont pas inventés.
    views = {'front', 'side', 'back', 'three_quarter'}
    poses = {'standing', 'airborne', 'crouching', 'lying', 'prone', 'flying', 'kneeling'}
    view = view if view in views else None
    pose = pose if pose in poses else None
    path = ['sprite'] + ([view] if view else []) + ([pose] if pose else [])
    return {'level_1': 'sprite', 'level_2': view, 'level_3': pose, 'path': path}


def prefix(hierarchy, canvas_size=(96, 96)):
    width, height = canvas_size
    return ', '.join(hierarchy['path'] + ['pixel art', f'{width}x{height}'])
