"""Compile the project's taxonomies into strict JSON Schema and llama.cpp GBNF."""
import copy
import json
from pathlib import Path

TAXONOMY_DIR = Path(__file__).resolve().parents[1]


def load_taxonomies():
    dimensions, leaves, documents = {}, {}, {}
    def walk(nodes, prefix=()):
        result = {}
        for name, node in nodes.items():
            path = [*prefix, name]
            if node['children']:
                result.update(walk(node['children'], path))
            else:
                result[name] = {'path': path, 'description': node['description']}
        return result
    for filename in ('taxonomies.json', 'taxonomies_couleurs.json'):
        document = json.loads((TAXONOMY_DIR / filename).read_text())
        documents[filename] = document
        dimensions.update(document['dimensions'])
        leaves.update({key: walk(tree) for key, tree in document['hierarchies'].items()})
    return dimensions, leaves, documents


def json_schema(dimensions, leaves, palette_size=None, transparent=None):
    fields = {}
    for field, options in leaves.items():
        values = list(options)
        if field == 'palette_size' and palette_size:
            values = [palette_size]
        if field == 'background_type' and transparent is not None:
            values = ['transparent'] if transparent else [v for v in values if v != 'transparent']
        if field == 'background_scene' and transparent is True:
            values = ['not_applicable']
        item = {'type': 'string', 'enum': values}
        if dimensions[field].get('cardinality') == 'multi_label':
            item = {'type': 'array', 'items': item, 'minItems': 1,
                    'maxItems': len(values), 'uniqueItems': True}
        fields[field] = item
    labels = {'type': 'object', 'properties': fields,
              'required': list(fields), 'additionalProperties': False}
    if transparent is False:
        branches = []
        for background in fields['background_type']['enum']:
            branch = copy.deepcopy(labels)
            branch['properties']['background_type']['enum'] = [background]
            branch['properties']['background_scene']['enum'] = (
                ['not_applicable'] if background == 'solid_color' else
                [v for v in leaves['background_scene'] if v != 'not_applicable'])
            branches.append(branch)
        labels = {'oneOf': branches}
    return {'type': 'object', 'additionalProperties': False,
            'properties': {
                # Describe first: avoid a caption that merely paraphrases the label choices.
                'caption_candidate': {'type': 'string', 'minLength': 1, 'maxLength': 640,
                                      'pattern': r'^A(n)? [^\n\r]{1,636}[.!?]$'},
                'labels': labels},
            'required': ['labels', 'caption_candidate']}


def literal(text):
    return json.dumps(text, ensure_ascii=False)


def grammar(dimensions, leaves, palette_size=None, transparent=None):
    # Base properties, with per-image exact constraints applied below.
    schema = json_schema(dimensions, leaves, palette_size, None)
    if transparent is not None:
        spec = schema['properties']['labels']['properties']['background_type']
        spec['enum'] = ['transparent'] if transparent else [v for v in spec['enum'] if v != 'transparent']
    rules = ['ws ::= [ \\t\\n\\r]{0,4}',
             r'char ::= [^"\\\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})',
             r'caption ::= "\"" ("A " | "An ") char{1,636} "." "\""']
    for field, spec in schema['properties']['labels']['properties'].items():
        name = field.replace('_', '-')
        if spec['type'] == 'array':
            values = [v for v in spec['items']['enum'] if v not in {'unknown', 'not_applicable'}]
            # Canonical subsets: nonempty, ordered, no duplicate and no unknown mixed with colors.
            for i in range(len(values) - 1, -1, -1):
                rule = literal(json.dumps(values[i]))
                if i + 1 < len(values):
                    tail = f'{name}-subset-{i + 1}'
                    rule = f'{rule} ("," ws {tail})? | {tail}'
                rules.append(f'{name}-subset-{i} ::= {rule}')
            alternatives = [f'{name}-subset-0'] + [literal(json.dumps(v)) for v in
                            spec['items']['enum'] if v in {'unknown', 'not_applicable'}]
            rules.append(f'{name} ::= "[" ws (' + ' | '.join(alternatives) + ') ws "]"')
        else:
            rules.append(f'{name} ::= ' + ' | '.join(literal(json.dumps(v)) for v in spec['enum']))
    # Encode the background/scene dependency directly, not just in post-validation.
    alternatives = []
    backgrounds = schema['properties']['labels']['properties']['background_type']['enum']
    for index, background in enumerate(backgrounds):
        terms = ['"{" ws']
        for i, field in enumerate(leaves):
            if i:
                terms.append('"," ws')
            terms.append(literal(json.dumps(field)) + ' ws ":" ws')
            if field == 'background_type':
                terms.append(literal(json.dumps(background)))
            elif field == 'background_scene' and background in {'transparent', 'solid_color'}:
                terms.append(literal(json.dumps('not_applicable')))
            elif field == 'background_scene' and background == 'complex':
                terms.append('(' + ' | '.join(literal(json.dumps(v)) for v in leaves[field]
                                             if v != 'not_applicable') + ')')
            else:
                terms.append(field.replace('_', '-'))
            terms.append('ws')
        terms.append('"}"')
        rules.append(f'labels-{index} ::= ' + ' '.join(terms))
        alternatives.append(f'labels-{index}')
    rules.insert(0, 'root ::= "{" ws ' + literal('"caption_candidate"') +
                 ' ws ":" ws caption ws "," ws ' + literal('"labels"') + ' ws ":" ws (' +
                 ' | '.join(alternatives) + ') ws "}" ws')
    return '\n'.join(rules) + '\n'


def caption_prompt():
    return ('Write an English training caption for the image. Describe shape, orientation and '
            'spatial arrangement of visible parts. Start with the view and overall silhouette, '
            'then describe two distinctive features and where they are relative to the body. '
            'Prefer concrete physical terms to broad category guesses. Mention colors as '
            'properties of structures. Do not invent identity, emotion, materials or gameplay '
            'role. Avoid claiming light emission from bright pixels. Write 1-2 complete '
            'sentences, aiming for 35-55 words for a complex subject, fewer for a simple icon, '
            'and at most 640 characters. Begin directly with "A " or "An " followed by the subject. '
            'Never use framing such as "The image displays", "This image shows", "The picture", '
            'or "There is". Do not refer to the image or picture itself. '
            'End with a period. Return JSON with caption_candidate.')


def prompt(dimensions, leaves, labels_only=False):
    specs = {key: {'definition': dimensions[key]['definition'],
                   'applicability': dimensions[key]['applicability'],
                   'cardinality': dimensions[key].get('cardinality', 'single_label'),
                   'values': {value: spec['description'] for value, spec in options.items()}}
             for key, options in leaves.items()}
    instructions = ('Annotate the visible image. Return only JSON with labels and caption_candidate. '
            'Choose allowed leaf values, unknown when uncertain, not_applicable when irrelevant. '
            'All dimensions are required. Respect the source transparency and exact palette measurement. '
            'A sheet or strip with several frames is not a single_sprite. '
            'For dominant_color_families choose visually important colors only, in the listed order, '
            'without duplicates; unknown/not_applicable must be alone. '
            'For transparent/solid_color background, background_scene is not_applicable. '
            'For complex background, choose a scene or unknown. '
            'Do not infer identity, game, emotion, '
            'material, narrative, gameplay role or rights. Do not treat pixel-art clusters as defects. '
            'Schema definitions: ' + json.dumps(specs, ensure_ascii=False, separators=(',', ':')) +
            '\nWrite caption_candidate FIRST, before labels. Build a coherent visual description '
            'from the pixels, not a paraphrase of taxonomy values. Start with overall silhouette '
            'and orientation, then describe 2-3 distinctive visible structures, their colors and '
            'where they are relative to the body. End with pose/background only if useful. '
            'Use spatial relationships rather than an unordered list of colors. '
            'Avoid generic phrases such as "with blue, red and purple colors". '
            'Do not invent features to fill the caption; use a short description for a simple icon. '
            'Use English, 1-2 complete sentences, aim for 35-55 words and at most 640 characters. Labels must still describe the '
            'image independently, not uncritically repeat guesses made in the caption. '
            'Start the caption directly with "A " or "An " and the subject. Never write '
            '"The image displays", "This image shows" or another framing phrase.')
    if labels_only:
        return instructions.split('\nWrite caption_candidate FIRST')[0].replace(
            'Return only JSON with labels and caption_candidate.', 'Return only JSON with labels.')
    return instructions


def validate(result, dimensions, leaves, palette_size, transparent):
    if not isinstance(result, dict) or set(result) != {'labels', 'caption_candidate'}:
        raise ValueError('Unexpected result fields')
    labels = result['labels']
    if not isinstance(labels, dict) or set(labels) != set(leaves):
        raise ValueError('Missing/extra taxonomy dimension')
    for field, options in leaves.items():
        value = labels[field]
        if dimensions[field].get('cardinality') == 'multi_label':
            if not isinstance(value, list) or not value or any(not isinstance(v, str) for v in value):
                raise ValueError(f'Invalid array for {field}')
            if len(set(value)) != len(value) or any(v not in options for v in value):
                raise ValueError(f'Invalid/duplicate leaf for {field}')
            if len(value) > 1 and set(value) & {'unknown', 'not_applicable'}:
                raise ValueError(f'Mixed sentinel for {field}')
        elif not isinstance(value, str) or value not in options:
            raise ValueError(f'Invalid leaf for {field}')
    if labels['palette_size'] != palette_size:
        raise ValueError('Palette disagrees with source pixels')
    if (labels['background_type'] == 'transparent') != transparent:
        raise ValueError('Transparency disagrees with alpha channel')
    if labels['background_type'] in {'transparent', 'solid_color'} and labels['background_scene'] != 'not_applicable':
        raise ValueError('Invalid background scene')
    if labels['background_type'] == 'complex' and labels['background_scene'] == 'not_applicable':
        raise ValueError('Complex background without a scene')
    if not isinstance(result['caption_candidate'], str) or not 1 <= len(result['caption_candidate'].strip()) <= 640:
        raise ValueError('Invalid caption')
    if not result['caption_candidate'].strip().endswith(('.', '!', '?')):
        raise ValueError('Caption must end with a complete sentence')
    if not result['caption_candidate'].startswith(('A ', 'An ')):
        raise ValueError('Caption must start directly with A/An and the subject')
    return result
