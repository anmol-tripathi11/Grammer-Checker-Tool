import json
from grammar_predictor import GrammarErrorPredictor

p = GrammarErrorPredictor()

print('LABEL_CLASSES:')
print(json.dumps(list(p.label_encoder.classes_), indent=2))
print('\nNumber of classes:', len(p.label_encoder.classes_))

samples = [
    'Yesterday I go to market.',
    'I have saw that movie.',
    'She will went home.'
]

print('\nSAMPLE PREDICTIONS:')
for s in samples:
    print('\nINPUT:', s)
    print(p.predict(s))
