#
# Entry point for a recognition grammar
#

import itertools

lexicon = {'a': {'a'}, 'b': {'b'}, 'c': {'c'}, 'd': {'d'},
           'the': {'D'},
           'dog': {'N'},
           'barks': {'V', 'V/INTR', '-COMP:T/inf'},
           'man': {'N'},
           'v': {'v', '#X'},
           'bites': {'V', '!COMP:D'}}

lexical_redundancy_rules = {'D': {'!COMP:N', '-COMP:Adv', '-SPEC:C', '-SPEC:T', '-SPEC:N', '-SPEC:V', '-SPEC:D', '-SPEC:P', '-SPEC:T/inf', '-SPEC:Adv'},
                            'V': {'-SPEC:C', '-SPEC:N', '-SPEC:T', '-SPEC:T/inf', '-COMP:A'},
                            'N': {'-COMP:A', '-SPEC:Adv', '-COMP:V', '-COMP:D', '-COMP:V', '-COMP:T', '-COMP:Adv', '-SPEC:V', '-SPEC:T', '-SPEC:C', '-SPEC:N', '-SPEC:D', '-SPEC:N', '-SPEC:P', '-SPEC:T/inf'},
                            'v': {'V', '!COMP:V', '!SPEC:D', '-COMP:Adv', '-COMP:A', '-COMP:v',  '-SPEC:T/inf'},
                            'V/INTR': {'-COMP:D', '-COMP:N', '!SPEC:D', '-COMP:T', '-SPEC:T/inf'}
                            }

# Major lexical categories assumed in this grammar
major_lexical_categories = ['C', 'N', 'v', 'V', 'T/inf', 'A', 'D', 'Adv', 'T', 'P', 'a', 'b', 'c', 'd']

# Class which stores and maintains the lexicon
class Lexicon:
    def __init__(self):
        self.lexical_entries = dict()
        self.compose_lexicon()

    # Composes the lexicon from the list of words and lexical redundancy rules
    def compose_lexicon(self):
        for lex in lexicon.keys():
            self.lexical_entries[lex] = lexicon[lex]
            for trigger_feature in lexical_redundancy_rules.keys():
                if trigger_feature in lexicon[lex]:
                    self.lexical_entries[lex] = self.lexical_entries[lex] | lexical_redundancy_rules[trigger_feature]

    def lexical_retrieval(self, name):
        ps = PhraseStructure()
        ps.features = self.lexical_entries[name]
        ps.phon = name
        ps.zero = True
        return ps


class PhraseStructure:
    def __init__(self, X=None, Y=None):
        self.features = set()
        self.phon = ''
        if X or Y:
            self.const = [X, Y]
        else:
            self.const = None
        self.mother_ = None
        if X:
            X.mother_ = self
        if Y:
            Y.mother_ = self

    def left(X):
        if X.const:
            return X.const[0]

    def right(X):
        if X.const:
            return X.const[-1]

    def mother(X):
        return X.mother_

    def copy(X):
        if not X.zero_level():
            Y = PhraseStructure(X.left().copy(), X.right().copy())
        else:
            Y = PhraseStructure()
        Y.copy_properties(X)
        return Y

    def copy_properties(Y, X):
        Y.features = X.features
        Y.phon = X.phon

    def zero_level(X):
        return not X.const

    def Merge(X, Y):
        return PhraseStructure(X, Y)

    def MergeRight(X, Y):
        N = X.mother()
        Z = X.Merge(Y)
        Z.mother_ = N
        if N:
            N.const = [N.left(), Z]
        return Z.root()

    def root(X):
        while X.mother():
            X = X.mother()
        return X

    def right_edge(X):
        right_edge = []
        while X:
            right_edge.append(X)
            X = X.right()
        return right_edge

    def MergePreconditions(X, Y):
        if X.zero_level():
            return X.complement_subcategorization(Y)
        elif Y.zero_level():
            return Y.complement_subcategorization(None)
        else:
            return Y.head().specifier_subcategorization(X)

    def sister(X):
        if X.mother():
            return next((const for const in X.mother().const if const != X), None)

    def complement(X):
        if X.sister() and X.mother().left() == X:
            return X.sister()

    def left_sister(X):
        if X.sister() and X.mother().right() == X:
            return X.sister()

    def head(X):
        if X.zero_level():
            return X
        for x in X.const:
            if x.zero_level():
                return x
        return x.head()

    def subcategorization(X):
        if X.zero_level():
            return X.complement_subcategorization(X.complement()) and X.specifier_subcategorization()
        else:
            return X.left().subcategorization() and X.right().subcategorization()

    def complement_subcategorization(X, Y):
        comps = {f.split(':')[1] for f in X.features if f.startswith('!COMP')}
        if comps and not (Y and Y.head().features & comps):
            return False
        noncomps = {f.split(':')[1] for f in X.features if f.startswith('-COMP')}
        if noncomps and (Y and Y.head().features & noncomps):
            return False
        return True

    def specifier_subcategorization(X, Y=None):
        if not Y:
            Y = X.specifier()
        specs = {f.split(':')[1] for f in X.features if f.startswith('!SPEC')}
        if specs and not (Y and Y.head().features & specs):
            return False
        nonspecs = {f.split(':')[1] for f in X.features if f.startswith('-SPEC')}
        if nonspecs and (Y and Y.head().features & nonspecs):
            return False
        return True

    def specifier(X):
        H = X
        while X.mother() and X.mother().head() == H:
            if X.left_sister() and not X.left_sister().zero_level():
                return X.left_sister()
            X = X.mother()

    def __str__(X):
        str = ''
        if X.zero_level():
            str += f'{X.phon}'
        else:
            str += '['
            if not X.zero_level():
                str += f'_{X.head().lexical_category()}P '
            for const in X.const:
                str += f'{const} '
            str += ']'
        return str

    def lexical_category(X):
        return next((f for f in X.features if f in major_lexical_categories), '?')


class SpeakerModel:
    def __init__(self):
        self.lexicon = Lexicon()
        self.grammatical = False
        self.n_sentences = 0
        self.prediction_errors = 0

    def derive(self, expected_grammaticality, sentence):
        self.n_sentences += 1
        self.grammatical = False
        print(f'\n({self.n_sentences}) {sentence}')
        sentence_lst = sentence.strip().split(' ')
        self.derivational_search_function(self.lexicon.lexical_retrieval(sentence_lst[0]), sentence_lst, 1)
        if expected_grammaticality != self.grammatical:
            print('\t=> Prediction error')
            self.prediction_errors += 1

    def derivational_search_function(self, X, sentence, index):
        if index == len(sentence):
            self.process_final_output(sentence, X)
        else:
            Y = self.lexicon.lexical_retrieval(sentence[index])
            for i, right_edge in enumerate(X.right_edge()):
                X_ = X.root().copy()
                right_edge_ = X_.right_edge()[i]
                new_X = right_edge_.MergeRight(Y)
                self.derivational_search_function(new_X, sentence, index + 1)

    def process_final_output(self, sentence, X):
        if not X.subcategorization():
            return
        print(f'\t{X}')
        self.grammatical = True


class LanguageData:
    def __init__(self):
        self.study_dataset = []
        self.log_file = None

    # Read the dataset
    def read_dataset(self, filename):
        numeration = []
        dataset = set()
        with open(filename) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    if line.startswith('END'):
                        break
                    if line.startswith('*'):
                        grammatical = False
                        line = line[1:]
                    else:
                        grammatical = True
                    self.study_dataset.append((grammatical, line))


def run_study(ld, sm):
    n_errors = 0
    sm.n_sentences = 0
    for expected_grammaticality, sentence in ld.study_dataset:
        sm.derive(expected_grammaticality, sentence)
    print(f'\nTotal errors: {sm.prediction_errors}')


ld = LanguageData()
ld.read_dataset('dataset.txt')
sm = SpeakerModel()
run_study(ld, sm)
