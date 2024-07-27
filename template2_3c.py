import itertools

lexicon = {'a': {'a'}, 'b': {'b'}, 'c': {'c'},
           'the': {'D'}, 'dog': {'N'}, 'bark': {'V', 'V/INTR'}, 'ing': {'N', '!wCOMP:V'},
           'man': {'N'}, 'bite': {'V', 'V/TR'}
           }

lexical_redundancy_rules = {'D': {'+COMP:N', '+SPEC:Ø'},
                            'V/TR': {'+COMP:D', '+SPEC:D'},
                            'V/INTR': {'+SPEC:D', '+COMP:Ø'},
                            'N': {'+COMP:Ø', '+SPEC:Ø'}}


def fformat(f):
    if f.startswith('+COMP') or f.startswith('+SPEC'):
        return f.split(':')[0], set(f.split(':')[1].split(','))
    return None, None

class Lexicon:
    """Stores lexical knowledge"""
    def __init__(self):
        self.speaker_lexicon = dict()
        self.compose_speaker_lexicon()

    def compose_speaker_lexicon(self):
        """Composes speaker lexicon from root lexicon and lexical redundancy rules"""
        for lex in lexicon.keys():
            self.speaker_lexicon[lex] = lexicon[lex]
            for trigger_feature in lexical_redundancy_rules:
                if trigger_feature in self.speaker_lexicon[lex]:
                    self.speaker_lexicon[lex] = self.speaker_lexicon[lex] | \
                                                lexical_redundancy_rules[trigger_feature]

    # Retrieves lexical items from the lexicon and wraps them into zero-level
    # phrase structure objects
    def retrieve(self, name):
        X0 = PhraseStructure()
        X0.features = self.speaker_lexicon[name]        # Retrieves lexical features from the lexicon
        X0.zero = True                                  # True = zero-level category
        X0.phonological_exponent = name                 # Spellout form is the same as the name
        return X0

class PhraseStructure:
    """Simple asymmetric binary-branching bare phrase structure formalism"""
    def __init__(self, X=None, Y=None):
        self.const = (X, Y)
        self.features = set()
        self.mother_ = None
        if X:
            X.mother_ = self
        if Y:
            Y.mother_ = self
        self.zero = False																																								
        self.phonological_exponent = ''

    def ccopy(X):
        if not X.terminal():
            Y = PhraseStructure(X.left().copy(), X.right().copy())
        else:
            Y = PhraseStructure()
        Y.copy_properties(X)
        return Y

    def copy_properties(Y, X):
        Y.phonological_exponent = X.phonological_exponent
        Y.features = X.features.copy()
        Y.zero = X.zero

    def mother(X):
        return X.mother_

    def left(X):
        return X.const[0]

    def right(X):
        return X.const[1]

    def isLeft(X):
        return X.mother() and X.mother().left() == X

    def isRight(X):
        return X.mother() and X.mother().right() == X

    # Determines whether X has a sister constituent and returns that constituent if present
    def sister(X):
        if X.isLeft():
            return X.mother().right()
        return X.mother().left()

    def complement(X):
        """Complement is a right sister of a zero-level objects"""
        if X.zero_level() and X.isLeft():
            return X.sister()

    # Standard Merge
    def Merge(X, Y):
        return PhraseStructure(X, Y)

    # Preconditions for Merge
    def MergePreconditions(X, Y):
        return not X.wcomplement_features() and \
               not Y.wcomplement_features() and \
               not X.selection_violation(Y)																	

    def selection_violation(X, Y):
        """Selection violation for both complement and specifier selection"""
        def satisfy(X, fset):
            return (not X and 'Ø' in fset) or (X and fset & X.head().features)
        return {f for x in X.Merge(Y).const for f in x.features if
                fformat(f)[0] == '+COMP' and not satisfy(x.complement(), fformat(f)[1])} or \
               (X.phrasal() and {f for f in Y.head().features if
                                 fformat(f)[0] == '+SPEC' and not satisfy(X, fformat(f)[1])})

        # Head Merge creates zero-level categories and implements feature inheritance
    def HeadMerge(X, Y):
        Z = X.Merge(Y)
        Z.zero = True
        Z.features = Y.features - {f for f in Y.features if f.startswith('!wCOMP:')}
        return Z

    # Preconditions for Head Merge (X Y)
    def HeadMergePreconditions(X, Y):
        return X.zero_level() and \
               Y.zero_level() and \
               Y.w_selects(X)

    # Word-internal selection between X and Y under (X Y),
    # where Y selects for X
    def w_selects(Y, X):
        return Y.wcomplement_features() and \
               Y.wcomplement_features() <= X.features

    # Returns a set of w-selection features
    def wcomplement_features(X):
        return {f.split(':')[1] for f in X.features if f.startswith('!wCOMP')}

    def zero_level(X):
        """Abstraction for the zero-property"""
        return X.zero

    def phrasal(X):
        return not X.zero_level()

    def linearize(X):
        """Linearizes phrase structure objects into sentences"""
        if X.zero_level():
            return X.linearize_word()[:-1] + ' '
        return ''.join([x.linearize() for x in X.const])

    def linearize_word(X):
        """Separate linearization algorithm for words"""
        if X.terminal():
            return X.phonological_exponent + '#'
        return ''.join([x.linearize_word() for x in X.const])

    # Terminal elements do not have daughter constituents
    def terminal(X):
        return len({x for x in X.const if x}) == 0

    def head(X):
        """Calculates the head of any phrase structure object"""
        for x in (X,) + X.const:
            if x and x.zero_level():
                return x
        return x.head()

    def __str__(X):
        """Simple printout function for phrase structure objects"""
        if X.terminal():
            return X.phonological_exponent
        elif X.zero_level():
            return '(' + ' '.join([f'{x}' for x in X.const]) + ')'
        return f'[_{X.head().lexical_category()}P ' + ' '.join([f'{x}' for x in X.const]) + ']'			

    # Defines the major lexical categories used in all printouts
    def lexical_category(X):
        return next((f for f in {'N', 'V', 'D', 'A', 'P', 'a', 'b', 'c'} if f in X.features), '?')


syntactic_operations = [(PhraseStructure.MergePreconditions, PhraseStructure.Merge, 2, 'Merge'),
                        (PhraseStructure.HeadMergePreconditions, PhraseStructure.HeadMerge, 2, 'Head Merge')]
N_sentences = 0

def derive(sWM):
    global N_sentences
    N_sentences = 0
    print('\nNumeration: {'+ ', '.join((str(x) for x in sWM)) + '}\n')
    derivational_search_function(sWM)

def derivational_search_function(sWM):
    if derivation_is_complete(sWM):
        process_final_output(sWM)
    else:
        for Preconditions, OP, n, name in syntactic_operations:
            for SO in itertools.permutations(sWM, n):
                if Preconditions(*SO):
                    new_sWM = {x for x in sWM if x not in set(SO)} | {OP(*SO)}
                    derivational_search_function(new_sWM)

def derivation_is_complete(sWM):
    return len(sWM) == 1

def process_final_output(sWM):
    global N_sentences
    N_sentences += 1
    X = sWM.pop()
    print(f'\t{N_sentences}. {X.linearize()}   {X}')

Lex = Lexicon()
Numeration_lst = [['a', 'b', 'c'],
                 ['the', 'dog', 'bite', 'the', 'man'],
                  ['the', 'dog', 'bark'],
                  ['the', 'dog', 'bark', 'the', 'man'],
                  ['the', 'bark', 'ing']
                  ]

for numeration in Numeration_lst:
    derive({Lex.retrieve(word) for word in numeration})
