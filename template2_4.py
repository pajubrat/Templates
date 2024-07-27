import itertools

# Template script for § 2.4 in Brattico, P. (2024). Computational generative grammar.

root_lexicon = {'a': {'a'}, 'b': {'b'}, 'c': {'c'}, 'd': {'d'},
           'the': {'D'},
           'dog': {'N'},
           'barks': {'V', 'V/INTR'},
           'ing': {'N', '!wCOMP:V', 'ε'},
           'man': {'N'},
           'bite': {'V', 'V/TR'},
           'v': {'v', 'V'},
           'does': {'T', 'Aux'},
           'ed': {'T', '!wCOMP:V'}
                }

lexical_redundancy_rules = {'D': {'+COMP:N', '+SPEC:Ø'},
                            'V/TR': {'+COMP:D', '+SPEC:X'},
                            'V/INTR': {'+SPEC:D', '+COMP:Ø'},
                            'T': {'+COMP:V'},
                            'v': {'+COMP:V', '+SPEC:D', '!wCOMP:V'},
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
        """Composes the lexicon from the list of words and (later) lexical redundancy rules"""
        for lex in root_lexicon.keys():
            self.speaker_lexicon[lex] = root_lexicon[lex]
            for trigger_feature in lexical_redundancy_rules:
                if trigger_feature in self.speaker_lexicon[lex]:
                    self.speaker_lexicon[lex] = self.speaker_lexicon[lex] | lexical_redundancy_rules[trigger_feature]

    def retrieve(self, name):
        """Retrieves lexical items from the speaker lexicon and wraps them
        into zero-level phrase structure objects"""
        X0 = PhraseStructure()
        X0.features = self.speaker_lexicon[name]        # Retrieves lexical features from the lexicon
        X0.zero = True                                  # True = zero-level category
        X0.phonological_exponent = name                 # Spellout form is the same as the name
        return X0

class PhraseStructure:
    """Simple asymmetric binary-branching bare phrase structure formalism"""
    log_report = ''
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
        self.elliptic = False

    def copy(X):
        """Copies whole constituent (recursively)"""
        if X.terminal():
            return PhraseStructure().copy_properties(X)
        else:
            return PhraseStructure(X.left().copy(), X.right().copy()).copy_properties(X)

    def copy_properties(target, source):
        """Copies the properties of a constituent"""
        target.phonological_exponent = source.phonological_exponent
        target.features = source.features.copy()
        target.elliptic = source.elliptic
        target.zero = source.zero
        return target

    def chaincopy(X):
        """Copies a constituent and adds anything assumed in the grammatical theory"""
        Y = X.copy()
        X.elliptic = True
        return Y

    def mother(X):
        return X.mother_

    def left(X):
        """Definition (abstraction) for the notion of left daughter"""
        return X.const[0]

    def right(X):
        """Definition(abstraction) for the notion of right daughter"""
        return X.const[1]

    def isLeft(X):
        return X.mother() and X.mother().left() == X

    def isRight(X):
        return X.mother() and X.mother().right() == X

    def sister(X):
        """Definition for sisterhood"""
        if X.isLeft():
            return X.mother().right()
        return X.mother().left()

    def complement(X):
        """Complement is right sister of a zero-level objects"""
        if X.zero_level() and X.isLeft():
            return X.sister()

    def Merge(X, Y):
        """Standard Merge"""
        return PhraseStructure(X, Y)

    def MergePreconditions(X, Y):
        """Preconditions for Merge"""
        return not Y.bound_morpheme() and \
               not X.selection_violation(Y)

    def MergeComposite(X, Y):
        """Composite Merge operation contains Head Movement (if applicable) and Merge"""
        return X.HeadMovement(Y).Merge(Y)

    def HeadMovementPreconditions(X, Y):
        """Preconditions for Head Movement after [X Y] are that X must be zero-level object and a
        bound morpheme"""
        return X.zero_level() and \
               X.bound_morpheme()

    def bound_morpheme(X):
        """Definition for bound morphemes"""
        return X.wcomplement_features()

    def HeadMovement(X, Y):
        """Head movement after [X Y] copies the head of Y and Head Merges it to X"""
        if X.HeadMovementPreconditions(Y):
            PhraseStructure.log_report += f'\nHead chain by {X}° targeting {Y.head()}° + '				
            return Y.head().chaincopy().HeadMerge(X)
        return X

    def selection_violation(X, Y):
        """Selection violation for both complement and specifier selection"""
        def satisfy(X, fset):
            return (not X and 'Ø' in fset) or (X and fset & X.head().features)

        return {f for x in X.Merge(Y).const for f in x.features if
                fformat(f)[0] == '+COMP' and not satisfy(x.complement(), fformat(f)[1])} or \
               (X.phrasal() and {f for f in Y.head().features if
                                 fformat(f)[0] == '+SPEC' and not satisfy(X, fformat(f)[1])})

    def HeadMergePreconditions(X, Y):
        """Preconditions for Head Merge (X Y)"""
        return X.zero_level() and \
               Y.zero_level() and \
               Y.w_selects(X) and 'ε' in X.features

    def HeadMerge(X, Y):
        """Creates a zero-level object from two zero-level objects.
        Features of the resulting object are inherited from the rightmost
        constituent"""
        Z = X.Merge(Y)
        Z.zero = True
        Z.features = Y.features - {f for f in Y.features if f.startswith('!wCOMP:')}
        return Z

    def w_selects(Y, X):
        """Word-internal selection between X and Y under (X Y), where Y selects for X"""
        return Y.wcomplement_features() and \
               Y.wcomplement_features() <= X.features

    def wcomplement_features(X):
        """Returns a set of w-selection features"""
        return {f.split(':')[1] for f in X.features if f.startswith('!wCOMP')}

    def zero_level(X):
        """Abstraction for the notion of zero-level object"""
        return X.zero

    def phrasal(X):
        return not X.zero_level()

    def linearize(X):
        """Linearizes phrase structure objects into sentences"""
        if X.elliptic:
            return ''
        if X.zero_level():
            return X.linearize_word()[:-1] + ' '
        return ''.join([x.linearize() for x in X.const])

    def linearize_word(X):
        """Separate linearization algorithm for words"""
        if X.terminal():
            return X.phonological_exponent + '#'
        return ''.join([x.linearize_word() for x in X.const])

    def terminal(X):
        return len({x for x in X.const if x}) == 0

    def head(X):
        """Head algorithm for phrase structure objects"""
        for x in (X,) + X.const:
            if x and x.zero_level():
                return x
        return x.head()

    def __str__(X):
        """Simple printout function for phrase structure objects"""
        if X.elliptic:
            return '__'
        if X.terminal():
            return X.phonological_exponent
        elif X.zero_level():
            return '(' + ' '.join([f'{x}' for x in X.const]) + ')'
        return '[' + ' '.join([f'{x}' for x in X.const]) + ']'

    def lexical_category(X):
        """Defines the major lexical categories used in all printouts"""
        return next((f for f in ['N', 'v', 'V', 'D', 'A', 'P', 'T', 'a', 'b', 'c'] if f in X.features), '?')


syntactic_operations = [(PhraseStructure.MergePreconditions,
                         PhraseStructure.MergeComposite,												
                         2,
                         'Merge'),
                        (PhraseStructure.HeadMergePreconditions,
                         PhraseStructure.HeadMerge,
                         2,
                         'Head Merge')]
N_sentences = 0
data = set()

def derive(sWM):
    """Wrapper for the derivation_search_function"""
    global N_sentences
    global data
    N_sentences = 0
    data = set()
    print('\nNumeration: {'+ ', '.join((str(x) for x in sWM)) + '}\n')
    derivational_search_function(sWM)
    PhraseStructure.log_report += '\nResults:\n'
    for i, s in enumerate(data, start=1):
        print_(f'{i}. {s} ')

def wcopy(sWM):
    return (x.copy() for x in sWM)

def print_lst(SO):
    return ', '.join([str(x) for x in SO])

def print_(s):
    """Prints for console and log file"""
    print(s)
    PhraseStructure.log_report += s

def derivational_search_function(sWM):
    if derivation_is_complete(sWM):
        process_final_output(sWM)
    else:
        for Preconditions, OP, n, name in syntactic_operations:
            for SO in itertools.permutations(sWM, n):
                if Preconditions(*SO):
                    new_sWM = {x for x in sWM if x not in set(SO)} | {OP(*wcopy(SO))}
                    PhraseStructure.log_report += f'\n{name}({print_lst(SO)})\n= ({print_lst(new_sWM)})\n\n'
                    derivational_search_function(new_sWM)

def derivation_is_complete(sWM):
    return len(sWM) == 1

def process_final_output(sWM):
    global N_sentences
    global data
    N_sentences += 1
    X = sWM.pop()
    data.add(f'{X.linearize()}  {X}')
    PhraseStructure.log_report += f'\t{N_sentences}. {X.linearize()}    {X}\n'


Lex = Lexicon()

Numeration_lst = [['the', 'dog', 'barks'],
                  ['the', 'dog', 'v', 'bite', 'the', 'man'],
                  ['the', 'dog', 'ed', 'v', 'bite', 'the', 'man']]

log_file = open('log.txt', 'w')

for numeration in Numeration_lst:
    PhraseStructure.log_report = '\n\n=====\nNumeration: {' + ', '.join(numeration) + '}\n'
    derive({Lex.retrieve(word) for word in numeration})
    log_file.write(PhraseStructure.log_report)
