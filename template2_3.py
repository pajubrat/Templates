import itertools

# Python template for § 2.3 in Brattico, P. (2024). Computational generative grammar

root_lexicon = {'a': set(), 'b': set(), 'c': set(), 'd': set()}

class Lexicon:
    """Stores lexical knowledge"""
    def __init__(self):
        self.speaker_lexicon = dict()
        self.compose_speaker_lexicon()

    def compose_speaker_lexicon(self):
        """Composes the lexicon from the list of words and
        (later) lexical redundancy rules"""
        for lex in root_lexicon.keys():
            self.speaker_lexicon[lex] = root_lexicon[lex]

    def retrieve(self, name):
        """Retrieves lexical items from the speaker lexicon and wraps them
        into zero-level phrase structure objects"""
        X0 = PhraseStructure()
        X0.features = self.speaker_lexicon[name]
        X0.zero = True
        X0.phonological_exponent = name
        return X0

class PhraseStructure:
    """Simple asymmetric binary-branching bare phrase structure formalism"""
    def __init__(self, X=None, Y=None, exponent=''):
        self.const = (X, Y)  # Left and right daughter constituents, in an ordered tuple
        self.features = set()  # Lexical features, a set
        self.mother = None  # Mother (immediately dominating) node
        self.zero = False
        if X:  # Create mother-of dependencies for daughters
            X.mother = self
        if Y:
            Y.mother = self
        self.phonological_exponent = exponent  # Phonological exponent, used for printout

    def left(X):
        """Definition (abstraction) for the notion of left daughter"""
        return X.const[0]

    def right(X):
        """Definition(abstraction) for the notion of right daughter"""
        return X.const[1]

    def Merge(X, Y):
        """Standard Merge"""
        return PhraseStructure(X, Y)

    def MergePreconditions(X, Y):
        """Preconditions for Merge (currently none)"""
        return True

    def HeadMerge(X, Y):
        """Creates a zero-level object from two zero-level objects.
        Features of the resulting object are inherited from the rightmost
        constituent"""
        Z = X.Merge(Y)
        Z.zero = True
        Z.features = Y.features
        return Z

    def HeadMergePreconditions(X, Y):
        """Preconditions for Head Merge (X Y): both must be zero-level objects"""
        return X.zero_level() and Y.zero_level()

    def zero_level(X):
        """Interface function for zero"""
        return X.zero

    def __str__(X):
        """Simple printout function for phrase structure objects"""
        if X.terminal():
            return X.phonological_exponent
        elif X.zero_level():
            return '(' + ' '.join([f'{x}' for x in X.const]) + ')'
        return '[' + ' '.join([f'{x}' for x in X.const]) + ']'

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

    def terminal(X):
        """Terminal elements do not have daughter constituents"""
        return len({x for x in X.const if x}) == 0
         
# This list contains all grammatical rules for the creation of complex phrase structure objects
# Currently only one operation, Merge. Each rule is defined by four properties:
# (1) Preconditions for application, defined as a function;
# (2) the operation itself, defined as a function;
# (3) number of input objects for the rule;
# (4) a name for the operation (not used in this version)

syntactic_operations = [(PhraseStructure.MergePreconditions, PhraseStructure.Merge, 2, 'Merge'),
                        (PhraseStructure.HeadMergePreconditions, PhraseStructure.HeadMerge, 2, 'Head Merge')]

N_sentences = 0

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
    print(f'{N_sentences}. {X.linearize()[:-1]}     {X}')

# Initialize the lexicon
Lex = Lexicon()

# Initial lexical feed (set of primitive constituents) for the derivation
Numeration = {Lex.retrieve('a'),
              Lex.retrieve('b'),
              Lex.retrieve('d'),
              Lex.retrieve('d')}

# Create all derivations from the numeration
derivational_search_function(Numeration)
