import itertools

# Asymmetric binary-branching phrase structure formalism, with
# a minimal set of properties

lexicon = {'a': set(),
           'b': set(),
           'c': set(),
           'd': set()}

# Class which stores and maintains the lexicon
class Lexicon:
    def __init__(self):
        self.lexical_entries = dict()   #   The lexicon is a dictionary
        self.compose_lexicon()          #   Creates the runtime lexicon

    # Composes the lexicon from the list of words and lexical redundancy rules
    def compose_lexicon(self):
        for lex in lexicon.keys():
            self.lexical_entries[lex] = lexicon[lex]

    # Retrieves lexical items from the lexicon and wraps them into zero-level
    # phrase structure objects
    def retrieve(self, name):
        X0 = PhraseStructure()
        X0.features = self.lexical_entries[name]        # Retrieves lexical features from the lexicon
        X0.zero = True                                  # True = zero-level category
        X0.phonological_exponent = name                                  # Spellout form is the same as the name
        return X0

class PhraseStructure:
    def __init__(self, X=None, Y=None):
        self.const = (X, Y)       			# Left and right daughter constituents, in an ordered tuple
        self.features = set()     			# Lexical features (not used in this script), in a set
        self.mother = None        			# Mother node (not used in this script)
        if X:
            X.mother = self
        if Y:
            Y.mother = self
        self.zero = False
        self.phonological_exponent = ''

    # Standard Merge
    def Merge(X, Y):
        return PhraseStructure(X, Y)

    # Preconditions for Merge (currently none)
    def MergePreconditions(X, Y):
        return True

    # Head Merge creates zero-level categories and implements feature inheritance
    def HeadMerge(X, Y):
        Z = X.Merge(Y)
        Z.zero = True
        Z.features = Y.features
        return Z

    # Preconditions for Head Merge (X Y)
    def HeadMergePreconditions(X, Y):
        return X.zero_level() and Y.zero_level()

    # Interface function for zero
    def zero_level(X):
        return X.zero

    # Maps phrase structure objects into linear lists of words
    def linearize(X):
        output_str = ''
        if X.zero_level():
            output_str += X.linearize_word('') + ' '
        else:
            for x in X.const:
                output_str += x.linearize()
        return output_str

    # Spellout algorithm for words, creates morpheme boundaries marked by symbol #
    def linearize_word(X, word_str):
        if X.terminal():
            if word_str:
                word_str += '#'
            word_str += X.phonological_exponent
        else:
            for x in X.const:
                word_str = x.linearize_word(word_str)
        return word_str

    # Terminal elements do not have daughter constituents
    def terminal(X):
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
    if derivation_is_complete(sWM):                                             #   Only one phrase structure object in working memory
        process_final_output(sWM)                                               #   Terminate processing and evaluate solution
    else:
        for Preconditions, OP, n, name in syntactic_operations:                 #   Examine all syntactic operations OP
            for SO in itertools.permutations(sWM, n):                           #   All n-tuples of objects in sWM
                if Preconditions(*SO):                                          #   Blocks illicit derivations
                    new_sWM = {x for x in sWM if x not in set(SO)} | {OP(*SO)}  #   Update sWM
                    derivational_search_function(new_sWM)                       #   Continue derivation, recursive branching

def derivation_is_complete(sWM):
    return len(sWM) == 1

def process_final_output(sWM):
    global N_sentences
    N_sentences += 1
    print(f'{N_sentences}. {sWM.pop().linearize()}')

# Initialize the lexicon
Lex = Lexicon()

# Numeration is the initial lexical feed (set of primitive constituents)
# for the derivation
Numeration = {Lex.retrieve('a'),
              Lex.retrieve('b'),
              Lex.retrieve('d'),
              Lex.retrieve('d')}

# Create all derivations from the numeration
derivational_search_function(Numeration)
