import itertools

# Asymmetric binary-branching phrase structure formalism, with
# a minimal set of properties

class PhraseStructure:
    def __init__(self, X=None, Y=None):
        self.const = (X, Y)       			# Left and right daughter constituents, in an ordered tuple
        self.features = set()     			# Lexical features (not used in this script), in a set
        self.mother = None        			# Mother node (not used in this script)
        if X:
            X.mother = self
        if Y:
            Y.mother = self

    # Standard Merge
    def Merge(X, Y):
        return PhraseStructure(X, Y)

    # Preconditions for Merge (currently none)
    def MergePreconditions(X, Y):
        return True

# This list contains all grammatical rules for the creation of complex phrase structure objects
# Currently only one operation, Merge. Each rule is defined by four properties:
# (1) Preconditions for application, defined as a function;
# (2) the operation itself, defined as a function;
# (3) number of input objects for the rule;
# (4) a name for the operation (not used in this version)
syntactic_operations = [(PhraseStructure.MergePreconditions, PhraseStructure.Merge, 2, 'Merge')]    #   Grammar

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
    # This function should contain all operations that must be done with
    # finished derivations (e.g., well-formedness conditions, printout, logging).
    # Currently no operations apart from one print command showing that
    # some derivation was completed.
    print('.')

# Here we create four primitive phrase structure objects
a = PhraseStructure()
b = PhraseStructure()
c = PhraseStructure()
d = PhraseStructure()

# Numeration is the initial lexical feed (set of primitive constituents)
# for the derivation
Numeration = {a, b, c, d}

# Create all derivations from the numeration
derivational_search_function(Numeration)
