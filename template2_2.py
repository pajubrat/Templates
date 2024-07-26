import itertools

# Template script for § 2.2, Brattico, P. (2024). Computational generative grammar.

class PhraseStructure:
    """Simple asymmetric binary-branching bare phrase structure formalism"""
    def __init__(self, X=None, Y=None, exponent=''):
        self.const = (X, Y)       		# Left and right daughter constituents, in an ordered tuple
        self.features = set()     		# Lexical features, a set
        self.mother = None        		# Mother (immediately dominating) node
        if X:                           # Create mother-of dependencies for daughters
            X.mother = self
        if Y:
            Y.mother = self
        self.phonological_exponent = exponent   # Phonological exponent, used for printout

    # Definition (abstraction) for the notion of left daughter
    def left(X):
        return X.const[0]

    # Definition (abstraction) for the notion of right daughter
    def right(X):
        return X.const[1]

    # Standard Merge
    def Merge(X, Y):
        return PhraseStructure(X, Y)

    # Preconditions for Merge (currently none)
    def MergePreconditions(X, Y):
        return True

   # Simple printout function for phrase structure objects
    def __str__(X):
        if not X.left() and not X.right():
            return X.phonological_exponent
        return '[' + ' '.join([f'{x}' for x in X.const]) + ']'

# All grammatical rules for the creation of complex phrase structure objects
# Currently only one operation, Merge. Each rule is defined by four properties:
# (1) Preconditions for application, defined as a function;
# (2) the operation itself, defined as a function;
# (3) number of input objects for the rule;
# (4) a name for the operation, for logging purposes
syntactic_operations = [(PhraseStructure.MergePreconditions, PhraseStructure.Merge, 2, 'Merge')]    #   Grammar

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
    """This function contains all postsyntactic operations that must be done with
    finished derivations."""
    print(f'{sWM.pop()}')

# Create three primitive phrase structure objects
a = PhraseStructure()
a.phonological_exponent = 'a'
b = PhraseStructure()
b.phonological_exponent = 'b'
c = PhraseStructure()
c.phonological_exponent = 'c'

# Initial lexical feed (set of primitive constituents from root lexicon) for the derivation
Numeration = {a, b, c}

# Create all derivations from the numeration
derivational_search_function(Numeration)
