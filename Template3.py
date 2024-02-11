#
# Starting point template for derivational search function
# for asymmetric binary-branching bare phrase structure
# with some advanced features such as head chains and
# standard EPP-induced A-chain
#

import itertools

# Lexicon with some features (move into external files)
lexicon = {'the': {'D'},
           'dog': {'N'},
           'man': {'N'},
           'T': {'T', '#X', 'EPP', '!SPEC:D'},
           'C': {'C'},
           'v': {'v', '#X'},
           'sit': {'V', '-COMP:D'},
           'bite': {'V', '!COMP:D'}}

lexical_redundancy_rules = {'D': {'!COMP:N', '-SPEC:C', '-SPEC:T', '-SPEC:N', '-SPEC:V', '-SPEC:D'},
                            'V': {'-SPEC:C', '-SPEC:N', '-SPEC:T', '-SPEC:V'},
                            'C': {'!COMP:T', '-SPEC:V', '-SPEC:D', '-SPEC:C', '-SPEC:N'},
                            'N': {'-COMP:V', '-COMP:D', '-COMP:V', '-COMP:T', '-SPEC:V', '-SPEC:T', '-SPEC:C', '-SPEC:N'},
                            'T': {'!COMP:V', '-SPEC:C', '-SPEC:T'},
                            'v': {'V', '!COMP:V', '!SPEC:D', '-COMP:v'}
                            }

major_lexical_categories = {'C', 'N', 'V', 'A', 'D', 'Adv', 'T', 'v'}

# Stores and maintains the lexicon
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

    def retrieve_lexical_item(self, name):
        return self.lexical_entries[name]

# Asymmetric binary-branching phrase structure formalism
class PhraseStructure:
    logging = None
    def __init__(self, X=None, Y=None):
        self.const = (X, Y)       # Left and right daughter constituents
        self.phon = ''            # Name
        self.features = set()      # Lexical features
        self.zero = False         # Zero-level categories
        self.silent = False       # Phonological silencing
        self.mother = None        # Mother nodes
        if X:
            X.mother = self
        if Y:
            Y.mother = self

    def copy(X):
        if not X.terminal():
            Y = PhraseStructure(X.const[0].copy(), X.const[1].copy())
        else:
            Y = PhraseStructure()
        Y.phon = X.phon
        Y.features = X.features
        Y.zero = X.zero
        Y.silent = X.silent
        return Y

    def chaincopy(X):
        Y = X.copy()
        X.silent = True
        return Y

    # Zero-level categories are phrase structure objects with less that two daughter constituents
    # or they are marked as such
    def zero_level(X):
        return X.zero or X.terminal()

    # Terminal elements do not have two daughter constituents
    def terminal(X):
        return not X.const[0] or not X.const[1]

    # Preconditions for Merge
    def MergePreconditions(X, Y):
        if X.zero_level():
            return X.complement_subcategorization(Y)
        else:
            return Y.head().specifier_subcategorization(X)

    # Merge
    def Merge(X, Y):
        return PhraseStructure(X, Y)

    # Merge with head and phrasal repair functions
    def Merge_(X, Y):
        return X.HeadRepair(Y).Merge(Y).PhrasalRepair()

    # Head repair for X before Merge
    def HeadRepair(X, Y):
        if X.zero_level() and X.bound_morpheme() and not Y.head().silent:
            PhraseStructure.logging.write(f'\tHead Chain ({X}, {Y.head()})\n')
            X = Y.head().chaincopy().HeadMerge_(X)
        return X

    # Phrasal repair for X before Merge
    def PhrasalRepair(X):
        H = X.head()
        if H.EPP() and H.complement() and H.complement().head().specifier() and not H.complement().head().specifier().silent:
            PhraseStructure.logging.write(f'\tEPP Chain ({H}, {H.complement().head().specifier()})')
            X = H.complement().head().specifier().chaincopy().Merge(X)  #   Phrasal copying
            PhraseStructure.logging.write(f' = {X}\n')
        return X

    # Head Merge creates a zero-level category and implements feature inheritance
    def HeadMerge_(X, Y):
        Z = X.Merge(Y)
        Z.zero = True
        Z.features = Y.features
        return Z

    # Determines whether X has a sister constituent and returns that constituent if present
    def sister(X):
        if X.mother:
            return next((const for const in X.mother.const if const != X), None)

    # Determines whether X has a right sister and return that constituent if present
    def complement(X):
        if X.sister() and X.mother.const[0] == X:
            return X.sister()

    def left_sister(X):
        if X.sister() and X.mother.const[1] == X:
            return X.sister()

    # Calculates the head of any phrase structure object X ("labelling algorithm")
    def head(X):
        if X.zero_level():                    #   X0: X0 is the head
            return X
        elif X.const[0].zero_level():         #   [X0 Y(P)]: X0 is the head
            return X.const[0]
        elif X.const[1].zero_level():         #   [XP Y0]: Y0 is the head
            return X.const[1]
        else:
            return X.const[1].head()          #   [XP YP]: return the head of YP

    # Verifies (recursively) that the configuration satisfies complement and
    # specifier subcategorization
    def subcategorization(X):
        if X.zero_level():
            return X.complement_subcategorization(X.complement()) and X.specifier_subcategorization()
        else:
            return X.const[0].subcategorization() and X.const[1].subcategorization()

    # Complement subcategorization under [X Y]
    def complement_subcategorization(X, Y):
        comps = {f.split(':')[1] for f in X.features if f.startswith('!COMP')}
        if comps and not (Y and Y.head().features & comps):
            return False
        noncomps = {f.split(':')[1] for f in X.features if f.startswith('-COMP')}
        if noncomps and (Y and Y.head().features & noncomps):
            return False
        return True

    # Specifier subcategorization under [_YP XP YP]
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

    # Definition for the notion of specifier based on
    # head algorithm
    def specifier(X):
        x = X.head()
        while x.mother and x.mother.head() == X.head():
            if x.mother.left_sister() and not x.mother.left_sister().zero_level():
                return x.mother.left_sister()
            x = x.mother

    # Maps phrase structure objects into linear lists of words
    # This function should contain all postsyntactic computations
    # involved in the implementation of the PF-spellout mapping
    def linearize(X):
        linearized_output_str = ''
        if not X.silent:
            if X.zero_level():
                linearized_output_str += X.linearize_word('') + ' '
            else:
                linearized_output_str += X.const[0].linearize()      # Recursion to left
                linearized_output_str += X.const[1].linearize()      # Recursion to right
        return linearized_output_str

    # Spellout algorithm for words
    def linearize_word(X, word_str):
        if X.terminal():
            if word_str:
                word_str += '#'
            word_str += X.phon
        else:
            word_str = X.const[0].linearize_word(word_str)
            word_str = X.const[1].linearize_word(word_str)
        return word_str

    # Definition for bound morpheme
    def bound_morpheme(X):
        return '#X' in X.features

    # Definition for EPP
    def EPP(X):
        return 'EPP' in X.features

    # Printout function
    def __str__(X):
        str = ''
        if X.silent:                    #   Phonologically silenced constituents are marked by __
            return '(__)'
        if X.terminal():                #   Terminal constituents are spelled out
            str = X.phon
        else:
            if X.zero_level():          #   Non-terminal zero-level categories use different brackets
                bracket = ('(', ')')
            else:
                bracket = ('[', ']')
            str += bracket[0]
            if not X.zero_level():
                str += f'_{X.head().lexical_category()}P '  #   Print information about heads and labelling
            for const in X.const:
                str = str + f'{const}'
                if X.const[1].terminal() and const != X.const[1]:   #   Space between terminal elements
                    str += ' '
            str += bracket[1]
        return str

    # Defines the major lexical categories used in all printouts
    def lexical_category(X):
        return next((f for f in X.features if f in major_lexical_categories), '?')

# Model of the speaker
class SpeakerModel:
    def __init__(self):
        # List of all syntactic operations available in the grammar
        self.external_syntactic_operations = [(PhraseStructure.MergePreconditions, PhraseStructure.Merge_, 'Merge')]
        self.n_derivations = 0                                          # Counts the number of derivations
        self.n_steps = 0                                                # NUmber of derivational steps
        self.output_data = set()                                        # Stores grammatical output data from the model
        self.lexicon = Lexicon()                                        # Add lexicon
        self.log_file = None                                            # Log file

    # Retrieves lexical items from the lexicon and wraps them into zero-level phrase structure
    # objects
    def LexicalRetrieval(self, name):
        ps = PhraseStructure()
        ps.features = self.lexicon.retrieve_lexical_item(name)  # Retrieves lexical features from the lexicon
        ps.phon = name                                          # Name for identification and easy recognition
        return ps

    # Wrapper function for the derivational search function
    # Performs initialization and maps the input into numeration
    def derive(self, numeration):
        self.n_steps = 0
        self.output_data = set()
        self.derivational_search_function([self.LexicalRetrieval(item) for item in numeration])

    # Derivational search function
    def derivational_search_function(self, sWM):
        if len(sWM) == 1:                                   #   Only one phrase structure object in working memory
            self.final_output(next(iter(sWM)))              #   Terminate processing and evaluate solution
        else:
            for Preconditions, OP, name in self.external_syntactic_operations:
                for X, Y in itertools.permutations(sWM, 2):
                    if Preconditions(X, Y):
                        Z = OP(X.copy(), Y.copy())                              #   Create new phrase structure object by applying an operation
                        new_sWM = {x for x in sWM if x not in {X, Y}} | {Z}     #   Populate syntactic working memory
                        self.consume_resource(name, [X, Y], Z, new_sWM)         #   Record resource consumption and write log entries
                        self.derivational_search_function(new_sWM)              #   Continue derivation

    # Resource recording
    def consume_resource(self, name, lst, result, sWM):
        self.n_steps += 1
        self.log_file.write(f'\t{name}({self.print_constituent_lst(lst)}) = {result}\n')
        self.log_file.write(f'\t...{self.print_constituent_lst(sWM)} ({self.n_steps})\n\n')

    # Processes final output
    def final_output(self, X):
        prefix = ''
        self.log_file.write(f'\t{X}')
        if X.subcategorization():  # Store only grammatical sentences
            output_sentence = f'{prefix}{X.linearize()}'
            print(f'\tAccepted: {output_sentence} {X}')   # Print the output
            self.log_file.write(f'\n\n\t ^ ACCEPTED, OUTPUT: {output_sentence}')
            self.output_data.add(output_sentence.strip())

        self.log_file.write('\n\n')

    def print_constituent_lst(self, sWM):
        str = ''
        for i, ps in enumerate(sWM):
            if ps.terminal():
                str += f'[{ps}]'
            else:
                str += f'{ps}'
            if i < len(sWM)-1:
                str += ', '
        return str

class LanguageData:
    def __init__(self):
        self.study_dataset = []           # Contains the datasets for one study, which contains tuples of numerations and datasets
        self.log_file = None

    def read_dataset(self, filename):
        numeration = []
        dataset = set()
        with open(filename) as f:
            lines = f.readlines()
            for line in lines:
                if line.strip() and not line.startswith('#'):
                    if line.startswith('Numeration='):
                        if numeration:
                            self.study_dataset.append((numeration, dataset))
                            dataset = set()
                        numeration = [word.strip() for word in line.split('=')[1].split(',')]
                    else:
                        dataset.add(line.strip())
            self.study_dataset.append((numeration, dataset))
            print(self.study_dataset)

    def start_logging(self):
        log_file = 'log.txt'
        log_file = open(log_file, 'w')
        PhraseStructure.logging = log_file
        return log_file

    def evaluate_experiment(self, output_from_simulation, gold_standard_dataset, n_steps):
        print(f'\tDerivational steps: {n_steps}')
        overgeneralization = output_from_simulation - gold_standard_dataset
        undergeneralization = gold_standard_dataset - output_from_simulation
        total_errors = len(overgeneralization) + len(undergeneralization)
        print(f'\tErrors {total_errors}')
        if total_errors > 0:
            print(f'\tShould not generate: {overgeneralization}')
            print(f'\tShould generate: {undergeneralization}')


def run_study(ld, sm):
    sm.log_file = ld.start_logging()
    n_dataset = 0
    for numeration, gold_standard_dataset in ld.study_dataset:
        n_dataset += 1
        print(f'Dataset {n_dataset}:')
        sm.log_file.write('\n---------------------------------------------------\n')
        sm.log_file.write(f'Dataset {n_dataset}:\n')
        sm.log_file.write(f'Numeration: {numeration}\n')
        sm.log_file.write(f'Predicted outcome: {gold_standard_dataset}\n\n\n')
        sm.derive(numeration)
        ld.evaluate_experiment(sm.output_data, gold_standard_dataset, sm.n_steps)


ld = LanguageData()
ld.read_dataset('dataset2.txt')
sm = SpeakerModel()
run_study(ld, sm)
