
#
# Starting point template for derivational search function
# for asymmetric binary-branching bare phrase structure
# with some advanced features such as head and phrasal chains;
# Agree is an implicit part of the minimal search
#

import itertools

# Lexicon with the following features
# !COMP: mandatory complement
# -COMP: illicit complement
# !SPEC: mandatory specifier
# -SPEC: illicit specifier
# EPP: standard EPP triggering A-movement
# #X: bound morpheme
# WH: Wh-operator feature
# SCOPE: Scope marking element for operators
# + major lexical categories
lexicon = {'a': {'a', 'f1', 'f2'},
           'b': {'b', 'f3', 'f4'},
           'c': {'c', 'f5', 'f6'},
           'd': {'d', 'f7', 'f8'},
           'the': {'D'},
           'dog': {'N'},
           'bark': {'V', 'V/INTR'},
           'ing': {'N', '!wCOMP:V', 'PC:#X'},
           'which': {'D', 'WH'},
           'man': {'N'},
           'angry': {'A', 'α:N', 'λ:L'},
           'frequently': {'Adv', 'α:V', 'λ:R'},
           'city': {'N'},
           'from': {'P'},
           'in': {'P', 'α:V', },
           'T': {'T', 'PC:#X', 'EPP', '!SPEC:D', '!wCOMP:V'},
           'T*': {'T', 'PC:#X', 'T', '!wCOMP:V'},
           'did': {'T', 'EPP'},
           'was': {'T', 'EPP'},
           'C': {'C'},
           'C(wh)': {'C', 'PC:#X', 'WH', 'SCOPE', 'SPEC:WH', '!wCOMP:T'},
           'v': {'v', 'PC:#X', '!wCOMP:V'},
           'v*': {'V', 'EPP', 'PC:#X', '!COMP:V', '-SPEC:v'},
           'that': {'C'},
           'believe': {'V', '!COMP:C'},
           'seem': {'V', 'EPP', '!SPEC:D', '!COMP:T/inf', 'RAISING'},
           'to': {'T/inf', '!COMP:V', '-COMP:RAISING', '-COMP:T', 'EPP'},
           'bite': {'V', '!COMP:D'}}

lexical_redundancy_rules = {'D': {'!COMP:N', '-COMP:Adv', '-SPEC:C', '-SPEC:T', '-SPEC:N', '-SPEC:V', '-SPEC:D', '-SPEC:P', '-SPEC:T/inf', '-SPEC:Adv'},
                            'V': {'-SPEC:C', '-SPEC:N', '-SPEC:T', '-SPEC:T/inf', '-COMP:A'},
                            'Adv': {'-COMP:D', '-COMP:N', '-SPEC:V', '-SPEC:v', '-SPEC:T', '-SPEC:D'},
                            'P': {'!COMP:D', '-COMP:Adv', '-SPEC:Adv', '-SPEC:C', '-SPEC:T', '-SPEC:N', '-SPEC:V', '-SPEC:v', '-SPEC:T/inf', 'λ:R'},
                            'C': {'!COMP:T', '-COMP:Adv', '-SPEC:V', '-SPEC:C', '-SPEC:N', '-SPEC:T/inf'},
                            'A': {'-COMP:D', '-SPEC:Adv', '-COMP:Adv', '-SPEC:D', '-SPEC:V', '-COMP:V', '-COMP:T', '-SPEC:T', '-SPEC:C', '-COMP:C'},
                            'N': {'-COMP:A', '-SPEC:Adv', '-COMP:V', '-COMP:D', '-COMP:V', '-COMP:T', '-COMP:Adv', '-SPEC:V', '-SPEC:T', '-SPEC:C', '-SPEC:N', '-SPEC:D', '-SPEC:N', '-SPEC:P', '-SPEC:T/inf'},
                            'T': {'!COMP:V', '-COMP:Adv', '-SPEC:C', '-SPEC:T', '-SPEC:V', '-SPEC:T/inf'},
                            'v': {'V', '!COMP:V', '!SPEC:D', '-COMP:Adv', '-COMP:A', '-COMP:v',  '-SPEC:T/inf', '!wCOMP:V'},
                            'V/INTR': {'-COMP:D', '-COMP:N', '!SPEC:D', '-COMP:T', '-SPEC:T/inf'}
                            }

# Major lexical categories assumed in this grammar
major_lexical_categories = ['C', 'N', 'v', 'V', 'T/inf', 'A', 'D', 'Adv', 'T', 'P', 'a', 'b', 'c', 'd']

def tcopy(SO):
    return tuple(x.copy() for x in SO)

# Class which stores and maintains the lexicon
class Lexicon:
    def __init__(self):
        self.lexical_entries = dict()   #   The lexicon is a dictionary
        self.compose_lexicon()          #   Creates the runtime lexicon by combining the lexicon and
                                        #   the lexical redundancy rules

    # Composes the lexicon from the list of words and lexical redundancy rules
    def compose_lexicon(self):
        for lex in lexicon.keys():
            self.lexical_entries[lex] = lexicon[lex]
            for trigger_feature in lexical_redundancy_rules.keys():
                if trigger_feature in lexicon[lex]:
                    self.lexical_entries[lex] = self.lexical_entries[lex] | lexical_redundancy_rules[trigger_feature]

    # Returns a lexical entry on the basis of the key [name]
    def retrieve_lexical_item(self, name):
        return self.lexical_entries[name]

#
# Asymmetric binary-branching phrase structure formalism
# together with several dependencies
#
class PhraseStructure:
    logging = None
    chain_index = 1
    logging_report = ''
    def __init__(self, X=None, Y=None):
        self.const = (X, Y)       # Left and right daughter constituents
        self.adjuncts = set()     # Adjunct pointers, for bookkeeping during derivation, not part of the theory
        self.phon = ''            # Name
        self.features = set()     # Lexical features
        self.zero = False         # Zero-level categories
        self.silent = False       # Phonological silencing
        self.chain_index = 0      # Marking chains in the output, not part of the theory
        self.mother = None        # Mother node
        if X:
            X.mother = self
        if Y:
            Y.mother = self

    # Definition for left constituent (abstraction)
    def left(self):
        return self.const[0]

    # Definition for right constituent (abstraction)
    def right(self):
        return self.const[1]

    def copy(X):
        if not X.terminal():
            Y = PhraseStructure(X.left().copy(), X.right().copy())
        else:
            Y = PhraseStructure()
        Y.copy_properties(X)
        return Y

    def copy_properties(Y, X):
        Y.phon = X.phon
        Y.features = X.features
        Y.zero = X.zero
        Y.chain_index = X.chain_index
        Y.silent = X.silent
        Y.adjuncts = X.adjuncts.copy()

    # Copying operation with phonological silencing
    # Implement chain numbers (if needed) inside this function
    def chaincopy(X):
        Y = X.copy()
        X.silent = True
        return Y

    # Zero-level categories are phrase structure objects with less that two daughter constituents
    # or they are marked as zero-level objects; these two are still kept separate since the
    # latter is currently an independent stipulation
    def zero_level(X):
        return X.zero or X.terminal()

    # Terminal elements do not have daughter constituents
    # Note: a constituent can be None, hence the search
    def terminal(X):
        for x in X.const:
            if x:
                return False
        return True

    # Standard bare Merge
    def Merge(X, Y):
        return PhraseStructure(X, Y)

    # Merge, with head and phrasal repair functions
    # Assumes that Move is part of Merge and derives the relevant
    # constructions without countercyclic operations
    def Merge_(X, Y):
        return {X.HeadMovement(Y).Merge(Y).PhrasalMovement()}

    # Preconditions for Merge
    def MergePreconditions(X, Y):
        if X.isRoot() and Y.isRoot():
            if Y.terminal() and Y.obligatory_wcomplement_features():
                return False
            if X.zero_level():                                                              #   Test if X selects Y
                return X.complement_subcategorization(Y)
            elif Y.zero_level():
                return Y.complement_subcategorization(None)                                 #   Test if Y requires a complement
            else:
                return Y.head().specifier_subcategorization(X)                              #   Test specifier subcategorization

    # Head repair for X before Merge
    def HeadMovement(X, Y):
        if X.zero_level() and X.bound_morpheme() and not Y.head().silent:           #   Preconditions
            PhraseStructure.logging_report += f'\tHead Chain ({X}, {Y.head()})\n'
            X = next(iter(Y.head().chaincopy().HeadMerge_(X)))                      #   Operation
        return X

    # Phrasal repair for X before Merge
    # We separate A- and A-bar chains explicitly
    def PhrasalMovement(X):
        H = X.head()
        target = None
        if H.complement():
            # Phrasal A-bar chains
            if H.scope_marker() and H.operator() and H.complement().minimal_search('WH') and not H.complement().minimal_search('WH').silent:
                target = H.complement().minimal_search('WH')
            # Phrasal A-chains
            elif H.EPP() and H.complement():
                if H.complement().head().specifier() and not H.complement().head().specifier().silent:
                    target = H.complement().head().specifier()
                elif H.complement().head().complement() and not H.complement().head().complement().silent:
                    target = H.complement().head().complement()
            # Copy the target and create a chain-index for better readability
            if target:
                target.babtize_chain()
                X = target.chaincopy().Merge(X)
                PhraseStructure.logging_report += f'\tChain ({H}, {target})\n'
        return X

    # Head Merge creates zero-level categories and implements feature inheritance
    def HeadMerge_(X, Y):
        Z = X.Merge(Y)
        Z.zero = True
        Z.features = Y.features
        Z.adjuncts = Y.adjuncts
        return {Z}

    # Preconditions for Head Merge (X Y)
    def HeadMergePreconditions(X, Y):
        return X.zero_level() and Y.zero_level() and Y.w_selects(X)

    # Word-internal selection between X and Y under (X Y), where
    # Y selects for X
    def w_selects(Y, X):
        return Y.leftmost().obligatory_wcomplement_features() and Y.leftmost().obligatory_wcomplement_features() <= X.rightmost().features

    def leftmost(X):
        while X.left():
            X = X.left()
        return X

    def rightmost(X):
        while X.right():
            X = X.right()
        return X

    # Adjunct Merge is a variation of Merge, but creates a parallel phrase structure
    def Adjoin_(X, Y):
        X.mother = Y           #   This is the actual operation
        Y.adjuncts.add(X)      #   For bookkeeping
        return {X, Y}

    # Preconditions for adjunct Merge
    def AdjunctionPreconditions(X, Y):
        return X.isRoot() and Y.isRoot() and X.head().adjoinable() and X.head().adjoinable() in Y.head().features

    def babtize_chain(X):
        if X.chain_index == 0:
            PhraseStructure.chain_index += 1
            X.chain_index = PhraseStructure.chain_index

    # Searches for a goal for phrasal movement, feature = target feature to be searched
    # This is also the kernel for Agree/probe-goal operation
    def minimal_search(X, feature):
        x = X
        while x:
            if x.zero_level():                          # From heads the search continues into complements
                x = x.complement()
            else:                                       # For complex constituents...
                for c in x.const:                       # examine both constituents and
                    if feature in c.head().features:    # return a constituent with the target feature, otherwise...
                        return c
                    if c.head() == x.head():            # search continues downstream inside the same projection
                        x = c

    # Determines whether X has a sister constituent and returns that constituent if present
    def sister(X):
        if X.mother:
            return next((const for const in X.mother.const if const != X), None)

    # Determines whether X has a right sister and return that constituent if present
    def complement(X):
        if X.sister() and X.mother.left() == X:
            return X.sister()

    # Left sister
    def left_sister(X):
        if X.sister() and X.mother.right() == X:
            return X.sister()

    # Calculates the head of any phrase structure object X ("labelling algorithm")
    # Returns the most prominent zero-level category inside X
    def head(X):
        for x in (X,) + X.const:            #   Order is from left to right
            if x and x.zero_level():        #   Returns the first zero-level object
                return x
        return x.head()                     #   Recursion

    # Verifies (recursively) that the configuration satisfies complement and
    # specifier subcategorization; only zero-level categories have subcategorization
    # requirements
    def subcategorization(X):
        if X.zero_level():
            return X.complement_subcategorization(X.complement()) and X.specifier_subcategorization() and X.w_subcategorization()
        else:
            return X.left().subcategorization() and X.right().subcategorization()    #   Recursion

    # Word-internal subcategorization that models morphotactic/morphological regularities
    # The recursive function looks for violations, otherwise returns True
    def w_subcategorization(X):
        if X.terminal():
            if X.obligatory_wcomplement_features():
                return False
        if X.left() and X.right():
            if not X.right().w_selects(X.left()):
                return False
        if X.left() and not X.left().terminal():
            if not X.left().w_subcategorization():
                return False
        if X.right() and not X.right().terminal():
            if not X.right().w_subcategorization():
                return False
        return True

    # Complement subcategorization under [X Y]
    # Returns False if subcategorization conditions are not met
    def complement_subcategorization(X, Y):
        comps = {f.split(':')[1] for f in X.features if f.startswith('!COMP')}
        if comps and not (Y and Y.head().features & comps):
            return False
        noncomps = {f.split(':')[1] for f in X.features if f.startswith('-COMP')}
        if noncomps and (Y and Y.head().features & noncomps):
            return False
        return True

    # Specifier subcategorization under [_YP XP YP]
    # Returns False if subcategorization conditions are not met
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

    # A generalized definition for the notion of specifier based on
    # head algorithm, allows specifier stacking and includes
    # left-adjoined phrases (if adjunction is part of the grammar)
    def specifier(X):
        x = X.head()
        while x and x.head() == X.head():
            if x.left_sister() and not x.left_sister().zero_level():
                return x.left_sister()
            x = x.mother

    # Maps phrase structure objects into linear lists of words
    # This function should contain all postsyntactic computations
    # involved in the implementation of the PF-spellout mapping
    def linearize(X):
        linearized_output_str = ''
        # Linearization of left adjuncts
        for x in (x for x in X.adjuncts if x.linearizes_left()):
            linearized_output_str += x.linearize()
        # Linearization of regular constituents
        if not X.silent:
            if X.zero_level():
                linearized_output_str += X.linearize_word('') + ' '
            else:
                for x in X.const:
                    linearized_output_str += x.linearize()
        # Linearization of right adjuncts
        for x in (x for x in X.adjuncts if x.linearizes_right()):
            linearized_output_str += x.linearize()
        return linearized_output_str

    # Spellout algorithm for words, creates morpheme boundaries marked by symbol #
    def linearize_word(X, word_str):
        if X.terminal():
            if word_str:
                word_str += '#'
            word_str += X.phon
        else:
            for x in X.const:
                word_str = x.linearize_word(word_str)
        return word_str

    # Definition for bound morpheme
    def bound_morpheme(X):
        return 'PC:#X' in X.features

    # Definition for EPP
    def EPP(X):
        return 'EPP' in X.features

    # Definition for operators
    def operator(X):
        return 'WH' in X.features

    # Definition for scope markers
    def scope_marker(X):
        return 'SCOPE' in X.features

    def linearizes_left(X):
        return 'λ:L' in X.head().features

    def linearizes_right(X):
        return 'λ:R' in X.head().features

    def isRoot(X):
        return not X.mother

    def obligatory_wcomplement_features(X):
        return {f.split(':')[1] for f in X.features if f.startswith('!wCOMP')}

    # Definition for adjoinability and returns the adjunction host
    def adjoinable(X):
        for f in X.features:
            if f.startswith('α:'):
                return f.split(':')[1]

    # Auxiliary printout function, to help eyeball the output
    def __str__(X):
        str = ''
        if X.silent:                    #   Phonologically silenced constituents are marked by __
            if X.zero_level():
                return '__ '
            else:
                return f'__:{X.chain_index} '
        if X.mother and X not in X.mother.const:                    # Adjunct printout, add the adjunction link
            if X.mother.zero_level():
                str += f'{X.mother.head().lexical_category()}|'
            else:
                str += f'{X.mother.head().lexical_category()}P|'
        if X.terminal():                #   Terminal constituents are spelled out
            str += X.phon
        else:
            if X.zero_level():          #   Non-terminal zero-level categories use different brackets
                bracket = ('(', ')')
            else:
                bracket = ('[', ']')
            str += bracket[0]
            if not X.zero_level():
                str += f'_{X.head().lexical_category()}P '  #   Print information about heads and labelling
            for const in X.const:
                str += f'{const} '
            str += bracket[1]
            if X.chain_index != 0:
                str += f':{X.chain_index} '
        for x in X.adjuncts:
            str += '*'
        return str

    # Defines the major lexical categories used in all printouts
    def lexical_category(X):
        return next((f for f in X.features if f in major_lexical_categories), '?')

#
# Model of the speaker which constitutes the executive layer
# In more realistic models the speaker models must be language-specific
#
class SpeakerModel:
    def __init__(self):
        # List of all syntactic operations available in the grammar
        self.syntactic_operations = [(PhraseStructure.MergePreconditions, PhraseStructure.Merge_, 2, 'Merge'),
                                     (PhraseStructure.HeadMergePreconditions, PhraseStructure.HeadMerge_, 2, 'Head Merge'),
                                     (PhraseStructure.AdjunctionPreconditions, PhraseStructure.Adjoin_, 2, 'Adjoin')]
        self.n_accepted = 0         # Counts the number of derivations
        self.n_steps = 0            # Number of derivational steps
        self.output_data = set()    # Stores grammatical output data from the model
        self.lexicon = Lexicon()    # Lexicon
        self.log_file = None        # Log file

    # Retrieves lexical items from the lexicon and wraps them into zero-level phrase structure
    # objects
    def LexicalRetrieval(self, name):
        ps = PhraseStructure()
        ps.features = self.lexicon.retrieve_lexical_item(name)  # Retrieves lexical features from the lexicon
        ps.phon = name                                          # Name for identification and easy recognition
        ps.zero = True                                          # True = zero-level category
        return ps

    # Wrapper function for the derivational search function
    # Performs initialization and maps the input into numeration
    def derive(self, numeration):
        self.n_steps = 0
        self.output_data = set()
        self.n_accepted = 0
        self.derivational_search_function([self.LexicalRetrieval(item) for item in numeration])

    # Derivational search function
    def derivational_search_function(self, sWM):
        if self.derivation_is_complete(sWM):                                                                #   Only one phrase structure object in working memory
            self.process_final_output(sWM)                                                                  #   Terminate processing and evaluate solution
        else:
            for Preconditions, OP, n, name in self.syntactic_operations:                                    #   Examine all syntactic operations OP
                for SO in itertools.permutations(sWM, n):                                                   #   All n-tuples of objects in sWM
                    if Preconditions(*SO):                                                                  #   Blocks illicit derivations
                        PhraseStructure.logging_report += f'\t{name}({self.print_lst(SO)})'                 #   Add line to logging report
                        new_sWM = {x for x in sWM if x not in set(SO)} | OP(*tcopy(SO))                     #   Update sWM
                        self.consume_resource(new_sWM, sWM)                                                 #   Record resource consumption and write log entries
                        self.derivational_search_function(new_sWM)                                          #   Continue derivation, recursive branching


    @staticmethod
    def derivation_is_complete(sWM):
        return len({X for X in sWM if X.isRoot()}) == 1

    @staticmethod
    def root_structure(sWM):
        return next((X for X in sWM if not X.mother))

    # Resource recording, this is what gets printed into the log file
    # Modify to enhance readability and to reflect the operations available
    # in the grammar
    def consume_resource(self, new_sWM, old_sWM):
        self.n_steps += 1
        self.log_file.write(f'{self.n_steps}.\n\n')
        self.log_file.write(f'\t{self.print_constituent_lst(old_sWM)}\n')
        self.log_file.write(f'{PhraseStructure.logging_report}')
        self.log_file.write(f'\n\t= {self.print_constituent_lst(new_sWM)}\n\n')
        PhraseStructure.logging_report = ''

    # Processes final output
    # This corresponds to the horizontal branches of the Y-architecture
    def process_final_output(self, sWM):
        PhraseStructure.chain_index = 0
        self.log_file.write(f'\t{self.print_constituent_lst(sWM)}\n')
        for X in sWM:
            if not X.subcategorization():
                self.log_file.write('\n\n')
                return
        self.n_accepted += 1
        prefix = f'{self.n_accepted}'
        output_sentence = f'{self.root_structure(sWM).linearize()}'
        print(f'\t({prefix}) {output_sentence} {self.print_constituent_lst(sWM)}')   # Print the output
        self.log_file.write(f'\t^ ACCEPTED: {output_sentence}')
        self.output_data.add(output_sentence.strip())
        self.log_file.write('\n\n')

    def print_lst(self, lst):
        str = ''
        for i, ps in enumerate(lst):
            if ps.terminal():
                str += f'{ps}°'
            else:
                str += f'{ps}'
            if i < len(lst) - 1:
                str += ', '
        return str

    # To help understand the output
    def print_constituent_lst(self, sWM):
        aWM = [x for x in sWM if not x.mother]
        iWM = [x for x in sWM if x.mother]
        str = f'{self.print_lst(aWM)}'
        if iWM:
            str += f' + {{ {self.print_lst(iWM)} }}'
        return str

#
# This class maintains all data used in the simulation
#
class LanguageData:
    def __init__(self):
        self.study_dataset = []           # Contains the datasets for one study, which contains tuples of numerations and datasets
        self.log_file = None

    # Read the dataset
    def read_dataset(self, filename):
        numeration = []
        dataset = set()
        with open(filename) as f:
            lines = f.readlines()
            for line in lines:
                if line.strip() and not line.startswith('#') and not line.startswith('END'):
                    if line.startswith('Numeration='):
                        if numeration:
                            self.study_dataset.append((numeration, dataset))
                            dataset = set()
                        numeration = [word.strip() for word in line.split('=')[1].split(',')]
                    else:
                        dataset.add(line.strip())
                if line.startswith('END'):
                    break
            self.study_dataset.append((numeration, dataset))

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

# Run one whole study as defined by the dataset file, itself containing
# numeration-target sentences blocks
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


ld = LanguageData()                 #   Instantiate the language data object
ld.read_dataset('dataset3.txt')     #   Name of the dataset file processed by the script, reads the file
sm = SpeakerModel()                 #   Create default speaker model, would be language-specific in a more realistic model
run_study(ld, sm)                   #   Runs the study
