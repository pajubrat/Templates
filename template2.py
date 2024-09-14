#
# Template script for Brattico, P. (2024). Computational generative grammar and complexity.
#
# Implements a simple entry-point script for the mechanical justification of grammars
#

import itertools

# Defines a simple root lexicon as a dictionary
lexicon = {'a': {'a'}, 'b': {'b'}, 'c': {'c'}, 'd': {'d'},
           'the': {'D'},
           'dog': {'N'},
           'bark': {'V', 'V/INTR'},
           'barks': {'V', 'V/INTR'},
           'ing': {'N', '+wCOMP:V', 'PC:#X', 'ε'},
           'bites': {'V', '+COMP:D', '+SPEC:D'},
           'bite': {'V', '+COMP:D'},
           'bite*': {'V', 'V/TR'},
           'which': {'D', 'WH'},
           'man': {'N'},
           'angry': {'A', 'α:N', 'λ:L'},
           'frequently': {'Adv', 'α:V', 'λ:R'},
           'city': {'N'},
           'from': {'P'},
           'in': {'P', 'α:V', },
           'ed': {'T', 'PC:#X', '+wCOMP:V'},
           'T': {'T', 'PC:#X', 'EPP', '+SPEC:D', '+wCOMP:V'},
           'T*': {'T', 'PC:#X', '+wCOMP:V'},
           'did': {'T', 'EPP'},
           'does': {'T'},
           'was': {'T', 'EPP'},
           'C': {'C'},
           'C(wh)': {'C', 'C(wh)', 'PC:#X', '+wCOMP:T', 'WH', 'SCOPE'},
           'v': {'v', 'PC:#X', '+wCOMP:V'},
           'v*': {'V', 'EPP', 'PC:#X', '+COMP:V', '-SPEC:v', '+wCOMP:V'},
           'that': {'C'},
           'believe': {'V', '+COMP:C'},
           'seem': {'V', 'EPP', '+SPEC:D', '+COMP:T/inf', 'RAISING'},
           'to': {'T/inf', '+COMP:V', '-COMP:RAISING', '-COMP:T', 'EPP'}}

# Lexical redundancy rules add features to lexical items based on their feature content
# This creates speaker lexicons
lexical_redundancy_rules = {'D': {'+COMP:N', '-COMP:Adv', '-SPEC:C', '-SPEC:T', '-SPEC:N', '-SPEC:V', '-SPEC:D', '-SPEC:P', '-SPEC:T/inf', '-SPEC:Adv'},
                            'V': {'-SPEC:C', '-SPEC:N', '-SPEC:T', '-SPEC:T/inf', '-COMP:A', '-COMP:N', '-COMP:T'},
                            'Adv': {'-COMP:D', '-COMP:N', '-SPEC:V', '-SPEC:v', '-SPEC:T', '-SPEC:D', '-COMP:Adv', '-COMP:A'},
                            'P': {'+COMP:D', '-COMP:Adv', '-SPEC:Adv', '-SPEC:C', '-SPEC:T', '-SPEC:N', '-SPEC:V', '-SPEC:v', '-SPEC:T/inf', 'λ:R'},
                            'C': {'+COMP:T', '-COMP:Adv', '-SPEC:V', '-SPEC:C', '-SPEC:N', '-SPEC:T/inf'},
                            'A': {'-COMP:D', '-SPEC:Adv', '-COMP:Adv', '-SPEC:D', '-SPEC:V', '-COMP:V', '-COMP:T', '-SPEC:T', '-SPEC:C', '-COMP:C'},
                            'N': {'-COMP:A', '-SPEC:Adv', '-COMP:V', '-COMP:D', '-COMP:V', '-COMP:T', '-COMP:Adv', '-SPEC:V', '-SPEC:T', '-SPEC:C', '-SPEC:N', '-SPEC:D', '-SPEC:N', '-SPEC:P', '-SPEC:T/inf'},
                            'T': {'+COMP:V', '-COMP:Adv', '-SPEC:C', '-SPEC:T', '-SPEC:V', '-SPEC:T/inf', '-ε'},
                            'v': {'V', '+COMP:V', '+SPEC:D', '-COMP:Adv', '-COMP:A', '-COMP:v',  '-SPEC:T/inf', '+wCOMP:V', '-ε'},
                            'V/INTR': {'-COMP:D', '+SPEC:D'},
                            'V/TR': {'-SPEC:D', '+COMP:D'}
                            }

major_lexical_categories = ['C', 'N', 'v', 'V', 'T/inf', 'A', 'D', 'Adv', 'T', 'P', 'a', 'b', 'c', 'd']

# Auxiliary functions used during during certain processing steps
def tcopy(SO):
    return tuple(x.copy() for x in SO)

def tset(X):
    if isinstance(X, set):
        return X
    else:
        return {X}

def print_lst(lst):
    return ', '.join([f'{x}' for x in lst])

def print_constituent_lst(sWM):
    str = f'{print_lst([x for x in sWM if not x.mother])}'
    if [x for x in sWM if x.mother]:
        str += f' + {{ {print_lst([x for x in sWM if x.mother])} }}'
    return str


class Lexicon:
    """Stores lexical knowledge independent of the syntactic phrase structure"""
    def __init__(self):
        self.speaker_lexicon = dict()   # The lexicon is a dictionary
        self.compose_speaker_lexicon()  # Creates the runtime lexicon by combining the root lexicon and
                                        # the lexical redundancy rules

    def compose_speaker_lexicon(self):
        """Composes the speaker lexicon from the list of words and lexical redundancy rules"""
        for lex in lexicon.keys():
            self.speaker_lexicon[lex] = lexicon[lex]
            for trigger_feature in lexical_redundancy_rules.keys():
                if trigger_feature in lexicon[lex]:
                    self.speaker_lexicon[lex] = self.speaker_lexicon[lex] | lexical_redundancy_rules[trigger_feature]

    def retrieve(self, name):
        """Retrieves lexical items from the speaker lexicon and wraps them
        into zero-level phrase structure objects"""
        X0 = PhraseStructure()
        X0.features = self.speaker_lexicon[name]
        X0.phonological_exponent = name
        X0.zero = True
        return X0


class PhraseStructure:
    """Simple asymmetric binary-branching bare phrase structure formalism"""
    logging = None          # class variable: whether logging is on or off, affects calculation speed
    chain_index = 1         # class variable: creates chain indexes
    logging_report = ''     # class variable: used to store logging information during the derivation
    def __init__(self, X=None, Y=None):
        self.const = (X, Y)             # Constituents are represented as tuples
                                        # Alternatives: list; axiomatic left/right constituents; dictionary
        self.features = set()           # Features (set of string literals)
        self.mother = None              # Phrase structure geometry...
        if X:                           # ...
            X.mother = self             # ...
        if Y:                           # ...
            Y.mother = self             # ...
        self.zero = False               # Whether the object is a zero-level object
        self.adjuncts = set()           # Stores the set of adjuncts adjoined to the constituent
                                        # This is an auxiliary structure that is not part of the theory and simplifies
                                        # certain calculations
        self.phonological_exponent = '' # Phonological exponent of the constituent, used in printout and linearization
        self.elliptic = False           # Ellipsis due to copying
        self.chain_index = 0            # Initial chain index

    def left(X):
        """Abstraction for the notion of left daughter"""
        return X.const[0]

    def right(X):
        """Abstraction for the notion of right daughter"""
        return X.const[1]

    def Merge(X, Y):
        """Standard Merge"""
        return PhraseStructure(X, Y)

    def isLeft(X):
        """Determines whether X is the left constituent"""
        return X.sister() and X.mother.left() == X

    def isRight(X):
        """Determines whether X is the right constituent"""
        return X.sister() and X.mother.right() == X

    def phrasal(X):
        """Defines the notion of phrasal constituent"""
        return X.left() and X.right()

    def copy(X):
        """Recursive copying for constituents"""
        if not X.terminal():
            Y = PhraseStructure(X.left().copy(), X.right().copy())
        else:
            Y = PhraseStructure()
        return Y.copy_properties(X)

    def copy_properties(Y, X):
        """Copies properties of X into Y and returns Y."""
        Y.phonological_exponent = X.phonological_exponent
        Y.features = X.features
        Y.zero = X.zero
        Y.chain_index = X.chain_index
        Y.elliptic = X.elliptic
        Y.adjuncts = X.adjuncts.copy()
        return Y

    def chaincopy(X):
        """Grammatical copying operation, with phonological silencing"""
        X.label_chain()     # Create chain information, not part of the theory
        Y = X.copy()        # Copying
        X.elliptic = True   # Mark the source elliptic
        return Y

    def zero_level(X):
        """Zero-level categories are considered primitive by phrasal syntactic rules"""
        return X.zero or X.terminal()

    def terminal(X):
        """Terminal elements do not have constituents"""
        return not X.right() and not X.left()

    def MergeComposite(X, Y):
        """
        Composite Merge operation contains head and phrasal movements (if applicable) and Merge.
        This format was adopted in the main article, but is not the only model possible. An alternative is
        to detach head movement and phrasal movement into their own operations that are applied independently
        by the derivational search function
        """
        return X.HeadMovement(Y).Merge(Y).PhrasalMovement()

    def MergePreconditions(X, Y):
        """
        Preconditions for Merge. These conditions apply at each merge operation and
        restrict the derivational search space.
        """
        if X.isRoot() and Y.isRoot():                                   # Both elements must be roots (no mother)
            if Y.terminal() and Y.obligatory_wcomplement_features():    # Y cannot be merged if it can only be a
                                                                        # word-internal constituent. Empirical assumption
                                                                        # that plays no role in the discussion in the
                return False                                            # main article.
            if X.zero_level():                                          # If X is zero-level, then Y must satisfy its
                return X.complement_subcategorization(Y)                # subcategorization features under [X Y(P)]
            elif Y.zero_level():                                        # If Y is zero-level, then
                return Y.complement_subcategorization(None)             # Y cannot require a complement,
            else:                                                       # else
                return Y.head().specifier_subcategorization(X)          # X must satisfy the specifier requirements
                                                                        # of the head of YP.

    def HeadMovement(X, Y):
        """Implementation for head movement"""
        if X.HeadMovementPreconditions(Y):
            PhraseStructure.logging_report += f'\n\t\t + Head chain by {X}° targeting {Y.head()}°'
            return Y.head().chaincopy().HeadMerge_(X)   # Under [X YP], pick the head Y, copy it and merge with X.
        return X    # If the preconditions are not satisfied, return an unmodified X

    def HeadMovementPreconditions(X, Y):
        """
        Head movement preconditions which apply during each Merge operation [X YP]:
        (i) X must be zero-level, phrasal constituents cannot trigger it;
        (ii) X must be bound morpheme, don't combine free morphemes;
        (iii) X cannot create complex heads directly without head movement (if they can, then that option must be used)
        """
        return X.zero_level() and \
               X.bound_morpheme() and \
               not X.mandateDirectHeadMerge()

    def PhrasalMovement(X):
        """Implementation for phrasal movement: first A-bar movement, then A-movement"""
        return X.phrasal_A_bar_movement().phrasal_A_movement()

    def phrasal_A_bar_movement(X):
        """
        Implementation for phrasal A-bar movement for phrase X. The conditions are:
        (i) The head of X is a scope-marking element (e.g. C; X = CP);
        (ii) the head of X is an operator (e.g., C[wh]);
        (iii) the complement of X contains an item with [wh];
        (iv) the item marked with [wh] has not already been moved;
        """
        if X.head().scope_marker() and \
                X.head().operator() and \
                X.head().complement() and \
                X.head().complement().minimal_search('WH') and \
                not X.head().complement().minimal_search('WH').elliptic:
            PhraseStructure.logging_report += f'\n\t\t + Phrasal A-bar chain by {X.head()}° targeting {X.head().complement().minimal_search("WH")}'
            return X.head().complement().minimal_search('WH').chaincopy().Merge(X)
        return X

    def phrasal_A_movement(X):
        """
        Implementation for phrasal A-movement. The operation applies iff
        (i) the head of X has the EPP-feature;
        (ii-iii) the head of X has a phrasal complement YP;
        (iv) YP contains a suitable element that can be moved.
        """
        if X.head().EPP() and X.head().complement() and X.head().complement().phrasal() and X.head().complement().goal_for_A_movement():
            PhraseStructure.logging_report += f'\n\t\t + Phrasal A chain by {X.head()}° targeting {X.head().complement().goal_for_A_movement()}'
            return X.head().complement().goal_for_A_movement().chaincopy().Merge(X)
        return X

    def referential(X):
        """Referential phrases are headed by D"""
        return 'D' in X.head().features

    def goal_for_A_movement(X):
        """
        Heuristic search algorithm for the potential targets for A-movement. This should
        be unified with minimal search.
        """
        return next((x for x in [X.left(), X.right()] if x.phrasal() and x.referential()), None)

    def HeadMerge_(X, Y):
        """Direct Head Merge creates zero-level objects from two zero-level objects"""
        Z = X.Merge(Y)
        Z.zero = True
        Z.features = Y.features     #   Feature inheritance
        Z.adjuncts = Y.adjuncts     #   Feature inheritance
        return Z

    def HeadMergePreconditions(X, Y):
        """Preconditions for direct Head Merge are that both objects must be
        zero-level objects, Y must select X and license the operation"""
        return X.zero_level() and \
               Y.zero_level() and \
               Y.w_selects(X) and \
               Y.mandateDirectHeadMerge()

    def w_selects(Y, X):
        """Word-internal selection (X Y) where Y w-selects X"""
        return Y.leftmost().obligatory_wcomplement_features() <= X.rightmost().features

    def leftmost(X):
        while X.left():
            X = X.left()
        return X

    def rightmost(X):
        while X.right():
            X = X.right()
        return X

    def Adjoin_(X, Y):
        """Adjunction creates asymmetric constituents with mother-of dependency without
        daughter dependency"""
        X.mother = Y
        Y.adjuncts.add(X)
        return {X, Y}

    def AdjunctionPreconditions(X, Y):
        return X.isRoot() and \
               Y.isRoot() and \
               X.head().license_adjunction() and \
               X.head().license_adjunction() in Y.head().features

    def label_chain(X):
        if X.chain_index == 0:
            PhraseStructure.chain_index += 1
            X.chain_index = PhraseStructure.chain_index

    def minimal_search(X, feature):
        """One implementation for minial search"""
        while X:
            if X.zero_level():                          # [X YP], search YP
                X = X.complement()
            else:
                for c in X.const:                       # [_LP XP Y(P)], search the constituent that
                    if feature in c.head().features:    # provides the label L.
                        return c
                    if c.head() == X.head():
                        X = c

    def sister(X):
        """Sisterhood"""
        if X.mother:
            return next((const for const in X.mother.const if const != X), None)

    def complement(X):
        """Complement is a right sister of a zero-level object"""
        if X.zero_level() and X.isLeft():
            return X.sister()

    # Calculates the head of any phrase structure object X ("labelling algorithm")
    # Returns the most prominent zero-level category inside X
    def head(X):
        for x in (X,) + X.const:
            if x and x.zero_level():
                return x
        return x.head()

    def subcategorization(X):
        """
        Recursive interface test for complement and specifier subcategorization; an example
        of how to test subcategorization as an output well-formedness condition
        """
        if X.zero_level():
            return X.complement_subcategorization(X.complement()) and \
                   X.specifier_subcategorization() and \
                   X.w_subcategorization()
        return X.left().subcategorization() and X.right().subcategorization()

    def w_subcategorization(X):
        """Word-internal subcategorization which applies when heads are merged directly
        to create zero-level constituents. This was not discussed in the main article."""
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

    def complement_subcategorization(X, Y):
        """Complement subcategorization under [X Y]"""
        if not Y:                                   # If there is no complement, then
            return not X.positive_comp_selection()  # there cannot be positive comp selection
        return (X.positive_comp_selection() <= Y.head().features) and \
               not (X.negative_comp_selection() & Y.head().features)
        # ^ Test both positive and negative complement selection

    def specifier_subcategorization(X, Spec=None):
        """Specifier subcategorization under [XP YP]"""
        if not Spec:
            if not X.specifier():
                return not X.positive_spec_selection()
            Spec = X.specifier()
        return X.positive_spec_selection() <= Spec.head().features and \
               not (X.negative_spec_selection() & Spec.head().features)
        # ^ Test positive and negative specifier selection

    def specifier(X):
        """
        Specifier of X is phrasal left constituent inside the project from X. The definition of
        specifier is controversial. This returns the left phrasal constituent closest to the head (of) X.
        """
        x = X.head()
        while x and x.mother and x.mother.head() == X:
            if x.mother.left() != X:
                return x.mother.left()
            x = x.mother

    def linearize(X):
        """Linearization algorithm that is unrealistic but written to be as easy as possible
        to understand"""
        stri = ''
        # Linearize left adjuncts first
        stri += ''.join([x.linearize() for x in X.adjuncts if x.linearizes_left()])
        # Linearize X if it is not elliptic
        if not X.elliptic:
            if X.zero_level():
                # Word-internal linearization by different function, which is realistic
                stri += X.linearize_word()[:-1] + ' '
            else:
                stri += ''.join([x.linearize() for x in X.const])
        # Linearize right adjuncts last
        stri += ''.join([x.linearize() for x in X.adjuncts if x.linearizes_right()])
        return stri

    def linearize_word(X):
        """Spellout algorithm for words, creates morpheme boundaries marked by symbol #
        This function would include all word-internal e.g. morphophonological operations that
        apply inside words.
        """
        if X.terminal():
            return X.phonological_exponent + '#'
        return ''.join([x.linearize_word() for x in X.const])

    def bound_morpheme(X):
        """Definition for bound morpheme"""
        return 'PC:#X' in X.features

    def EPP(X):
        """Definition for EPP"""
        return 'EPP' in X.features

    def operator(X):
        """Definition for operators"""
        return 'WH' in X.features

    def scope_marker(X):
        """Definition for scope markers"""
        return 'SCOPE' in X.features

    def linearizes_left(X):
        """Adjunct linearization to left is controlled by a lexical feature"""
        return 'λ:L' in X.head().features

    def linearizes_right(X):
        """Adjunct linearization to right is controlled by a lexical feature"""
        return 'λ:R' in X.head().features

    def isRoot(X):
        """Roots are constituents that do not have mothers"""
        return not X.mother

    def mandateDirectHeadMerge(X):
        return 'ε' in X.features

    # Functions for parsing complex features
    #
    def obligatory_wcomplement_features(X):
        return {f.split(':')[1] for f in X.features if f.startswith('+wCOMP')}

    def positive_spec_selection(X):
        return {f.split(':')[1] for f in X.features if f.startswith('+SPEC')}

    def negative_spec_selection(X):
        return {f.split(':')[1] for f in X.features if f.startswith('-SPEC')}

    def positive_comp_selection(X):
        return {f.split(':')[1] for f in X.features if f.startswith('+COMP')}

    def negative_comp_selection(X):
        return {f.split(':')[1] for f in X.features if f.startswith('-COMP')}

    def license_adjunction(X):
        return next((f.split(':')[1] for f in X.features if f.startswith('α:')), None)

    def __str__(X):
        """Simple printout function for phrase structure objects"""
        if X.elliptic:
            return '__' + X.get_chain_subscript()
        if X.terminal():
            return X.phonological_exponent
        elif X.zero_level():
            return '(' + ' '.join([f'{x}' for x in X.const]) + ')'
        return f'[_{X.head().lexical_category()}P ' + ' '.join([f'{x}' for x in X.const]) + ']' + X.get_chain_subscript()

    def get_chain_subscript(X):
        if X.chain_index != 0:
            return ':' + str(X.chain_index)
        return ''

    def lexical_category(X):
        """# Defines the major lexical categories used in all printouts"""
        return next((f for f in major_lexical_categories if f in X.features), '?')

#
# Model of the speaker which constitutes the executive layer
# In more realistic models the speaker models must be language-specific
#
class SpeakerModel:
    def __init__(self):
        # List of all syntactic operations available in the grammar
        self.syntactic_operations = [(PhraseStructure.MergePreconditions, PhraseStructure.MergeComposite, 2, 'Merge'),
                                     (PhraseStructure.HeadMergePreconditions, PhraseStructure.HeadMerge_, 2, 'Head Merge'),
                                     (PhraseStructure.AdjunctionPreconditions, PhraseStructure.Adjoin_, 2, 'Adjoin')]
        self.n_accepted = 0         # Number of accepted sentences
        self.n_steps = 0            # Number of steps consumed
        self.output_data = set()    # A set containing all output data
        self.lexicon = Lexicon()    # Lexicon
        self.log_file = None        # Logging

    def derive(self, numeration):
        """Finds all derivations from numeraton"""
        self.n_steps = 0
        self.output_data = set()
        self.n_accepted = 0
        self.derivational_search_function([self.lexicon.retrieve(item) for item in numeration])

    def derivational_search_function(self, sWM):
        """Derivational search function"""
        if self.derivation_is_complete(sWM):
            self.process_final_output(sWM)
        else:
            for Preconditions, OP, n, name in self.syntactic_operations:
                for SO in itertools.permutations(sWM, n):
                    if Preconditions(*SO):
                        PhraseStructure.logging_report += f'\n\t{name}({print_lst(SO)})'
                        new_sWM = {x for x in sWM if x not in set(SO)} | tset(OP(*tcopy(SO)))
                        self.consume_resource(new_sWM, sWM)
                        self.derivational_search_function(new_sWM)

    @staticmethod
    def derivation_is_complete(sWM):
        return len({X for X in sWM if X.isRoot()}) == 1

    @staticmethod
    def root_structure(sWM):
        return next((X for X in sWM if not X.mother))

    def consume_resource(self, new_sWM, old_sWM):
        """
        Resource recording, this is what gets printed into the log file.
        Modify to enhance readability and to reflect the operations available in the grammar
        """
        self.n_steps += 1
        self.log_file.write(f'{self.n_steps}.\n\n')
        self.log_file.write(f'\t{print_constituent_lst(old_sWM)}\n')
        self.log_file.write(f'{PhraseStructure.logging_report}')
        self.log_file.write(f'\n\t= {print_constituent_lst(new_sWM)}\n\n')
        PhraseStructure.logging_report = ''

    def process_final_output(self, sWM):
        PhraseStructure.chain_index = 0
        self.log_file.write(f'\t{print_constituent_lst(sWM)}\n')
        for X in sWM:
            if not X.subcategorization():
                self.log_file.write('\n\n')
                return
        self.n_accepted += 1
        prefix = f'{self.n_accepted}'
        output_sentence = f'{self.root_structure(sWM).linearize()}'
        print(f'\t({prefix}) {output_sentence} {print_constituent_lst(sWM)}')   # Print the output
        self.log_file.write(f'\t^ ACCEPTED: {output_sentence}')
        self.output_data.add(output_sentence.strip())
        self.log_file.write('\n\n')


class LanguageData:
    """Stores and manipulates all data used in the simulation"""
    def __init__(self):
        self.study_dataset = []     # dataset is a list of (numeration, set of target sentences) tuples
        self.log_file = None

    # Read the dataset
    def read_dataset(self, filename):
        numeration = []
        dataset = set()
        with open(filename) as f:
            lines = f.readlines()
            for line in lines:
                if line.strip() and not line.startswith('#') and not line.startswith('END'):
                    line = line.strip()
                    if line.startswith('Numeration='):
                        if numeration:
                            self.study_dataset.append((numeration, dataset))
                            dataset = set()
                        # Items in the numeration are separated by comma
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
        errors = len(overgeneralization) + len(undergeneralization)
        print(f'\tErrors {errors}')
        if errors > 0:
            print(f'\tShould not generate: {overgeneralization}')
            print(f'\tShould generate: {undergeneralization}')
        return errors

# Run one whole study as defined by the dataset file, itself containing
# numeration-target sentences blocks
def run_study(ld, sm):
    sm.log_file = ld.start_logging()
    n_dataset = 0       #   Number of datasets in the experiment (counter)
    n_total_errors = 0  #   Count the number of errors in the whole experiment (counter)
    for numeration, gold_standard_dataset in ld.study_dataset:
        n_dataset += 1
        print(f'Dataset {n_dataset}:')
        sm.log_file.write('\n---------------------------------------------------\n')
        sm.log_file.write(f'Dataset {n_dataset}:\n')
        sm.log_file.write(f'Numeration: {numeration}\n')
        sm.log_file.write(f'Predicted outcome: {gold_standard_dataset}\n\n\n')
        sm.derive(numeration)
        n_total_errors += ld.evaluate_experiment(sm.output_data, gold_standard_dataset, sm.n_steps)
    print(f'\nTotal errors: {n_total_errors}\n')
    sm.log_file.write(f'\nTotal errors: {n_total_errors}')


ld = LanguageData()                         #   Instantiate the language data object
ld.read_dataset('dataset_template2.txt')    #   Name of the dataset file processed by the script, reads the file
sm = SpeakerModel()                         #   Create default speaker model, would be language-specific in a more realistic model
run_study(ld, sm)                           #   Runs the study
