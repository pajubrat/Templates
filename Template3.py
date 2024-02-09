#
# Starting point template for derivational search function
# for asymmetric binary-branching bare phrase structure
# with some more advanced features
#

import itertools

# Lexicon with some features (move into external files)
lexicon = {'the': {'D'},
           'dog': {'N'},
           'man': {'N'},
           'T': {'T', '!COMP:V', '#X'},
           'bite': {'V', '!COMP:D', '!SPEC:D'}}

lexical_redundancy_rules = {'D': {'!COMP:N', '-SPEC:D'},
                            'N': {'-COMP:V', '-COMP:D'}}

major_lexical_categories = {'N', 'V', 'A', 'D', 'Adv', 'T'}

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
    def __init__(self, X=None, Y=None):
        self.const = (X, Y)       # Left and right daughter constituents
        self.phon = ''            # Name
        self.features = None      # Lexical features
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

    def terminal(X):
        return not X.const[0] or not X.const[1]

    # Merge
    def Merge(X, Y):
        return PhraseStructure(X, Y)

    def MergePreconditions(X, Y):
        return not X.zero_level() or X.complement_subcategorization(Y)

    # Wrapper function for Merge
    def Merge_(X, Y):
        return X.HeadRepair(Y).Merge(Y)

    def HeadRepair(X, Y):
        if X.zero_level() and X.bound_morpheme() and not Y.head().silent:
            return Y.head().chaincopy().HeadMerge_(X)
        return X

    def HeadMerge_(X, Y):
        if X.zero_level() and Y.zero_level() and Y.bound_morpheme():
            Z = X.Merge(Y)
            Z.zero = True
            Z.features = Y.features
            return Z

    # Determines whether X has a sister constituent and returns that constituent if present
    def sister(X):
        if X.mother:
            return next((const for const in X.mother.const if const != X), None)

    # Determines whether X has a right sister and return that constituent if present
    def right_sister(X):
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
            return X.const[1].head()    #   [XP YP]: return the head of YP

    # Verifies (recursively) that all zero-level categories of X satisfies all subcategorization
    # features
    def subcategorization(X):
        if X.zero_level():
            return X.complement_subcategorization(X.right_sister()) and X.specifier_subcategorization()
        else:
            return X.const[0].subcategorization() and X.const[1].subcategorization()

    def complement_subcategorization(X, Y):
        comps = {f.split(':')[1] for f in X.features if f.startswith('!COMP')}  # Subcat features
        noncomps = {f.split(':')[1] for f in X.features if f.startswith('-COMP')}
        if comps and not (Y and Y.head().features & comps):
            return False
        if noncomps and (Y and Y.head().features & noncomps):
            return False
        return True

    def specifier_subcategorization(X):
        specs = {f.split(':')[1] for f in X.features if f.startswith('!SPEC')}
        nonspecs = {f.split(':')[1] for f in X.features if f.startswith('-SPEC')}
        return not ((specs and not (X.specifier() and X.specifier().head().features & specs)) or
                    (nonspecs and (X.specifier() and X.specifier().head().features & nonspecs)))

    def specifier(X):
        node = X.head()
        while node.mother and node.mother.head() == X.head():
            if node.mother.left_sister():
                return node.mother.left_sister()
            node = node.mother

    # Maps phrase structure objects into linear lists of words
    def linearize(X):
        linearized_output_str = ''                               #   Holds the list
        if X.zero_level():
            if not X.silent:
                linearized_output_str += X.linearize_word('') + ' '
        else:
            linearized_output_str += X.const[0].linearize()      # Recursion to left
            linearized_output_str += X.const[1].linearize()      # Recursion to right
        return linearized_output_str

    def linearize_word(X, word_str):
        if X.terminal():
            if word_str:
                word_str += '#'
            word_str += X.phon
        else:
            word_str = X.const[0].linearize_word(word_str)
            word_str = X.const[1].linearize_word(word_str)
        return word_str

    def bound_morpheme(X):
        return '#X' in X.features

    # Printout function
    def __str__(X):
        str = ''
        if X.silent:                    #   Phonologically silenced constituents are marked by __
            return '_'
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
        self.derivational_search_function([self.LexicalRetrieval(item) for item in numeration])

    # Derivational search function
    def derivational_search_function(self, sWM):
        if len(sWM) == 1:                                   #   Only one phrase structure object in working memory
            self.final_output(next(iter(sWM)))              #   Terminate processing and evaluate solution
        else:
            for Preconditions, OP, name in self.external_syntactic_operations:
                for X, Y in itertools.permutations(sWM, 2):
                    if Preconditions(X, Y):
                        Z = OP(X.copy(), Y.copy())
                        new_sWM = {x for x in sWM if x not in {X, Y}} | {Z}
                        self.consume_resource(name, [X, Y], Z, new_sWM)
                        self.derivational_search_function(new_sWM)

    def consume_resource(self, name, lst, result, sWM):
        self.n_steps += 1
        self.log_file.write(f'{self.n_steps}.\n\t{name}({self.print_constituent_lst(lst)}) = {result}\n')
        self.log_file.write(f'\tSyntactic working memory: {self.print_constituent_lst(sWM)}\n\n')

    # Processes final output
    def final_output(self, X):
        prefix = ''
        self.log_file.write(f'\t{X}')
        if X.subcategorization():  # Store only grammatical sentences
            output_sentence = f'{prefix}{X.linearize()}'
            self.log_file.write(f'\n\n\t ^ ACCEPTED, OUTPUT: {output_sentence}')
            self.output_data.add(output_sentence.strip())
            print(f'{output_sentence} {X}')   # Print the output
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
        self.dataset = set()        # Contains the dataset with target sentences
        self.numeration = []        # Contains the numeration
        self.log_file = None

    def read_dataset(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            for line in lines:
                if line.strip() and not line.startswith('#'):                            #   Ignore empty lines and comments
                    if ';' in line:                                                      #   Numeration is recognized by semicolon, which
                        self.numeration = [word.strip() for word in line.split(';')]     #   separates the words
                    else:
                        self.dataset.add(line.strip())                                   #   Everything else is a sentence

    def start_logging(self):
        log_file = 'log.txt'
        log_file = open(log_file, 'w')
        log_file.write(f'Numeration: {self.numeration}\n')
        log_file.write(f'Predicted outcome: {self.dataset}\n\n\n')
        return log_file

    def evaluate_experiment(self, output_from_simulation):
        overgeneralization = output_from_simulation - self.dataset
        undergeneralization = self.dataset - output_from_simulation
        total_errors = len(overgeneralization) + len(undergeneralization)
        print(f'Errors {total_errors}')
        if total_errors > 0:
            print(f'Should not generate: {overgeneralization}')
            print(f'Should generate: {undergeneralization}')


ld = LanguageData()                     #   Process the dataset
ld.read_dataset('dataset2.txt')         #   Read the dataset'
sm = SpeakerModel()                     #   Create a speaker model
sm.log_file = ld.start_logging()        #   Begins logging
sm.derive(ld.numeration)                #   Create derivation from the numeration
ld.evaluate_experiment(sm.output_data)  #   Evaluate results
