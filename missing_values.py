# Reproduce functionality of Calculate Missing Values in ShowTell
import struct
import re
from csv import DictReader, DictWriter
from difflib import SequenceMatcher
import argparse
from tkinter import filedialog, Tk
import os.path

"""
    The Dict.bin file is organized into SECTORSIZE byte sectors.  (SECTORSIZE == 32)
     o  The "WordOffset" table above (4096 long integers)
     o  Size of the file (4 bytes)
     o  Size of the rules (4 bytes)
     o  The rules for words not in dictionary
     o  The pronunciations.  Each sector ends with a possible link
        to a continuation sector.  Each possible pronunciation is followed
        by a two-byte occurrence count.
     o  The words in English text pointed to by WordOffset table.  Each sector
        ends with a possible line to a continuation sector.  Each word is
        followed by a link to the possible pronunciations
     o  Expansion area for new words and new pronunciations.  The original
        data always continued to the next successive sector.  Only later expansion
        results in the use of non-contiguous sectors.
"""

# Phonetic Constants from Showandtell
NUMCON = 24   # number of Y line consonants
# excludes glottal stop which cannot occur on Y line
MAXCON = NUMCON + 1  # includes glottal stop

# define constants for CharTypes table
TYPE_RARE = 0o400000
TYPE_OTHER = 0o000000
TYPE_DIACRITIC = 0o100000
TYPE_CONSONANT = 0o200000
TYPE_VOWEL = 0o300000
MASK_KIND = 0o300000
SHIFT_KIND = 15
TEST_PHONEME = 0o200000
TYPE_VOICED = 0o020000
SHIFT_VOICE = 13
TYPE_OBSTRUENT = TYPE_CONSONANT+0o004000
TYPE_SONORANT = TYPE_CONSONANT+0o000000+TYPE_VOICED
MASK_CLASS = 0o004000
SHIFT_CLASS = 11
TYPE_STOP = TYPE_OBSTRUENT+0o000000
TYPE_FRICATIVE = TYPE_OBSTRUENT+0o001000
TYPE_AFFRICATIVE = TYPE_OBSTRUENT+0o002000
TYPE_NASAL = TYPE_SONORANT+0o000000
TYPE_GLIDE = TYPE_SONORANT+0o001000
TYPE_LIQUID = TYPE_SONORANT+0o002000
MASK_MANNER = 0o007000
SHIFT_MANNER = 9
TYPE_EARLY_8 = 0o000100
TYPE_MIDDLE_8 = 0o000200
TYPE_LATE_8 = 0o000300
MASK_STAGE = 0o000300
TYPE_BILABIAL = 0o10000000
TYPE_LABIO_DENTAL = 0o20000000
TYPE_DENTAL = 0o30000000
TYPE_ALVEOLAR = 0o40000000
TYPE_PALATAL = 0o50000000
TYPE_VELAR = 0o60000000
TYPE_GLOTTAL = 0o70000000
MASK_PLACE = 0o70000000
SHIFT_PLACE = 21
TYPE_DIPHTHONG = TYPE_VOWEL+0o040000
TYPE_RHOTACIZED = TYPE_VOWEL+0o020000
TYPE_PHONEMIC_PLACE = 0o20000000
TYPE_PHONEMIC_HEIGHT = 0o10000000
MASK_PHONEMIC = 0o30000000
TYPE_FRONT = 0o002000
TYPE_CENTRAL = 0o004000
TYPE_BACK = 0o006000
MASK_FRONT_BACK = 0o006000
SHIFT_FRONT_BACK = 10
TYPE_HIGH = 0o000400
TYPE_ANY_MID = 0o001000
TYPE_HIGH_MID = 0o001100
TYPE_MID = 0o001200
TYPE_LOW_MID = 0o001300
TYPE_LOW = 0o001400
MASK_HIGH_LOW = 0o001400
SHIFT_HIGH_LOW = 8
MASK_HI_LOW_FINE = 0o001700
MASK_DISPLAY_IX = 0o1700000000
SHIFT_DISPLAY_IX = 24
# types for RE/QRE phoneme groups
TYPE_RE_FRIC = 0o02000000000
TYPE_RE_RHOT = 0o04000000000
TYPE_RE_L = 0o06000000000
TYPE_RE_NG = 0o10000000000
MASK_RE = 0o16000000000
SHIFT_RE = 28

MASK_INDEX = 0o0000077
MARK_STRESS_JUNC = 0o004000
MARK_INTRUSIVE = 0o002000
TYPE_ON_GLIDE = TYPE_DIACRITIC+0o000100+MARK_INTRUSIVE
TYPE_TIE = TYPE_DIACRITIC+0o000200
TYPE_UNSURE = TYPE_DIACRITIC+0o000300
TYPE_LIP = TYPE_DIACRITIC+0o000400
TYPE_NASALITY = TYPE_DIACRITIC+0o000500
TYPE_STRESS = TYPE_DIACRITIC+0o000600+MARK_STRESS_JUNC
TYPE_CONFIG = TYPE_DIACRITIC+0o000700
TYPE_POSITION = TYPE_DIACRITIC+0o001000
TYPE_SOURCE = TYPE_DIACRITIC+0o001100
TYPE_SYLLABIC = TYPE_DIACRITIC+0o001200
TYPE_OFF_GLIDE = TYPE_DIACRITIC+0o001300+MARK_INTRUSIVE
TYPE_STOP_REL = TYPE_DIACRITIC+0o001400
TYPE_TIMING = TYPE_DIACRITIC+0o001500
TYPE_JUNCTURE = TYPE_DIACRITIC+0o001600+MARK_STRESS_JUNC
MASK_DIACRITIC = 0o001700
SHIFT_DIACRITIC = 6
TYPE_SPACE = TYPE_OTHER+0o000100
TYPE_MISSING = TYPE_OTHER+0o000200
TYPE_UNINTEL = TYPE_OTHER+0o000300

CharTypes = (
 0, 0, 0, 0, 0, 0, 0, 0, 0,
 TYPE_OTHER + (8 << SHIFT_DISPLAY_IX),  # tab displayed as Slash
 0, 0, 0, 0, 0, 0,  # most control chars.
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # are not used

 TYPE_SPACE,  # Space
 TYPE_NASALITY + 1,  # !  Nasalized
 TYPE_SOURCE + 35,  # "  Glottalized
 TYPE_NASALITY + 3,  # #  Denasalized

 TYPE_LIP + 4,  # $  Rounded Vowel
 TYPE_LIP + 5,  # %  Unrounded Vowel
 TYPE_LIP + 7,  # &  Nonlabialized Consonant
 TYPE_RARE + TYPE_FRICATIVE + TYPE_DENTAL +
 TYPE_LATE_8 + 17,  # '  /th/

 TYPE_STOP_REL + 9,  # (  Aspirated (h)
 TYPE_STOP_REL + 10,  # )  Unaspirated (=)
 TYPE_LIP + 8,  # *  Inverted
 TYPE_TIE + 43,  # +  Tie

 TYPE_RARE + TYPE_VOWEL + TYPE_LOW_MID + TYPE_BACK + 21 +
 (7 << SHIFT_DISPLAY_IX),  # ,  /D/
 TYPE_MISSING,  # -  Omitted (_)
 TYPE_RARE + TYPE_AFFRICATIVE + TYPE_PALATAL + TYPE_RE_FRIC +
 TYPE_MIDDLE_8 + 14,  # .  /tsh/
 TYPE_RARE + TYPE_VOWEL + TYPE_HIGH_MID + TYPE_CENTRAL + 20 +
 (6 << SHIFT_DISPLAY_IX),  # /  /3/
 # _________________________________________

 TYPE_RARE + TYPE_VOWEL + TYPE_LOW +
 TYPE_CENTRAL + 19,  # 0  /a/
 TYPE_VOWEL + TYPE_MID +
 TYPE_CENTRAL + 7,  # 1  /4/ (schwa)
 TYPE_VOWEL + TYPE_LOW_MID +
 TYPE_FRONT + 12,  # 2  /E/
 TYPE_RARE + TYPE_VOWEL + TYPE_MID +
 TYPE_FRONT + 0,  # 3  /e/

 TYPE_VOWEL + TYPE_LOW_MID +
 TYPE_CENTRAL + 9,  # 4  /^/
 TYPE_RHOTACIZED + TYPE_MID + TYPE_RE_RHOT +
 TYPE_CENTRAL + 18,  # 5  /4^/
 TYPE_RHOTACIZED + TYPE_MID + TYPE_RE_RHOT +
 TYPE_CENTRAL + 17,  # 6  /3^/
 TYPE_RARE + TYPE_VOWEL + TYPE_HIGH_MID +
 TYPE_BACK + 14,  # 7  /U/

 TYPE_VOWEL + TYPE_HIGH_MID +
 TYPE_FRONT + 6,  # 8  /I/
 TYPE_RARE + TYPE_VOWEL + TYPE_LOW_MID +
 TYPE_BACK + 16,  # 9  /c/
 TYPE_SOURCE + 34,  # :  Partially devoiced
 TYPE_FRICATIVE + TYPE_VOICED + TYPE_DENTAL +
 TYPE_LATE_8 + 20,  # /dh/

 TYPE_ON_GLIDE + 44,  # <  Upper left intrusive
 TYPE_RARE + TYPE_STOP +
 TYPE_GLOTTAL + 24,  # =  /?/ (glottal stop)
 TYPE_OFF_GLIDE + 45,  # >  Upper right intrusive
 TYPE_SYLLABIC + 42,  # ?  Syllabic
 # _________________________________________

 TYPE_NASALITY + 2,  # @  Nasal emission
 TYPE_POSITION + 26,  # A  Centralized
 TYPE_STRESS + 17,  # B  Primary stress (1)
 TYPE_SOURCE + 38,  # C  Whistled

 TYPE_POSITION + 28,  # D  Advanced tongue body
 TYPE_JUNCTURE + 12,  # E  Open juncture
 TYPE_POSITION + 29,  # F  Raised tongue body
 TYPE_POSITION + 30,  # G  Lowered tongue body

 TYPE_POSITION + 31,  # H  Fronted (<)
 TYPE_CONFIG + 20,  # I  Dentalized
 TYPE_POSITION + 32,  # J  Backed (>)
 TYPE_CONFIG + 25,  # K  Derhotacized

 TYPE_SOURCE + 33,  # L  Partially voiced
 TYPE_STRESS + 19,  # M  Tertiary stress (3)
 TYPE_STRESS + 18,  # N  Secondary stress (2)
 TYPE_CONFIG + 21,  # O  Palatalized
 # _________________________________________

 TYPE_CONFIG + 22,  # P  Lateralized
 TYPE_TIMING + 40,  # Q  Lengthened (:)
 TYPE_JUNCTURE + 13,  # R  Internal open juncture
 TYPE_POSITION + 27,  # S  Retracted tongue body

 TYPE_JUNCTURE + 14,  # T  Falling terminal junc.
 TYPE_JUNCTURE + 16,  # U  Checked/held juncture
 TYPE_SOURCE + 39,  # V  Weak
 TYPE_TIMING + 41,  # W  Shortened (>)

 TYPE_SOURCE + 37,  # X  Frictionalized
 TYPE_JUNCTURE + 15,  # Y  Rising terminal junc.
 TYPE_SOURCE + 36,  # Z  Breathy
 TYPE_RARE + TYPE_DIPHTHONG + TYPE_BACK + TYPE_PHONEMIC_HEIGHT + TYPE_LOW + 13 +
 (2 << SHIFT_DISPLAY_IX),  # [  /@U/

 TYPE_UNINTEL,  # \  Unintelligible (*)
 TYPE_DIPHTHONG + TYPE_PHONEMIC_PLACE + TYPE_BACK +
 TYPE_PHONEMIC_HEIGHT + TYPE_LOW + 8 +
 (1 << SHIFT_DISPLAY_IX),  # ]  /@I/
 TYPE_LIP + 6,  # ^  Labialized consonant
 TYPE_STOP_REL + 11,  # _  Unreleased
 # _________________________________________

 TYPE_RARE + TYPE_DIPHTHONG + TYPE_MID + TYPE_PHONEMIC_PLACE + TYPE_BACK + 15 +
 (5 << SHIFT_DISPLAY_IX),  # `  /cI/
 TYPE_RARE + TYPE_NASAL + TYPE_VELAR + TYPE_RE_NG +
 TYPE_MIDDLE_8 + 9,  # a  /ng/
 TYPE_STOP + TYPE_VOICED + TYPE_BILABIAL +
 TYPE_EARLY_8 + 1,  # b  /b/
 TYPE_RARE + TYPE_FRICATIVE + TYPE_PALATAL + TYPE_RE_FRIC +
 TYPE_LATE_8 + 16,  # c  /sh/

 TYPE_STOP + TYPE_VOICED + TYPE_ALVEOLAR +
 TYPE_EARLY_8 + 5,  # d  /d/
 TYPE_VOWEL + TYPE_BACK +
 TYPE_LOW + 2,  # e  /@/
 TYPE_FRICATIVE + TYPE_LABIO_DENTAL +
 TYPE_MIDDLE_8 + 12,  # f  /f/
 TYPE_STOP + TYPE_VOICED + TYPE_VELAR +
 TYPE_MIDDLE_8 + 11,  # g  /g/

 TYPE_FRICATIVE + TYPE_GLOTTAL +
 TYPE_EARLY_8 + 7,  # h  /h/
 TYPE_VOWEL + TYPE_HIGH +
 TYPE_FRONT + 1,  # i  /i/
 TYPE_RARE + TYPE_AFFRICATIVE + TYPE_VOICED + TYPE_PALATAL + TYPE_RE_FRIC +
 TYPE_MIDDLE_8 + 15,  # j  /dzh/
 TYPE_STOP + TYPE_VELAR +
 TYPE_MIDDLE_8 + 10,  # k  /k/

 TYPE_LIQUID + TYPE_ALVEOLAR + TYPE_RE_L +
 TYPE_LATE_8 + 21,  # l  /l/
 TYPE_NASAL + TYPE_BILABIAL +
 TYPE_EARLY_8 + 0,  # m  /m/
 TYPE_NASAL + TYPE_ALVEOLAR +
 TYPE_EARLY_8 + 3,  # n  /n/
 TYPE_RARE + TYPE_VOWEL + TYPE_MID +
 TYPE_BACK + 4,  # o  /o/
 # _________________________________________

 TYPE_STOP + TYPE_BILABIAL +
 TYPE_EARLY_8 + 6,  # p  /p/
 TYPE_VOWEL + TYPE_LOW +
 TYPE_FRONT + 10,  # q  /ae/
 TYPE_LIQUID + TYPE_PALATAL + TYPE_RE_RHOT +
 TYPE_LATE_8 + 22,  # r  /r/
 TYPE_FRICATIVE + TYPE_ALVEOLAR + TYPE_RE_FRIC +
 TYPE_LATE_8 + 18,  # s  /s/

 TYPE_STOP + TYPE_ALVEOLAR +
 TYPE_MIDDLE_8 + 8,  # t  /t/
 TYPE_VOWEL + TYPE_HIGH +
 TYPE_BACK + 3,  # u  /u/
 TYPE_RARE + TYPE_FRICATIVE + TYPE_VOICED + TYPE_LABIO_DENTAL +
 TYPE_MIDDLE_8 + 13,  # v  /v/
 TYPE_GLIDE + TYPE_BILABIAL +
 TYPE_EARLY_8 + 4,  # w  /w/

 TYPE_RARE + TYPE_FRICATIVE + TYPE_VOICED + TYPE_PALATAL + TYPE_RE_FRIC +
 TYPE_LATE_8 + 23,  # x  /zh/
 TYPE_RARE + TYPE_GLIDE + TYPE_PALATAL +
 TYPE_EARLY_8 + 2,  # y  /j/
 TYPE_FRICATIVE + TYPE_VOICED + TYPE_ALVEOLAR + TYPE_RE_FRIC +
 TYPE_LATE_8 + 19,  # z  /z/
 TYPE_CONFIG + 23,  # {  Rhotacized

 TYPE_UNSURE,  # |  Box (unsure)
 TYPE_CONFIG + 24,  # }  Velarized
 TYPE_DIPHTHONG + TYPE_MID + TYPE_FRONT + 11 +
 (3 << SHIFT_DISPLAY_IX),  # ~  /eI/
 TYPE_DIPHTHONG + TYPE_MID + TYPE_BACK + 5 +
 (4 << SHIFT_DISPLAY_IX)  # DEL /oU/
)

PhonemeNames = (
 "m",  # 0
 "b",  # 1
 "j",  # 2
 "n",  # 3
 "w",  # 4
 "d",  # 5
 "p",  # 6
 "h",  # 7
 "t",  # 8
 "ng",  # 9
 "k",  # 10
 "g",  # 11
 "f",  # 12
 "v",  # 13
 "tsh",  # 14
 "dzh",  # 15
 "sh",  # 16
 "th",  # 17
 "s",  # 18
 "z",  # 19
 "dh",  # 20
 "l",  # 21
 "r",  # 22
 "zh",  # 23
 "?",  # 24
 "e",  # 0 (vowels)
 "i",  # 1
 "@",  # 2
 "u",  # 3
 "o",  # 4
 "oU",  # 5
 "I",  # 6
 "4",  # 7
 "@I",  # 8
 "^",  # 9
 "ae",  # 10
 "eI",  # 11
 "E",  # 12
 "@U",  # 13
 "U",  # 14
 "cI",  # 15
 "c",  # 16
 "3^",  # 17
 "4^",  # 18
 "a",  # 19
 "3",  # 20
 "D"  # 21
)


def match_count(matcher):
    """return the number of matches in a difflib.SequenceMatcher by summing the third element of matching blocks"""
    return sum([b[2] for b in matcher.get_matching_blocks()])


class PhoneDict:
    def __init__(self, dictionary_dir=None):
        self.SECTORSIZE = 32  # bytes
        self.LONGSIZE = 4
        self.SHORTSIZE = 2
        self.WORDOFFSETSIZE = 4096

        # if dictionary_dir is not specified, default to directory containing this script
        if not dictionary_dir:
            dictionary_dir = os.path.dirname(__file__)
        dictionary_filename = os.path.join(dictionary_dir, 'Dict.bin')
        dictfix_filename = os.path.join(dictionary_dir, 'DictFix.txt')

        with open(dictionary_filename, 'rb') as f:
            self.data = f.read()

        self.dictfix = {}

        with open(dictfix_filename, 'r') as f:
            for line in f:
                key, val = line.strip().split('\t')
                # Convert phonetic representation into internal phonetic codes
                val_ph = ''.join([self.get_ph(p) for p in val.split('-')])
                self.dictfix[key.upper()] = val_ph

        self.wordoffset = struct.unpack('l' * self.WORDOFFSETSIZE, self.data[:self.LONGSIZE * self.WORDOFFSETSIZE])
        self.rules = self.read_and_format_rules()

        self.sector_chars_left = 0
        self.cursor = 0

    def read_and_format_rules(self):
        # Rules are provided as pattern/phoneme pairs.
        rules_size_address = self.LONGSIZE * (self.WORDOFFSETSIZE + 1)
        rules_size = struct.unpack('l', self.data[rules_size_address:rules_size_address + self.LONGSIZE])[0]
        rules_address = rules_size_address + self.LONGSIZE
        # Read rules as bytes
        rules = struct.unpack('{}s'.format(rules_size), self.data[rules_address:rules_address + rules_size])[0]
        # Convert to utf8 and split at null bytes
        rules = rules.decode('utf8').split('\x00')
        # Assemble tuples ( pattern, phonemes )
        rules = list(zip(rules[::2], rules[1::2]))
        return rules

    def get_next_sector_char(self):
        """ The dictionary is organized into sectors.  This function retrieves the next character in the current
        sector, if any.  Follow links to continuation sectors if present. """
        self.sector_chars_left -= 1
        if self.sector_chars_left < 0:  # end of sector
            if self.sector_chars_left < -1:
                return 0
            offset = struct.unpack('l', self.data[self.cursor:self.cursor+self.LONGSIZE])[0]
            if not offset:  # no continuation to next sector
                return 0
            self.cursor = offset  # start of next sector
            self.sector_chars_left = self.SECTORSIZE - self.LONGSIZE - 1
        ch = struct.unpack('s', self.data[self.cursor:self.cursor + 1])[0]
        return ch

    def get_next_sector_short(self):
        # read the next SHORTSIZE bytes of the dictionary, interpreted as a short integer
        return self.get_next_sector_bytes(self.SHORTSIZE, 'h')

    def get_next_sector_long(self):
        # read the next LONGSIZE bytes of the dictionary, interpreted as a long integer
        return self.get_next_sector_bytes(self.LONGSIZE, 'l')

    def get_next_sector_bytes(self, size, unpack_code):
        # read the next SIZE bytes of the dictionary, interpreted as a struct.unpack(unpack_code)
        bytes_ = b''
        for i in range(size):
            b = self.get_next_sector_char()
            bytes_ += b
            self.cursor += 1
        return struct.unpack(unpack_code, bytes_)[0]

    def get_pronunciation(self, address):
        # Get the pronunciation
        pronounce = b''
        self.cursor = address
        self.sector_chars_left = self.SECTORSIZE - self.LONGSIZE
        b = self.get_next_sector_char()
        while b != b'\x00':
            pronounce += b
            self.cursor += 1
            b = self.get_next_sector_char()

        # Get the occurrence count for the pronunciation
        self.cursor += 1  # skip the null byte and read the next thing...
        occurrence_count = self.get_next_sector_short()
        # print(occurrence_count)

        # There could be additional sets of [pronunciation]\x00[occurrence count] but we probably just care about the
        # first one here.
        pronounce = pronounce.decode('utf8')
        return pronounce, occurrence_count

    def get_word_list(self, address):
        # Find the list of words in the dictionary which have a common hash (which points to the same offset)
        words = []

        self.cursor = address
        # Read a word
        self.sector_chars_left = self.SECTORSIZE - self.LONGSIZE
        b = self.get_next_sector_char()
        while b:
            word = b''
            while b != b'\x00':
                word += b
                self.cursor += 1
                b = self.get_next_sector_char()
            if word:
                # skip the null byte
                self.cursor += 1

                # get the offset for this word (link to pronunciation)
                offsetbytes = b''
                for i in range(self.LONGSIZE):
                    b = self.get_next_sector_char()
                    offsetbytes += b
                    self.cursor += 1
                offset = struct.unpack('l', offsetbytes)[0]

                words.append((word.decode('utf8'), offset))

            b = self.get_next_sector_char()
        return words

    def get_pronounce_link(self, address, target):
        """ Search the list of words in the dictionary which have a common hash.
        Stop reading if the target word is found, and return just the offset for the pronunciation
        """
        self.cursor = address
        # Read a word
        self.sector_chars_left = self.SECTORSIZE - self.LONGSIZE
        b = self.get_next_sector_char()
        while b:
            word = b''
            while b != b'\x00':
                word += b
                self.cursor += 1
                b = self.get_next_sector_char()
            if word:
                # skip the null byte
                self.cursor += 1

                # get the offset for this word (link to pronunciation)
                offsetbytes = b''
                for i in range(self.LONGSIZE):
                    b = self.get_next_sector_char()
                    offsetbytes += b
                    self.cursor += 1
                offset = struct.unpack('l', offsetbytes)[0]
                if word.decode('utf8') == target:
                    return offset
            b = self.get_next_sector_char()
        # no match found
        return 0

    def lookup_sentence(self, sentence):
        words = sentence.split(' ')
        pwords = [self.lookup_word(word) for word in words]
        return ' '.join(pwords), ' '.join([self.phonetic_display(p) for p in pwords])

    def lookup_word(self, word):

        word = self.clean_word(word)
        if word in self.dictfix.keys():
            return self.dictfix[word]

        hashvalue = self.get_word_hash(word)
        address = self.wordoffset[hashvalue]
        pronounce_link = self.get_pronounce_link(address, word)
        if pronounce_link:
            pronunciation, occurrence_count = self.get_pronunciation(pronounce_link)
        else:
            # Word not in dictionary.  Use rules to derive a pronunciation
            # print('{} not in dict'.format(word))
            pronunciation = self.apply_rules(word)
        return pronunciation

    def phonetic_display(self, pronunciation):
        """ Convert internal phonetic representation to display representation """
        return '-'.join(self.get_pseudo(p) for p in self.clean_pronunciation(pronunciation))

    def get_pseudo(self, ch):
        """ Based on CExperimentDoc::GetPseudo"""
        if ch == ' ':
            scode = ' '
        else:
            if ch == '\\':  # unintelligible
                scode = '*'
            elif ch == '-':  # omission
                scode = ''
            else:
                isvowel = CharTypes[ord(ch)] & MASK_KIND == TYPE_VOWEL
                ixPhoneme = CharTypes[ord(ch)] & MASK_INDEX
                if isvowel:
                    ixPhoneme += MAXCON
                scode = PhonemeNames[ixPhoneme]
        return scode

    def get_ph(self, ch):
        """ Convert from pseudo to internal phonetic representation

        this is needed for statistics to be correctly computed for words in dictfix, and possibly if rules are used.
        """
        ixPhoneme = PhonemeNames.index(ch)
        if ixPhoneme < MAXCON:
            nKindIndex = TYPE_CONSONANT + ixPhoneme
        else:
            nKindIndex = TYPE_VOWEL + ixPhoneme - MAXCON

        for i in range(32, 129):
            if CharTypes[i] & (MASK_KIND + MASK_INDEX) == nKindIndex:
                return chr(i)

    def apply_rules(self, word):
        """ Use rules to derive pronunciation of word not in dictionary """
        yline = '^' + word + '$'

        # [rule for rule in self.rules if '"' in rule[0]]  #  [('""ED$', '?D$'), ('""', '?')]

        for pattern, repl in self.rules:
            # don't apply ED$ -> D$ to words like SPED
            if pattern == 'ED$' and len(yline) <= 6 and yline[1] not in 'AEIUO':
                continue
            yline = self.wildcard_sub(yline, pattern, repl)

        # now remove ^ and $ around word and insert syllabics
        yword = ''
        PutSyllabicHereIfNeeded = 0
        state = 0  # 0 == looking for syllable start; 1 == looking for vowel; -1 == looking for syllable end
        for ch in yline[1:-1]:
            if ch == '*':
                ch = '\\'  # convert to Y/Z line (what does this mean?)
            elif ch == '$':
                ch = ' '
            yword += ch

            chartype = CharTypes[ord(ch)]
            kind = chartype & MASK_KIND
            assert kind != TYPE_DIACRITIC
            if kind == TYPE_OTHER or (state and (chartype & (MASK_KIND + MASK_MANNER)) == TYPE_STOP):
                # end of syllable found
                if state > 0:  # if no vowel found
                    yword = yword[:PutSyllabicHereIfNeeded] + '?' + yword[PutSyllabicHereIfNeeded:]
                state = 0  # looking for new syllable
            elif kind == TYPE_VOWEL or ch == 's' or ch == 'z':
                state = -1  # found a vowel
            elif not state:
                PutSyllabicHereIfNeeded = len(yword)
                state = 1  # consonant starting syllable
        return yword

    @staticmethod
    def wildcard_sub(word, old, new):
        """ If 'old' is in word, replace with with 'new' using wildcard substitution """
        if old[:2] == '""':
            pattern = r'(.)\1' + re.escape(old[2:])
        else:
            pattern = re.escape(old).replace(r'\?', '(.)')
        repl = new.replace('?', r'\1')
        return re.sub(pattern, repl, word)

    @staticmethod
    def clean_pronunciation(pronunciation):
        """ Exclude diacritics"""
        cleaned = ''.join([p for p in pronunciation if CharTypes[ord(p)] & TEST_PHONEME])
        return cleaned

    @staticmethod
    def clean_word(word):
        # Convert to upper case, and strip out characters except for ' in 'LL
        s = word.upper()
        s = re.sub(r'[^A-Z\']', '', s)  # strip out non-alpha, non-apostrophe characters
        s = re.sub(r"'(?!LL)", '', s)  # strip out apostrophes not followed by LL (as in we'll, she'll, etc.)
        return s

    @staticmethod
    def get_word_hash(word):
        # Compute hash to lookup offset in WordOffset table
        hashvalue = 0
        for c in word:
            hashvalue = hashvalue * 3 + ord(c)
        hashvalue = hashvalue & 0o7777
        return hashvalue


class ShowTellLine:
    """ Compute statistics for a line in ShowAndTell"""

    def __init__(self, sentence, response, phonetic_sentence, phonetic_response):
        self.sentence = sentence
        self.response = response
        self.phonetic_sentence = phonetic_sentence
        self.phonetic_response = phonetic_response

        self.stats = self.compute_statistics()

    def compute_statistics(self):
        """ Compute fundamental statistics from text and phonetic sentence and response"""
        stats = {}

        swords = self.sentence.lower().split(' ')
        rwords = self.response.lower().split(' ')
        rwords = [w for w in rwords if w != '*']

        s_pwords = self.phonetic_sentence.split(' ')  # sentence: phonetic words
        r_pwords = self.phonetic_response.split(' ')  # response: phonetic words
        s_phonemes = ''.join(s_pwords)  # sentence: phonemes (no spaces)
        r_phonemes = ''.join(r_pwords)  # response: phonemes (no spaces)

        # Partition phonetic representations into vowels/consonants
        s_vowels = ''
        s_consonants = ''
        for p in s_phonemes:
            if CharTypes[ord(p)] & MASK_KIND == TYPE_VOWEL:
                s_vowels += p
            else:
                s_consonants += p
        r_vowels = ''
        r_consonants = ''
        for p in r_phonemes:
            if CharTypes[ord(p)] & MASK_KIND == TYPE_VOWEL:
                r_vowels += p
            else:
                r_consonants += p

        s_phonemes_init = ''.join([w[:1] for w in s_pwords])
        r_phonemes_init = ''.join([w[:1] for w in r_pwords])
        s_phonemes_fin = ''.join([w[-1:] for w in s_pwords])
        r_phonemes_fin = ''.join([w[-1:] for w in r_pwords])

        s_graph_init = ''.join([w[:1] for w in swords])
        r_graph_init = ''.join([w[:1] for w in rwords])
        s_graph_fin = ''.join([w[-1:] for w in swords])
        r_graph_fin = ''.join([w[-1:] for w in rwords])

        matcher = SequenceMatcher(autojunk=False)

        # Count of words in stimulus
        stats['SWord'] = len(swords)

        # Count of words intelligible (not *) in response
        stats['RWord'] = len(rwords)

        # Count of matching words (in sequential order)
        matcher.set_seqs(swords, rwords)
        stats['MWord'] = match_count(matcher)

        # Count of matching words regardless of order
        matcher.set_seqs(sorted(swords), sorted(rwords))
        stats['MWordA'] = match_count(matcher)

        # Count of matching phonemes regardless of order
        matcher.set_seqs(sorted(s_phonemes), sorted(r_phonemes))
        stats['MPhoA'] = match_count(matcher)

        # Count of vowels in stimulus
        stats['SVwl'] = len(s_vowels)

        # Count of vowels in response
        stats['RVwl'] = len(r_vowels)

        # Count of matching vowels (in sequential order)
        matcher.set_seqs(s_vowels, r_vowels)
        stats['MVwl'] = match_count(matcher)

        # Count of matching vowels regardless of order
        matcher.set_seqs(sorted(s_vowels), sorted(r_vowels))
        stats['MVwlA'] = match_count(matcher)

        # Count of consonants in stimulus
        stats['SCon'] = len(s_consonants)

        # Count of consonants in response
        stats['RCon'] = len(r_consonants)

        # Count of matching consonants (in sequential order)
        matcher.set_seqs(s_consonants, r_consonants)
        stats['MCon'] = match_count(matcher)

        # Count of matching consonants regardless of order
        matcher.set_seqs(sorted(s_consonants), sorted(r_consonants))
        stats['MConA'] = match_count(matcher)

        # Count of matched phonemes that are initial in stimulus and response (in sequential order)
        matcher.set_seqs(s_phonemes_init, r_phonemes_init)
        stats['MInit'] = match_count(matcher)

        # Count of matched phonemes that are initial in stimulus and response regardless of order
        matcher.set_seqs(sorted(s_phonemes_init), sorted(r_phonemes_init))
        stats['MInitA'] = match_count(matcher)

        # Count of matched phonemes that are final in stimulus and response (in sequential order)
        matcher.set_seqs(s_phonemes_fin, r_phonemes_fin)
        stats['MFin'] = match_count(matcher)

        # Count of matched phonemes that are final in stimulus and response regardless of order
        matcher.set_seqs(sorted(s_phonemes_fin), sorted(r_phonemes_fin))
        stats['MFinA'] = match_count(matcher)

        # Count of matched letters that are initial in stimulus and response (in sequential order, independent match)
        matcher.set_seqs(s_graph_init, r_graph_init)
        stats['MIgraph'] = match_count(matcher)

        # Count of matched letters that are initial in stimulus and response regardless of order
        matcher.set_seqs(sorted(s_graph_init), sorted(r_graph_init))
        stats['MIgraphA'] = match_count(matcher)

        # Count of matched letters that are final in stimulus and response (in sequential order, independent match)
        matcher.set_seqs(s_graph_fin, r_graph_fin)
        stats['MFgraph'] = match_count(matcher)

        # Count of matched letters that are final in stimulus and response regardless of order
        matcher.set_seqs(sorted(s_graph_fin), sorted(r_graph_fin))
        stats['MFgraphA'] = match_count(matcher)

        # Count of * for unintelligible in response
        stats['RUnintel'] = self.response.count('*')

        # Count of matching word breaks using match done for all characters in sequential order
        stats['MBreak'] = len([p for p in zip(self.sentence, self.response) if p == (' ', ' ')])

        # Second-order fields
        stats['%MWord'] = 200 * stats['MWord'] / (stats['SWord'] + stats['RWord'])
        stats['%CWord'] = 100 * stats['MWord'] / stats['SWord']
        stats['%MWordA'] = 200 * stats['MWordA'] / (stats['SWord'] + stats['RWord'])
        stats['%CWordA'] = 100 * stats['MWordA'] / stats['SWord']

        stats['SPho'] = stats['SCon'] + stats['SVwl']
        stats['RPho'] = stats['RCon'] + stats['RVwl']
        stats['MPho'] = stats['MCon'] + stats['MVwl']
        stats['%MPho'] = 200 * stats['MPho'] / (stats['SPho'] + stats['RPho'])
        stats['%CPho'] = 100 * stats['MPho'] / stats['SPho']
        stats['%MPhoA'] = 200 * stats['MPhoA'] / (stats['SPho'] + stats['RPho'])
        stats['%CPhoA'] = 100 * stats['MPhoA'] / stats['SPho']

        stats['%MVwl'] = 200 * stats['MVwl'] / (stats['SVwl'] + stats['RVwl'])
        stats['%CVwl'] = 100 * stats['MVwl'] / stats['SVwl']
        stats['%MVwlA'] = 200 * stats['MVwlA'] / (stats['SVwl'] + stats['RVwl'])
        stats['%CVwlA'] = 100 * stats['MVwlA'] / stats['SVwl']

        stats['%MCon'] = 200 * stats['MCon'] / (stats['SCon'] + stats['RCon'])
        stats['%CCon'] = 100 * stats['MCon'] / stats['SCon']
        stats['%MConA'] = 200 * stats['MConA'] / (stats['SCon'] + stats['RCon'])
        stats['%CConA'] = 100 * stats['MConA'] / stats['SCon']

        stats['%MInit'] = 200 * stats['MInit'] / (stats['SWord'] + stats['RWord'])
        stats['%CInit'] = 100 * stats['MInit'] / stats['SWord']
        stats['%MInitA'] = 200 * stats['MInitA'] / (stats['SWord'] + stats['RWord'])
        stats['%CInitA'] = 100 * stats['MInitA'] / stats['SWord']

        stats['%MFin'] = 200 * stats['MFin'] / (stats['SWord'] + stats['RWord'])
        stats['%CFin'] = 100 * stats['MFin'] / stats['SWord']
        stats['%MFinA'] = 200 * stats['MFinA'] / (stats['SWord'] + stats['RWord'])
        stats['%CFinA'] = 100 * stats['MFinA'] / stats['SWord']

        stats['%MIgraph'] = 200 * stats['MIgraph'] / (stats['SWord'] + stats['RWord'])
        stats['%CIgraph'] = 100 * stats['MIgraph'] / stats['SWord']
        stats['%MIgraphA'] = 200 * stats['MIgraphA'] / (stats['SWord'] + stats['RWord'])
        stats['%CIgraphA'] = 100 * stats['MIgraphA'] / stats['SWord']

        stats['%MFgraph'] = 200 * stats['MFgraph'] / (stats['SWord'] + stats['RWord'])
        stats['%CFgraph'] = 100 * stats['MFgraph'] / stats['SWord']
        stats['%MFgraphA'] = 200 * stats['MFgraphA'] / (stats['SWord'] + stats['RWord'])
        stats['%CFgraphA'] = 100 * stats['MFgraphA'] / stats['SWord']

        stats['SBreak'] = stats['SWord'] - 1
        stats['RBreak'] = stats['RWord'] + stats['RUnintel'] - 1
        stats['RAWord'] = stats['RWord'] + stats['RUnintel']
        try:
            stats['%MBreak'] = 200 * stats['MBreak'] / (stats['SBreak'] + stats['RBreak'])
        except ZeroDivisionError:
            stats['%MBreak'] = 0

        return stats

    def statistics(self):
        fields = [
            'SWord', 'RWord', 'MWord', 'MWordA', '%MWord', '%CWord', '%MWordA', '%CWordA',
            'SPho', 'RPho', 'MPho', 'MPhoA', '%MPho', '%CPho', '%MPhoA', '%CPhoA',
            'SVwl', 'RVwl', 'MVwl', 'MVwlA', '%MVwl', '%CVwl', '%MVwlA', '%CVwlA',
            'SCon', 'RCon', 'MCon', 'MConA', '%MCon', '%CCon', '%MConA', '%CConA',
            'MInit', '%MInit', '%CInit', 'MInitA', '%MInitA', '%CInitA', 'MFin', '%MFin', '%CFin',
            'MFinA', '%MFinA', '%CFinA', 'MIgraph', '%MIgraph', '%CIgraph', 'MIgraphA', '%MIgraphA', '%CIgraphA',
            'MFgraph', '%MFgraph', '%CFgraph', 'MFgraphA', '%MFgraphA', '%CFgraphA',
            'RUnintel', 'RAWord', 'MBreak', '%MBreak'
        ]

        statistics = {}
        for f in fields:
            if '%' in f:
                statistics[f] = '{:.1f}'.format(self.stats.get(f, 0))
            else:
                statistics[f] = '{}'.format(self.stats.get(f, 0))
        return statistics


def process_line(phonedict, line):
    line['psentence'], line['Phonetic Sentence'] = phonedict.lookup_sentence(line['Sentence'])
    line['presponse'], line['Phonetic Response'] = phonedict.lookup_sentence(line['Response'])

    stline = ShowTellLine(sentence=line['Sentence'],
                          response=line['Response'],
                          phonetic_sentence=line['psentence'],  # use internal phonetic representation
                          phonetic_response=line['presponse'])

    line.update(stline.statistics())
    return line


def test_compute_missing_values():

    class Args:
        directory = ''
        files = ['test_data/original.txt']
        output_dir = 'test_output'
        dictionary_dir = ''

    compute_missing_values(Args())


def compute_missing_values(args):

    if args.directory:
        # get a list of candidate files
        pattern = re.compile('(Control[_ ]File|Research_Responses|Parent_Responses|Training_Responses)-')
        filenames = [os.path.join(args.directory, f) for f in os.listdir(args.directory) if pattern.search(f)]
    elif args.files:
        filenames = args.files
    else:
        # Neither input files nor a directory were supplied.  Prompt for filenames
        filenames = filedialog.askopenfilenames(title='Select files for missing values computation')
    if not filenames:
        print('No input found')
        return

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = filedialog.askdirectory(title='Select directory for processed output')
    if not os.path.isdir(output_dir):
        print('Output directory not found: {}'.format(output_dir))
        return

    # Load phonetic dictionary
    pd = PhoneDict(dictionary_dir=args.dictionary_dir)

    # Extra fields for missing values
    extra_fields = [
        # 'Line', 'Date', 'Listener',
        # 'Age', 'Sex', 'Race', 'Ethnicity'  # These four columns may or may not be present
        # 'Directory', 'Control', 'File', 'Repeat', 'Shows', 'Sentence', 'Response',
        'Time', 'Confident', 'Understood', 'Phonetic Sentence', 'Phonetic Response',
        'SWord', 'RWord', 'MWord', 'MWordA', '%MWord', '%CWord', '%MWordA', '%CWordA',
        'SPho', 'RPho', 'MPho', 'MPhoA', '%MPho', '%CPho', '%MPhoA', '%CPhoA',
        'SVwl', 'RVwl', 'MVwl', 'MVwlA', '%MVwl', '%CVwl', '%MVwlA', '%CVwlA',
        'SCon', 'RCon', 'MCon', 'MConA', '%MCon', '%CCon', '%MConA', '%CConA',
        'MInit', '%MInit', '%CInit', 'MInitA', '%MInitA', '%CInitA', 'MFin', '%MFin', '%CFin',
        'MFinA', '%MFinA', '%CFinA', 'MIgraph', '%MIgraph', '%CIgraph', 'MIgraphA', '%MIgraphA', '%CIgraphA',
        'MFgraph', '%MFgraph', '%CFgraph', 'MFgraphA', '%MFgraphA', '%CFgraphA',
        'RUnintel', 'RAWord', 'MBreak', '%MBreak']

    for input_filename in filenames:
        output_filename = os.path.join(output_dir, os.path.basename(input_filename))

        with open(input_filename) as csvfile, open(output_filename, 'w') as output_file:
            reader = DictReader(csvfile, dialect='excel-tab')
            writer = DictWriter(output_file, fieldnames=reader.fieldnames + extra_fields, dialect='excel-tab',
                                lineterminator='\n', extrasaction='ignore')
            writer.writeheader()
            for line in reader:
                newline = process_line(pd, line)
                writer.writerow(newline)


if __name__ == '__main__':
    # Process command line arguments
    parser = argparse.ArgumentParser(
        description='Add missing values to showtell listener response files.')

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "-d", "--dir",
        dest='directory',
        action='store',
        default="",
        help='Directory containing input files.')

    input_group.add_argument(
        "-i", "--input",
        dest='files',
        action='store',
        nargs='+',
        default="",
        help='Files to process.  Defaults to asking user if no directory or file is provided.')

    parser.add_argument(
        "-o", "--output",
        dest='output_dir',
        action='store',
        default="",
        help='Directory for output files. Defaults to asking user if no directory is provided.')

    parser.add_argument(
        "--dict",
        dest='dictionary_dir',
        action='store',
        default='',
        help='Directory containing Dict.bin and DictFix.txt phonetic dictionary files.  Defaults to script directory.'
    )

    args = parser.parse_args()

    # Suppress tkinter root window
    root = Tk()
    root.withdraw()

    compute_missing_values(args)
