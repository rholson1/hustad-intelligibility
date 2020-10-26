#!/usr/bin/env python

"""intelligibility: average ShowAndTell Listener scores, grouping by utterance length

Written for Python 3.2

Created Feb 21, 2012 by Robert H. Olson, Ph.D., rolson@waisman.wisc.edu
Changes:
Feb 23, 2012: Loop over unique values of SWord rather than max:min:-1
Apr 19, 2013: Add option to aggregate over sentences rather than word counts
"""
import sys
import os.path
import argparse
import re
import numpy as np
import tkinter
from tkinter.constants import *
import tkinter.filedialog as fd
from tkinter.messagebox import askyesno, showinfo
import difflib
from itertools import groupby, chain
from collections import defaultdict


def float_or_default(s, default=0):
    """ Try to apply the float function to a variable.  If conversion fails with a ValueError, return the default """
    try:
        value = float(s)
    except ValueError:
        value = default
    return value


class Word:
    """ Store the number of occurrences and correct responses to a word, grouped by utterance length
    """
    def __init__(self, word, phone):
        self.word = word
        self.phone = phone
        self.utterance = [0, 0, 0, 0, 0, 0, 0]
        self.correct = [0, 0, 0, 0, 0, 0, 0]

    def update(self, utterance_length, correct):
        try:
            self.utterance[utterance_length - 1] += 1
            if correct:
                self.correct[utterance_length - 1] += 1
        except IndexError:
            while len(self.utterance) < utterance_length:
                self.utterance.append(0)
                self.correct.append(0)
            self.update(utterance_length, correct)  # try again


def sequence_match(seq1, seq2):
    """
    Compare two sequences, returning a boolean list with elements corresponding to elements of seq1, True if there
    is a corresponding element in seq2.
    """
    diff = difflib.SequenceMatcher(None, seq1, seq2)
    correct = []
    cursor = 0
    for block in diff.get_matching_blocks():
        while cursor < block[0]:
            correct.append(False)
            cursor += 1
        while cursor < block[0] + block[2]:
            correct.append(True)
            cursor += 1
    return correct


def parse_intel_control_filename(fname):
    """ Intelligibility control filenames are formatted as:
    [initials]v[visit#] Control File-[datetime].txt
    """
    path, basename = os.path.split(fname)
    try:
        # base_prefix = re.match('(.+)[_ ]Control[_ ]File-', basename).group(1)
        match = re.match('(.+)[_ ](Control[_ ]File|Research_Responses|Parent_Responses)-(.*)\.txt', basename)
        base_prefix = match.group(1)
        listener_type = match.group(2)
        cf_number = match.group(3)
    except:
        raise Exception('Unexpected format for control file name: {}'.format(basename))
    return {'prefix': os.path.join(path, base_prefix), 'filename': fname, 'cf_number': cf_number, 'visit': base_prefix,
            'listener_type': listener_type}


def file_to_key(fname):
    """ Convert a filename to a key based on visit and control file number
    Works both for listener response files and for question files (- and !)
    """
    path, basename = os.path.split(fname)
    try:
        # base_prefix = re.match('(.+)[_ ]Control[_ ]File-', basename).group(1)
        match = re.match('(.+)[_ ](Control[_ ]File|Research_Responses|Parent_Responses)[-!](.*)\.txt', basename)
        base_prefix = match.group(1)
        cf_number = match.group(3)
    except:
        raise Exception('Unexpected format for file name: {}'.format(basename))
    return '{}-{}'.format(base_prefix, cf_number)


def group_filenames(fnames):
    """ Given a list of filenames, create subgroups with matching initials and visit number.
    Only include filenames containing the substring 'Control File-' (that is, intelligibility control files).
    """

    pattern = re.compile('(Control[_ ]File-|Research_Responses-|Parent_Responses-)')
    f_list = [parse_intel_control_filename(f) for f in fnames if pattern.search(f)]
    prefixes = set([f['prefix'] for f in f_list])
    grouped_list = [{'prefix': p,
                     'filenames': [f['filename'] for f in f_list if f['prefix'] == p]} for p in prefixes]
    return grouped_list


def readfiles(filenames):
    """Generator iterates through lines of supplied files successively, returning the line together with the name of
     the source file"""
    for f in filenames:
        for line in open(f):
            yield line, f


def merge_iwpm_files(path):
    """ Given a directory path, concatenate IWPM files (sentence, word, or both).  If present, combine with WPM files.
    """

    def clean_wpm_key(k):
        # remove special characters from the key used to match WPM data with IWPM data.
        remove_chars = {ord(s): None for s in '()[]_-'}
        an_key = k.translate(remove_chars)
        # remove trailing letters which follow a number
        key = re.sub('(?<=[0-9])[a-z]$', '', an_key)
        # convert to upper case
        key = key.upper()
        return key

    # output files
    fname_sentence = os.path.join(path, 'combined_sentence_IWPM.txt')
    fname_word = os.path.join(path, 'combined_word_IWPM.txt')
    fname_all = os.path.join(path, 'combined_all_stimuli.txt')

    all_files = os.listdir(path)
    sentence_files = [os.path.join(path, f) for f in all_files if 'sentenceIWPMsummary.txt' in f]
    word_files = [os.path.join(path, f) for f in all_files if 'wordIWPMsummary.txt' in f]

    pattern = re.compile('[^i]wpm', flags=re.IGNORECASE)  # look for filenames containing WPM but not IWPM (case insensitive)
    wpm_files = [os.path.join(path, f) for f in all_files if pattern.search(f)]

    # Create a WPM dictionary where the key is TOCS_numb (the second column of the WPM file
    wpm_dict = {clean_wpm_key(L.split('\t')[1]): L.strip().split('\t') for L, f in readfiles(wpm_files) if '\t' in L}

    # Also add in syllables per second if files are present
    pattern = re.compile('syllsPerSecond', flags=re.IGNORECASE)  # look for filenames containing syllsPerSecond
    sylls_files = [os.path.join(path, f) for f in all_files if pattern.search(f)]
    # Create a sylls dictionary where the key is TOCS_numb (the second column of the sylls file
    sylls_dict = {clean_wpm_key(L.split('\t')[1]): L.strip().split('\t') for L, f in readfiles(sylls_files) if '\t' in L}

    # Also add in articulation rate files, if present
    pattern = re.compile('articRate', flags=re.IGNORECASE)  # look for filenames containing articRate
    artic_files = [os.path.join(path, f) for f in all_files if pattern.search(f)]
    artic_dict = {clean_wpm_key(L.split('\t')[1]): L.strip().split('\t') for L, f in readfiles(artic_files) if '\t' in L}

    # WPM file has 5 cols: Sentence, TOCS_numb, UttDur, TotalWords, WPM
    # Syll file has 7 cols: Sentence, TOCS_numb, UttDur, TotalWords, WPM, TotalSylls, SyllsPerSecond
    # Artic file has 7 cols: Sentence, TOCS_numb, UttDur, TotalWords, WPM, Pauses>0.15s, CumPauseDur
    # Note Syll file is a superset of the WPM file!
    header = ['Child', '# Listeners', 'Avg SWord', 'Sentence', 'Time Sum', 'MWordA Sum', '%CWordA', 'MWordA Avg', 'SentenceFile']
    header += ['Sentence', 'TOCS_numb', 'UttDur', 'TotalWords', 'WPM']
    if sylls_files:
        header += ['TotalSylls', 'SyllsPerSecond']
    if artic_files:
        header += ['Pauses>0.15s', 'CumPauseDur']

    with open(fname_sentence, 'w') as of_sentence, open(fname_word, 'w') as of_word, open(fname_all, 'w') as of_all:
        of_all.write('\t'.join(header) + '\n')
        of_sentence.write('\t'.join(header) + '\n')
        of_word.write('\t'.join(header) + '\n')

        for line, f in readfiles(word_files):
            of_word.write(line)
            of_all.write(line)

        for line, f in readfiles(sentence_files):
            lineparts = line.split('\t')
            if lineparts[8].strip() == 'SentenceFile':
                key = 'TOCSnumb'
            else:
                key = os.path.splitext(os.path.basename(lineparts[8]))[0]  # Converts AGv07TOCSS\AGv07sT01.wav to AGv07sT01
                key = clean_wpm_key(key)

            first_part = wpm_dict.get(key, '') or sylls_dict.get(key, '') or artic_dict.get(key, '')
            if first_part:
                new_line_parts = first_part[:5]  # first 5 cols
                if sylls_files:
                    new_line_parts += sylls_dict.get(key, '')[5:7] or ['', '']
                if artic_files:
                    new_line_parts += artic_dict.get(key, '')[5:7] or ['', '']
                new_line = '\t'.join([line.strip()] + new_line_parts) + '\n'
            else:
                new_line = line.strip() + '\n'

            # if key in sylls_dict:
            #     new_line = '\t'.join([line.strip(), sylls_dict.get(key, '')]) + '\n'
            # elif key in wpm_dict:
            #     new_line = '\t'.join([line.strip(), wpm_dict.get(key, '')]) + '\n'
            # else:
            #     new_line = line.strip() + '\n'
            of_sentence.write(new_line)
            of_all.write(new_line)


def splitvisit(childvisit):
    """ Given a child-visit combination (e.g. AG(c)v01) return the child identifier and visit number"""

    try:
        child, visit = childvisit.split('v', 1)
    except ValueError:
        child = childvisit
        visit = '1'
        #raise Exception('Child/Visit identifier does not contain a "v": {}'.format(childvisit))

    # strip out parentheses, if present
    child = child.translate({ord(s): None for s in '()'})

    return [child, visit]


def combined_iwpm_to_intelligibility(path, ask=True, exclude=True):
    """ Process the combined_*_IWPM.txt files located in the supplied path"""

    def process_group(group):
        sum_words = 0.
        sum_duration = 0.
        sum_mwordaavg = 0.
        sum_sylls = 0.
        sum_cumpausedur = 0.
        for line in group:
            if line[0] == 'Child':  # Do not process header line
                continue
            if line[sentence_col].upper() == 'A DRESS':
                continue
            sum_words += float(line[sword_col])
            sum_mwordaavg += float(line[mwordaavg_col])
            # Utterance duration only applies to sentences
            if float(line[sword_col]) > 1:
                try:
                    sum_duration += float(line[uttdur_col])
                except (IndexError, TypeError):
                    pass
                try:
                    sum_sylls += float(line[totsylls_col])
                except (IndexError, TypeError):
                    pass
                try:
                    sum_cumpausedur += float(line[cumpause_col])
                except (IndexError, TypeError):
                    pass

        wpm = sum_words / sum_duration * 60 if sum_duration > 0 else 'nan'
        iwpm = sum_mwordaavg / sum_duration * 60 if sum_duration > 0 else 'nan'
        intelligibility = sum_mwordaavg / sum_words * 100 if sum_words > 0 else 'nan'
        sylls_per_second = sum_sylls / sum_duration if sum_duration > 0 else 'nan'
        artic_rate = sum_sylls / (sum_duration - sum_cumpausedur) if sum_duration > 0 else 'nan'

        return {'sum_words': sum_words,
                'sum_duration': sum_duration,
                'sum_mwordaavg': sum_mwordaavg,
                'sum_sylls': sum_sylls,
                'wpm': wpm,
                'iwpm': iwpm,
                'intelligibility': intelligibility,
                'sylls_per_second': sylls_per_second,
                'artic_rate': artic_rate}

    if ask:
        exclude_small_sentence_sets = askyesno(title='Exclusion Rules', message='If there are four or fewer sentences of a given length, exclude from calculations?')
    else:
        exclude_small_sentence_sets = exclude

    for ftype in (('sentence', 'IWPM'), ('word', 'IWPM'), ('all', 'stimuli')):
        filename_in = os.path.join(path, 'combined_{}_{}.txt'.format(ftype[0], ftype[1]))
        filename_out = os.path.join(path, 'combined_{}_intelligibility.txt'.format(ftype[0]))
        filename_validation = os.path.join(path, 'combined_{}_intelligibility_validation.txt'.format(ftype[0]))

        # Read combined IWPM file into array
        header = []
        data = []
        firstline = True
        for line in open(filename_in):
            lineparts = line.split('\t')
            if firstline:
                header = lineparts[:1] + ['ChildID', 'VisitID'] + lineparts[1:]
                firstline = False
            else:
                data.append(lineparts[:1] + splitvisit(lineparts[0]) + lineparts[1:])

        # Sort data by child / visit / Avg SWORD
        child_col = header.index('ChildID')
        visit_col = header.index('VisitID')
        sword_col = header.index('Avg SWord')
        mwordaavg_col = header.index('MWordA Avg')
        sentence_col = header.index('Sentence')

        # UttDur is only present if there is a wpm file (i.e., not for words)
        uttdur_col = header.index('UttDur') if 'UttDur' in header else None
        totwords_col = header.index('TotalWords') if 'TotalWords' in header else None
        totsylls_col = header.index('TotalSylls') if 'TotalSylls' in header else None
        cumpause_col = header.index('CumPauseDur') if 'CumPauseDur' in header else None

        cvs_key = lambda s: '-'.join([s[child_col], s[visit_col], s[sword_col]])
        cv_key = lambda s: '-'.join([s[child_col], s[visit_col]])

        data.sort(key=cvs_key)

        if exclude_small_sentence_sets:
            filtered_data = [list(g) for k, g in groupby(data, cvs_key)]
            filtered_data = [g for g in filtered_data if len(g) > 4]
            data = list(chain.from_iterable(filtered_data))

        data.sort(key=cvs_key)

        # Data validation
        with open(filename_validation, 'w') as f_valid:
            f_valid.write('\t'.join(['Message'] + header))  # write header for reference

            # is there a way to validate naming convention?

            # Check: Avg Sword == TotalWords
            for line in data:
                if len(line) > totwords_col:  # If the TotalWords column exists...
                    if float_or_default(line[sword_col]) != float_or_default(line[totwords_col]):
                        f_valid.write('\t'.join(['Avg Sword != TotalWords'] + line))

            # Check: sentence has wpm data (use presence of totwords to check)
            for line in data:
                if line[0] != 'Child':
                    if float_or_default(line[sword_col]) > 1 and line[sentence_col].upper() != 'A DRESS':  # i.e., a sentence
                        if len(line) < totwords_col:  # missing wpm data
                            f_valid.write('\t'.join(['Missing wpm data'] + line))

            # Warn if fewer than four sentences of a given length
            for k, g in groupby(data, cvs_key):
                if len(list(g)) <= 4:
                    kp = k.split('-')
                    f_valid.write('\t'.join(['Four or fewer sentences of length {}'.format(kp[2]), 'v'.join(kp[:2]), kp[0], kp[1]]) + '\n')

            # Warn if sentence has more than 7 words
            for line in data:
                if line[0] != 'Child':
                    if float_or_default(line[sword_col]) > 7:
                        f_valid.write('\t'.join(['More than 7 words in sentence'] + line))

        with open(filename_out, 'w') as f_out:
            f_out.write('\t'.join([
                'Child',
                'Visit',
                'Avg SWord',
                'Sum Words',
                'Sum Duration',
                'Sum MWordA Avg',
                'WPM',
                'IWPM',
                'Intelligibility',
                'SyllsPerSecond',
                'ArticRate'
            ]) + '\n')

            # Process data by group (Child/Visit/Avg SWORD)
            for k, g in groupby(data, cvs_key):
                group_calcs = process_group(g)
                f_out.write('\t'.join(
                    k.split('-') +
                    [str(group_calcs[s]) for s in
                    ('sum_words', 'sum_duration', 'sum_mwordaavg', 'wpm', 'iwpm', 'intelligibility', 'sylls_per_second', 'artic_rate')])
                            + '\n')

            # Process data by group (Child/Visit)
            for k, g in groupby(data, cv_key):
                group_calcs = process_group([line for line in g if float_or_default(line[sword_col]) > 1])
                f_out.write('\t'.join(
                    k.split('-') + ['s2-s7'] +
                    [str(group_calcs[s]) for s in
                     ('sum_words', 'sum_duration', 'sum_mwordaavg', 'wpm', 'iwpm', 'intelligibility', 'sylls_per_second', 'artic_rate')])
                            + '\n')

            # Process data by group (Child/Visit)
            for k, g in groupby(data, cv_key):
                group_calcs = process_group(g)
                f_out.write('\t'.join(
                    k.split('-') + ['s1-s7'] +
                    [str(group_calcs[s]) for s in
                     ('sum_words', 'sum_duration', 'sum_mwordaavg', 'wpm', 'iwpm', 'intelligibility', 'sylls_per_second', 'artic_rate')])
                            + '\n')


def is_sentence_file(filename):
    """ Check if a file is a sentence file or a word file by looking for presence of 's' or 'w' in filename """
    match = re.search('(w|s\d+)t\d+[a-z]?(_.......)?\.wav', os.path.basename(filename).lower())
    if match:
        return match.group(1)[0] == 's'  # True for a sentence file
    else:
        raise Exception('Stimulus filename has unexpected format: {}'.format(filename))


def main(directory="", exclude="ask"):
    # Suppress tkinter root window
    root = tkinter.Tk()
    root.withdraw()

    # Resolve directory to search. Use the ask user if blank directory used argument
    if directory == "":
        working_dir = fd.askdirectory(
            title='Select the directory containing listener files to be processed')
        if not working_dir:
            print("Selection canceled...")
            return
    else:
        working_dir = os.path.abspath(directory)

    print("Directory is {}".format(working_dir))

    if not os.path.isdir(working_dir):
        print("Directory not found: {}".format(working_dir))
        return

    missing_values = []  # list of prefixes of files that probably need to have missing values computed

    # Older version prompted for files. We search a directory instead.
    # fnames = askopenfilenames(title='Select Listener File',
    #                    filetypes=[('Text Files','.txt'),
    #                                ('All Files','.*')])
    # fnames = root.tk.splitlist(fnames)

    pattern = re.compile('(Control[_ ]File-|Research_Responses-|Parent_Responses-)')
    fnames = [os.path.join(working_dir, f) for f in os.listdir(working_dir) if pattern.search(f)]

    pl_file = initialize_perceptual_learning(working_dir)
    articulation = read_articulation(working_dir)

    # Ask user about creation of overall IWPM files

    # Regroup filenames so that files with matching initials and visit number are concatenated and processed together.
    file_sets = group_filenames(fnames)

    for file_set in file_sets:
        prefix = file_set['prefix']  # Path and file prefix used for output files
        # Store data in parts:
        ShortListener = []  # just the listener initials
        Listener = []  # listener augmented by control file number
        Repeat = []
        Data = []
        SWord = []
        Sentence = []
        PSentence = []  # phonetic sentences
        SentenceFile = []  # sentence file names
        SentenceFileSentence = {}
        Correct = []  # is response completely correct?
        SourceFile = []
        Response = []  # typed response (not phonetic)

        word_dict = {}

        rfname = prefix + '_reliability.txt'
        print(os.path.basename(prefix), flush=True)

        # Files may have an extra 4 columns with demographic information
        offset = 0

        with open(rfname, 'w') as rf:
            #with open(fname, 'r') as f:
                for line, sourcefile in readfiles(file_set['filenames']):
                    fileparts = parse_intel_control_filename(sourcefile)
                    cf_number = fileparts['cf_number']
                    lineparts = line.split('\t')
                    if len(lineparts) > 1:
                        if lineparts[0].isnumeric():
                            if len(lineparts) < 16 + offset:
                                # Probably forgot to compute missing values
                                missing_values.append(os.path.basename(prefix))
                                continue

                            # Any SENTENCE where speaker said only ONE word (SWord == 1) should be ignored
                            try:
                                if is_sentence_file(lineparts[5+offset]) and int(lineparts[15+offset]) == 1:
                                    continue
                            except:
                                print('Line: {}'.format(line))
                                continue

                            # This is a data line
                            ShortListener.append(lineparts[2])
                            Listener.append('{}-{}'.format(lineparts[2], cf_number))
                            Repeat.append(lineparts[6+offset])
                            Data.append([lineparts[10+offset]] + lineparts[15+offset:])
                            SWord.append(lineparts[15+offset])
                            Sentence.append(lineparts[8+offset].capitalize())
                            PSentence.append(lineparts[13+offset])
                            # SentenceFile[Sentence[-1]] = lineparts[5+offset]  # CHANGE
                            SentenceFile.append(os.path.basename(lineparts[5+offset]))
                            SentenceFileSentence[SentenceFile[-1]] = Sentence[-1]

                            # store data for word-level analysis
                            sentence_parts = lineparts[8+offset].split()
                            sentence_length = len(sentence_parts)
                            psentence_parts = lineparts[13+offset].split()
                            response_parts = lineparts[14+offset].split()

                            # If the phonetic representation has different number of words, use text instead.
                            if len(sentence_parts) != len(psentence_parts):
                                psentence_parts = lineparts[8 + offset].lower().split()
                                response_parts = lineparts[9 + offset].lower().split()

                            correct = sequence_match(psentence_parts, response_parts)  # Score response
                            Correct.append(all(correct))

                            SourceFile.append(sourcefile)
                            Response.append(lineparts[9 + offset])

                            for idx, word in enumerate(sentence_parts):
                                pword = psentence_parts[idx]  # phonetic word
                                cap_word = word.capitalize()
                                if cap_word not in word_dict:
                                    word_dict[cap_word] = Word(cap_word, pword)
                                word_dict[cap_word].update(sentence_length, correct[idx])

                        else:
                            # This is a header line
                            has_demographic_info = lineparts[3] == 'Age'
                            offset = 4 if has_demographic_info else 0
                            header = ['Child', 'Listener', '# of Sentences'] + [lineparts[10 + offset]] + lineparts[15+offset:]
                            #if aggregation_mode == 'Sentence':
                            #    header.insert(3, 'Phonetic Sentence')
                            #    header.insert(3, 'Sentence')
                        if lineparts[6+offset] != '0':
                            # Write lines where Repeat != 0 to the reliability file
                            rf.write(line)
                    else:
                        # Throw source filenames into reliability file, too
                        rf.write(line)

        Listener = np.array(Listener)
        Listeners = np.unique(Listener)
        ShortListener = np.array(ShortListener)
        ShortListeners = np.unique(ShortListener)
        if len(Listeners) > len(ShortListeners):
            print('Warning: Duplicate listener initials. {}'.format(' '.join(Listeners)))

        Sentence = np.array(Sentence)
        Sentences = np.unique(Sentence)
        SentenceFile = np.array(SentenceFile)
        SentenceFiles = np.unique(SentenceFile)
        Correct = np.array(Correct)
        SourceFile = np.array(SourceFile)
        Response = np.array(Response)

        PSentence = np.array(PSentence)
        Repeat = np.fromiter(Repeat,'int')
        Data = np.array([[float(z) for z in x] for x in Data])
        SWord = np.fromiter(SWord,'int')
        WorkingSet = [r in [0, 1] for r in Repeat]

        basename = os.path.basename(prefix)
        ofname_wc = prefix + '_intellxlistenerxlength.txt'
        ofname_sentence = prefix + '_intellxutterance.txt'
        sfname_wc = prefix + '_intellxlistener_summmary.txt'
        sfname_sentence = prefix + '_intellxutterance_summary.txt'

        sfname_sentence_sentence = prefix + '_intellxutterance_sentenceIWPMsummary.txt'
        sfname_sentence_word = prefix + '_intellxutterance_wordIWPMsummary.txt'

        word_fname = prefix + '_intellxword.txt'

        # Generate perceptual learning output file
        perceptual_learning(pl_file, header[3:], Sentence, SentenceFile, Correct, Repeat, Data, SourceFile, Response, articulation)

        # Insert a column for %CWordA_SD after %CWordA
        CWordA_SD_col = header.index('%CWordA') + 1
        header.insert(CWordA_SD_col, '%CWordA_SD')

        #if aggregation_mode == 'Word Count':
        with open(sfname_wc, 'w') as sf:
            with open(ofname_wc, 'w') as f:
                # Write header line
                f.write('\t'.join(header))
                sf.write('\t'.join([
                    'Child',
                    '# Listeners',
                    '# Sentences',
                    'Avg Sent Length',
                    '\t'.join(header[3:])
                ]))

                # Loop over word counts/utterance
                for wc in np.unique(SWord)[::-1]:
                    WordSet = SWord == wc

                    # Loop over Listeners
                    blockdata = []
                    exemplars_total = 0.0
                    for L in Listeners:
                        exemplars = np.logical_and( np.logical_and( WordSet, WorkingSet ), Listener == L )
                        if np.any(exemplars):
                            bd = np.mean(Data[exemplars, :], 0)
                            # Insert %CWordA_SD data into bd
                            sd_data = np.std(Data[exemplars, CWordA_SD_col - 1])
                            bd = np.insert(bd, CWordA_SD_col - 3, sd_data)

                            blockdata.append(bd)
                            # Write data to file
                            f.write('\t'.join([basename, L, str(exemplars.sum())]))
                            f.write('\t')
                            f.write('\t'.join(str(s) for s in bd))
                            f.write('\n')
                            exemplars_total += exemplars.sum()
                    # Write average across listeners
                    #bd = np.mean(np.array(blockdata), 0)
                    blockdata = np.array(blockdata)
                    bd = [np.mean(blockdata[:, idx]) if '%' in h else np.sum(blockdata[:, idx])
                          for idx, h in enumerate(header[3:])]

                    summary_line = [
                        basename,
                        str(len(Listeners)),
                        str(exemplars_total / len(Listeners)),  # Avg number of sentences
                        str(bd[1] / len(Listeners)),  # Mean SWord
                        '\t'.join(str(s) for s in bd),
                        '\n']
                    f.write('\t'.join(summary_line[:3] + summary_line[4:]))  # Exclude Avg Sentence Length (col 3)
                    f.write('\n')

                    # write average line to summary file as well
                    sf.write('\t'.join(summary_line))

        #elif aggregation_mode == 'Sentence':
        with open(sfname_sentence_sentence, 'w') as ssf:
            with open(sfname_sentence_word, 'w') as swf:
                with open(sfname_sentence, 'w') as sf:
                    with open(ofname_sentence, 'w') as f:
                        # Write header line
                        header.insert(3, 'Phonetic Sentence')
                        header.insert(3, 'Sentence')
                        f.write('\t'.join(header))
                        summary_header_list = ['Child', '# Listeners', 'Avg SWord'] + \
                                              header[3:5] + \
                                              [h + ' Sum' if '%' not in h else h for h in header[5:]]
                        sf.write('\t'.join(summary_header_list))
                        partition_idx = [summary_header_list.index(h) for h in ('Child', '# Listeners', 'Avg SWord', 'Sentence', 'Time Sum', 'MWordA Sum', '%CWordA')]
                        partition_header = '\t'.join([summary_header_list[idx] for idx in partition_idx] + ['MWordA Avg', 'SentenceFile']) + '\n'
                        ssf.write(partition_header)
                        swf.write(partition_header)
                        # Loop over sentences

                        # sort sentences by word count first
                        Sentences = Sentences[np.argsort([s.count(' ') for s in Sentences], kind='mergesort')]

                        # sort sentencefiles by word count first
                        SentenceFiles = SentenceFiles[np.argsort([SentenceFileSentence[s].count(' ') for s in SentenceFiles], kind='mergesort')]

                        for sentencefile in SentenceFiles:
                            SentenceSet = SentenceFile == sentencefile
                            s = SentenceFileSentence[sentencefile]
                        #for s in Sentences:
                        #    SentenceSet = Sentence == s

                            # Loop over listeners
                            blockdata = []
                            exemplars_total = 0.0
                            for L in Listeners:
                                exemplars = np.logical_and( np.logical_and( SentenceSet, WorkingSet), Listener == L )
                                if np.any(exemplars):
                                    bd = Data[exemplars, :].flatten('F')  # flatten by column
                                    bd = np.insert(bd, CWordA_SD_col - 3, np.nan)
                                    blockdata.append(bd)
                                    # Write data to file
                                    f.write('\t'.join([basename, L, str(exemplars.sum()), s, PSentence[exemplars][0]]))
                                    f.write('\t')
                                    f.write('\t'.join(str(s) if str(s) != 'nan' else '' for s in bd))
                                    f.write('\n')
                                    exemplars_total += exemplars.sum()
                            # Write average across listeners
                            #bd = np.mean(np.array(blockdata), 0)
                            blockdata = np.array(blockdata)
                            try:
                                bd = [np.mean(blockdata[:, idx]) if '%' in h else np.sum(blockdata[:, idx])
                                      for idx, h in enumerate(header[5:])]
                            except IndexError:
                                print(Listeners)
                                # print(blockdata)
                                print('Length of blockdata elements: {}'.format([len(zz) for zz in blockdata]))
                                # print(list(enumerate(header[5:])))

                                print('Problem computing statistics for sentence "{}" in {}'.format(s, os.path.basename(prefix)))

                                #raise Exception('Problem computing statistics for sentence "{}" in {}'.format(s, os.path.basename(prefix)))
                            if blockdata.ndim == 1:
                                print('Skipping "{}"'.format(s))
                                continue

                            bd[CWordA_SD_col - 3] = np.std(blockdata[:, CWordA_SD_col - 4])

                            summary_line = [
                                basename,
                                str(len(Listeners)),
                                str(exemplars_total / len(Listeners)),
                                s,
                                PSentence[SentenceSet][0],
                                '\t'.join(str(s) for s in bd),
                                '\n'
                            ]

                            f.write('\t'.join(summary_line))
                            f.write('\n')
                            # write average line to summary file as well
                            summary_average_line = '\t'.join(summary_line[:2] + [str(float(bd[1]) / len(Listeners))] + summary_line[3:])

                            if float(bd[1]) / len(Listeners) > 10:
                                print('Suspicious value for mean SWORD')
                                print('bd[1]: {}'.format(bd[1]))
                                print(', '.join([str(s) for s in blockdata[:, 1]]))
                                print('len(Listeners): {}'.format(len(Listeners)))
                                print(summary_average_line)

                            summary_average_list = summary_average_line.split('\t')
                            sf.write(summary_average_line)
                            summary_partition_list = [summary_average_list[idx] for idx in partition_idx]
                            # append MWordA Avg
                            summary_partition_list.append(str(float(summary_partition_list[5]) / float(summary_partition_list[1])))
                            # summary_partition_list.append(SentenceFile[s])
                            summary_partition_list.append(sentencefile)
                            if (float(bd[1]) / len(Listeners) < 2) or s.lower() == 'a dress':
                                swf.write('\t'.join(summary_partition_list) + '\n')
                            else:
                                ssf.write('\t'.join(summary_partition_list) + '\n')

        # Write out word-level analysis

        # Determine max line length
        max_length = 7
        for k in word_dict:
            max_length = max(max_length, len(word_dict[k].utterance))

        with open(word_fname, 'w') as f:
            # write header line
            f.write('\t'.join(['Child', '# Listeners', 'Word',
                               '\t'.join(['UL {}'.format(i + 1) for i in range(max_length)]),
                               'Total Utterances',
                               '\t'.join(['Correct {}'.format(i + 1) for i in range(max_length)]),
                               'Total Correct',
                               '% Correct']))
            f.write('\n')
            for word in sorted(word_dict.keys()):
                utterance_total = sum(word_dict[word].utterance)
                correct_total = sum(word_dict[word].correct)

                linelist = [
                    basename,
                    str(len(Listeners)),
                    word,
                    '\t'.join([str(s) for s in word_dict[word].utterance])]
                if max_length > len(word_dict[word].utterance):
                    linelist.append('\t'.join(['0' for i in range(max_length - len(word_dict[word].utterance))]))
                linelist += [str(utterance_total), '\t'.join([str(s) for s in word_dict[word].correct])]
                if max_length > len(word_dict[word].correct):
                    linelist.append('\t'.join(['0' for i in range(max_length - len(word_dict[word].correct))]))
                linelist += [str(correct_total),
                             str(100. * correct_total / utterance_total)]

                f.write('\t'.join(linelist))
                f.write('\n')

    if fnames:
        path = os.path.dirname(fnames[0])
        merge_iwpm_files(path)
        ask_arg = exclude == "ask"
        exclude_arg = exclude == "yes"
        combined_iwpm_to_intelligibility(path, ask_arg, exclude_arg)

    if missing_values:
        showinfo('Check input', 'Compute missing values for: {}'.format(', '.join(set(missing_values))))


def initialize_perceptual_learning(working_dir):
    """Initialize the output file used for the combined perceptual learning file (across visits/listeners)"""
    fname = os.path.join(working_dir, 'combined_perceptual_listening.txt')
    with open(fname, 'w') as f:
        f.write('TPC = Total Phonemes Correct; TVC = Total Vowels Correct; TCC = Total Consonants Correct; Reliability N = Nth presentation of reliability samples\n')
        f.write('\t'.join(['Visit', 'Child chronological age'] + ['L{} {}'.format(L, s) for L in (1, 2, 3) for s in
                          ('First task', 'CF Number', 'Listener Type', 'WTOCS Count',
                           'WTOCS 1/3', 'WTOCS 2/3', 'WTOCS 3/3', 'TPC 1/3', 'TPC 2/3', 'TPC 3/3',
                           'WTOCS overall intell', 'WTOCS reliability 1 intell', 'WTOCS reliability 2 intell',
                           'WTOCS reliability 1 TPC', 'WTOCS reliability 2 TPC', 'WTOCS avg precision rating',  # precision == articulation
                           'WTOCS TPC', 'WTOCS TVC', 'WTOCS TCC',
                           'STOCS overall intell', 'STOCS reliability 1 intell', 'STOCS reliability 2 intell',
                           'STOCS reliability 1 TPC', 'STOCS reliability 2 TPC',
                           'STOCS TPC', 'STOCS TVC', 'STOCS TCC'
                           )]
                          ))

    return fname


def ca_from_visit(visit):
    """Given a visit name, return a chronological age in months.  For now, only works on TD kids who have the age at
    visit encoded in the visit name as YYMM_F_XXXv01"""
    try:
        match = re.match('\d{4}', visit).group()
    except AttributeError:
        return 0
    y = int(match[:2])
    m = int(match[2:])
    return 12 * y + m


def constant_factory(value):
    return lambda: value


def perceptual_learning(pl_filename, header, sentence, filenames, correct, repeat, data, sourcefile, response, articulation):
    """Write perceptual learning files
    One file per listener, plus one summary file that contains WTOCS intelligibility in thirds and TPC in thirds
    TPC == total phonemes correct, calculated as sum(MPhoA)/sum(SPho) over the set of interest
    """

    # Exclude 'a dress' and 'dee' from consideration
    word_filter = np.array([s.lower() not in ('a dress', 'dee') for s in sentence])  # always, for wtocs
    first_utterance = np.isin(repeat, (0, 1))  # most of the time, except for comparison of repeats

    all_stocs = np.array([is_sentence_file(f) for f in filenames])
    all_wtocs = np.logical_not(all_stocs)

    col = {s: header.index(s) for s in ('SVwl', 'MVwl', 'MVwlA', 'SCon', 'MCon', 'MConA', 'MPhoA', 'SPho')}

    last_prefix = ''
    # group output by listener (aka by source filename)
    sources = np.unique(sourcefile)
    for source in sources:
        listener_set = sourcefile == source
        listener_all_wtocs = np.logical_and(listener_set, all_wtocs)  # select the wtocs rows in art
        listener_all_stocs = np.logical_and(listener_set, all_stocs)

        # single_word_set = first utterances of wtocs by this listener, excluding 'a dress', 'dee'
        repeat_set = np.logical_and(listener_all_wtocs, word_filter)  # repeat in (0, 1, 2)
        single_word_set = np.logical_and(repeat_set, first_utterance)  # repeat in (0, 1)

        single_sentence_set = np.logical_and(listener_all_stocs, first_utterance)  # repeat in (0, 1)
        sentence_repeat_set = {}
        sentence_reliability = {}
        for i in (1, 2):
            sentence_repeat_set[i] = np.logical_and(listener_all_stocs, repeat == i)
            sentence_reliability[i] = {}
            for s in ('MPhoA', 'SPho'):
                sentence_reliability[i][s] = data[sentence_repeat_set[i], col[s]]

        # get articulation for wtocs (all wtocs for this source)
        art = articulation.get(file_to_key(source), None)
        art_stim = [os.path.basename(s) for s in art['stimulus']] if art and art['stimulus'] else None
        art = np.array([float(a) for a in art['articulation']]) if art and art['articulation'] else None

        # Want to exclude articulation values from training items (in articulation but not in listener response files)
        lrf_filenames = [os.path.basename(s) for s in filenames[listener_all_wtocs]]
        art_stim_filter = np.array([a in lrf_filenames for a in art_stim]) if art_stim else None
        art = art[art_stim_filter] if art_stim_filter is not None else None

        art_filter = single_word_set[listener_all_wtocs]
        art_value = art[art_filter] if art is not None else None

        parts = parse_intel_control_filename(source)
        listener_filename = parts['prefix'] + '_perceptual_learning-' + parts['cf_number'] + '.txt'

        wtocs_first = all_wtocs[listener_set][0]
        wtocs_count = sum(single_word_set)
        stocs_count = sum(single_sentence_set)

        # extract variables needed to compute TPC (total phonemes correct)
        MPhoA = data[single_word_set, col['MPhoA']]
        MPhoAr = data[repeat_set, col['MPhoA']]
        SPho = data[single_word_set, col['SPho']]
        SPhor = data[repeat_set, col['SPho']]
        D = {s: data[single_word_set, col[s]] for s in ('SVwl', 'MVwl', 'MVwlA', 'SCon', 'MCon', 'MConA')}
        sws_correct = correct[single_word_set]

        s_data = data[single_sentence_set, :]
        sss_correct = correct[single_sentence_set]

        # compute intelligibility of wtocs in thirds
        limits = [int(np.ceil(wtocs_count * x / 3.)) for x in range(4)]
        chunk_size = [len(sws_correct[limits[i]:limits[i + 1]]) for i in range(3)]
        chunk_correct = [sum(sws_correct[limits[i]:limits[i + 1]]) for i in range(3)]
        chunk_mphoa = [sum(MPhoA[limits[i]:limits[i + 1]]) for i in range(3)]
        chunk_spho = [sum(SPho[limits[i]:limits[i + 1]]) for i in range(3)]

        # Accuracy of repeated items with separation
        repeat_accuracy = {}
        repeat_agg = {1: defaultdict(int), 2: defaultdict(int)}
        for i, (word, c, r) in enumerate(zip(sentence[repeat_set],
                                             correct[repeat_set],
                                             repeat[repeat_set])):
            if r == 1:  # first repeat
                repeat_accuracy[word] = {'word': word, 'correct 1': c, 'pos 1': i}
            elif r == 2:  # second repeat
                repeat_accuracy[word].update({'correct 2': c, 'pos 2': i})
            if r in (1, 2):
                repeat_agg[r]['correct'] += c
                repeat_agg[r]['count'] += 1
                repeat_agg[r]['mphoa'] += MPhoAr[i]
                repeat_agg[r]['spho'] += SPhor[i]

        # Accuracy of repeated items with separation: stocs
        s_MPhoAr = data[listener_all_stocs, col['MPhoA']]
        s_SPhor = data[listener_all_stocs, col['SPho']]

        s_repeat_accuracy = {}
        s_repeat_agg = {1: defaultdict(int), 2: defaultdict(int)}
        for i, (word, c, r) in enumerate(zip(sentence[listener_all_stocs],
                                             correct[listener_all_stocs],
                                             repeat[listener_all_stocs])):
            if r == 1:  # first repeat
                s_repeat_accuracy[word] = {'word': word, 'correct 1': c, 'pos 1': i}
            elif r == 2:  # second repeat
                s_repeat_accuracy[word].update({'correct 2': c, 'pos 2': i})
            if r in (1, 2):
                s_repeat_agg[r]['correct'] += c
                s_repeat_agg[r]['count'] += 1
                s_repeat_agg[r]['mphoa'] += s_MPhoAr[i]
                s_repeat_agg[r]['spho'] += s_SPhor[i]

        # write to listener file and combined file
        with open(listener_filename, 'w') as lf, open(pl_filename, 'a') as combined_f:
            lf.write('{}tocs first\n'.format('w' if wtocs_first else 's'))
            lf.write('wtocs count\t{}\n'.format(wtocs_count))

            for i in range(3):
                lf.write('wtocs {}/3\t{:.2f}\n'.format(i+1, chunk_correct[i]/chunk_size[i]))

            for i in range(3):
                lf.write('TPC {}/3\t{:.2f}\n'.format(i + 1, chunk_mphoa[i] / chunk_spho[i]))

            if parts['prefix'] != last_prefix:
                combined_f.write('\n{}'.format(parts['visit']))
                # try to write chronological age from the visit id
                combined_f.write('\t{}'.format(ca_from_visit(parts['visit'])))

                last_prefix = parts['prefix']

            combined_f.write('\t' + '\t'.join(['{}tocs'.format('w' if wtocs_first else 's'), parts['cf_number'], parts['listener_type'], str(wtocs_count)] +
                                       ['{:.2f}'.format(chunk_correct[i] / chunk_size[i]) for i in range(3)] +
                                       ['{:.2f}'.format(chunk_mphoa[i] / chunk_spho[i]) for i in range(3)] +
                                       ['{:.2f}'.format(sum(chunk_correct) / sum(chunk_size))] +  # WTOCS overall intell
                                       ['{:.2f}'.format(repeat_agg[r]['correct'] / repeat_agg[r]['count']) for r in (1, 2)] +  # Reliability intell
                                       ['{:.2f}'.format(repeat_agg[r]['mphoa'] / repeat_agg[r]['spho']) for r in (1, 2)] +  # Reliability TPC
                                       ['{:.2f}'.format(np.mean(art_value) if art_value is not None else 0)] + # articulation (aka precision)
                                       ['{:.2f}'.format(sum(chunk_mphoa) / sum(chunk_spho))] +  # WTOCS TPC
                                       ['{:.2f}'.format(sum(D['MVwlA']) / sum(D['SVwl'])),  # WTOCS TVC
                                        '{:.2f}'.format(sum(D['MConA']) / sum(D['SCon']))] +  # WTOCS TCC
                                       ['{:.2f}'.format(v) for v in [
                                           sum(sss_correct) / stocs_count,  # STOCS intelligibility
                                           sum(correct[sentence_repeat_set[1]]) / sum(sentence_repeat_set[1]) if sum(sentence_repeat_set[1]) else -1,  # STOCS Reliability 1 Intell
                                           sum(correct[sentence_repeat_set[2]]) / sum(sentence_repeat_set[2]) if sum(sentence_repeat_set[2]) else -1,  # STOCS Reliability 2 Intell
                                           sum(sentence_reliability[1]['MPhoA']) / sum(sentence_reliability[1]['SPho']) if sum(sentence_reliability[1]['SPho']) else -1,  # STOCS Reliability 1 TPC
                                           sum(sentence_reliability[2]['MPhoA']) / sum(sentence_reliability[2]['SPho']) if sum(sentence_reliability[2]['SPho']) else -1,  # STOCS Reliability 2 TPC
                                           sum(s_data[:, col['MPhoA']]) / sum(s_data[:, col['SPho']]) if sum(s_data[:, col['SPho']]) else -1,  # STOCS TPC
                                           sum(s_data[:, col['MVwlA']]) / sum(s_data[:, col['SVwl']]) if sum(s_data[:, col['SVwl']]) else -1,  # STOCS TVC
                                           sum(s_data[:, col['MConA']]) / sum(s_data[:, col['SCon']]) if sum(s_data[:, col['SCon']]) else -1  # STOCS TCC
                                       ]]
                                       ))

            # WTOCS repeat accuracy with separation
            lf.write('\t'.join(['Word', 'Correct 1', 'Correct 2', 'Separation']) + '\n')
            for k, v in repeat_accuracy.items():
                lf.write('\t'.join([str(s) for s in [v['word'], v['correct 1'], v.get('correct 2', 'missing'), v.get('pos 2', 0) - v['pos 1']]]) + '\n')

            lf.write('\t'.join(['Repeat Total', 'Repeat 1', 'Repeat 2']) + '\n')
            lf.write('\t'.join(
                ['Intelligibility'] + ['{:.2f}'.format(repeat_agg[r]['correct'] / repeat_agg[r]['count']) for r in (1, 2)]) + '\n')
            lf.write('\t'.join(
                ['TPC'] + ['{:.2f}'.format(repeat_agg[r]['mphoa'] / repeat_agg[r]['spho']) for r in (1, 2)]) + '\n')

            # STOCS repeat accuracy with separation
            lf.write('\t'.join(['Sentence', 'Correct 1', 'Correct 2', 'Separation']) + '\n')
            for k, v in s_repeat_accuracy.items():
                lf.write('\t'.join([str(s) for s in [v['word'], v['correct 1'], v.get('correct 2', 'missing'), v.get('pos 2', 0) - v['pos 1']]]) + '\n')

            lf.write('\t'.join(['Repeat Total', 'Repeat 1', 'Repeat 2']) + '\n')
            lf.write('\t'.join(
                ['Intelligibility'] + ['{:.2f}'.format(s_repeat_agg[r]['correct'] / s_repeat_agg[r]['count'] if s_repeat_agg[r]['count'] else -1) for r in (1, 2)]) + '\n')
            lf.write('\t'.join(
                ['TPC'] + ['{:.2f}'.format(s_repeat_agg[r]['mphoa'] / s_repeat_agg[r]['spho'] if s_repeat_agg[r]['spho'] else -1) for r in (1, 2)]) + '\n')

            # Single word accuracy (vowels, consonants), articulation (from ! file)
            lf.write('\t'.join(['Word', 'Response', 'Correct', 'TPC', 'SVwl', 'MVwl', 'MVwlA', 'SCon', 'MCon', 'MConA', 'Articulation']) + '\n')

            for i in range(wtocs_count):
                lf.write('\t'.join([
                    str(s) for s in [
                        sentence[single_word_set][i],
                        response[single_word_set][i],
                        sws_correct[i],
                        '{:.2f}'.format(MPhoA[i] / SPho[i])
                    ]
                ] + ['{:.2f}'.format(D[s][i]) for s in ('SVwl', 'MVwl', 'MVwlA', 'SCon', 'MCon', 'MConA')] +
                                   ['{:.2f}'.format(art_value[i]) if art_value is not None else '']) + '\n')

            # Single sentence accuracy (vowels, consonants)
            lf.write('\t'.join(['Word', 'Response', 'Correct', 'TPC', 'SVwl', 'MVwl', 'MVwlA', 'SCon', 'MCon', 'MConA']) + '\n')

            for i in range(stocs_count):
                lf.write('\t'.join([
                    str(s) for s in [
                        sentence[single_sentence_set][i],
                        response[single_sentence_set][i],
                        sss_correct[i],
                        '{:.2f}'.format(s_data[i, col['MPhoA']] / s_data[i, col['SPho']])
                    ]
                ] + ['{:.2f}'.format(s_data[i, col[s]]) for s in ('SVwl', 'MVwl', 'MVwlA', 'SCon', 'MCon', 'MConA')]) + '\n')


def read_articulation(working_dir):
    """Read articulation from the question response files in a supplied directory"""
    pattern = re.compile('(Control[_ ]File|Research_Responses|Parent_Responses)!')
    fnames = [os.path.join(working_dir, f) for f in os.listdir(working_dir) if pattern.search(f)]
    print(fnames)

    art = {}
    for fname in fnames:
        articulation = []
        stimulus = []
        for line in open(fname, 'r'):
            lineparts = line.split('\t')
            try:
                if lineparts[1] == 'articulation':
                    articulation.append(lineparts[2])
                    stimulus.append(lineparts[3].strip())
            except IndexError:
                continue

        art[file_to_key(fname)] = {'articulation': articulation, 'stimulus': stimulus}
    return art


if __name__ == '__main__':
    # Process command line arguments
    parser = argparse.ArgumentParser(
        description='Compute intelligibility results in a directory.')

    parser.add_argument(
        "-d", "--dir",
        dest='directory',
        action='store',
        default="",
        help='Directory to check. Defaults to asking user if no directory is provided.')

    parser.add_argument(
        "-e", "--exclude",
        dest='exclude',
        action='store',
        default="ask",
        choices=['yes', 'no', 'ask'],
        help='Whether to exclude sentences when there 4 or fewer ones of a given length. Defaults to asking the user.')

    parser.add_argument(
        "-c", "--confirm",
        dest='confirm',
        action='store',
        default="yes",
        choices=['yes', 'no'],
        help='Whether to require a keypress to end program. Defaults to yes (input is required).')

    args = parser.parse_args()
    d = args.directory
    exclude = args.exclude
    try:
        main(d, exclude)
    except:
        import sys
        print(sys.exc_info()[0])
        import traceback
        print(traceback.format_exc())
    finally:
        print("Press Enter to continue ...")
        if args.confirm == "yes":
            input()

else:
    print('intelligibility loaded as a module')

