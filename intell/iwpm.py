import ntpath
import re
import io
import numpy as np
from itertools import chain, groupby
from collections import defaultdict

from .utils import readfiles, float_or_default
from .file import File

from tkinter.messagebox import askyesno


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


def merge_iwpm_files(sentence_files, word_files, wpm_files, sylls_files, artic_files):
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



    # all_files = os.listdir(path)
    # sentence_files = [os.path.join(path, f) for f in all_files if 'sentenceIWPMsummary.txt' in f]
    # word_files = [os.path.join(path, f) for f in all_files if 'wordIWPMsummary.txt' in f]

    # pattern = re.compile('[^i]wpm', flags=re.IGNORECASE)  # look for filenames containing WPM but not IWPM (case insensitive)
    # wpm_files = [os.path.join(path, f) for f in all_files if pattern.search(f)]

    # Create a WPM dictionary where the key is TOCS_numb (the second column of the WPM file
    wpm_dict = {clean_wpm_key(L.split('\t')[1]): L.strip().split('\t') for L, f in readfiles(wpm_files) if '\t' in L}

    # Also add in syllables per second if files are present
    # pattern = re.compile('syllsPerSecond', flags=re.IGNORECASE)  # look for filenames containing syllsPerSecond
    # sylls_files = [os.path.join(path, f) for f in all_files if pattern.search(f)]
    # Create a sylls dictionary where the key is TOCS_numb (the second column of the sylls file
    sylls_dict = {clean_wpm_key(L.split('\t')[1]): L.strip().split('\t') for L, f in readfiles(sylls_files) if '\t' in L}

    # Also add in articulation rate files, if present
    # pattern = re.compile('articRate', flags=re.IGNORECASE)  # look for filenames containing articRate
    # artic_files = [os.path.join(path, f) for f in all_files if pattern.search(f)]
    artic_dict = {clean_wpm_key(L.split('\t')[1]): L.strip().split('\t') for L, f in readfiles(artic_files) if '\t' in L}

    # WPM file has 5 cols: Sentence, TOCS_numb, UttDur, TotalWords, WPM
    # Syll file has 7 cols: Sentence, TOCS_numb, UttDur, TotalWords, WPM, TotalSylls, SyllsPerSecond
    # Artic file has 7 cols: Sentence, TOCS_numb, UttDur, TotalWords, WPM, Pauses>0.15s, CumPauseDur
    # Note Syll file is a superset of the WPM file!
    # header = ['Child', '# Listeners', 'Avg SWord', 'Sentence', 'Time Sum', 'MWordA Sum', '%CWordA', 'MWordA Avg', 'SentenceFile']
    # header += ['Sentence', 'TOCS_numb', 'UttDur', 'TotalWords', 'WPM']
    header_suffix = ['Sentence', 'TOCS_numb', 'UttDur', 'TotalWords', 'WPM']
    if sylls_files:
        header_suffix += ['TotalSylls', 'SyllsPerSecond']
    if artic_files:
        header_suffix += ['Pauses>0.15s', 'CumPauseDur']


    of_sentence = io.StringIO()
    of_word = io.StringIO()
    of_all = io.StringIO()

    first = True
    for line, f in readfiles(word_files):
        if first:
            line = line.strip() + '\t' + '\t'.join(header_suffix) + '\n'
            first = False
        of_word.write(line)
        of_all.write(line)

    first = True
    for line, f in readfiles(sentence_files):
        lineparts = line.rstrip().split('\t')
        if first:
            # Assumes the first line is a header line
            line = line.strip() + '\t' + '\t'.join(header_suffix) + '\n'
            sfcol = lineparts.index('SentenceFile')
            first = False

        if lineparts[sfcol] == 'SentenceFile':
            key = 'TOCSnumb'
        else:
            key = ntpath.splitext(ntpath.basename(lineparts[sfcol]))[0]  # Converts AGv07TOCSS\AGv07sT01.wav to AGv07sT01
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
            new_line = line.rstrip() + '\n'

        of_sentence.write(new_line)
        of_all.write(new_line)

    of_word.seek(0)
    of_sentence.seek(0)
    of_all.seek(0)

    return [
        File(of_word, name='combined_word_IWPM.txt'),
        File(of_sentence, name='combined_sentence_IWPM.txt'),
        File(of_all, name='combined_all_stimuli.txt')
    ]



def combined_iwpm_to_intelligibility(merged_files, ask=True, exclude=True):
    """ Process the combined_*_IWPM.txt files located in the supplied path"""

    def process_group(group):
        sum_words = 0.
        sum_duration = 0.
        sum_mwordaavg = 0.
        sum_sylls = 0.
        sum_cumpausedur = 0.
        for line in group:
            if line[0] in ('Child', 'Phase'):  # Do not process header line
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

    output_files = []

    if ask:
        exclude_small_sentence_sets = askyesno(title='Exclusion Rules', message='If there are four or fewer sentences of a given length, exclude from calculations?')
    else:
        exclude_small_sentence_sets = exclude

    for ftype in (('sentence', 'IWPM'), ('word', 'IWPM'), ('all', 'stimuli')):

        file_in = [f for f in merged_files if f.name == 'combined_{}_{}.txt'.format(ftype[0], ftype[1])][0]
        f_out = io.StringIO()
        f_valid = io.StringIO()

        #filename_in = os.path.join(path, 'combined_{}_{}.txt'.format(ftype[0], ftype[1]))
        #filename_out = os.path.join(path, 'combined_{}_intelligibility.txt'.format(ftype[0]))
        #filename_validation = os.path.join(path, 'combined_{}_intelligibility_validation.txt'.format(ftype[0]))

        # Read combined IWPM file into array
        # Insert 2 new columns for ChildID and VisitID
        header = []
        phaseheader = []
        data = []
        for line in file_in:
            lineparts = line.rstrip().split('\t')
            if lineparts[:1] == ['Child'] and header == []:
                header = lineparts[:1] + ['ChildID', 'VisitID'] + lineparts[1:]
            elif lineparts[:1] == ['Phase']:
                phaseheader = lineparts[:1] + ['', ''] + lineparts[1:]
            else:
                data.append(lineparts[:1] + splitvisit(lineparts[0]) + lineparts[1:])

        # Create index vectors for partitioning data by phase
        header = np.array(header)
        postfix = np.argmax(header == 'SentenceFile')  # find the end of the phase data

        phasecols = defaultdict(list)
        if phaseheader:
            phaseheader = np.array(phaseheader)
            phases = sorted([p for p in set(phaseheader[3:postfix]) if p])
            for phase in phases:
                phasecols[phase] = np.nonzero(phaseheader == phase)
        else:
            phases = ['single']
            phasecols['single'] = list(range(3, postfix))  # include all "phase" columns

        prefix = [0, 1, 2]
        pdata = defaultdict(list)

        # Partition by phase
        for line in data:
            line = np.array(line)
            for P in phases:
                pdata[P].append(np.hstack((
                    line[prefix],
                    line[phasecols[P]],
                    line[postfix:],
                    np.zeros(5)  # pad with zeros to avoid index errors
                )))

        group_calcs = defaultdict(dict)
        group_calcs27 = defaultdict(dict)
        group_calcs17 = defaultdict(dict)

        for P in phases:
            pheader = np.hstack((header[prefix], header[phasecols[P]], header[postfix:]))

            # get index of important columns from the header

            # Sort data by child / visit / Avg SWORD
            child_col = np.argmax(pheader == 'ChildID')
            visit_col = np.argmax(pheader == 'VisitID')
            sword_col = np.argmax(pheader == 'Avg SWord')
            mwordaavg_col = np.argmax(pheader == 'MWordA Avg')
            sentence_col = np.argmax(pheader == 'Sentence')

            # UttDur is only present if there is a wpm file (i.e., not for words)
            uttdur_col = np.argmax(pheader == 'UttDur') or None
            totwords_col = np.argmax(pheader == 'TotalWords') or None
            totsylls_col = np.argmax(pheader == 'TotalSylls') or None
            cumpause_col = np.argmax(pheader == 'CumPauseDur') or None

            cvs_key = lambda s: '-'.join([s[child_col], s[visit_col], s[sword_col]])
            cv_key = lambda s: '-'.join([s[child_col], s[visit_col]])

            pdata[P].sort(key=cvs_key)

            if exclude_small_sentence_sets:
                filtered_data = [list(g) for k, g in groupby(pdata[P], cvs_key)]
                filtered_data = [g for g in filtered_data if len(g) > 4]
                pdata[P] = list(chain.from_iterable(filtered_data))

            pdata[P].sort(key=cvs_key)

            # Data validation
            #-----------------------------------------
            f_valid.write(f'Phase: {P}\n')
            f_valid.write('\t'.join(np.hstack((['Message'], pheader))) + '\n')  # write header for reference

            # is there a way to validate naming convention?

            # Check: Avg Sword == TotalWords
            for line in pdata[P]:
                if len(line) > totwords_col:  # If the TotalWords column exists...
                    if float_or_default(line[sword_col]) != float_or_default(line[totwords_col]):
                        f_valid.write('\t'.join(np.hstack((['Avg Sword != TotalWords'], line))) + '\n')

            # Check: sentence has wpm data (use presence of totwords to check)
            for line in pdata[P]:
                if line[0] != 'Child':
                    if float_or_default(line[sword_col]) > 1 and line[sentence_col].upper() != 'A DRESS':  # i.e., a sentence
                        if len(line) < totwords_col:  # missing wpm data
                            f_valid.write('\t'.join(np.hstack((['Missing wpm data'], line))) + '\n')

            # Warn if fewer than four sentences of a given length
            for k, g in groupby(pdata[P], cvs_key):
                if len(list(g)) <= 4:
                    kp = k.split('-')
                    f_valid.write('\t'.join(['Four or fewer sentences of length {}'.format(kp[2]), 'v'.join(kp[:2]), kp[0], kp[1]]) + '\n')

            # Warn if sentence has more than 7 words
            for line in pdata[P]:
                if line[0] != 'Child':
                    if float_or_default(line[sword_col]) > 7:
                        f_valid.write('\t'.join(np.hstack((['More than 7 words in sentence'], line))) + '\n')

            #----- end of data validation --------------


            # process data by group and store
            for k, g in groupby(pdata[P], cvs_key):
                group_calcs[k][P] = process_group(g)
            for k, g in groupby(pdata[P], cv_key):
                group_calcs27[k][P] = process_group([line for line in g if float_or_default(line[sword_col]) > 1])
            for k, g in groupby(pdata[P], cv_key):
                group_calcs17[k][P] = process_group(g)


        # Write intelligibility to output file, interleaving phases
        phase_cols = [
            'Sum Words',
            'Sum Duration',
            'Sum MWordA Avg',
            'WPM',
            'IWPM',
            'Intelligibility',
            'SyllsPerSecond',
            'ArticRate'
        ]
        calc_keys = ('sum_words', 'sum_duration', 'sum_mwordaavg', 'wpm', 'iwpm', 'intelligibility', 'sylls_per_second', 'artic_rate')
        groupdata_default = {k: 0 for k in calc_keys}

        intelligibility_header = np.hstack((['Child', 'Visit', 'Avg SWord'], np.repeat(phase_cols, len(phases))))
        phase_header = np.hstack((('Phase', '', ''), np.tile(phases, len(phase_cols))))


        f_out.write('\t'.join(intelligibility_header) + '\n')
        if len(phases) > 1:
            f_out.write('\t'.join(phase_header) + '\n')

        for k in sorted(group_calcs.keys()):
            if k.split('-')[2] != '0.0':
                f_out.write('\t'.join(
                    k.split('-') +
                    list(chain(*zip(*[[str(group_calcs[k].get(P, groupdata_default)[s]) for s in calc_keys] for P in phases])))
                ) + '\n')

        for k in sorted(group_calcs27.keys()):
            f_out.write('\t'.join(
                k.split('-') + ['s2-s7'] +
                list(chain(*zip(*[[str(group_calcs27[k].get(P, groupdata_default)[s]) for s in calc_keys] for P in phases])))
            ) + '\n')

        for k in sorted(group_calcs17.keys()):
            f_out.write('\t'.join(
                k.split('-') + ['s1-s7'] +
                list(chain(*zip(*[[str(group_calcs17[k].get(P, groupdata_default)[s]) for s in calc_keys] for P in phases])))
            ) + '\n')

        output_files.extend([
            File(f_out, name='combined_{}_intelligibility.txt'.format(ftype[0])),
            File(f_valid, name='combined_{}_intelligibility_validation.txt'.format(ftype[0]))
        ])

    return output_files