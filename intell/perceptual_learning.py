import os
import io
import re
from .utils import file_to_key, is_sentence_file, parse_intel_control_filename
from .file import File
import numpy as np
from collections import defaultdict


def initialize_perceptual_learning():
    """Initialize the output file used for the combined perceptual learning file (across visits/listeners)"""
    # fname = os.path.join(working_dir, 'combined_perceptual_listening.txt')
    # with open(fname, 'w') as f:

    f = io.StringIO()

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

    return f


def read_articulation(articulation_files):
    """Read articulation from the question response files in a supplied directory"""
    # pattern = re.compile(r'(Control[_ ]File|Research_Responses|Parent_Responses)!')
    # fnames = [os.path.join(working_dir, f) for f in os.listdir(working_dir) if pattern.search(f)]
    # print(fnames)

    art = {}
    # for fname in fnames:
    for file in articulation_files:
        articulation = []
        stimulus = []
        for line in file:
            lineparts = line.split('\t')
            try:
                if lineparts[1] == 'articulation':
                    articulation.append(lineparts[2])
                    stimulus.append(lineparts[3].strip())
            except IndexError:
                continue

        art[file_to_key(file.name)] = {'articulation': articulation, 'stimulus': stimulus}
    return art


def ca_from_visit(visit):
    """Given a visit name, return a chronological age in months.  For now, only works on TD kids who have the age at
    visit encoded in the visit name as YYMM_F_XXXv01"""
    try:
        match = re.match(r'\d{4}', visit).group()
    except AttributeError:
        return 0
    y = int(match[:2])
    m = int(match[2:])
    return 12 * y + m


def perceptual_learning(combined_f, header, sentence, filenames, correct, repeat, data, sourcefile, response, articulation):
    """Write perceptual learning files
    One file per listener, plus one summary file that contains WTOCS intelligibility in thirds and TPC in thirds
    TPC == total phonemes correct, calculated as sum(MPhoA)/sum(SPho) over the set of interest
    """
    output_files = []

    # Exclude 'a dress' from consideration
    word_filter = np.array([s.lower() not in ('a dress',) for s in sentence])  # always, for wtocs
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

        # single_word_set = first utterances of wtocs by this listener, excluding 'a dress'
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
        lf = io.StringIO()
        #with open(listener_filename, 'w') as lf, open(pl_filename, 'a') as combined_f:
        lf.write('{}tocs first\n'.format('w' if wtocs_first else 's'))
        lf.write('wtocs count\t{}\n'.format(wtocs_count))

        for i in range(3):
            lf.write('wtocs {}/3\t{:.2f}\n'.format(i+1, chunk_correct[i]/chunk_size[i] if chunk_size[i] else -1))

        for i in range(3):
            lf.write('TPC {}/3\t{:.2f}\n'.format(i + 1, chunk_mphoa[i] / chunk_spho[i] if chunk_spho[i] else -1))

        if parts['prefix'] != last_prefix:
            combined_f.write('\n{}'.format(parts['visit']))
            # try to write chronological age from the visit id
            combined_f.write('\t{}'.format(ca_from_visit(parts['visit'])))

            last_prefix = parts['prefix']

        combined_f.write('\t' + '\t'.join(['{}tocs'.format('w' if wtocs_first else 's'), parts['cf_number'], parts['listener_type'], str(wtocs_count)] +
                                   ['{:.2f}'.format(chunk_correct[i] / chunk_size[i] if chunk_size[i] else -1) for i in range(3)] +
                                   ['{:.2f}'.format(chunk_mphoa[i] / chunk_spho[i] if chunk_spho[i] else -1) for i in range(3)] +
                                   ['{:.2f}'.format(sum(chunk_correct) / sum(chunk_size) if sum(chunk_size) else -1)] +  # WTOCS overall intell
                                   ['{:.2f}'.format(repeat_agg[r]['correct'] / repeat_agg[r]['count']) if repeat_agg[r]['count'] else '-1' for r in (1, 2)]  +  # Reliability intell
                                   ['{:.2f}'.format(repeat_agg[r]['mphoa'] / repeat_agg[r]['spho']) if repeat_agg[r]['spho'] else '-1' for r in (1, 2)] +  # Reliability TPC
                                   ['{:.2f}'.format(np.mean(art_value) if art_value is not None else 0)] + # articulation (aka precision)
                                   ['{:.2f}'.format(sum(chunk_mphoa) / sum(chunk_spho) if sum(chunk_spho) else -1)] +  # WTOCS TPC
                                   ['{:.2f}'.format(sum(D['MVwlA']) / sum(D['SVwl']) if sum(D['SVwl']) else -1),  # WTOCS TVC
                                    '{:.2f}'.format(sum(D['MConA']) / sum(D['SCon']) if sum(D['SCon']) else -1)] +  # WTOCS TCC
                                   ['{:.2f}'.format(v) for v in [
                                       sum(sss_correct) / stocs_count if stocs_count else -1,  # STOCS intelligibility
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
            ['Intelligibility'] + ['{:.2f}'.format(repeat_agg[r]['correct'] / repeat_agg[r]['count']) if repeat_agg[r]['count'] else '-1' for r in (1, 2)]) + '\n')
        lf.write('\t'.join(
            ['TPC'] + ['{:.2f}'.format(repeat_agg[r]['mphoa'] / repeat_agg[r]['spho']) if repeat_agg[r]['spho'] else '-1' for r in (1, 2)]) + '\n')

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

        lf.seek(0)
        output_files.append(File(lf, name=listener_filename))


    return combined_f, output_files
