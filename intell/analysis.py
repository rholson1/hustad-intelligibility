import numpy as np
import ntpath
import io
from collections import defaultdict
from itertools import chain
from .file import File


def word_count_analysis(prefix, header, SWord, Listeners, WorkingSet, Listener, Data, CWordA_SD_col, Phase, Phases):

    # Expand header according to the number of phases
    phase_col_count = len(header) - 2  # number of columns that should be expanded by the number of phases
    expanded_header = np.hstack((header[:2], np.repeat(header[2:], len(Phases))))
    phase_header = np.hstack((('Phase', ''), np.tile(Phases, phase_col_count)))
    summary_phase_header = np.hstack((('Phase', ''), np.tile(Phases, phase_col_count + 1)))

    basename = ntpath.basename(prefix)

    sf = io.StringIO()
    f = io.StringIO()

    # Write header line
    f.write('\t'.join(expanded_header) + '\n')
    if len(Phases) > 1:
        f.write('\t'.join(phase_header) + '\n')
    sf.write('\t'.join(
        ['Child', '# Listeners'] +
        ['# Sentences'] * len(Phases) +
        ['Avg Sent Length'] * len(Phases) +
        ['\t'.join(expanded_header[2+len(Phases):]) + '\n']
    ))

    if len(Phases) > 1:
        sf.write('\t'.join(summary_phase_header) + '\n')

    # Loop over word counts/utterance
    for wc in np.unique(SWord)[::-1]:
        WordSet = SWord == wc

        # Loop over Listeners
        blockdata = defaultdict(list)
        exemplars_total = defaultdict(float)
        for L in Listeners:
            bd = {}
            exemplar_count = defaultdict(int)
            for P in Phases:
                exemplars = np.logical_and.reduce((WordSet, WorkingSet, Listener == L, Phase == P))
                if np.any(exemplars):
                    bd[P] = np.mean(Data[exemplars, :], 0)
                    # Insert %CWordA_SD data into bd
                    sd_data = np.std(Data[exemplars, CWordA_SD_col - 1])
                    bd[P] = np.insert(bd[P], CWordA_SD_col - 3, sd_data)
                else:
                    bd[P] = np.zeros(Data.shape[1] + 1)

                blockdata[P].append(bd[P])
                exemplar_count[P] = exemplars.sum()
                exemplars_total[P] += exemplar_count[P]

            # interleave phase data
            linedata = [basename, L] + [exemplar_count[P] for P in Phases] + \
                       list(chain(*zip(*[bd[P] for P in Phases])))
            f.write('\t'.join([str(s) for s in linedata]) + '\n')


        # Write average across listeners
        # bd = np.mean(np.array(blockdata), 0)
        for P in Phases:
            blockdata[P] = np.array(blockdata[P])
            bd[P] = [np.mean(blockdata[P][:, idx]) if '%' in h else np.sum(blockdata[P][:, idx])
                     for idx, h in enumerate(header[3:])]

        summary_line = (
                [basename, len(Listeners)] +
                # Total number of sentences
                [exemplars_total[P] for P in Phases] +
                # Average sentence length
                [bd[P][1] / len(Listeners) for P in Phases] +
                # average across listeners, interleaved by phase
                list(chain(*zip(*[bd[P] for P in Phases])))
        )
        summary_line = [str(s) for s in summary_line]

        # Exclude Avg Sentence Length columns
        f.write('\t'.join(summary_line[:2+len(Phases)] + summary_line[2+2*len(Phases):]) + '\n')

        # write average line to summary file as well
        sf.write('\t'.join(summary_line) + '\n')


    sf.seek(0)
    f.seek(0)
    sfname_wc = basename + '_intellxlistener_summmary.txt'
    ofname_wc = basename + '_intellxlistenerxlength.txt'
    return [File(f, ofname_wc), File(sf, sfname_wc)]




def sentence_analysis(prefix, header, Sentences, SentenceFiles, SentenceFileSentence, SentenceFile, Listeners, WorkingSet, Listener, Data, CWordA_SD_col, PSentence, Phase, Phases):

    # Add extra columns to header for sentence, phonetic sentence.  These do not need to be expanded.
    header.insert(2, 'Phonetic Sentence')
    header.insert(2, 'Sentence')

    # Every header column after the first 4 should be expanded based on the number of phases
    fixed_cols = 4  # do not expand fixed columns
    phase_col_count = len(header) - fixed_cols  # number of columns that should be expanded by the number of phases
    expanded_header = np.hstack((header[:fixed_cols], np.repeat(header[fixed_cols:], len(Phases))))
    phase_header = np.hstack((('Phase',), ('',) * (fixed_cols - 1), np.tile(Phases, phase_col_count)))

    summary_header_list = list(expanded_header.copy())
    for i, v in enumerate(summary_header_list):
        if v == '# of Sentences':
            summary_header_list[i] = 'Avg SWord'
        elif v == 'Listener':
            summary_header_list[i] = '# Listeners'
        elif i >= fixed_cols and '%' not in v:
            summary_header_list[i] = v + ' Sum'

    # IWPM summary files are based on a partition of the intellxutterance files
    partition_idx = [i for i, v in enumerate(summary_header_list) if v in
                     ('Child', '# Listeners', 'Avg SWord', 'Sentence', 'Time Sum', 'MWordA Sum', '%CWordA')]
    mworda_cols = [i for i, v in enumerate(summary_header_list) if v == 'MWordA Sum']

    partition_header_list = np.hstack((
        [summary_header_list[idx] for idx in partition_idx],
        ['MWordA Avg'] * len(Phases),
        ['SentenceFile']
    ))
    partition_phase_header = np.hstack((
        [phase_header[i] for i in partition_idx],
        Phases,
        ['SentenceFile']
    ))

    basename = ntpath.basename(prefix)

    ssf = io.StringIO()
    swf = io.StringIO()
    sf = io.StringIO()
    f = io.StringIO()

    # Write header lines
    f.write('\t'.join(expanded_header) + '\n')
    if len(Phases) > 1:
        f.write('\t'.join(phase_header) + '\n')

    sf.write('\t'.join(summary_header_list) + '\n')
    if len(Phases) > 1:
        sf.write('\t'.join(phase_header) + '\n')

    ssf.write('\t'.join(partition_header_list) + '\n')
    if len(Phases) > 1:
        ssf.write('\t'.join(partition_phase_header) + '\n')

    swf.write('\t'.join(partition_header_list) + '\n')
    if len(Phases) > 1:
        swf.write('\t'.join(partition_phase_header) + '\n')

    # Loop over sentences

    # sort sentences by word count first
    Sentences = Sentences[np.argsort([s.count(' ') for s in Sentences], kind='mergesort')]

    # sort sentencefiles by word count first
    SentenceFiles = SentenceFiles[
        np.argsort([SentenceFileSentence[s].count(' ') for s in SentenceFiles], kind='mergesort')]

    for sentencefile in SentenceFiles:
        SentenceSet = SentenceFile == sentencefile
        s = SentenceFileSentence[sentencefile]

        # Loop over listeners
        blockdata = defaultdict(list)
        exemplars_total = defaultdict(float)
        for L in Listeners:
            bd = {}
            exemplar_count = defaultdict(int)
            exemplars_all = np.logical_and.reduce((SentenceSet, WorkingSet, Listener == L))
            for P in Phases:
                exemplars = np.logical_and(exemplars_all, Phase == P)
                if np.any(exemplars):
                    bd[P] = Data[exemplars, :].flatten('F')  # flatten by column
                    bd[P] = np.insert(bd[P], CWordA_SD_col - 3, np.nan)
                else:
                    bd[P] = np.zeros(Data.shape[1] + 1)

                blockdata[P].append(bd[P])
                exemplar_count[P] = exemplars.sum()
                exemplars_total[P] += exemplar_count[P]

            # interleave phase data
            try:
                linedata = (
                        [basename, L] +
                        [s, PSentence[exemplars_all][0]] +
                        [exemplar_count[P] for P in Phases] +
                        list(chain(*zip(*[bd[P] for P in Phases])))
                )
                f.write('\t'.join([str(s) if str(s) != 'nan' else '' for s in linedata]) + '\n')
            except IndexError:
                print(f'IndexError when looking for sentence {s} for listener {L}')

            # Write data to file
            # f.write('\t'.join([basename, L, str(exemplars.sum()), s, PSentence[exemplars][0]]))
            # f.write('\t')
            # f.write('\t'.join(str(s) if str(s) != 'nan' else '' for s in bd))
            # f.write('\n')
            # exemplars_total += exemplars.sum()

        # Write average across listeners
        # bd = np.mean(np.array(blockdata), 0)
        for P in Phases:
            blockdata[P] = np.array(blockdata[P], ndmin=2)
            try:
                bd[P] = [np.mean(blockdata[P][:, idx]) if '%' in h else np.sum(blockdata[P][:, idx])
                         for idx, h in enumerate(header[5:])]
            except IndexError:
                print(Listeners)
                print('Length of blockdata elements: {}'.format([len(zz) for zz in blockdata[P]]))
                print('Problem computing statistics for sentence "{}" in {}'.format(s, ntpath.basename(prefix)))

            # Compute Standard Deviation of CWordA
            bd[P][CWordA_SD_col - 3] = np.std(blockdata[P][:, CWordA_SD_col - 4])

        # Child, Listener, Sentence, Phonetic Sentence, # of sentences [phase1], # of sentences [phase2],...
        summary_list = (
            [basename, len(Listeners)] +
            # sentence, phonetic sentence
            [s, PSentence[SentenceSet][0]] +
            # Number of sentences (by phase)
            [exemplars_total[P] / len(Listeners) for P in Phases] +
            # blockdata, interleave by phase
            list(chain(*zip(*[bd[P] for P in Phases])))
        )
        summary_list = [str(s) for s in summary_list]

        f.write('\t'.join(summary_list) + '\n')

        # summary_average_list is like summary_list, except that it replaces # of sentences with Avg SWord
        summary_average_list = (
            [basename, len(Listeners)] +
            # sentence, phonetic sentence
            [s, PSentence[SentenceSet][0]] +
            # Avg SWord (by phase)
            [float(bd[P][1]) / len(Listeners) for P in Phases] +
            # blockdata, interleave by phase
            list(chain(*zip(*[bd[P] for P in Phases])))
        )
        summary_average_list = [str(s) for s in summary_average_list]

        # write average line to summary file as well
        summary_average_line = '\t'.join(summary_average_list) + '\n'

        # if float(bd[1]) / len(Listeners) > 10:
        #     print('Suspicious value for mean SWORD')
        #     print('bd[1]: {}'.format(bd[1]))
        #     print(', '.join([str(s) for s in blockdata[:, 1]]))
        #     print('len(Listeners): {}'.format(len(Listeners)))
        #     print(summary_average_line)

        sf.write(summary_average_line)

        # IWPM files are partition of summary file with added MWordA Avg and sentencefile
        summary_partition_list = [summary_average_list[idx] for idx in partition_idx]
        # append MWordA Avg = MWordA Sum / # Listeners
        listener_count = float(summary_partition_list[1])
        for i in mworda_cols:
            summary_partition_list.append(str(float(summary_average_list[i]) / listener_count))
        summary_partition_list.append(sentencefile)

        if ' ' not in s or s.lower() == 'a dress':
            swf.write('\t'.join(summary_partition_list) + '\n')
        else:
            ssf.write('\t'.join(summary_partition_list) + '\n')


    ofname_sentence = basename + '_intellxutterance.txt'
    sfname_sentence = basename + '_intellxutterance_summary.txt'
    sfname_sentence_sentence = basename + '_intellxutterance_sentenceIWPMsummary.txt'
    sfname_sentence_word = basename + '_intellxutterance_wordIWPMsummary.txt'

    ssf.seek(0)
    swf.seek(0)
    sf.seek(0)
    f.seek(0)

    return [
        File(ssf, name=sfname_sentence_sentence),
        File(swf, name=sfname_sentence_word),
        File(sf, name=sfname_sentence),
        File(f, name=ofname_sentence)
    ]


def word_analysis(prefix, word_dict, Listeners):
    basename = ntpath.basename(prefix)
    word_fname = basename + '_intellxword.txt'
    # Determine max line length
    max_length = 7
    for k in word_dict:
        max_length = max(max_length, len(word_dict[k].utterance))

    f = io.StringIO()

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

    f.seek(0)
    return [File(f, name=word_fname)]