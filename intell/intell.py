import io
import os
import numpy as np

from .perceptual_learning import initialize_perceptual_learning, read_articulation, perceptual_learning
from .utils import group_filenames, parse_intel_control_filename, readfiles, is_sentence_file, sequence_match
from .file import File
from .data import Word
from .analysis import word_analysis, sentence_analysis, word_count_analysis
from .iwpm import merge_iwpm_files, combined_iwpm_to_intelligibility


def compute_intelligibility(files, articulation_files, wpm_files, sylls_files, artic_files, exclude):

    output_files = []

    missing_values = []  # list of prefixes of files that probably need to have missing values computed
    messages_stream = io.StringIO()  # for error messages


    # for perceptual learning
    pl_file = initialize_perceptual_learning()
    articulation = read_articulation(articulation_files)


    # Regroup filenames so that files with matching initials and visit number are concatenated and processed together.
    file_sets = group_filenames(files.keys())

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
        Phase = []  # phase in listener training task

        word_dict = {}

        # Find location of columns based on header
        col = {}
        prev_timestamp = None
        prev_sentence = None

        rf_stream = io.StringIO()  # Reliability file

        file_set_files = [files[f] for f in file_set['filenames']]

        for line, sourcefile in readfiles(file_set_files):
            fileparts = parse_intel_control_filename(sourcefile)
            cf_number = fileparts['cf_number']
            lineparts = line.strip().split('\t')
            if len(lineparts) > 1:
                if lineparts[0].isnumeric():
                    if 'SWord' not in col.keys():
                        # Probably forgot to compute missing values
                        missing_values.append(os.path.basename(prefix))
                        continue

                    # Discard lines which repeat a timestamp from a previous line.
                    if 'Timestamp' in col.keys():
                        if lineparts[col['Timestamp']] == prev_timestamp and float(prev_timestamp) > 0:
                            if lineparts[col['Sentence']] == prev_sentence:
                                continue
                            else:
                                messages_stream.write('Repeated lines have the same timestamp but different sentences.'
                                                      f'  Timestamp {prev_timestamp} in {sourcefile}/n')

                        else:
                            prev_timestamp = lineparts[col['Timestamp']]
                            prev_sentence = lineparts[col['Sentence']]

                    # Any SENTENCE where speaker said only ONE word (SWord == 1) should be ignored
                    try:
                        if is_sentence_file(lineparts[col['File']]) and int(lineparts[col['SWord']]) == 1:
                            continue
                    except Exception as e:
                        messages_stream.write(f'{e}/nLine: {line}')
                        #print('Line: {}'.format(line))
                        continue

                    # This is a data line
                    ShortListener.append(lineparts[col['Listener']])
                    Listener.append('{}-{}'.format(lineparts[col['Listener']], cf_number))
                    Phase.append(lineparts[col['Phase']] if 'Phase' in col.keys() else 'Single')
                    Repeat.append(lineparts[col['Repeat']])
                    Data.append([lineparts[col['Time']]] + lineparts[col['SWord']:])
                    SWord.append(lineparts[col['SWord']])
                    Sentence.append(lineparts[col['Sentence']].capitalize())
                    PSentence.append(lineparts[col['Phonetic Sentence']])

                    SentenceFile.append(os.path.basename(lineparts[col['File']]))
                    SentenceFileSentence[SentenceFile[-1]] = Sentence[-1]

                    # store data for word-level analysis
                    sentence_parts = lineparts[col['Sentence']].split()
                    sentence_length = len(sentence_parts)
                    psentence_parts = lineparts[col['Phonetic Sentence']].split()
                    response_parts = lineparts[col['Phonetic Response']].split()

                    # If the phonetic representation has different number of words, use text instead.
                    if len(sentence_parts) != len(psentence_parts):
                        psentence_parts = lineparts[col['Sentence']].lower().split()
                        response_parts = lineparts[col['Response']].lower().split()

                    correct = sequence_match(psentence_parts, response_parts)  # Score response
                    Correct.append(all(correct))

                    SourceFile.append(sourcefile)
                    Response.append(lineparts[col['Response']])

                    for idx, word in enumerate(sentence_parts):
                        pword = psentence_parts[idx]  # phonetic word
                        cap_word = word.capitalize()
                        if cap_word not in word_dict:
                            word_dict[cap_word] = Word(cap_word, pword)
                        word_dict[cap_word].update(sentence_length, correct[idx])

                else:
                    # This is a header line
                    col = {a: i for i, a in enumerate(lineparts)}  # store the column numbers
                    if 'SWord' not in col.keys():
                        # Probably forgot to compute missing values
                        missing_values.append(os.path.basename(prefix))
                        continue
                    # has_demographic_info = lineparts[3] == 'Age'
                    # offset = 4 if has_demographic_info else 0
                    header = ['Child', 'Listener', '# of Sentences', 'Time'] + lineparts[col['SWord']:]

                if lineparts[col['Repeat']] != '0':
                    # Write lines where Repeat != 0 to the reliability file
                    rf_stream.write(line)
            else:
                # Throw source filenames into reliability file, too
                rf_stream.write(line)

        rf_stream.seek(0)
        rf_name = prefix + '_reliability.txt'
        #print(os.path.basename(prefix), flush=True)

        output_files.append(File(rf_stream, name=rf_name))

        if not Listener:
            continue
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
        Repeat = np.fromiter(Repeat, 'int')
        Data = np.array([[float(z) for z in x] for x in Data])
        SWord = np.fromiter(SWord, 'int')
        WorkingSet = [r in [0, 1] for r in Repeat]


        # Generate perceptual learning output file.  Ignore phases for this file.
        pl_file, pl_files = perceptual_learning(pl_file, header[3:], Sentence, SentenceFile, Correct, Repeat, Data, SourceFile, Response, articulation)
        output_files.extend(pl_files)

        # Insert a column for %CWordA_SD after %CWordA
        CWordA_SD_col = header.index('%CWordA') + 1
        header.insert(CWordA_SD_col, '%CWordA_SD')

        Phase = np.array(Phase)
        Phases = np.unique(Phase)

        # if aggregation_mode == 'Word Count':
        wc_files = word_count_analysis(prefix, header, SWord, Listeners, WorkingSet, Listener, Data, CWordA_SD_col, Phase, Phases)
        output_files.extend(wc_files)

        #elif aggregation_mode == 'Sentence':
        sentence_files = sentence_analysis(prefix, header, Sentences, SentenceFiles, SentenceFileSentence, SentenceFile, Listeners, WorkingSet, Listener, Data, CWordA_SD_col, PSentence, Phase, Phases)
        output_files.extend(sentence_files)

        # Write out word-level analysis
        word_files = word_analysis(prefix, word_dict, Listeners)
        output_files.extend(word_files)


    # gather files required as input to merge_iwpm_files

    # files generated in previous steps
    sentence_files = [f for f in output_files if 'sentenceIWPMsummary.txt' in f.name]
    word_files = [f for f in output_files if 'wordIWPMsummary.txt' in f.name]

    # wpm_files, sylls_files, artic_files provided separately
    merged_files = merge_iwpm_files(sentence_files, word_files, wpm_files, sylls_files, artic_files)
    output_files.extend(merged_files)

    ask_arg = exclude == "ask"
    exclude_arg = exclude == "yes"
    intelligibility_files = combined_iwpm_to_intelligibility(merged_files, ask_arg, exclude_arg)
    output_files.extend(intelligibility_files)

    if missing_values:
        #showinfo('Check input', 'Compute missing values for: {}'.format(', '.join(set(missing_values))))
        messages_stream.write('Compute missing values for: {}'.format(', '.join(set(missing_values))) + '/n')


    pl_file.seek(0)
    output_files.append(File(pl_file, name='combined_perceptual_listening.txt'))

    messages_stream.seek(0)
    output_files.append(File(messages_stream, name='messages.txt'))

    return output_files

