import re
import ntpath
import difflib


def float_or_default(s, default=0):
    """ Try to apply the float function to a variable.  If conversion fails with a ValueError, return the default """
    try:
        value = float(s)
    except ValueError:
        value = default
    return value


def readfiles(filenames):
    """Generator iterates through lines of supplied files or Django file objects successively, returning the line
    together with the name of the source file"""
    for f in filenames:
        if hasattr(f, 'file'):
            # Django file object
            for line in f.file:
                yield line, f.name
        else:
            # filename
            for line in open(f):
                yield line, f


def parse_intel_control_filename(fname):
    """ Intelligibility control filenames are formatted as:
    [initials]v[visit#] Control File-[datetime].txt
    """
    path, basename = ntpath.split(fname)
    try:
        # base_prefix = re.match('(.+)[_ ]Control[_ ]File-', basename).group(1)
        match = re.match('(.+)[_ ](Control[_ ]File|Research_Responses|Parent_Responses|Training_Responses)-(.*)\.txt', basename)
        base_prefix = match.group(1)
        listener_type = match.group(2)
        cf_number = match.group(3)
    except:
        raise Exception('Unexpected format for control file name: {}'.format(basename))
    return {'prefix': ntpath.join(path, base_prefix), 'filename': fname, 'cf_number': cf_number, 'visit': base_prefix,
            'listener_type': listener_type}


def file_to_key(fname):
    """ Convert a filename to a key based on visit and control file number
    Works both for listener response files and for question files (- and !)
    """
    path, basename = ntpath.split(fname)
    try:
        # base_prefix = re.match('(.+)[_ ]Control[_ ]File-', basename).group(1)
        match = re.match('(.+)[_ ](Control[_ ]File|Research_Responses|Parent_Responses|Training_Responses)[-!](.*)\.txt', basename)
        base_prefix = match.group(1)
        cf_number = match.group(3)
    except:
        raise Exception('Unexpected format for file name: {}'.format(basename))

    # Check to see if base_prefix ends with a visit number: vNN
    # If not, assume visit 1 and append v01
    if re.search('v\d\d$', base_prefix) is None:
        base_prefix = base_prefix + 'v01'

    return '{}-{}'.format(base_prefix, cf_number)


def group_filenames(fnames):
    """ Given a list of filenames, create subgroups with matching initials and visit number.
    Only include filenames containing the substring 'Control File-' (that is, intelligibility control files).
    """

    pattern = re.compile('(Control[_ ]File|Research_Responses|Parent_Responses|Training_Responses)-')
    f_list = [parse_intel_control_filename(f) for f in fnames if pattern.search(f)]
    prefixes = sorted(set([f['prefix'] for f in f_list]))
    grouped_list = [{'prefix': p,
                     'filenames': [f['filename'] for f in f_list if f['prefix'] == p]} for p in prefixes]
    return grouped_list


def is_sentence_file(filename):
    """ Check if a file is a sentence file or a word file by looking for presence of 's' or 'w' in filename """
    match = re.search(r'(w|s\d+)t\d+[a-z]?(_.......)?\.wav', ntpath.basename(filename).lower())
    if match:
        return match.group(1)[0] == 's'  # True for a sentence file
    else:
        raise Exception('Stimulus filename has unexpected format: {}'.format(filename))


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



