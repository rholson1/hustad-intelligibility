# Refactor intelligibility so that it can operate on file-like objects, not just files in a filesystem

import argparse
import tkinter
import tkinter.filedialog as fd
import os
import re
from intell.file import File
from intell.intell import compute_intelligibility


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


    pattern = re.compile(r'(Control[_ ]File|Research_Responses|Parent_Responses|Training_Responses)-\d+\.')
    fnames = [os.path.join(working_dir, f) for f in os.listdir(working_dir) if pattern.search(f)]
    files = {fname: File(open(fname), name=fname) for fname in fnames}

    # for perceptual learning
    pl_pattern = re.compile(r'(Control[_ ]File|Research_Responses|Parent_Responses)!')
    articulation_fnames = [os.path.join(working_dir, f) for f in os.listdir(working_dir) if pl_pattern.search(f)]
    articulation_files = {fname: File(open(fname), name=fname) for fname in articulation_fnames}

    # Other files
    all_files = os.listdir(working_dir)

    pattern = re.compile('[^i]wpm', flags=re.IGNORECASE)  # look for filenames containing WPM but not IWPM (case insensitive)
    wpm_filenames = [os.path.join(working_dir, f) for f in all_files if pattern.search(f)]
    #wpm_files = {fname: File(open(fname), name=fname) for fname in wpm_filenames}
    wpm_files = [File(open(fname), name=fname) for fname in wpm_filenames]

    pattern = re.compile('syllsPerSecond', flags=re.IGNORECASE)  # look for filenames containing syllsPerSecond
    sylls_filenames = [os.path.join(working_dir, f) for f in all_files if pattern.search(f)]
    sylls_files = {fname: File(open(fname), name=fname) for fname in sylls_filenames}

    pattern = re.compile('articRate', flags=re.IGNORECASE)  # look for filenames containing articRate
    artic_filenames = [os.path.join(working_dir, f) for f in all_files if pattern.search(f)]
    artic_files = {fname: File(open(fname), name=fname) for fname in artic_filenames}

    output = compute_intelligibility(files, articulation_files, wpm_files, sylls_files, artic_files, exclude)


    for f in output:
        filename = os.path.basename(f.name)
        with open(os.path.join(working_dir, filename), 'w') as of:
            f.seek(0)
            for line in f.file:
                of.write(line)
        #print(os.path.basename(f.name))


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

    debugging = True
    if debugging:
        main(d, exclude)
    else:
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



