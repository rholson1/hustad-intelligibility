Intelligibility and Missing Values
==================================

This repository contains code used to process listener responses from WISCLab listening tasks.

missing_values.py
-----------------
*missing_values.py* takes as input listener response files (from ShowTell or from online listening tasks) and adds
"missing values" in the sense of the "Add Missing Values" function from ShowTell.  The resulting outputs are called
"statistics" files.  Various scores are computed based on
the phonetic representation of the utterance and the listener's response.  This script differs from ShowTell in that it
retains extra columns in the input -- notably including the Phase column which is included in listener response files
produced from the Listener Training task.  Additionally, *missing_values.py* uses a different matching algorithm to
evaluate some of the measures (e.g., the number of matching words).  *missing_values.py* uses difflib.SequenceMatcher.



intelligibility.py
------------------
*intelligibility.py* computes various measures of intelligibility from statistic files.
