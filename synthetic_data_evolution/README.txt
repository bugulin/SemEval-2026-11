TL;DR: for the presentation, all the relevant info is in the raw_data_summary.json files and in the validity_check_stats.txt files.

PIPELINE: on raw_data, run the correctness_checker script. The output is in the raw_data_output.json file. This file contains the flags whether ur not the syllogism is of correct form, uncertain, uncertain_count, or invalid form. Also, this script produces the raw_data_summary.json file containing the basic statistics. On the raw_data_output.json file, run the purifier script to get the raw_data_purified.json set -- this is not that important here, but it takes the syllogisms of correct forms only and converts them to the original form without all the fancy flags and moods so we can train on them. On raw_data_output.json, run the logic_parser script to get the raw_data_formalized.json file. This file contains the symbolic representation of the syllogism of style Aab, Abc, Aac and so on. NOTE: the order of the terms a,b,c is actually reversed compared to Smith's book Aristotle's Logic, 2004, Chapters 4 and 5. Finally, on the raw_data_dormalized.json run the validity checker to get the validity checked. The statistics after this step is in the validity_check_stats.txt file, and the corresponding output is in the raw_data_validity_check.json file.

1: contains all the relevant outputs for the 1st try. 

2: contains all the relevant outputs for the 2nd try. 

3: contains all the relevant outputs for the 3rd try. 

scripts: contains the used scripts.