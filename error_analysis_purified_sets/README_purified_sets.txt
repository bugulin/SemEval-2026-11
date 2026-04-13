This dir contains the following files:
-train_data_original_purified: original syllogisms from semeval that syllogism_correctness_checker.py flagged as "valid_form"
-train_data_original_purified_uncertain: original syllogisms from semeval that syllogism_correctness_checker.py flagged as "uncertain". These are mostly of the valid form, they were just written in a more exotic language so they did not match any of the given patterns. Those are definitely worth including into the train set.
-train_data_original_purified_uncertain_count: original syllogisms from semeval that syllogism_correctness_checker.py flagged as "uncertain_count" - i.e. the checker found more than 3 terms. These are 50/50 valid/invalid, and I would be careful about including them to the train set.
-train_data_original_purified_uncertain_count: syllogisms from synthetic dataset that syllogism_correctness_checker.py flagged as "valid_form"

Uncertain and uncertain_count categories from synthetic dataset are missing because they contained mostly invalid syllogisms, so I am strongly against including them to the train set. However, if you want, I can add them for educational purposes.

