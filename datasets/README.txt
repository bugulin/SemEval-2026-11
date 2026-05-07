train.json is the train split.

validation.json is the validation split.

Merged_train_hand.json is the hand-edited set containing all train and validation samples before the split.

merger.py and splitter.py scripts were used to create the splits; the files I have used as arguments are in the mess folder, but you do not need to explicitly access it unless you want to.

I also include the test split, but we should not use it until the very end. Thus, it is in the separate folder.

There are approx. 1250 train samples and 137 validation samples.