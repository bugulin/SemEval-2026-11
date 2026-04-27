This folder contains the original scripts with some bugs fixed.

-setup_inference.py:
Contains bugfixes for LoRA_llama_setup_for_inference.py. However, I don't think I have made any changes in this file.

-setup_training.py:
Contains bugfixes for LoRA_training_setup.py. The following changes were made:
-properly distinguished train and test sets.
-added model argument to setup so we train the correct model (personal preference)
-proper path handling to files.

--llm_fne_tune (2).ipynb:
Contains bug fixes for the llm_fine_tune.ipynb.
-instead of going into the original code, I have added 2 cells at the end. So, if you want to run it, just run the pip imports, then the cell after "RUN ONLY THIS" text cell and in the end, run the eval. The cell after "RUN ONLY THIS" text cell contains the code from setup_train.py file.
-what has changed compared to the original training:
-properly distinguished train and test sets.
-added model argument to setup so we train the correct model (personal preference)
-proper path handling to files.
-!!!!!!!! removed the 
	model = peft.get_peft_model(model, peft_config)
 line that has wrapped the original llama model into NEW UNTRAINED LoRA weights after training. As a result, it seemed that the model was not finetuned at all -- the finetuned LoRA weight were thrown away in this single line.
-added eval cell.

-training_eval.ipynb:
-new file that contains only the cells you should run if you want to obtain a finetuned model and evaluate it. These are precisely the cells discussed above.

-eval.txt:
log of the eval cell. The only important info is contained in the last 4 rows, which show that our finetuning makes sense and improves the model.