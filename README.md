# UCL COMPGI10 - Bioinformatics
Accompanying code for the COMPGI10 Bioinformatics project undertaken at University College London.

The script can be run directly from console.


## Output of the python script
Produces the sequence, protein locations and accompanying probabilities for each entry in the blinded set


## Python modules required
numpy, collections, itertools, sklearn


## Functions used
read_fasta:
This function takes as inputs the FASTA files, and outputs a 2 element tuple, with the first element being the sequence name/identifier, and the 2nd element being the sequence itself.

print_statistics: 
This function takes as inputs both the ground truth labels as well as the predicted labels. It then outputs the macro/micro/weight versions of the precision, recall and F1 metrics between the 2 sequences.



