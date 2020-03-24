# Compressed kNN
Repository for the Compressed kNN (CS-kNN) algorithm implemented in MOA.

For more informations about MOA, check out the official website: 
http://moa.cms.waikato.ac.nz

## Citing CS-kNN
To cite the CS-kNN in a publication, please cite the following paper: 
> Maroua Bahri, Albert Bifet, Silviu Maniu, Rodrigo Fernandes de Mello, Nikolaos Tziortziotis.
> Compressed k-Nearest Neighbors Ensembles for Evolving Data Streams. In the 24th European Conference on Artificial Intelligence (ECAI), 2020.

## Important source files
The implementation and related codes used in this work are the following: 
* CS-kNN.java: the compressed sensing k-nearest neighbors using the random projection internally.
* CS-filter.java: the compressed sensing used as a filter.

## How to execute it
To test the CS-kNN, you can copy and paste the following command in the interface (right click the configuration text edit and select "Enter configuration”).
Sample command: 

`EvaluatePrequential -l (lazy.CS_kNN -d 5 -f 500) -s (ArffFileStream -f /pathto/tweet500.arff) -e BasicClassificationPerformanceEvaluator`

Explanation: this command executes CS-kNN prequential evaluation precising the output and input dimensionality, d and f respectively on the tweet500 dataset (-f tweet1.arff). 
**Make sure to extract the tweet500.arff dataset, and setting -f to its location (pathto), before executing the command.**

## Datasets used in the original paper
The datasets used in this work are compressed and available at the root directory. 
