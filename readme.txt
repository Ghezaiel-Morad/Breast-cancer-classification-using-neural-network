#######################################################################################

#										      #
#										      #
#										      #	
# NEURAL NETWORK MODELLING FOR BREAST CANCER SUBTYPES PREDICTION USING RNASeq DATASET

 #
#										      #
#										      #
#	Authors : 
VICARI Célia, GHEZAIEL Morad 
Master 2 Bioinformatics DLAD	      #
				     		
#										      #
#######################################################################################

####################################################################################

###

1. Data: 

Informations related to datasets are in /doc/data_informations.txt. User have to fill the data folder with data.csv and 
labels.csv.

2. Aim:
The aim of is program is to predict breast cancer subtypes giving rna-seq data (expressions levels) using a neural network. 



3. Run : Main program
main.py data/data.csv data/labels.csv
 .
 

See the figures in /output file.


You can get more informations about the protocol, the program or the data on the /doc file.

General information :
	- This program consists in a neural network based pipeline analysis for breast cancer subtype prediction using RNA-Seq datas
	- 3 pipelines are availables:
		1) Activation function comparaison 
		2) Optimizer comparaison
		3) General presentation of the model (launched by default)

One can skip the 2 first pipelines by answering "n" in the prompt.

The Master report is stored in the doc folder.

