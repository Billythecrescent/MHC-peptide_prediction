
# MHC-peptide_prediction

Object: peptide-MHC binding prediction

Final-year project of Computational Biology major in Sichuan University (SCU).

[Abstract] Whether for the treatment of tumors or the prevention of infectious diseases, T cell epitope vaccines are playing an increasingly important role. In the process of T cell antigen presentation, the combination of the major histocompatibility complex (MHC) and the polypeptide is the most selective process. Since screening MHC binding peptides in experiments is cumbersome and inefficient, using computers to predict MHC binding peptides has gradually become an important research method for T cell epitope screening. Although there are many prediction algorithms at this stage, the best ones are still based on artificial neural networks, especially NetMHC series algorithms. In this study, the principle of this kind of algorithm was analyzed in detail, and the program was implemented using the Python programming language. In addition, this article shows the evolution of the algorithm in different stages. Based on the crystal structure analysis of MHC and peptides, the article emphasized the importance of MHC pseudo-sequence selection to extract the binding characteristics of MHC and peptides. Additionally, it improved the algorithm one the alignment and encoding of peptide sequences and the selection of MHC pseudo-sequence, and therefore showed the effect of different pseudo-sequences on the model's predictability. To test the performance of the algorithm in the actual context, this study used the verified tumor epitope data set and the vaccinia virus epitope data set that passed the immune experiment, and compared the actual prediction ability of different algorithms. It also discussed how to choose a suitable prediction model and how to choose MHC pseudo-sequences in the actual T cell epitope prediction process, which not only promoted the development of MHC binding peptide prediction algorithms, but also provided a reference on how to choose a T cell epitope prediction tools.

[Key Words] Neural Network; MHC-I; MHC-II; T-cell epitope; Regression Model

`/data` is the directory containing all the data needed

`/models` is the directory containing all the saved predictor models

`/matrices` is the directory containing all the encoding matrices (BLOSUM)

`/MHCI_BP_predictor` is the project using BP neural network to predict peptide-MHCI binding

`/MHCII_BP_predictor` is the project using BP neural network to predict peptide-MHCII binding

`/MHCI_CNN_predictor` is the project using BP neural network to predict peptide-MHCII binding