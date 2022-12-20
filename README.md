# guide
#For science use
#This is a step by step instruction for manuscrpit "A bibliometric analysis of 16,826 triple-negative breast cancer (TNBC) publications using multiple machine learning algorithms: Progress in the past 17 years"

Supplemental information 1 Step by step instruction

Step 1:  Confirmation of research purpose
To identify the purpose of the research, this study uses machine learning methods to explore the current research status and deficiencies from a macro perspective on TNBC publications. Then based on the research purpose, select the appropriate database and search terms, search steps, and inclusion and exclusion criteria. We describe the specific details in Table 1.

Step2：Publication access
Confirm the search strategy, the database, and the API used based on the research purpose. This study uses the PubMed database https://pubmed.ncbi.nlm.nih.gov/ to obtain as many publications about TNBC as possible, and the download method R were used.

Use our R code


Step 3: Inclusion and exclusion criteria
After obtaining publications, the required publications were further included and excluded. Repeat steps 1 and 2 until it fits research purpose. We put our inclusion and exclusion criteria in Table 2.

Step 4：General literature information studies
	We have attached a partial example of downloading the original file in the file. Each publication in the study included year; title; Abstract; Author; Affiliation; Country; MedlineTA; Substance; CitationSubset; MeshHeadingList; Reference_ArticleId; Reference_title; Publication_Type; received_date; accepted_date; pubmed_date; medline_date; entrez_date; revised_date. Based on the above information, data extraction and basic literature information analysis can be carried out. Because the original content is too large, we only use it here as an example. For details, please refer to the supplementary materials we uploaded, and Figure 1 is the original data.
  
Step 5: Topic Modeling, LDA Analysis
The LDA algorithm is an unsupervised text analysis algorithm and a text topic model [1]。LDA is a very typical bag of words model. Based on the example theory, each document is a collection of phrases, and there is no order or sequence relationship between words. A document can contain multiple topics, and each word in the document is generated by one of the topics. Based on this way of thinking, as long as you use our algorithm, you can get the result. We use Python, and you can directly use the sample data we uploaded for analysis. The specific code is as follows:

Than use our uploaded LDA code:


According to the LDA result, you only need to adjust the k value in the code. K represents the total number of topics. The selection of the k value is based on multiple indicators, and you can select according to your results. The results will finally output two files, the calculation result and the visualization result, based on Gepia software, which can be displayed in various ways. The results can refer to Figure 5 and Figure 6 of the manuscripte.

