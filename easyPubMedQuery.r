library(easyPubMed)
myQuery <- '("triple negative breast neoplasms"[MeSH Terms] OR ("triple"[All Fields] AND "negative"[All Fields] AND "breast"[All Fields] AND "neoplasms"[All Fields]) OR "triple negative breast neoplasms"[All Fields])triple negative breast cancer AND ("2005/01/01"[PDAT] : "2022/01/01"[PDAT])'
myIdList <- get_pubmed_ids(myQuery)

fdt_files <- batch_pubmed_download(pubmed_query_string = myQuery,
                                   format = "xml",
                                   batch_size = 4000,
                                   dest_file_prefix = "fdt",
                                   encoding = "UTF-8");

fdt_list <- lapply(fdt_files, table_articles_byAuth, 
                   included_authors = "last", getKeywords = TRUE)
