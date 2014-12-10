#Structured Reading 
==========

Implementation of structured reading active learning 


## How to run?

From the command line:

```bash
$ python experiment.py --train data_name --config config_file.cfg
```

## Configuration File

Configuration file has the same format as an INI file. Sections ```[sections]``` and ```key=value``` or ```key:value```. 

The sections are: 
* student
* expert
* experiment
* data 

Example:

```
[student]
type           : "sequential"
model          : "lrl2"
parameter      : 1
utility      : "rnd"
snippet      : "sr"
                
			
[expert]        
type           : "sent"
model          : "lrl2"
penalty        : 1
costmodel      : 1
                

[data]
data            : "imdb"
limit			: 2
split			: 0.5
min_size		: 10
vectorizer 		: "tfidf"
categories		: "religion"

[experiment]   
bootstrap      : 50
stepsize       : 10
maxiter        : 125
budget         : 1250
cheatingmode   : False
trials         : 5
folds          : 1
fileprefix     : "RND-F1-LIM2-"
outputdir      : "../results"
seed           : 876543210
limit          : 2
costfunction   : "unit"
```

* **student section**
* type: sequential, joint
* model: lr, lrl2, mnb
* utility: rnd, unc
* snippet: sr, rnd, first1


* **expert section**
* type: sent(ence), true(labels), pred(icting)
* model: lr, lrl2, mnb

* **data section **
* vectorizer: tfidf, bow
* categories: only applies for some data. options: None, religion (20news)
* data: imdb, sraa, 20news

