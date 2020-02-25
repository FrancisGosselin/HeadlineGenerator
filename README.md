# HeadlineGenerator
Language modeling with n-gram model. The data used to train the model is from Kaggle: https://www.kaggle.com/therohk/million-headlines.

## Requierements
python3, pip3, and pandas
    pip3 install pandas

## Usage
Download the training data, either from this  git repository or the url above

### Train and save
Train the model with a ngram value > 2 and save it:

    python3 model.py --train ./abcnews-date-text.csv  --ngram 4 --save quadgram
    
```
[##############################]  100% trained
Training Done!
```
A ngram of level 4 takes 232Mb of space

### Load a model and generate

Load the model and print 5 random headlines:

    python3 model.py --load quadgram --print 5
```
Loading model...
 fiji opposition move threats against plans 
 kim honan attends first charges 
 elderly man to announce candidates say they admire rakti 
 veterans support ruled inadmissible evidence inquiry futile says harvest on sunday league world cup spot 
 serena williams
```
### Generate headline from input

Loads the model and generate headlines from manual input until user inputs *quit*

    python3 model.py --load quadgram --manual
    
```
Loading model...
Enter a begging of a headline: donald trump
 donald trump at us house fires still trying to put on tiger snakes give business post antibiotic 
Enter a begging of a headline: jesus christ
 jesus christ former police 
Enter a begging of a headline: south park
 south park in armadale police handling of torres strait ferries back bank boss 
Enter a begging of a headline: quit
```

