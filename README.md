# Question-type-classification

First install the dependencies required to run these files by  pip install -r requirements.txt

# How to train the model
Run the Train.py file using python3 Train.py
By running this file it creates two pickel files one is to store the model and the other one is to store the clean text

# How to test the model with query

Run the command to get the output of the question type which is predicted by the model
python3 Test.py "query question"

# example
To test the query "What is the Kashmir issue"
python3 Test.py "What is the Kashmir issue"

# expected output
Question : What is the Kashmir issue
predicted : what

