from pytorch_pretrained_bert.tokenization import BertTokenizer
import pickle
tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased", do_lower_case=True)
filehandler = open("tokenizer.pkl", 'wb')
pickle.dump(tokenizer, filehandler)
