import tokenization

vocab_file = '../vocab.korean_morp.list'

tokenizer = tokenization.BasicTokenizer(vocab_file=vocab_file) #,add_special_token=False)

tokens = tokenizer.tokenize('ETRI/SL 에서/JKB 한국어/NNP BERT/SL 언어/NNG 모델/NNG 을/JKO 배포/NNG 하/XSV 었/EP 다/EF ./SF')


token_list=[]

token_list.append('[CLS]')
token_list.extend(tokens)
token_list.append("[SEP]")

print(token_list)
print(tokenizer.convert_tokens_to_ids(token_list))