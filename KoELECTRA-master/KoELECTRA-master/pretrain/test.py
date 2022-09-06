from model import tokenization
import pretrain_data
import configure_pretraining

from konlpy.tag import Mecab

mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")
print(mecab.pos('안녕 하세요'))
#
# vocab_file = './vocab.korean_morp.list'
# tokenizer = tokenization.FullTokenizer(
#     vocab_file=vocab_file,
#     do_lower_case=False)
#
# inputs = ["ETRI/SL 에서/JKB 한국어/NNP BERT/SL 언어/NNG 모델/NNG 을/JKO 배포/NNG 하/XSV 었/EP 다/EF ./SF"]
# print(pretrain_data.get_input_fn(configure_pretraining, True))
# token = tokenizer.tokenize('ETRI/SL 에서/JKB 한국어/NNP BERT/SL 언어/NNG 모델/NNG 을/JKO 배포/NNG 하/XSV 었/EP 다/EF ./SF')
# ids = tokenizer.convert_tokens_to_ids(token)

# print(token)
# print(ids)