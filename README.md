# POS_embedding

## RoBERTA segmentation layer => POS layer  
RoBERTA의 경우 NSP을 안하기 때문에 segmentation layer가 필요없다.  
형태소 정보를 넣기위해 segmentation layer정보 대신 형태소 정보가 있는 POS를 넣는다.  
이때 RoBERTA의 tokenizer와 형태소 분석을 했을때 단어의 길이가 안맞는 문제가 발생한다. 이를 tokenizer길이에 맞게 형태소 정보의 길이를 맞춰준다. (roberta_embedding.py)  
  
RoBERTA pretraining colab: https://colab.research.google.com/drive/1jmJSloiSgNKhHPspGkfe4L0v9w0GrtKD  

pretraining한 모델을 klue-baseline dp로 결과를 확인한다.  
colab: https://colab.research.google.com/drive/1mWTHATolfdNbsI2aUGW1HqgNBzR4g35l#scrollTo=N0LzZ-8ZtxP0  
결과가 생각보다 안나옴
Why? 영어 RoBERTA의 vocab size와 klue/RoBERTA의 vocab size mismatch로 추측...  

## vocab 수정
기존 ETRI의 KorBERT의 경우 vocab size가 30349였다. 이를 모두의 말뭉치로 해서 vocab에 없을 경우 추가했다.  
vocab size를 키웠을 경우 성능을 비교해 보기  
형태소분석해서 vocab에 추가하는걸 KLTagger로 해서 성능 비교해 보기  

### KoELECTRA 
KoELECTRA tokenizer를 KorBERT형태로 바꾸고 입력이 들어왔을 경우 Mecab으로 형태소 붙여서 변하게 만듬  
model embedding layer에 POS embedding layer 추가  
작동해볼려고 했지만 TPU를 GCP에 돌려야 하는데 제대로 안됨..  
***Exception: Parameter "name" value does not match the pattern "^projects/[^/]+/locations/[^/]+/nodes/[^/]+$"***
