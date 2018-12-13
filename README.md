## japan_sentence_generator

- 목적 : 고객의 질문(트위터 또는 사이트 질문사항)에 응답해주는 직원들의 타이핑 수를 줄여주어 업무 효율을 증대시키기 위해(은근 타자를 많이 치심...)

- 설명 : 일본어 corpus를 학습하여 문맥을 이해하고, 학습된 정보를 바탕으로 문장을 생성하고 이를 추천해준다.(일본어를 요구하셔서 일본어만 전처리하여 학습하였음. 추가 언어는 필요하신 분이 있을 때, 전처리하여 학습 예정입니다.)

- 작업환경 : 
1. python 3.6
2. tensorflow 1.12

### 전처리만 실행
input data와 output data를 argument로 지정해줘야 합니다.
그러면 output data로 정제된 data가 생성됩니다.

```
python japan_corpus_preprocessing.py --input data\sns_text.txt --output data\re_sns_text.txt
```
전처리는 문단으로 input data가 들어가는 것을 문장으로 바꿔주기 위해 사용했습니다.

또한 트위터나 개인 사이트가 input data에 들어갈 때에 성능이 저하되기 때문에 의미없는 데이터를 정제하였습니다.
(정제가 부족하더라도 이해해주세요... 프로젝트 시간이 얼마 없어서 일본어 corpus 분석을 많이 못했어요...)

