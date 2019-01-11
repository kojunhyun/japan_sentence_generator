## japan_sentence_generator

- 목적 : 고객의 질문(트위터 또는 사이트 질문사항)에 응답해주는 직원들의 타이핑 수를 줄여주어 업무 효율을 증대시키기 위해(은근 타자를 많이 치심...)

- 설명 : 일본어 corpus를 학습하여 문맥을 이해하고, 학습된 정보를 바탕으로 문장을 생성하고 이를 추천해준다.(일본어를 요구하셔서 일본어만 전처리하여 학습하였음. 추가 언어는 필요하신 분이 있을 때, 전처리하여 학습 예정입니다.)

- 작업환경 : 
1. python 3.6
2. tensorflow 1.3

### 전처리만 실행
input data와 output data를 argument로 지정해줘야 합니다.
그러면 output data로 정제된 data가 생성됩니다.

```
python japan_corpus_preprocessing.py --input data\sns_text.txt --output data\re_sns_text.txt
```
전처리는 문단으로 input data가 들어가는 것을 문장으로 바꿔주기 위해 사용했습니다.
(문단을 문장으로 나누어 input data에 넣기위한 작업)

또한 트위터나 개인 사이트가 input data에 들어갈 때에 성능이 저하되기 때문에 개인정보같은 부분을 정제하였습니다.
(정제가 부족하더라도 이해해주세요... 프로젝트 시간이 얼마 없어서 일본어 corpus 분석을 많이 못했어요...)



tensorflow 0.12 version → 1.3 version upgrade
data pre-processing(text 정제, NFCK 정규화, SNS long-sentence → short sentence 변환) 전, 후 성능비교
pre-processing 전 : 추천 문자에서 end-of-sentence를 예측을 못하여 문자 무한생성
pre-processing 후 : 짧은 문장으로 추천 문장 return
1개의 추천 문장 → 10개의 추천 문장

- 2017.11.27 ~ 2017.12.01
##### 현재 version 모델에서 입력데이터의 placehold가 없어서 실시간 테스트에서 output를 출력할 수 없는상황
##### 입력데이터의 placehold를 추가하고, 실시간 테스트를 위한 output을 출력하기 위해 api 구조 재구축
##### 현재 요청사항이 있어 문장생성 api version up 중단 후 api 제공을 위한 작업 중

- 2017.12.04 ~ 2017.12.08
### 입력데이터의 placehold를 추가하고, 실시간 테스트를 위한 output을 출력하기 위해 api 구조 재구축 
##### tensorflow 1.3 version에서 사용할 수 있게 tensorflow 함수 및 데이터 shape 파악
##### 기존 input, target data가 class의 private로 정의되어있어 real testing에 문제 발생 (real testing에서는 target이 정의되지 않기 때문)
##### 기존 문장 1개 생성 → 문장 10개 생성
(1) sns 코멘트를 기준으로 생각했을 때, 1개보다는 여러개의 문장을 추천하는 것이 취지에 맞을 것으로 판단
(2) 확률상 높은 순위 10개를 기준으로 next character 예측

<img src=https://github.com/kojunhyun/japan_sentence_generator/blob/master/fig/10sentence_test.PNG>
