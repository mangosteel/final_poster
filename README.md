https://like-grapejuice.tistory.com/633 # 여기 참고해서 github 파일 받고 오류 고치기! 

# 기존에는 없던 imu센서 추가및 센서 데이터 받아오는 코드 만듬.. 그리고 actor , critic loss를 step/epi별로 출력하도록 수정함 (600step = 1epi)

# 그리고 기존에 존재하던 nvidia localhost가 만들어지던 nucles가 25년도로 없어질 예정이라 hub workstation으로 변경해야한다.

# 어차피 10월 1일에 공식적으로 nucelus서버가 삭제되지만, 지금은 로컬서버를 uninstall 하고 다시 깔면 되긴함. 하지만 불안정함!

https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html # 패치 url

![Screenshot from 2025-02-03 15-16-32](https://github.com/user-attachments/assets/5eb9eab5-63ef-4548-9b3d-15cbe5ec6c00)
그리고 여기서 허브 파일을 다운받고 파일을 압축해제한 다음  script 폴더로 가서 install.sh 를 실행시킨다. 이때 명령어는 ./install.sh 이다

그렇게 되면.. isaacsim파일이 설치되어 있는 pkg 폴더로 hub폴더가 다운로드 되고 이런다음 isaccsim을 실행시키면 cache : hub가 뜬다. 그러면 localhost서버가 이전과는 다르게 활성화 되어있다.


![허브 된듯?](https://github.com/user-attachments/assets/4eb17f52-72e9-4719-a4e2-c5d01269fedc)

![ngc에서  에서 허브파일받고 install sh하면 아이작심에 설치됨 굳이 확장자 필요없음](https://github.com/user-attachments/assets/1172380a-cc1f-492b-ac4f-d7c86a6d0d81)

![오케이 기존의 neclues서버를 hub로 대체](https://github.com/user-attachments/assets/a481dce0-867c-4d28-95c9-15f0e26417e3)

# 이런식으로 터미널에출력된다
![조금 더 예븐 터미널 출력값?](https://github.com/user-attachments/assets/e63de6cf-2c1d-47d9-9b2f-56cfd12f5848)

