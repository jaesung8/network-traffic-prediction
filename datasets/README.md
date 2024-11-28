shoreset path값으로 계산함. 


예를 들어 demandMatrix-abilene-zhang-5min-20040301-0000.txt 파일을 읽어서 처리한다고 하면, 

  ATLAM5_ATLAng ( ATLAM5 ATLAng ) 1 0.522208 UNLIMITED 이면 
  
  ATLAM5 = ATLAM5 + 0.522208
  ATLAng = ATLAng + 0.522208
  
  ATLAM5_CHINng ( ATLAM5 CHINng ) 1 1.641339 UNLIMITED
  
  ATLAM5 = ATLAM5 + 1.641339
  CHINng = CHINng + 1.641339
  
  ATLAM5_DNVRng ( ATLAM5 DNVRng ) 1 0.335728 UNLIMITED
  
  ATLAM5 = ATLAM5 + 0.335728
  DNVRng = DNVRng + 0.335728
  
  ATLAM5_HSTNng ( ATLAM5 HSTNng ) 1 0.413032 UNLIMITED
  
  ATLAM5 = ATLAM5 + 0.413032
  HSTNng = HSTNng + 0.413032
  
  ... 
  ...

  print(ATLAM5 CHINng DNVRng ...)  # 이런식으로 한줄 프린트 됨.
  
  demandMatrix-abilene-zhang-5min-20040301-0005.txt 
  
  demandMatrix-abilene-zhang-5min-20040301-0010.txt
  
  demandMatrix-abilene-zhang-5min-20040301-0015.txt 
  
  ...
