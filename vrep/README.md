* DDPG_version  
>>>--用來測試的版本 
* sac_version 
>>>--用來測試的版本
* DDPG_good_version  
>>>--訓練三軸到固定點 
* sac_good_version
>>>--訓練三軸到隨機點 
* secene 是場景

### git 指令

1.git clone 到自己的git  
2.cd要修改的資料夾  
3.git status 查看有沒有檔案沒更新到  
4.git add <<"檔名">>  
5.git commit -m"增加說明"  
6.git log查看歷史修正  
7.git remote -v差看可以push到的位置  
8.git push origin將修改完的上傳到git  

*******如果git push 不上去*******  
線上版本比電腦的新  
sol:先拉在推 git pull --rebase
