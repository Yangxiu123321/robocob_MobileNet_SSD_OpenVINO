#!/bin/bash
#echo  "123123"|sudo ./run.sh

#开始保存日志文件


#sudo -S ./run.sh << EOF 
#123123
#EOF
#script screen.log

# 通过命令打开终端
# sudo -S ./run.sh > mod.txt << EOF
sudo -S ./run_ssd.sh << EOF
123123
EOF
#结束保存日志文件
exit
