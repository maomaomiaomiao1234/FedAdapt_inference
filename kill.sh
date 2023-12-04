###################################################################
# File Name: kill.sh
# Author: whang1234
# mail: @163.com
# Created Time: 2023年12月04日 星期一 14时01分58秒
#=============================================================
#!/bin/bash
sudo lsof -t -i:1998 | xargs -r sudo kill -9
sudo lsof -t -i:1999 | xargs -r sudo kill -9
sudo lsof -t -i:2000 | xargs -r sudo kill -9

