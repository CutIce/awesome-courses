# !nvidia-smi

# 挂载google drive
from google.colab import drive
drive.mount('/content/drive/')

# 更改目录
import os
os.chdir("tmp")

# 训练数据移动
!cp /content/drive/Sharedrives/deeplearning_data/conternet-pytorch.zip  /content
!unzip -d /content/centernet


# 防止断联
# 在colab页面 按下F12或者Ctrl+Shift+I
# 在console输入以下代码并回车
# 关闭浏览器需要重新设置
function ConnectionButton(){
    console.log("Connect pushed");
    document.querySelector("#connect").click()
}
setInterval(ConnectionButton, 60000);


# 占用1个g的gpu
import torch
a = torch.Tensor([1000, 1000, 1000]).cuda()