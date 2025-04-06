#关于网络模型（所有模型位于models文件夹下，）：
1.shellnet：只保留了分类代码，注释掉分割代码，如果使用分割代码，自行开启即可。
2.pointnet_cls:是pointnet的分类代码。
3.pct:是清华大学提出的point cloud transformer。
4.pointnet2_msg和pointnet2_ssg是pointnet++提出的两种架构。



# -----如果电脑系统windows11，cuda选择cuda_12.5.1_555.85_windows。cudnn选择的是windows-x86_64-8.9.7.29_cuda12-archive。pytorch只能使用2.3.0 
# condarc文件内容
channels:
  - defaults
show_channel_urls: true
default_channels:
    - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
    - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
    conda-forge: http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
    msys2: http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
    bioconda: http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
    menpo: http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
    pytorch: http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
    pytorch-lts: http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
    simpleitk: http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
    deepmodeling: http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/
ssl_verify: false
    
#1 安装GPU版本的pytorch，使用梯子下载 
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
#2 安装可视化的进度条 库
pip install tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple

#3 安装处理h5数据的库
pip install h5py -i https://pypi.tuna.tsinghua.edu.cn/simple

#4 查看参数量和FLOPs
pip install thop -i https://pypi.mirrors.ustc.edu.cn/simple/

#5 安装sklearn，用于计算总体准确率和平均准确率
pip install scikit-learn -i https://pypi.mirrors.ustc.edu.cn/simple/

#6.安装 tensorboardX，用于查看迭代的波动
pip install tensorboardX -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple



# -----如果电脑系统windows10
之前的镜像源不好安装各种pytorch，如果想要安装对应的pytorch可以采用，分别下载pytorch、torchvision和cudatoolkit的版本。
例如：显卡是3060，使用的cuda版本是11.6（cuda_11.6.2_511.65_windows.exe） 使用的cudnn（cudnn-windows-x86_64-8.9.7.29_cuda11-archive），需要安装pytorch的1.13.1版本
但使用pytorch中的conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia命令，无法下载。则可以使用以下方式
打开 https://download.pytorch.org/whl/torch_stable.html网址，分别下载pytorch==1.13.1，torchvision==0.14.1，torchaudio==0.13.1的wheel文件，CUDA11.7版本，
包名后面的数字是包的版本号，cp37指的是python版本号，win指的是Windows系统，linux系统选另一个。下载完成后，打开conda的环境，使用pip install 文件夹名称.whl

#1 安装GPU版本的pytorch
conda install pytorch==1.6.0 cudatoolkit=10.1

## 2 安装可视化的进度条 库
pip install tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple

#3 安装处理h5数据的库
pip install h5py -i https://pypi.tuna.tsinghua.edu.cn/simple

#4 查看参数量和FLOPs
pip install thop -i https://pypi.mirrors.ustc.edu.cn/simple/

#5 安装sklearn，用于计算总体准确率和平均准确率
pip install scikit-learn -i https://pypi.mirrors.ustc.edu.cn/simple/

#6.安装 tensorboardX，用于查看迭代的波动
pip install tensorboardX -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple



#-----如果使用linux系统
第一步：安装conda软件
 1. 进入conda文件所在的目录（例如conda文件名  Anaconda3-2021.05-Linux-x86_64.sh）
 2. shell终端进如该文件的的目录 赋予conda文件的读取权限，chmod +x Anaconda3-2021.05-Linux-x86_64.sh      
 3. 依次添加命令 ./Anaconda3-2021.05-Linux-x86_64.sh 
 4. 点击Enter（回车键） 一直到输入yes 开始安装
 5. 安装完成后 会提示，Anaconda will now be installed into this location? 这是配置环境变量的意思，输入 yes
 6. 此时conda已经安装完成，需要关闭shell的窗口，重新进入 ，然后输入 conda info -e 查看虚拟环境。如果有base环境，说明conda已经安装成功
第二步：配置国内镜像源，使得库的下载速度更快。
 1.创建condarc        conda config --set show_channel_urls yes
 2.在终端窗口输入vim ~/.condarc       将里面的文件全部删除，然后将下面的镜像源复制到窗口中

注：镜像源可能失效
channels:
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/menpo/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
show_channel_urls: true

 3.将鼠标定位到窗口中，然后按esc键，输入  :wq!   进行强制保存文件
  
#1 安装GPU版本的pytorch
conda install pytorch==1.6.0 cudatoolkit=10.1

#2 安装可视化的进度条 库
pip install tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple

#3 安装处理h5数据的库
pip install h5py -i https://pypi.tuna.tsinghua.edu.cn/simple

#4 查看参数量和FLOPs
pip install thop -i https://pypi.mirrors.ustc.edu.cn/simple/

#5 安装sklearn，用于计算总体准确率和平均准确率
pip install scikit-learn -i https://pypi.mirrors.ustc.edu.cn/simple/

#6. 如果需要查看训练的状态，请安装 tensorboardX，用于查看迭代的波动
pip install tensorboardX -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple

#额外需要其他库的，选择下面
#pip install hydra-core --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple
#pip install omegaconf -i https://pypi.tuna.tsinghua.edu.cn/simple

#国内镜像安装opencv
#pip install opencv-contrib-python -i https://pypi.mirrors.ustc.edu.cn/simple/

# 点云数据集样本可视化方式 Visualization
（1）Using show3d_balls.py

（2）使用mayavi库和依赖库
#pip install mayavi -i https://pypi.tuna.tsinghua.edu.cn/simple
#pip install pyqt5 -i https://pypi.tuna.tsinghua.edu.cn/simple

（3）使用open3D
pip install open3d



# S3SISD分割数据集使用方法
（1）在飞浆下载S3DISD数据集m然后数据全部放在data/s3dis/Stanford3dDataset_v1.2_Aligned_Version/ 要看清楚是在data下新建s3dis文件。
（2）进入data_utils文件夹，执行collect_indoor3d_data.py，将在data/s3dis/Stanford3dDataset_v1.2_Aligned_Version/生成npy文件，应该把这些npy文件全部放到/data/stanford_indoor3d路径下。
（3）Area_5/hallway_6中多了一个额外的字符，不符合编码规范，需要手动删除。经查找具体位置为：Stanford3dDataset_v1.2_Aligned_Version\Area_5\hallway_6\Annotations\ceiling_1.txt中的第180389行数字185后（百度搜索）

```
## windows参考https://blog.csdn.net/qq_38939905/article/details/121961058
（1）下载VS2019，新建DLL项目，设置头文件和源文件，生成相应python解释器的编码。
```

