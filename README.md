# DynamicBEV
 


 ## Train
启动训练，有三种方式：  
  1. python train.py  
  2. python train.py  -- your-params.json地址
  3. python search_hyperparameters.py

## Test
启动测试，有两种方式：
  1. python evaluate.py  
  2. python evaluate.py  -- your-params.json地址

## Debug
调试不可用search_hyperparamters.py起，用train.py和evaluate.py

## Description
  1. 方式1启动时，用指定params启动训练或测试时，需要在params.json里指定所有需要用到的参数，  
例如数据集地址：experiments/params.json-->data_dir
  2. 方式2启动时，需要在parameters.py里指定所有需要设置的参数，它是去覆盖（和增加）已有的默认experiments/params.json参数。
  3. 方式3启动时，是通过search_pyperparamters.py调用train.py，它至少需要传入 gpu_used/model_dir/exp_name/tb_path和exp_id参数

## Others
  (1). supervised & unsupervised 即有监督和无监督训练，用参数model_train_type指定  
  (2). 所用摄像头个数（数据集读几路摄像头数据）用参数camera_list指定  
  (3). 参数dataloader_task_mode用来指定读数据时读哪些需要的参数。目前为了兼容任意切换有监督和无监督，采用默认的配置：全读。

## Model
  backbone架构采用分类网络  
  若是无监督训练，则有第二阶段，需要在第一阶段模型backbone推理外，再执行。  
  第一阶段输出 预测的点的偏移  
  第二阶段输出 warp后的图 

## Dataset
  暂时放在/home/data/lwb/data/dybev/下  
  (1). v1目录，是用去畸变图上做扰动造的数据集  
  (2). v2目录，是用鸟瞰图上做扰动造的数据集  
  但v1和v2都是每个摄像头只用一张输入图造的，从v3开始是多张输入图来造  
  造数据脚本dataset/data_maker.py  
  (3). v3  

## TensorboardX  
`tensorboard --logdir_spec exp_1:add_1,exp_2:add_2 --bind_all --port xxxx`  
`tensorboard --log_dir=tf_log_path`  


## Homography
P1 -> P2 -> P3, warp img1 to img3  
H_13 = H_23 @ H_12  
warp(H_13, img1) => img3  
because H@p = p':  
H<u>n</u> @ H<u>n-1</u> @ H<u>n-2</u> @ ... @ H<u>2</u> @ H<u>1</u> @ P = P'   
==> H<u>n</u> @ H<u>n-1</u> @ H<u>n-2</u> @ ... @ (H<u>2</u> @ (H<u>1</u> @ P)) = P'   
from right to left ←, not left to right, 反向找点  


H_b2u_pred = H_p2u_pred @ H_b2p_pred # b to p to u  
H_u2b_pred = torch.inverse(H_b2u_pred) , or   
H_u2b_pred = H_p2b_pred @ H_u2p_pred  # u to p to b  
运算顺序：从右往左，左连乘矩阵  


Homograpy, 是warp过去，但是却是倒着找点

## Unit test
For net, `python model/net_unit_test.py`, the result is under the dir of 'model': `vis`


## Net second stage pipeline

1. `train & test: undist -> bev perturbed -> bev pred`  
(1). data['homo_u2b'] = data['H_bev_pert_pred_to_origin'] @ data['H_undist_to_bev_pert']  
(2). data['coords_bev_origin_pred'] = data["coords_bev_perturbed"] - data['offset_pred']  
　　data['homo_u2b'] = dlt_homo(data["coords_undist"], data['coords_bev_origin_pred'])  
*** result(1) != result(2) 
2. `inference:  undist -> bev origin -> bev pred`  
data['homo_u2b'] = data['H_bev_pert_pred_to_origin'] @ data['H_undist_to_bev_origin']
