python train_qua.py pixel_obs=false action_repeat=1 task=quadruped_run agent=ddpg_rotate aug_ratio=4  seed=1


train_xx.py: the run file for each task
task=xxx : the task name. Available: 
                     1.quadruped_run 2.reacher_hard 3.cheetah_run 4.cheetah3d_run
                     5.hopper_hop 6.hopper3d_hop 7.Humanoid_stand 8.humanoid_run
                     9.walker_run 10.walker3d_run
agent=xxx:  the method you will use: 
                    1.ddpg_our: the original DDPG
                    2.ddpg_rotate:  the DDPG + aug
                    3.ddpg_rad:  DDPG + RAS
                    4.ddpg_guass: DDPG + GN

aug_ratio: The rotate ratio in the batch 
                    0:no rotate. 
                    1:100% rotate. 
                    2:50% rotate 
                    3:50% rotate 
                    4:25% rotate  
                                      
The The results will be saved at ./exp



The code is adapted from "Continuous MDP Homomorphisms and Homomorphic Policy Gradient" by Sahand Rezaei-Shoshtari, Rosie Zhao, Prakash Panangaden, David Meger, and Doina Precup, presented at the Advances in Neural Information Processing Systems (NeurIPS) conference in 2022. We gratefully acknowledge their significant contributions to this field.
