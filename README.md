## TODO plan for Unet
1. config      # not a good way to do this
2. unet from scratch        # unet 
3. dataloader  # add basic dataset class for segmentations
4. loss         
5. optimizers
7. training  # 1. train step 2. loader 3. save checkpoint 4. save train and val loss (todo : log)

8. ViT from scratch 
9. ViT from scratch # qvk not clear
10. add cityscapes dataset for segmentation, can run but net output and target shape confused

11. loss function pytorch bug # https://clay-atlas.com/blog/2020/05/16/pytorch-cn-error-runtimeerror-ambiguous/
12. running successfully!!  # total label 35， **solve CUDA ERROR: device-side assert triggered at**  problem, reason:number of classes and network out of classes is not identical. ![loss](/imgs/first_run.png)
13. loss is decreasing but very slowly, accurary is not right!,more evaluation.
14. 200 epoch without too much data arguementation ![loss](/imgs/result.png)
15. add inference
16. inference structure and training structure and trace