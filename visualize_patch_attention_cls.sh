MODEL_PATH="/home/jinfang/project/new_CarotidPlaqueStabilityClassification/output/output_patch_feature_cls/train_cls_20260203_174531/models/best_model.pth"
# #   可视化单个切片:                                                                                                                                                                                        
# python visualize_patch_attention_cls.py \                                                                                                                                                              
#       --model-path $MODEL_PATH \                                                                                                                      
#       --patient-name "患者名" \                                                                                                                                                                          
#       --slice-idx 50                                                                                                                                                                                     
                                                                                                                                                                                                         
#   可视化该患者所有有patch的切片 (不指定 --slice-idx):                                                                                                                                                    
python visualize_patch_attention_cls.py  --model-path $MODEL_PATH  --patient-name "刁天朝"                                                                                                                  
      