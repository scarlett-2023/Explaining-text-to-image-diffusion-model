import subprocess
print("Step 1: 分析高importance虚词...")
subprocess.run(["python", "importance_analysis.py"])
print("Step 2: Finetune多模型...")
subprocess.run(["python", "finetune_multi_model.py"])
print("Step 3: 多模型评估与报告生成...")
subprocess.run(["python", "eval_multi_model_pipeline.py"])
print("全部完成！请查阅 full_pipeline_outputs/ 下的报告与可视化。")