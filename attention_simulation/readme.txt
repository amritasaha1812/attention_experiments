model1 -> using standard attention
model2 -> using change of variables with Parent_final_score = parent_local_score + sum_{child} softmax(child_final_score)
model3 -> using change of variables with Parent_final_score = parent_local_score + max_{child} softmax(child_final_score)


to create a model, create an empty dir (say model1)
fill the params.json in that model1 dir (example provided in example_model_params.json)
to train: python train.py model1/params.json
