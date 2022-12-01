import neptune.new as neptune

run = neptune.init_run(project="genome/lnc_detection")
run["parameters"] = {"lr": 0.001, "optim": "Adam"} # parameters
run["f1_score"] = 0.66 # metrics
run["roc_curve"].upload("roc_curve.png") # charts
run["model"].upload("model.h5") # models
