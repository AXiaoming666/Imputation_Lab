import os
import subprocess


def check_if_run(args: dict) -> bool:
    result_path = f"./results/{args['dataset']}/"
    if args["missing_rate"] == 0:
        if not os.path.exists(result_path + "pred_true.npy"):
            return False
        result_path += f"ground/{args['forecast_model']}/"
        if not os.path.exists(result_path + "pred.npy"):
            return False
        if not os.path.exists(result_path + "pred_metrics.npy"):
            return False
    else:
        result_path = "./results/{}/{}_{}_{}_{}/".format(
            args["dataset"],
            args["missing_rate"],
            args["missing_type"],
            args["complete_num"] if args["dataset"] != "traffic" else args["complete_rate"],
            args["imputer"]
        )
        if not os.path.exists(result_path + "imputed_set.npy"):
            return False
        if not os.path.exists(result_path + "imputed_metrics.npy"):
            return False
        result_path += args["forecast_model"] + "/"
        if not os.path.exists(result_path + "pred_metrics.npy"):
            return False
        if not os.path.exists(result_path + "pred.npy"):
            return False
    return True

def run(args: dict) -> None:
    command = ["/home/cxz/anaconda3/envs/myenv/bin/python", "run.py",
               "--dataset", args["dataset"],
               "--forecast_model", args["forecast_model"]]
    if args["missing_rate"] != 0:
        command += ["--missing_rate", str(args["missing_rate"]),
                    "--missing_type", args["missing_type"],
                    "--imputer", args["imputer"]]
        if args["dataset"] != "traffic":
            command += ["--complete_num", str(args["complete_num"])]
        else:
            command += ["--complete_rate", str(args["complete_rate"])]
    else:
        command += ["--missing_rate", str(args["missing_rate"])]
    
    subprocess.run(command, check=True)

if __name__ == "__main__":
    args = {}
    
    args["missing_rate"] = 0
    for dataset in ["exchange_rate", "illness", "traffic"]:
        args["dataset"] = dataset
        for forecast_model in ["TimesNet", "iTransformer", "TimeXer"]:
            args["forecast_model"] = forecast_model
            if check_if_run(args):
                print("Already run")
            else:
                run(args)
                print("Run successfully")
    
    args["dataset"] = "exchange_rate"
    for missing_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        args["missing_rate"] = missing_rate
        for missing_type in ["MCAR", "MAR", "F-MNAR", "D-MNAR"]:
            args["missing_type"] = missing_type
            for complete_num in [0, 1, 2, 3, 4, 5, 6, 7]:
                args["complete_num"] = complete_num
                for imputer in ["Mean", "Forward", "KNN", "XGBoost", "IIM"]:
                    args["imputer"] = imputer
                    for forecast_model in ["TimesNet", "iTransformer", "TimeXer"]:
                        args["forecast_model"] = forecast_model
                        if check_if_run(args):
                            print("Already run")
                        else:
                            run(args)
                            print("Run successfully")
    
    args["dataset"] = "illness"
    for missing_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        args["missing_rate"] = missing_rate
        for missing_type in ["MCAR", "MAR", "F-MNAR", "D-MNAR"]:
            args["missing_type"] = missing_type
            for complete_num in [0, 1, 2, 3, 4, 5, 6]:
                args["complete_num"] = complete_num
                for imputer in ["Mean", "Forward", "KNN", "XGBoost", "IIM"]:
                    args["imputer"] = imputer
                    for forecast_model in ["TimesNet", "iTransformer", "TimeXer"]:
                        args["forecast_model"] = forecast_model
                        if check_if_run(args):
                            print("Already run")
                        else:
                            run(args)
                            print("Run successfully")
    
    args["dataset"] = "traffic"
    for missing_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        args["missing_rate"] = missing_rate
        for missing_type in ["MCAR", "MAR", "F-MNAR", "D-MNAR"]:
            args["missing_type"] = missing_type
            for complete_rate in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                args["complete_rate"] = complete_rate
                for imputer in ["Mean", "Forward", "KNN", "XGBoost", "IIM"]:
                    args["imputer"] = imputer
                    for forecast_model in ["TimesNet", "iTransformer", "TimeXer"]:
                        args["forecast_model"] = forecast_model
                        if check_if_run(args):
                            print("Already run")
                        else:
                            run(args)
                            print("Run successfully")