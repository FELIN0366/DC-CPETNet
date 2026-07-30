import argparse
_Channel_NAME = {"HF_MetSyn": ['METS', 'HR', 'VO2', 'VO2/kg', 'VCO2', 'RER',
                                   'VE', 'VE/VO2', 'VE/VCO2', 'RR', 'VTex', 'VTin']}
_CLASS_NAME = {"HF_MetSyn": ["HF", "MetSyn"]}
_CLASS_NAME_SHORT = {"HF_MetSyn": ["HF", "MetSyn"]}

_MODEL_HYPER_PARAMS = {
    "HF_MetSyn":{
            "LSTM":{
                "epochs":200,
                "batch_size":5,
                "lr":0.005,
                "hidden_dim":5,
                "n_layer":2
            },
            "ANN":{
                "epochs":200,
                "batch_size":5,
                "lr":0.001,
                "hidden_dim":16,
                "n_layer":2
            },
            "CNN":{
                "epochs":500,
                "batch_size":5,
                "lr":0.001,
                "hidden_dim":32,
                "n_layer":2
            },
            "GCN":{
                "epochs":200,
                "batch_size":5,
                "lr":0.001, # 0.1,
                "hidden_dim":32,
                "n_layer":2
            },
            "GALSTM":{
                "epochs":200,
                "batch_size":5,
                "lr":0.01,
                "hidden_dim":32,
                "n_layer":2
            },
            "GAT":{
                "epochs":2,
                "batch_size":4,
                "lr":5e-3,  # 5e-3
                "hidden_dim":8,
                "n_layer":2
            },
            "STGNnet":{
                "epochs":1000,
                "batch_size":5,
                "lr":0.001, # 0.1,
                "hidden_dim":16,
                "n_layer":2
            },
            "STGnet0606":{
                "epochs":1000,
                "batch_size":5,
                "lr":0.001, # 0.1,
                "hidden_dim":16,
                "n_layer":2
            },
            "STFinalNet":{
                "epochs":50,
                "batch_size":5,
                "lr":0.001, # 0.1,
                "hidden_dim":16,
                "n_layer":1
            }
                }
}


def build_args(model_name=None, dataset="HF_MetSyn"):
    parser = argparse.ArgumentParser(
        "This script is used for the CPET Classification.")
    parser.add_argument("--mode", default="train", type=str)
    parser.add_argument("--model_name", default="GCN", type=str)
    parser.add_argument("--model_file", default=None, type=str)
    args = parser.parse_args()

    args.gpu = 0
    args.dataset = dataset
    args.class_name = _CLASS_NAME_SHORT[args.dataset]
    args.class_name_long = _CLASS_NAME[args.dataset]
    args.class_dict = {action_cls: idx for idx, action_cls in
                       enumerate(args.class_name)}
    if args.dataset == "HF_MetSyn":
        args.data_root = "./HF_MetSyn"
        args.output_root = "./output"
        args.repeat = 2  # 这是什么
        args.part_actions = args.class_name[:]

    args.n_class = len(args.part_actions)
    args.action_dict = {action_cls: idx for idx, action_cls in
                        enumerate(args.part_actions)}
    print("number of class:{}\naction list:{}".format(args.n_class,
                                                      args.part_actions))
    # prepare data
    args.L_win = 162  # 更新为第一段数据统计值
    args.stride = 1
    args.channels = list(range(12))  # choose part of channels
    print("length of window:{}\nstride of window:{}".format(args.L_win,
                                                            args.stride))

    # train_test_subject
    args.subindex = 0
    args.test_ratio = 0.3

    if model_name is not None:
        args.model_name = model_name

    if model_name !="LDA":
        args.epochs=_MODEL_HYPER_PARAMS[args.dataset][args.model_name]["epochs"]
        args.batch_size=_MODEL_HYPER_PARAMS[args.dataset][args.model_name]["batch_size"]
        args.lr=_MODEL_HYPER_PARAMS[args.dataset][args.model_name]["lr"]
        args.hidden_dim=_MODEL_HYPER_PARAMS[args.dataset][args.model_name]["hidden_dim"]
        args.n_layer=_MODEL_HYPER_PARAMS[args.dataset][args.model_name]["n_layer"]

    return args

if __name__ == "__main__":
    build_args("GCN")