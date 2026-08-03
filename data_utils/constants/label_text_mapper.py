SEED_onehot = {
    (1, 0, 0): "sad",
    (0, 1, 0): "neutral",
    (0, 0, 1): "happy",
}


SEED = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

# SEED = {
#     0: "happy",
#     1: "neutral",
#     2: "sad",
# }

# SEED = {
##乱序
#     0: "neutral",
#     1: "sad",
#     2: "happy",
# }

# SEED = {
#     #无语义关联组合1
#     0: "volcano", #火山
#     1: "teaspoon",  #茶匙
#     2: "butterfly",   #蝴蝶
# }

# SEED = {
#     #无语义关联组合2
#     0: "glacier", #火冰川
#     1: "keyboard ",  #键盘
#     2: "cactus",   #仙人掌
# }

# SEED = {
#     #无语义关联组合3
#     0: "asteroid", #小行星
#     1: "toothbrush ",  #牙刷
#     2: "jellyfish",   #水母
# }




SEEDiv = {
    0: "neutral",
    1: "sad",
    2: "fear",
    3: "happy"
}

SEEDv = {
    0: "disgust",
    1: "fear",
    2: "sad",
    3: "neutral",
    4: "happy",
}

FACED_onehot = {
    (1, 0, 0, 0, 0, 0, 0, 0, 0): "anger",
    (0, 1, 0, 0, 0, 0, 0, 0, 0): "disgust",
    (0, 0, 1, 0, 0, 0, 0, 0, 0): "fear",
    (0, 0, 0, 1, 0, 0, 0, 0, 0): "sadness",
    (0, 0, 0, 0, 1, 0, 0, 0, 0): "neutral",
    (0, 0, 0, 0, 0, 1, 0, 0, 0): "amusement",
    (0, 0, 0, 0, 0, 0, 1, 0, 0): "inspiration",
    (0, 0, 0, 0, 0, 0, 0, 1, 0): "joy",
    (0, 0, 0, 0, 0, 0, 0, 0, 1): "tenderness",
}

FACED = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "sadness",
    4: "neutral",
    5: "amusement",
    6: "inspiration",
    7: "joy",
    8: "tenderness",
}


DEAP= {
    0: "negative",
    1: "positive"
}


DREAMER={
    0: "negative",
    1: "positive"
}


LabelMapper = {
    "seed": SEED,
    "seed_onehot": SEED_onehot,
    "seediv": SEEDiv,
    "seedv": SEEDv,
    "faced": FACED,
    "faced_onehot": FACED_onehot,
    "deap": DEAP,
    "dreamer":DREAMER
}

def getLabelMapper(dataset="seed", onehot=False):
    if dataset.startswith("faced"):
        return LabelMapper[dataset[:5]+"_onehot"] if onehot else LabelMapper[dataset[:5]]
    elif dataset.startswith("seediv"):
        return LabelMapper[dataset[:6] + "_onehot"] if onehot else LabelMapper[dataset[:6]]
    elif dataset.startswith("seedv"):
        return LabelMapper[dataset[:5] + "_onehot"] if onehot else LabelMapper[dataset[:5]]
    elif dataset.startswith("dreamer"):
        return LabelMapper[dataset[:7] + "_onehot"] if onehot else LabelMapper[dataset[:7]]
    else:
        return LabelMapper[dataset[:4] + "_onehot"] if onehot else LabelMapper[dataset[:4]]  # deap
