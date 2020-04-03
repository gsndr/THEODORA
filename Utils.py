import matplotlib.pyplot as plt

def getResult(cm, N_CLASSES):
    tp = cm[0][0]  # attacks true
    fn = cm[0][1]  # attacs predict normal
    fp = cm[1][0]  # normal predict attacks
    tn = cm[1][1]  # normal as normal
    attacks = tp + fn
    normals = fp + tn
    OA = (tp + tn) / (attacks + normals)
    AA = ((tp / attacks) + (tn / normals)) / N_CLASSES
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F1 = 2 * ((P * R) / (P + R))
    FAR = fp / (fp + tn)
    TPR = tp / (tp + fn)
    r = (tp, fn, fp, tn, OA, AA, P, R, F1, FAR, TPR)
    return r




def getXY(train,  cls):
    clssList = train.columns.values
    target=[i for i in clssList if i.startswith(cls)]

    # remove label from dataset to create Y ds
    train_Y=train[target]
    # remove label from dataset
    train_X = train.drop(target, axis=1)
    train_X=train_X.values


    return train_X, train_Y