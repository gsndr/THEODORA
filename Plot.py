import matplotlib.pyplot as plt





def printPlotLoss(history,d, path):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(path+"plotLossAIDA" + str(d) + ".png")
    plt.close()
    # plt.show()


def printPlotAccuracy(history, d ,path):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(path+"plotAccuracy" + str(d) + ".png")
    plt.close()

def plotCluster(df, pathPlot, cls):
    df_normal = df[(df[cls] == 1)]
    normal = df_normal['Prob1'] >= 0.50
    df_normal = df_normal[normal]
    l = list(range(df_normal.shape[0]))
    print(len(l))
    df_normal = df_normal.reindex(l)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.xlabel("Samples", fontsize=18)
    plt.ylabel("Confidence", fontsize=18)

    plt.scatter(df_normal.index, df_normal['Prob1'], c='#0066cc', marker=".")

    plt.savefig(pathPlot + 'svc_clusterNormal.png')
    #plt.show()
    plt.close()

def plotClusterChange(df, pathPlot, threshold):
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.xlabel("Samples", fontsize=18)
    plt.ylabel("Confidence", fontsize=18)

    plt.scatter(df.index, df['Prob1'], c='#0066cc', marker=".")

    plt.title('cluster1 - Threshold: %s' % threshold + ' - #LabelChanged: %s' % df.shape[0])
    plt.savefig(pathPlot + 'svc-c1-normal-threshold.png')
    plt.close()
