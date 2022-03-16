"""Data reading and printing utils."""

from texttable import Texttable
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from matplotlib.pyplot import figure

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def create_numeric_mapping(node_properties):
    """
    Create node feature map.
    :param node_properties: List of features sorted.
    :return : Feature numeric map.
    """
    return {value:i for i, value in enumerate(node_properties)}

def loss_plot_write(write_path,list_loss,type='train_MSE'):
    prefix = "graphSample-500-" #+str(len(list_loss))#data_path.split("/")[-1]

            # for x,y in zip(xs,ys):

    plt.plot(list_loss, 'r-')

    plt.title("%s_%s_BC " % (prefix,type), fontsize=18)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    # label = ["k=(5,10)", "k=(10,15)", "k=(15,5)", "k=(5,rank)", "k=(10, rank)", "k=(15,rank)"]
    # plt.annotate(label[x-1], # this is the text
    # (x,y),# these are the coordinates to position the label
    # textcoords="offset points", # how to position the text
    # xytext=(0,10), # distance from text to points (x,y)
    #  ha='center') # horizontal alignment can be left, right or center
    # plt.legend(loc="upper left")
    # plt.xticks(xs, label, fontsize=13)
        #plt.show()
    plt.savefig("%s%s_%s_loss_plot.png"%(write_path, prefix, type), dpi=300, bbox_inches='tight')
    plt.close()

# def loss_write():
#      if mse_test<best_mse_test:
#         file = open("FasBetModel.pickle","wb")
#         pickle.dump(model,file)
#         file.close()
#         best_mse_test = mse_test
#
#     prefix = data_path.split("/")[-1]
#     if e % 20 == 0:
#         generate_plots = True
#         if generate_plots:
#             # for x,y in zip(xs,ys):
#
#             plt.plot(test_mse_list, 'r-')
#
#             plt.title("%s_test_BC " % prefix, fontsize=18)
#             plt.xlabel('Epoch', fontsize=16)
#             plt.ylabel('MSE', fontsize=16)
#             # label = ["k=(5,10)", "k=(10,15)", "k=(15,5)", "k=(5,rank)", "k=(10, rank)", "k=(15,rank)"]
#             # plt.annotate(label[x-1], # this is the text
#             # (x,y),# these are the coordinates to position the label
#             # textcoords="offset points", # how to position the text
#             # xytext=(0,10), # distance from text to points (x,y)
#             #  ha='center') # horizontal alignment can be left, right or center
#             # plt.legend(loc="upper left")
#             # plt.xticks(xs, label, fontsize=13)
#         #plt.show()
#         plt.savefig("%s_test_BC_MSE.png" % prefix, dpi=300, bbox_inches='tight')
#     plt.close()
