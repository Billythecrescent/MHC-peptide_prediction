'''
File: drawCurve.py
Author: Mucun Hou
Date: Apr 28, 2020
Description: This script is for the drawing of curves
'''

import os.path, re
from math import log
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import time, datetime

import PredictionFunction as PF

module_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) #code\MHC-peptide_prediction
current_path = os.path.dirname(os.path.abspath(__file__)) #code\MHC-peptide_prediction\MHCI_BP_predictor
model_path = os.path.join(module_path,"models") #code\MHC-peptide_prediction\models
data_path = os.path.join(module_path,"data") #code\MHC-peptide_prediction\data
result_dir_prefix = os.path.join(current_path, "results")

# this indicates the colors to be used in curve plots
colors = {'mhci1': 'olive', 'mhci3_Random': 'green', 'mhci2': 'cyan',"panExistGeneral_NetMHC": "olive", "panExistGeneral_globalCore": "brown", \
    "panExistSpecific_NetMHC": "magenta", "panExistSpecific_H-2": "darkkhaki", "panExistSpecific_globalCore": "royalblue", "NetMHCpan4": "goldenrod",
          'mhci3_Exist': 'navy', 'NetMHC4': 'orange', "panExistSpecific_HLA": "cyan", "panExistGeneral_HLA": "pink", "NetMHCpan4'": "orange", \
    "NetMHCii": "orange", "mhcii": "navy"}

# this indicates the real name of the method with proper format. this is used in the final plots.
method_names_proper = {'NetMHC4': 'NetMHC-4.0', 'mhci1': 'cMHCi1', "panExistGeneral_NetMHC": "MHCipan_ExistStart_General_NetMHC", \
    "panExistGeneral_globalCore": "MHCipan_ExistStart_General_globalCore", "panExistSpecific_NetMHC": "MHCipan_ExistStart_Specific_NetMHC", \
    "panExistSpecific_H-2": "MHCipan_ExistStart_Specific_H-2", "panExistSpecific_globalCore": "MHCipan_ExistStart_Specific_globalCore", \
    "NetMHCpan4": "NetMHCpan-4.0", 'mhci2': 'cMHCi2', 'mhci3_Exist': 'cMHCi3_ExistStart', 'mhci3_Random': 'cMHCi3_RandomStart', \
    "panExistSpecific_HLA": "MHCipan_ExistStart_Specific_HLA", "panExistGeneral_HLA": "MHCipan_ExistStart_General_HLA", "NetMHCpan4'": "NetMHCpan-4.0_general", \
    "NetMHCii": "NetMHCII-2.3", "mhcii": "cMHCii"}

def locate_y_coordinate_for_specific_x_coordinate(data_df, zoom_level):
    """
    Identify the y coordinate for the specific x coordinate. This is needed in case of zooming in, to avoid breaking the line before the end of the plot area
    Args:
        zoom_level: The x coordinate upto which the plot needs to be zoomed in
    """
    y2 = np.interp(zoom_level, data_df['x_axis_item'], data_df['y_axis_item'])
    return y2

def locate_y_coordinate_for_specific_x_coordinate_for_x_percent(method, item, score_direction, data_sorted, score_at_x_percent_item, percent_value, type):
    """
    Identify the y coordinate for the specific x coordinate. This is for the functions identifying % peptides to capture epitopes/reponse;
    identify % epitopes/response captured by top x peptides etc.
    Args:
        item: epitope/response
        type: to identify % peptides to capture x% epitopes/response or % epitopes/response in top peptides
    """
    if type=='x_percent':
        if score_direction == 'direct':
            x_axis_item_plus_1 = data_sorted.query(method + '_score>' + str(score_at_x_percent_item))['cum_sum_'+item+'_percent'].max()
            x_axis_item_minus_1 = data_sorted.query(method + '_score<' + str(score_at_x_percent_item))['cum_sum_'+item+'_percent'].min()
            y_axis_item_plus_1 = data_sorted.query(method + '_score>' + str(score_at_x_percent_item))['count_percent'].max()
            y_axis_item_minus_1 = data_sorted.query(method + '_score<' + str(score_at_x_percent_item))['count_percent'].min()
        else:
            x_axis_item_plus_1 = data_sorted.query(method + '_score>' + str(score_at_x_percent_item))['cum_sum_'+item+'_percent'].min()
            x_axis_item_minus_1 = data_sorted.query(method + '_score<' + str(score_at_x_percent_item))['cum_sum_'+item+'_percent'].max()
            y_axis_item_plus_1 = data_sorted.query(method + '_score>' + str(score_at_x_percent_item))['count_percent'].min()
            y_axis_item_minus_1 = data_sorted.query(method + '_score<' + str(score_at_x_percent_item))['count_percent'].max()
    else:
        if score_direction == 'direct':
            y_axis_item_plus_1 = data_sorted.query(method + '_score>' + str(score_at_x_percent_item))['cum_sum_' + item + '_percent'].max()
            y_axis_item_minus_1 = data_sorted.query(method + '_score<' + str(score_at_x_percent_item))['cum_sum_' + item + '_percent'].min()
            x_axis_item_plus_1 = data_sorted.query(method + '_score>' + str(score_at_x_percent_item))['count_percent'].max()
            x_axis_item_minus_1 = data_sorted.query(method + '_score<' + str(score_at_x_percent_item))['count_percent'].min()
        else:
            y_axis_item_plus_1 = data_sorted.query(method + '_score>' + str(score_at_x_percent_item))['cum_sum_' + item + '_percent'].min()
            y_axis_item_minus_1 = data_sorted.query(method + '_score<' + str(score_at_x_percent_item))['cum_sum_' + item + '_percent'].max()
            x_axis_item_plus_1 = data_sorted.query(method + '_score>' + str(score_at_x_percent_item))['count_percent'].min()
            x_axis_item_minus_1 = data_sorted.query(method + '_score<' + str(score_at_x_percent_item))['count_percent'].max()
    if np.isnan(x_axis_item_minus_1):
        x_axis_item_minus_1 = 1.0
    if np.isnan(y_axis_item_minus_1):
        y_axis_item_minus_1 = 1.0
    if np.isnan(x_axis_item_plus_1):
        x_axis_item_plus_1 = 1.0
    if np.isnan(y_axis_item_plus_1):
        y_axis_item_plus_1 = 1.0
    x_axis_item2 = [x_axis_item_minus_1, x_axis_item_plus_1]  # x_axis_item 1 point above & below the needed x_axis_item
    y_axis_item2 = [y_axis_item_minus_1, y_axis_item_plus_1]  # y_axis_item 1 point above & below the needed x_axis_item

    y2 = np.interp(percent_value/100, sorted(x_axis_item2), sorted(y_axis_item2))

    return y2

def make_curves(methods, zoom_or_full, x_axis_type, data_for_curves_dict, title, plot_file_name):
    """
    Make curves - ROC curve from FPR/TPR data, % response vs % peptides etc.
    Args:
        zoom_or_full: shows whether normal full length plot or zoomed in version
        x_axis_type: FPR (for ROC curves) or % peptides (for response based curves)
        title: plot title
    Returns:
        Generates figures
        2.2.roc_curves_binary_classification_based.png
        2.2.roc_curves_binary_classification_based_selected_methods.png
        2.2.roc_curves_binary_classification_based_zoomed_in.png
        2.2.roc_curves_binary_classification_based_zoomed_in_selected_methods.png
        3.2.curves_response_based.png
        8.2.roc_curves_mass_spec_and_netmhcpan_4_l.png
    """
    plt.figure(num=None, figsize=(10, 8), dpi=300)
    zoom_level = 0.02
    if zoom_or_full == 'full':
        plt.plot((0.0, 1.0), (0.0, 1.0), ls="--", c=".3", label='Random (0.500)')
    else:
        plt.plot((0.0, zoom_level), (0.0, zoom_level), ls="--", c=".3", label='Random (0.500)')
    for i in range(len(methods)):
        method = methods[i]
        color = colors[method]
        x_axis_item = data_for_curves_dict[method]['x_axis_item']
        y_axis_item = data_for_curves_dict[method]['y_axis_item']
        result_auc = data_for_curves_dict[method]['result_auc']
        count_of_x_axis_items_at_zoom_level = len([x for x in x_axis_item if x <= zoom_level])
        x_y_df = pd.DataFrame(data={'x_axis_item': x_axis_item, 'y_axis_item': y_axis_item})
        if zoom_or_full == 'full':
            plt.plot(x_axis_item, y_axis_item, color=color, label='%s (%0.3f)' % (method_names_proper[method], result_auc))
            plt.xticks(np.arange(0.0, 1.1, 0.1), fontsize=12)
        else: # here it's zoomed in version
            y_coordinate_for_zoom_level = locate_y_coordinate_for_specific_x_coordinate(x_y_df, zoom_level)
            x_axis_item_new = x_axis_item[:count_of_x_axis_items_at_zoom_level]
            y_axis_item_new = y_axis_item[:count_of_x_axis_items_at_zoom_level]
            x_axis_item_new.append(zoom_level)
            y_axis_item_new.append(y_coordinate_for_zoom_level)
            plt.plot(x_axis_item_new, y_axis_item_new, color=color, label='%s (%0.3f)' % (method_names_proper[method], result_auc))
        if x_axis_type == 'fpr':
            plt.xlabel('False positive rate', fontsize=12)
            plt.ylabel('True positive rate', fontsize=12)
            plt.yticks(np.arange(0.0, 1.1, 0.1), fontsize=12)
        else: # here it's % peptides
            plt.xlabel('% peptides', fontsize=12)
            plt.ylabel('% response', fontsize=12)
            plt.gca().set_yticklabels(["{0:.0f}%".format(x*100) for x in np.arange(0.0, 1.1, 0.1)], fontsize=12)
            plt.yticks(np.arange(0.0, 1.1, 0.1), fontsize=12)
    plt.ylim(-0.05, 1.05)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], title='Methods (AUC)', loc='lower right')
    plt.title(title)
    plt.savefig(result_dir_prefix+plot_file_name)
    plt.close()
    print('->', plot_file_name, u'\u2713')

def draw():
    # dataName = "VACV"
    dataName = "Tumor"
    # methods = ["mhci1", "mhci2", "mhci3_Random", "mhci3_Exist", "NetMHC4"]
    # methods = ["panExistGeneral_NetMHC", "panExistGeneral_globalCore", "panExistGeneral_HLA", "panExistSpecific_NetMHC", \
    #     "panExistSpecific_HLA", "panExistSpecific_globalCore", "NetMHCpan4", "NetMHCpan4'"]
    methods = ["NetMHCii", "mhcii"]
    data_for_curve_dict = {}
    prediction_path = os.path.join(current_path, "prediction_data")
    for method in methods:
        file_path = os.path.join(prediction_path, method+'_'+dataName+'_result.csv')
        dataset = pd.read_csv(file_path)
        # dataset = dataset.loc[dataset['immunogenicity'] != 'minor']
        labels = dataset.binder
        predicted_scores = dataset.log50k
        dic = PF.GenerateDataForCurveDict(method, labels, predicted_scores)
        data_for_curve_dict.update(dic)
    make_curves(methods, 'full', 'fpr', data_for_curve_dict, 'MHCii ROC curves (epitope binary classification based; all methods)','roc_curves_binary_classification_based.png')
    make_curves(methods, 'zoom', 'fpr', data_for_curve_dict, 'MHCii ROC curves (epitope binary classification based; all methods; zoomed-in to FPR = 0.02)','roc_curves_binary_classification_based_zoomed_in.png')


draw()