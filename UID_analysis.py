import argparse
import sys
import re
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import normaltest, ttest_rel, wilcoxon
from sklearn.preprocessing import StandardScaler
from IPython.display import Markdown, display
# from src.uid import *
from tqdm import tqdm
# import nltk
# nltk.download('averaged_perceptron_tagger')
# nltk.download('averaged_perceptron_tagger_eng')
# from nltk.tag import pos_tag

# Model experiments
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, log_loss
import statsmodels.api as sm
SEED = 17776

# import stanza
# stanza.download('en', download_method=None)
# nlp = stanza.Pipeline(
#     lang="en",
#     processors="tokenize,mwt,pos,lemma,depparse"
# )

import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter, LogFormatter
from matplotlib import rc
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# === Constants === #

context_lvls = ['document', 'sentence']

#=== Metrics & Features ===#

surprisal_metric_names = [
    # 'surp_mean', # Mean surprisal
    'surp_slor', # SLOR
]

uid_metric_names = [
    'uid_std', # Std. of surprisal
    # 'uid_mad', # mean avg. deviation of surprisal
    'uid_pwd', # Local variance (pairwise euc. distance)
    # 'uid_range', # Range of surprisal
    # 'uid_len' # Length of sentence (units)
    # 'uid_slope', # Slope of a linear function fit to surp values
]

surprisal_metrics_formatted = [
    # 'Mean Surp.',
    '-SLOR',
]

uid_metrics_formatted = [
    'Surp. STD',
    # 'Surp. MAD',
    'Local Var.',
    # 'Surp. Range',
    # 'Length',
    # 'LinReg Slope',
]

metrics = surprisal_metric_names + uid_metric_names
metrics_formatted = surprisal_metrics_formatted + uid_metrics_formatted

metrics_mapping = dict(zip(metrics, metrics_formatted))

util_features = [
        'doc_id',
        'sent_idx',
        'context',
        'factual',
        'conversion'
]

ling_features = []
for name in ['agent', 'patient']:
    ling_features.extend([
        # f"{name}",
        f"{name}_len", 
        f"{name}_unigram_logprob",
        f"{name}_is_pronoun",
        f"{name}_is_plural",
        f"{name}_is_animate",
        f"{name}_is_definite",
        # f"{name}_is_prop"
        ])

ling_features_formatted = [
        "Agt Len", 
        "Agt Log Prob",
        "Agt Pronom",
        "Agt Plural",
        "Agt Animate",
        "Agt Definite",
        # "Agt Proper"
        "Pnt Len", 
        "Pnt Log Prob",
        "Pnt Pronom",
        "Pnt Plural",
        "Pnt Animate",
        "Pnt Definite",
        # "Pnt Proper"
]

uid_features = [
        'uid_std',
        # 'uid_mad',
        'uid_pwd',
        # 'uid_range',
        # 'uid_cv'
]

uid_features_formatted = [
        'Surp. STD',
        # 'Surp. MAD',
        'Local Var.',
        # 'Surp. Range',
        # 'Surp. CV', # Coeff. of Variation
]

surp_features = [
        # 'surp_mean',
        'surp_slor'
]

surp_features_formatted = [
        # 'Mean Surp.',
        '-SLOR',
]

format_map = {'const': 'Intercept'}
format_map.update(dict(zip(
        ling_features, ling_features_formatted
)))
format_map.update(dict(zip(
        uid_features, uid_features_formatted
)))
format_map.update(dict(zip(
        surp_features, surp_features_formatted
)))

# === Colormaps === #

factuality_cmap = {
    True: "#7EACB5",
    False: "#BF4646",
}

conversion_cmap = {
    "P to A": "tomato",
    "A to P": "steelblue",
    "Both": "#FFF8DE"
}

context_cmap = {
    "document": "#195C9F",
    "sentence": "#F6C28B",
    "documentA to P": "#195C9F",
    "documentP to A": "#328BAE",
    "sentenceA to P": "#F6C28B",
    "sentenceP to A": "#FDE2C4",
    "other": "#2F3D4C"
}

feature_cmap = {'const': "#3D3B40"}
ling_color = "#11235A"
uid_color = "#8f97cf"
surp_color = "#ab9ab5"
feature_cmap.update(dict(zip(
    ling_features, [ling_color] * len(ling_features)
    )))
feature_cmap.update(dict(zip(
    uid_features, [uid_color] * len(uid_features)
    )))
feature_cmap.update(dict(zip(
    surp_features, [surp_color] * len(surp_features)
    )))
feature_legend_cmap = {
    "Baseline Features": ling_color,
    "UID Features": uid_color,
    "Surprisal Features": surp_color
}

passive_cmap = {
    True: "tomato",
    False: "steelblue",
    "Both": "#FFF8DE",
}
passive_cmap.update({
        "Passive": passive_cmap[True],
        "Active": passive_cmap[False],
        "P to A": passive_cmap[True],
        "A to P": passive_cmap[False],
})


# === Seaborn defaults === #
sns.set_palette("Set2")
bg_color = "#FAF9F6"
grid_color = "#E5E4E2"
sns.set(rc={
    'axes.facecolor':bg_color, 
    'figure.facecolor':bg_color,
    'grid.color': grid_color,
    'axes.edgecolor': grid_color,
    'lines.markeredgecolor': grid_color,
    'scatter.edgecolors': grid_color,
})

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": "DejaVu Serif",
})

save_kwargs = {
    'dpi': 100,
    'bbox_inches': 'tight',
    'transparent': True
}

# === Utils === #

def extract_sents_of_interest(doc_sents: pd.DataFrame):
    converted_indices = doc_sents['conv_idx'].unique()
    sents_of_interest = []
    for i, sent in doc_sents.iterrows():
        if ((sent['sent_idx'] in converted_indices)
            and 
            (sent['conv_idx'] in [sent['sent_idx'], -1])):
            sents_of_interest.append(sent)
    return pd.DataFrame(sents_of_interest)

def check_passive(f_cf_pair: pd.DataFrame):
    conversion = f_cf_pair['conversion'].unique()
    if 'a>p' in conversion:
        return f_cf_pair.assign(passive=[False if fact else True for fact in f_cf_pair['factual']])
    elif 'p>a' in conversion:
        return f_cf_pair.assign(passive=[True if fact else False for fact in f_cf_pair['factual']])
    
def check_conversion(row):
    mapping = {
        "p>a": "P to A",
        "a>p": "A to P"
    }
    if row['conversion'] != 'og':
        return mapping[row['conversion']]
    if row['passive']:
        return 'P to A'
    return 'A to P'

def perm_test(data, 
              label_column, 
              value_column, 
              n_permutations, 
              agg_fn=np.mean, 
              test_stat=lambda a, b: np.abs(a - b),
              compare_fn=lambda sim, obs: sim >= obs,
              ):
    value_a, value_b = data[label_column].unique()
    print("Labels in order: ", value_a, value_b)
    obs_a = agg_fn(data.loc[data[label_column] == value_a, value_column])
    obs_b = agg_fn(data.loc[data[label_column] == value_b, value_column])
    obs_stat = test_stat(obs_a, obs_b)
    
    sim_stats = []
    for _ in range(n_permutations):
        sim_labels = np.random.permutation(data[label_column])
        sim_a = agg_fn(data.loc[sim_labels == value_a, value_column])
        sim_b = agg_fn(data.loc[sim_labels == value_b, value_column])
        sim_stat = test_stat(sim_a, sim_b)
        sim_stats.append(sim_stat)
    sim_stats = np.array(sim_stats)
    return obs_stat, sim_stats, (compare_fn(sim_stats, obs_stat)).mean()

def get_idx(lst, sublst):
    idx = 0
    start_idx = 0
    end_idx = -1
    while idx < len(lst):
        if lst[idx] == sublst[0]:
            start_idx = idx
            while idx - start_idx < len(sublst) and lst[idx] == sublst[idx - start_idx]:
                idx += 1
            if idx == start_idx + len(sublst):
                end_idx = start_idx + len(sublst) - 1
                break
            else:
                continue
        idx += 1
    if end_idx == -1:
        return None
    return start_idx, end_idx

def check_proper(s):
    # s_stanza = nlp(s).sentences[0].words
    # s_word = [word for word in s_stanza if word.head==0][0]
    # if s_word.upos == 'PROPN':
    #     return True
    # return False
    s = str(s)
    tagged = pos_tag(s.split())
    for tag in tagged:
        if tag[1] == 'NNP':
            return True
    return False

def str_to_list(s):
    return eval(s.replace('\0', ''))

def evaluate_model(model, X, y, plot=None, output_dir=None, plot_suffix=None):
    y_score = model.predict(X)
    y_pred = y_score > 0.5
    cm = confusion_matrix(y, y_pred)
    # fpr, tpr, thresh = roc_curve(y, y_score)
    roc_auc = roc_auc_score(y, y_score)
    acc = accuracy_score(y, y_pred)
    naive = max(np.mean(y), np.mean(1 - np.array(y)))
    log_likelihood = -log_loss(y, y_score, normalize=False)
    act_acc = cm[0][0] / (cm[0][0] + cm[0][1])
    pass_acc = cm[1][1] / (cm[1][0] + cm[1][1])
    macro_acc = (act_acc + pass_acc) / 2

    print("Naive Acc: %.4f" % naive)
    print("Accuracy: %.4f" % acc)
    print("Macro Acc: %.4f" % macro_acc)
    print("ROC-AUC: %.4f" % roc_auc)
    print("Log like.: %.4f" % log_likelihood)
    if plot is not None:
        fig, axs = plt.subplots()
        sns.heatmap(cm, annot=True, ax=axs)
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.title("Confusion Matrix")
        fig.savefig(output_dir / ("%s_test_cm%s" % (plot, plot_suffix)),
                    **save_kwargs)
        results_file = output_dir / ("%s_test_results%s.txt" % (plot, plot_suffix))
        with open(results_file, 'w') as outfile:
            outfile.writelines(["Naive Acc: %.4f\n" % naive,
                "Accuracy: %.4f\n" % acc,
                "Macro Acc: %.4f\n" % macro_acc,
                "ROC-AUC: %.4f\n" % roc_auc,
                "Log like.: %.4f\n" % log_likelihood,
            ])
    
    return cm, roc_auc, acc

def create_legend(cmap):
    handles = []
    for label, color in cmap.items():
        patch = mpatches.Patch(color=color, label=label)
        handles.append(patch)
    return plt.legend(handles=handles)

def cp_plot(data, feature, model, grid_size=100):
    feature_range = None
    all_cp_predictions = []
    for i in range(data.shape[0]):
        observation = data.iloc[[i]]
        feature_range, cp_pred = cp_plot_single(data, 
                                                feature, 
                                                observation, 
                                                model, 
                                                grid_size=grid_size)
        all_cp_predictions.append(cp_pred)
    all_cp_predictions = np.vstack(all_cp_predictions)
    mean_cp_predictions = all_cp_predictions.mean(axis=0)
    return feature_range, mean_cp_predictions

def cp_plot_single(data, feature, observation, model, grid_size=100):
    feature_min = data[feature].min()
    feature_max = data[feature].max()
    feature_range = np.arange(feature_min, feature_max,
                              (feature_max - feature_min) / grid_size)
    cp_observations = [observation.copy(deep=True).assign(**{feature:feat_val}) 
                       for feat_val in feature_range]
    cp_predictions = model.predict_proba(pd.concat(cp_observations))[:, 1]
        
    return feature_range, cp_predictions

def get_uid_df(data_path):
    print(" - Building UID dataframe:")
    # Read data 
    uid_df = pd.read_csv(data_path).iloc[:, 1:]
    _, uid_unit, uid_level = Path(data_path).stem.split("_")[:3]
    plot_suffix = f"_{uid_unit}_{uid_level}"
    
    # Process csv data
    uid_df[['factual', 'doc_name', 'conv_idx', 'conversion']] = uid_df['doc_id'].str.split("::", expand=True)
    uid_df['conv_idx'] = uid_df['conv_idx'].astype(int)
    uid_df['surp_slor'] *= -1
    uid_df['raw_surps'] = uid_df['raw_surps'].apply(str_to_list)
    uid_df['raw_uni_surps'] = uid_df['raw_uni_surps'].apply(str_to_list)
    uid_df['units'] = uid_df['units'].apply(str_to_list)
    print(" - - Shape:", uid_df.shape)
    # print(uid_df.columns)
    return uid_df, plot_suffix

def get_cf_comparison(uid_df):
    sents_of_interest = uid_df.groupby("doc_name").apply(extract_sents_of_interest).reset_index().drop(columns="level_1")
    # print(sents_of_interest.shape)
    print(" - Building cf comparison:")
    cf_comparison = sents_of_interest[sents_of_interest['context'].isin(context_lvls)]#.drop(columns=['raw_surps', 'raw_uni_surps'])
    # print(cf_comparison.shape)
    cf_comparison['factual'] = cf_comparison['factual'] == 'f'
    cf_comparison = cf_comparison.groupby(['doc_id', 'sent_idx', 'context']).first().sort_values(by=['doc_name', 'sent_idx']).reset_index()
    # print(cf_comparison.shape)
    cf_comparison = cf_comparison.groupby(['doc_name', 'sent_idx']).filter(lambda g: g.shape[0] >= len(context_lvls) * 2)
    # print(cf_comparison.shape)
    cf_comparison = cf_comparison.groupby(['doc_name', 'sent_idx']).apply(check_passive).reset_index()
    # print(cf_comparison.shape)
    # print(cf_comparison.apply(check_conversion, axis=1).shape)
    cf_comparison['conversion'] = cf_comparison.apply(check_conversion, axis=1)
    
    # print(" - - Processing Agent Prop:")
    # cf_comparison['agent_is_prop'] = cf_comparison['agent'].apply(check_proper)
    # print(" - - Processing Patient Prop:")
    # cf_comparison['patient_is_prop'] = cf_comparison['patient'].apply(check_proper)
    
    cf_comparison = cf_comparison[((cf_comparison['patient_unigram_logprob'] != -np.inf) & (cf_comparison['agent_unigram_logprob'] != -np.inf))]
    
    standard_scaler = StandardScaler()
    to_standardize = (['patient_len',
                         'agent_len',
                         'patient_unigram_logprob',
                         'agent_unigram_logprob']
                        + uid_features
                        + surp_features
                        + metrics)
    
    cf_comparison[to_standardize] = standard_scaler.fit_transform(cf_comparison[to_standardize])
    
    return cf_comparison

def get_pw_diffs(cf_comparison):
    print("Building pw. diffs:")
    get_diff = lambda s: s.diff().iloc[1]
    get_diff_flipped = lambda s: -s.diff().iloc[1]
    agg_map = dict(zip(metrics + ['uid_len'], [get_diff] * (len(metrics) + 1)))
    passthrough = [
        'passive'
    ] + ling_features
    agg_map.update(dict(zip(passthrough, [lambda s: s.iloc[0]] * len(passthrough))))
    agg_map.update({
        'conversion': lambda s: 'A to P' if 'A to P' in s.unique() else 'P to A',
    })
    pw_diffs = (cf_comparison
                .sort_values(by='factual', ascending=False) # Always Counterfactual - Factual, shows shift
                .groupby(['doc_name','sent_idx', 'context'])
                [metrics + passthrough + ['conversion', 'uid_len']]
                .aggregate(
                    agg_map
                    )
                .reset_index())
    pw_diffs = pw_diffs[pw_diffs['surp_slor'] != 0]
    return pw_diffs

def get_pw_diffs_regression(cf_comparison):
    print("Building regression pw diffs")
    # take pairwise differences, but *Active - Passive*
    get_diff = lambda s: s.diff().iloc[1]
    pass_through = lambda s: s.iloc[0]
    agg_map = dict(zip(uid_features + surp_features, [get_diff] * len(uid_features + surp_features)))
    agg_map.update(dict(zip(ling_features, [pass_through] * len(ling_features))))
    agg_map.update({
        'conversion': lambda s: 'A to P' if 'A to P' in s.unique() else 'P to A',
        # 'passive': lambda s: s.iloc[0],
    })
    pw_diffs = (cf_comparison
                .sort_values(by='passive', ascending=False) # Always Active - Passive
                .groupby(['doc_name','sent_idx', 'context'])
                [uid_features + surp_features + ling_features + ['conversion']]
                .aggregate(
                    agg_map
                    )
                .reset_index())
    pw_diffs = pw_diffs[pw_diffs['surp_slor'] != 0]
    pw_diffs['passive'] = pw_diffs['conversion'] == 'P to A'
    pw_diffs = pw_diffs.reset_index(drop=True)
    return pw_diffs

def plot_f_v_cf(cf_comparison, plot_suffix, output_dir, context=None):
    context = context or context_lvls[0]
    print("Plotting f v. cf with %s context" % context)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 5), sharey=True, width_ratios=[1, 2])
    hue_order = [True, False]
    uid_unit, uid_level = plot_suffix.split("_")[1:]
    # surp metrics
    df = cf_comparison[cf_comparison['context']==context]
    sns.boxenplot(
        data=df.melt(
            id_vars=['factual'],
            value_vars=surprisal_metric_names,
            var_name='Metric',
            value_name='Value'
            ),
        hue='factual', y='Value', x='Metric', ax=axs[0], palette=factuality_cmap, hue_order=hue_order,
        showfliers=False
    )
    
    axs[0].set_title(f"Surprisal metrics")
    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].set_xticks(ticks=surprisal_metric_names, labels=surprisal_metrics_formatted)
    axs[0].legend(title="Factual")
    # axs[0].get_legend().remove()

    # uid metrics
    sns.boxenplot(data=df.melt(
        id_vars=['factual'],
        value_vars=uid_metric_names,
        var_name='Metric',
        value_name='Value'
        ),
            hue='factual', y='Value', x='Metric', ax=axs[1], palette=factuality_cmap, hue_order=hue_order,
            showfliers=False)
    axs[1].set_title(f"UID metrics")
    handles, labels = axs[1].get_legend_handles_labels()
    # axs[1].legend(title="Factual", bbox_to_anchor=(1.2, 1.0))
    axs[1].set_xticks(ticks=uid_metric_names, labels=uid_metrics_formatted)
    # axs[1].set_xlabel("")
    axs[1].get_legend().remove()

    # entire plot
    plt.suptitle(f"Factual vs. Counterfactual\n(Context={context}, Unit={uid_unit}, Level={uid_level})")
    plt.tight_layout()
    fig.savefig(output_dir / ("fact_cfact_metrics_%s%s" % (context, plot_suffix)),
                **save_kwargs)

def plot_diffs(pw_diffs, plot_suffix, output_dir, context='document'):
    context = context or context_lvls[0]
    print("Plotting pw diffs with %s context" % context)
    uid_unit, uid_level = plot_suffix.split("_")[1:]
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True, width_ratios=[1, 2])
    # surp metrics
    df = pw_diffs[pw_diffs['context']==context]
    sns.boxplot((df.loc[
                (df['uid_len'] > -5) & (df['uid_len'] < 5),
                # :,
                ['doc_name', 'sent_idx', 'conversion', 'context'] + surprisal_metric_names]
                .melt(id_vars=['doc_name', 'sent_idx', 'conversion', 'context'])), 
                y='value', x='variable', 
                # color=conversion_cmap['Both'],
                # hue='context', palette=context_cmap,
                hue='conversion', palette=conversion_cmap,
                showfliers=False,
                # inner=None
                ax=axs[0]
                )
    axs[0].axhline(y=0, color='#BF4646', 
                # linestyle='dotted', 
                alpha=0.5,
                zorder=5)
    axs[0].set_title("Surprisal")
    axs[0].set_ylabel(r'$\Delta_{\text{metric}}$')
    axs[0].set_xlabel('')
    # axs[0].legend(title="Context Level")
    axs[0].get_legend().remove()
    axs[0].set_xticks(ticks=surprisal_metric_names, labels=surprisal_metrics_formatted)

    # uid metrics
    sns.boxplot((df[['doc_name', 'sent_idx', 'conversion', 'context'] + uid_metric_names]
                .melt(id_vars=['doc_name', 'sent_idx', 'conversion', 'context'])), 
                y='value', x='variable', 
                # color=conversion_cmap['Both'],
                # hue='context', palette=context_cmap,
                hue='conversion', palette=conversion_cmap,
                showfliers=False,
                # inner=None
                ax=axs[1]
                )
    axs[1].axhline(y=0, color='#BF4646', 
                # linestyle='dotted', 
                alpha=0.5,
                zorder=5)
    axs[1].set_title("Uniformity$^{-1}$")
    axs[1].set_ylabel('')
    axs[1].set_xlabel('')
    # axs[1].get_legend().remove()
    axs[1].legend(title="Conversion", bbox_to_anchor=(1,1))
    axs[1].set_xticks(ticks=uid_metric_names, labels=uid_metrics_formatted)
    # ax.grid(axis='x')
    # full plot
    # plt.suptitle(f"Pairwise Differences (Counterfactual - Factual)")
    plt.tight_layout()
    plt.savefig(output_dir / ("pw_diffs_%s%s" % (context, plot_suffix)),
                **save_kwargs)
    
def wilcoxon_test(pw_diffs, cf_comparison, plot_suffix, output_dir, context=None):
    context = context or context_lvls[0]
    print("Running wilcoxon test with %s context" % context)
    results = []
    df = pw_diffs[pw_diffs['context'] == context]
    for conv in ['P to A', 'A to P']:
        for metric in metrics:
            metric_diffs = df.loc[df['conversion']==conv, metric]
            # normal_res = normaltest(metric_diffs)
            factuals = cf_comparison.loc[cf_comparison['factual'] 
                                        & (cf_comparison['conversion']==conv)]
            cfactuals = cf_comparison.loc[~cf_comparison['factual']
                                        & (cf_comparison['conversion']==conv)]
            N = cfactuals.shape[0]
            wilcoxon_res = wilcoxon(factuals[metric], cfactuals[metric],
                                alternative="less", method='asymptotic'
                                )
            results.append({
                'Metric': metrics_mapping[metric],
                'Conversion Type': conv,
                'Mean Change': metric_diffs.mean(),
                'T': wilcoxon_res.statistic,
                'z': wilcoxon_res.zstatistic,
                'Effect Size': np.abs(wilcoxon_res.zstatistic) / np.sqrt(N),
                'Wilcoxon P-val': wilcoxon_res.pvalue
            })
    results = pd.DataFrame(results)
    results.to_csv(output_dir / ("wilcox_res_%s%s.csv" % (context, plot_suffix)))
    
def setup_regression_data(pw_diffs, 
                          bootstrap=False, 
                          resample=False, 
                          test_size=0.2,
                          use_ling=True,
                          use_uid=True,
                          use_surp=True,
                          context=None,
                          override_features=[]):
    context = context or context_lvls[0]
    print("Building datasets with %s context" % context)
    assert use_ling or use_uid or use_surp, "No data chosen"
    if override_features:
        use_uid=False
        use_surp=False
    # define features, target, and naive baseline
    X = pw_diffs.loc[
        pw_diffs['context']==context,
        []
        + (ling_features if use_ling else [])
        + (uid_features if use_uid else [])
        + (surp_features if use_surp else [])
        + override_features
        ]
    y = pw_diffs.loc[
        pw_diffs['context']==context,
        'passive'
    ]

    # bootstrap to resolve class imbalance
    if bootstrap:
        indices = pd.Series(X.index)
        n = X.shape[0]
        sampled_passive = indices[y].sample(n // 2, replace=True)
        sampled_active = indices[~y].sample(n // 2, replace=True)
        X = pd.concat([X.iloc[sampled_passive], X.iloc[sampled_active]])
        y = pd.concat([y.iloc[sampled_passive], y.iloc[sampled_active]])
        
    # resample to resolve class imbalance
    if resample:
        indices = pd.Series(X.index)
        n = min(y.sum(), (~y).sum())
        sampled_passive = indices[y]
        sampled_active = indices[~y].sample(n, replace=False)
        X = pd.concat([X.iloc[sampled_passive], X.iloc[sampled_active]]).reset_index(drop=True)
        y = pd.concat([y.iloc[sampled_passive], y.iloc[sampled_active]]).reset_index(drop=True)
        
    X = X.replace([-np.inf], np.nan).fillna(np.log(1e-10))
    X = sm.add_constant(X, has_constant='add')
    X = X.astype(np.float64)
    y = y.astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        shuffle=True, stratify=y,
                                                        random_state=SEED)
    naive_baseline_accuracy = max(y.mean(), (1 - y).mean())
    return X, y, X_train, X_test, y_train, y_test, naive_baseline_accuracy

def plot_feature_corr(X, plot_suffix, output_dir, plot_title="modeling_feat_corrs"):
    fig, axs = plt.subplots(1, 2, figsize=(10, 10), width_ratios=[30, 1])
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    data = X.drop(columns=['const'])
    data.columns = [format_map[col] for col in data.columns]
    data = data.corr()
    mask = np.triu(np.ones_like(data, dtype=bool))
    sns.heatmap(data, mask=mask,
                annot=True, fmt=".2f", ax=axs[0], cbar_ax=axs[-1], cmap=cmap,
                vmin=-1, vmax=1)
    axs[0].set_title("Feature Correlations")
    axs[0].grid(visible=False)
    plt.tight_layout()
    fig.savefig(output_dir / ("/plots/%s%s" % (plot_title, plot_suffix)),
                dpi=100, bbox_inches='tight', transparent=True)
    
def logistic_regression_experiments(X, y, X_train, X_test, y_train, y_test, naive_baseline_accuracy,
                                    output_dir, plot_suffix, cp_plots=False):
    print(" - Running Logistic Regression Model experiments")
    logreg = sm.Logit(y_train, X_train).fit()
    print(" - - Naive baseline accuracy: %.4f" % naive_baseline_accuracy)
    print(" - - Training metrics:")
    evaluate_model(logreg, X_train, y_train)
    print(" - - Test metrics:")
    evaluate_model(logreg, X_test, y_test, 
                   plot="lr", 
                   output_dir=output_dir, plot_suffix=plot_suffix)
    
    # Plots confidence intervals for coefficients from statsmodels
    ci = logreg.conf_int()
    ci.columns = ['2.5%', '97.5%']
    ci['coef'] = logreg.params
    ci = ci.reset_index().rename(columns={'index': 'feature'})
    ci.to_csv(output_dir / ("lr_model_params%s.csv" % plot_suffix))
    # ci = ci[ci['feature'] != 'Intercept']
    fig = plt.figure(figsize=(8, 6))
    sns.scatterplot(x=ci['coef'], y=ci['feature'], 
                    hue=ci['feature'], palette=feature_cmap, zorder=20)
    x_coords = np.arange(len(ci))
    y_coords = ci['coef']
    xerr = ci['coef'] - ci['2.5%']
    plt.errorbar(x=y_coords, y=x_coords, xerr=xerr, fmt='none', ecolor='grey', capsize=5)
    plt.axvline(0, color='grey', linestyle='--')
    plt.title("Logistic Regression Coefficients with 95% CI (Predicting Likelihood of Passive)")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    ticks = list(X.columns)
    plt.yticks(
        ticks = ticks,
        labels = [format_map[tick] for tick in ticks] 
    )
    plt.tight_layout()
    create_legend(feature_legend_cmap)
    fig.savefig(output_dir / ("modeling_logreg_coeffs%s" % plot_suffix),
                dpi=100, bbox_inches='tight', transparent=True)
    
    logreg_skl = LogisticRegression()
    logreg_skl.fit(X_train, y_train)
    
    # CP plots
    if cp_plots:
        cp_features_list = ['agent_len', 'uid_std', 'uid_pwd', 'surp_slor']
        lr_cp_plots = dict()
        for feature in tqdm(cp_features_list, desc="Calculating Cet. Par. Plots..."):
            X_sample = X.sample(6000, replace=True)
            value, proba = cp_plot(X_sample, feature, logreg_skl, grid_size=100)
            lr_cp_plots[feature] = pd.DataFrame({
                'value': value,
                'proba': proba
            })
            
        n_cols = 2
        n_rows = np.ceil(len(cp_features_list) / n_cols).astype(int)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows),
                                sharey=True)
        observations = X.copy(deep=True)
        observations['pred_proba'] = rf.predict_proba(observations)[:, 1]
        observations['og_form'] = ["Passive" if y_val == 1 else "Active" for y_val in y]
        axs = np.array(axs).ravel()
        for i, pair in enumerate(lr_cp_plots.items()):
            feat_name, data = pair
            sns.lineplot(data, x='value', y='proba', ax=axs[i],
                        label='Cet. Par. Line', color='#647FBC')
            sns.scatterplot(observations, x=feat_name, y='pred_proba',
                            hue='og_form',
                            ax=axs[i], alpha=0.3)
            axs[i].set_title(format_map[feat_name])
            axs[i].set_ylabel('Predicted P(Passive)')
            axs[i].set_xlabel('Feature value')
            axs[i].axhline(y=y.mean(), color='red', linestyle='--', alpha=0.5, label='Chance')
            if i == n_cols - 1:
                axs[i].legend(bbox_to_anchor=(1, 1))
            else:
                axs[i].get_legend().remove()
        fig.suptitle("Ceteris Paribus Plots (Logistic)")
        fig.tight_layout()
        fig.savefig(output_dir / ("cp_logreg%s" % plot_suffix),
                    **save_kwargs)
    return logreg

def random_forest_experiments(X, y, X_train, X_test, y_train, y_test, naive_baseline_accuracy,
                              output_dir, plot_suffix, cp_plots=False):
    print(" - Running Random Forest model experiments")
    n_trials = 10
    results = {
        'Feature': [],
        'Importance': []
    }
    for _ in tqdm(range(n_trials)):
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X_train, y_train)
        results['Feature'].extend(X.columns)
        results['Importance'].extend(rf.feature_importances_)
    results = pd.DataFrame(results)
    
    # Evaluate
    print("Naive baseline accuracy: %.2f" % naive_baseline_accuracy)
    print("=" * 10)
    print("Training metrics:")
    evaluate_model(rf, X_train, y_train)
    print("=" * 10)
    print("Test metrics:")
    evaluate_model(rf, X_test, y_test,
                   plot="rf", 
                   output_dir=output_dir, plot_suffix=plot_suffix)
    
    # Importance plot
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(results[results['Feature'] != 'const'], x='Feature', y='Importance',
                errorbar=('ci', 95), 
                hue='Feature', palette=feature_cmap,
                zorder=20)
    ticks = list(X.columns)
    ticks.remove('const')
    plt.xticks(
        ticks = ticks,
        labels = [format_map[tick] for tick in ticks],
        rotation=90)
    plt.grid(axis='y', zorder=-1)
    for spine in ['left', 'top', 'right']:
        ax.spines[spine].set_visible(False)
    create_legend(feature_legend_cmap)
    plt.title(f"Feature Importance across {n_trials} RF Models")
    fig.savefig(output_dir / ("rf_feat_importance%s" % plot_suffix),
                **save_kwargs)

    # CP Plot
    if cp_plots:
        cp_features_list = ['agent_len', 'uid_std', 'uid_pwd', 'surp_slor']
        rf_cp_plots = dict()
        for feature in tqdm(cp_features_list, desc="Calculating Cet. Par. Plots..."):
            X_sample = X.sample(6000, replace=True)
            value, proba = cp_plot(X_sample, feature, rf, grid_size=100)
            rf_cp_plots[feature] = pd.DataFrame({
                'value': value,
                'proba': proba
            })
        
        n_cols = 2
        n_rows = np.ceil(len(cp_features_list) / n_cols).astype(int)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows),
                                sharey=True)
        observations = X.copy(deep=True)
        observations['pred_proba'] = rf.predict_proba(observations)[:, 1]
        observations['og_form'] = ["Passive" if y_val == 1 else "Active" for y_val in y]
        axs = np.array(axs).ravel()
        for i, pair in enumerate(rf_cp_plots.items()):
            feat_name, data = pair
            sns.lineplot(data, x='value', y='proba', ax=axs[i],
                        label='Cet. Par. Line', color='#647FBC')
            sns.scatterplot(observations, x=feat_name, y='pred_proba',
                            hue='og_form',
                            ax=axs[i], alpha=0.3)
            axs[i].set_title(format_map[feat_name])
            axs[i].set_ylabel('Predicted P(Passive)')
            axs[i].set_xlabel('Feature value')
            axs[i].axhline(y=y.mean(), color='red', linestyle='--', alpha=0.5, label='Chance')
            if i == n_cols - 1:
                axs[i].legend(bbox_to_anchor=(1, 1))
            else:
                axs[i].get_legend().remove()
        fig.suptitle("Avg. Ceteris Paribus Plots for All Observations")
        fig.tight_layout()
        fig.savefig(output_dir / ("cp_rf%s" % plot_suffix),
                    **save_kwargs)
    return rf
    
def save_logreg(logreg, output_dir, plot_suffix):
    model = logreg.conf_int()
    model.columns = ['2.5%', '97.5%']
    model['coef'] = logreg.params
    model.to_csv(output_dir / "lr_model_params%s.csv" % plot_suffix)


def main():
    parser = argparse.ArgumentParser(description='Run analysis scripts for results of active/passive/process data.')
    parser.add_argument("data_dir", type=str, default=None, help="Path to folder containing csv files to process.")
    parser.add_argument("output_dir", type=str, default=None, help="Output directory for plots/results")
    parser.add_argument("--uid_df_path", type=str, default=None, help="Path to pre-saved uid_df csv file.")
    parser.add_argument("--cf_comparison_path", type=str, default=None, help="Path to pre-saved cf_comparison csv file.")
    parser.add_argument("--pw_diffs_path", type=str, default=None, help="Path to pre-saved pw_diffs csv file.")
    parser.add_argument("--pw_diffs_reg_path", type=str, default=None, help="Path to pre-saved pw_diffs csv file for regression.")
    
    args, unk = parser.parse_known_args()
    
    # Handle unknown args and save jic
    extra_args = {}
    for arg in unk:
        # edge case handling
        if '=' in arg:
            key, value = arg.split('=', 1)
            # Convert value to appropriate type
            if value.lower() == 'true':
                extra_args[key] = True
            elif value.lower() == 'false':
                extra_args[key] = False
            elif value.isdigit():
                extra_args[key] = int(value)
            elif re.match(r'^-?\d+\.\d+$', value):
                extra_args[key] = float(value)
            else:
                extra_args[key] = value
                
    print("Processing files from %s\nPlots will be saved to %s\n=======" % (args.data_dir, args.output_dir))
    data_paths = Path(args.data_dir).glob('*.csv')
    output_dir = Path(args.output_dir)
    
    all_cf_comparisons = []
    all_pw_diffs = []
    all_uid_dfs = []
    plot_suffix = ""
    if args.uid_df_path:
        print("Reading in data from file %s" % args.uid_df_path)
        uid_df = pd.read_csv(args.uid_df_path)
        uid_unit, uid_level = Path(args.uid_df_path).stem.split("_")[1:3]
        plot_suffix=("_%s_%s" % (uid_unit, uid_level))
    else:
        for idx, data_path in enumerate(data_paths):
            print("Reading in data from doc %d: %s" % (idx, data_path))
            # Get dataframes
            uid_df, plot_suffix = get_uid_df(data_path)
            all_uid_dfs.append(uid_df)
            # all_cf_comparisons.append(cf_comparison)
            # all_pw_diffs.append(pw_diffs)
            print("Complete")
            print()
        
        # cf_comparison = pd.concat(all_cf_comparisons)
        # pw_diffs = pd.concat(all_pw_diffs)
        uid_df = pd.concat(all_uid_dfs)
        uid_unit, uid_level = plot_suffix.split("_")[1:]
        csv_path = output_dir / ("cf_%s_%s_uid_full.csv" % (uid_unit, uid_level))
        uid_df.to_csv(csv_path)
        print(f"Data saved to {csv_path}")
    if args.cf_comparison_path:
        cf_comparison = pd.read_csv(args.cf_comparison_path)
    else:
        cf_comparison = get_cf_comparison(uid_df)
        cf_comparison.to_csv(output_dir / ("cf_comparison%s.csv" % plot_suffix))
        
    if args.pw_diffs_path:
        pw_diffs = pd.read_csv(args.pw_diffs_path)
    else:
        pw_diffs = get_pw_diffs(cf_comparison)
        pw_diffs.to_csv(output_dir / ("pw_diffs%s.csv" % plot_suffix))
    
    plot_f_v_cf(cf_comparison, plot_suffix, output_dir)
    plot_diffs(pw_diffs, plot_suffix, output_dir)
    wilcoxon_test(pw_diffs, cf_comparison, plot_suffix, output_dir)
    plot_diffs(pw_diffs, plot_suffix, output_dir, context='sentence')
    wilcoxon_test(pw_diffs, cf_comparison, plot_suffix, output_dir, context='sentence')
    
    
    if args.pw_diffs_reg_path:
        pw_diffs_regression = pd.read_csv(args.pw_diffs_reg_path)
    else:
        pw_diffs_regression = get_pw_diffs_regression(cf_comparison)
        pw_diffs_regression.to_csv(output_dir / ("pw_diffs_reg%s.csv" % plot_suffix))
    
    
    # == UID FEATURE COMPARISON == #
    # = Surp. STD = #
    X, y, X_train, X_test, y_train, y_test, naive_baseline_accuracy = setup_regression_data(pw_diffs_regression,
                                                                                            override_features=['uid_std'])
    logistic_regression_experiments(X, y, X_train, X_test, y_train, y_test, naive_baseline_accuracy, 
                                    output_dir, ("_uid_std" + plot_suffix))
    # = Local Var. = # 
    X, y, X_train, X_test, y_train, y_test, naive_baseline_accuracy = setup_regression_data(pw_diffs_regression,
                                                                                            override_features=['uid_pwd'])
    logistic_regression_experiments(X, y, X_train, X_test, y_train, y_test, naive_baseline_accuracy, 
                                    output_dir, ("_uid_pwd" + plot_suffix))
    # = -SLOR = #
    X, y, X_train, X_test, y_train, y_test, naive_baseline_accuracy = setup_regression_data(pw_diffs_regression,
                                                                                            override_features=['surp_slor'])
    logistic_regression_experiments(X, y, X_train, X_test, y_train, y_test, naive_baseline_accuracy, 
                                    output_dir, ("_surp_slor" + plot_suffix))
    
    # == CONTEXT LEVEL COMPARISON == #
    # = with context = #
    X, y, X_train, X_test, y_train, y_test, naive_baseline_accuracy = setup_regression_data(pw_diffs_regression,
                                                                                            use_ling=False,
                                                                                            context='document')
    logistic_regression_experiments(X, y, X_train, X_test, y_train, y_test, naive_baseline_accuracy, 
                                    output_dir, ("_context" + plot_suffix))
    # w/o context
    X, y, X_train, X_test, y_train, y_test, naive_baseline_accuracy = setup_regression_data(pw_diffs_regression,
                                                                                            use_ling=False,
                                                                                            context='sentence')
    logistic_regression_experiments(X, y, X_train, X_test, y_train, y_test, naive_baseline_accuracy, 
                                    output_dir, ("_no_context" + plot_suffix))
    # == FULL MODEL == #
    X, y, X_train, X_test, y_train, y_test, naive_baseline_accuracy = setup_regression_data(pw_diffs_regression)
    logistic_regression_experiments(X, y, X_train, X_test, y_train, y_test, naive_baseline_accuracy, 
                                    output_dir, ("_full" + plot_suffix))
    random_forest_experiments(X, y, X_train, X_test, y_train, y_test, naive_baseline_accuracy, 
                                output_dir, plot_suffix, cp_plots=False)
    
    print("N factual:", len(cf_comparison['doc_name'].unique()))
    print(cf_comparison['factual'].value_counts())
    print(pw_diffs['conversion'].value_counts())
if __name__ == "__main__":
    main()