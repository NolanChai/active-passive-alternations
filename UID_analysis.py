import argparse
import sys
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
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.tag import pos_tag

# import stanza
# stanza.download('en', download_method=None)
# nlp = stanza.Pipeline(
#     lang="en",
#     processors="tokenize,mwt,pos,lemma,depparse"
# )

from re import X
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter, LogFormatter
from matplotlib import rc
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# === Constants === #

context_lvls = ['document']

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

passive_cmap = {
    True: "#1A3263",
    False: "#FAB95B"
}

#=== Metric Names ===#

surprisal_metric_names = [
    'surp_mean', # Mean surprisal
    'surp_slor', # SLOR
]

uid_metric_names = [
    'uid_std', # Std. of surprisal
    'uid_mad', # mean avg. deviation of surprisal
    'uid_pwd', # Local variance (pairwise euc. distance)
    'uid_range', # Range of surprisal
    # 'uid_len' # Length of sentence (units)
    # 'uid_slope', # Slope of a linear function fit to surp values
]

surprisal_metrics_formatted = [
    'Mean Surp.',
    '-SLOR',
]

uid_metrics_formatted = [
    'Surp. STD',
    'Surp. MAD',
    'Local Var.',
    'Surp. Range',
    # 'Length',
    # 'LinReg Slope',
]

metrics = surprisal_metric_names + uid_metric_names
metrics_formatted = surprisal_metrics_formatted + uid_metrics_formatted

metrics_mapping = dict(zip(metrics, metrics_formatted))
# metrics_cmap = dict(zip([
#     'surp_slor', ''
# ]))

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
    "font.serif": "Times",
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

def get_cf_comparison(data_path):
    print(" - Building cf comparison:")
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
    
    sents_of_interest = uid_df.groupby("doc_name").apply(extract_sents_of_interest).reset_index().drop(columns="level_1")

    cf_comparison = sents_of_interest[sents_of_interest['context'].isin(context_lvls)]#.drop(columns=['raw_surps', 'raw_uni_surps'])
    cf_comparison['factual'] = cf_comparison['factual'] == 'f'
    cf_comparison = cf_comparison.groupby(['doc_id', 'sent_idx', 'context']).first().sort_values(by=['doc_name', 'sent_idx']).reset_index()
    cf_comparison = cf_comparison.groupby(['doc_name', 'sent_idx']).filter(lambda g: g.shape[0] >= len(context_lvls) * 2)
    cf_comparison = cf_comparison.groupby(['doc_name', 'sent_idx']).apply(check_passive).reset_index()
    cf_comparison['conversion'] = cf_comparison.apply(check_conversion, axis=1)
    
    print(" - - Processing Agent Prop:")
    cf_comparison['agent_is_prop'] = cf_comparison['agent'].apply(check_proper)
    print(" - - Processing Patient Prop:")
    cf_comparison['patient_is_prop'] = cf_comparison['patient'].apply(check_proper)
    
    standard_scaler = StandardScaler()
    cf_comparison[metrics] = standard_scaler.fit_transform(cf_comparison[metrics])
    
    return cf_comparison, plot_suffix

def get_pw_diffs(cf_comparison):
    print(" - Building pw. diffs:")
    get_diff = lambda s: s.diff().iloc[1]
    get_diff_flipped = lambda s: -s.diff().iloc[1]
    agg_map = dict(zip(metrics + ['uid_len'], [get_diff] * (len(metrics) + 1)))
    passthrough = [
        'agent_is_prop',
        'patient_is_prop',
        'passive'
    ]
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

def plot_f_v_cf(cf_comparison, plot_suffix, output_dir):
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 5), sharey=True, width_ratios=[1, 2])
    hue_order = [True, False]
    uid_unit, uid_level = plot_suffix.split("_")[1:]
    # surp metrics
    sns.boxenplot(
        data=cf_comparison.melt(
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
    sns.boxenplot(data=cf_comparison.melt(
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
    plt.suptitle(f"Factual vs. Counterfactual\n(Context={' or '.join(context_lvls)}, Unit={uid_unit}, Level={uid_level})")
    plt.tight_layout()
    fig.savefig(output_dir / ("fact_cfact_metrics%s" % plot_suffix),
                **save_kwargs)

def plot_diffs(pw_diffs, plot_suffix, output_dir):
    uid_unit, uid_level = plot_suffix.split("_")[1:]
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True, width_ratios=[1, 2])
    # surp metrics
    sns.boxplot((pw_diffs.loc[
                (pw_diffs['uid_len'] > -5) & (pw_diffs['uid_len'] < 5),
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
    sns.boxplot((pw_diffs[['doc_name', 'sent_idx', 'conversion', 'context'] + uid_metric_names]
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
    plt.savefig(output_dir / ("pw_diffs_context%s" % plot_suffix),
                **save_kwargs)
    
def wilcoxon_test(pw_diffs, cf_comparison, plot_suffix, output_dir):
    results = []
    for conv in ['P to A', 'A to P']:
        for metric in metrics:
            metric_diffs = pw_diffs.loc[pw_diffs['conversion']==conv, metric]
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
    results.to_csv(output_dir / ("wilcox_res_%s.csv" % plot_suffix))

def main():
    parser = argparse.ArgumentParser(description='Run analysis scripts for results of active/passive/process data.')
    parser.add_argument("data_dir", type=str, help="Path to folder containing csv files to process.")
    parser.add_argument("output_dir", type=str, help="Output directory for plots/results")
    
    
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
    plot_suffix = ""
    for idx, data_path in enumerate(data_paths):
        print("Reading in data from doc %d: %s" % (idx, data_path))
        # Get dataframes
        cf_comparison, plot_suffix = get_cf_comparison(data_path)
        pw_diffs = get_pw_diffs(cf_comparison)
        all_cf_comparisons.append(cf_comparison)
        all_pw_diffs.append(pw_diffs)
        print("Complete")
        print()
    cf_comparison = pd.concat(all_cf_comparisons)
    pw_diffs = pd.concat(all_pw_diffs)
    cf_comparison.to_csv(output_dir / ("cf_comparison%s.csv" % plot_suffix))
    pw_diffs.to_csv(output_dir / ("pw_diffs%s.csv" % plot_suffix))
    
    plot_f_v_cf(cf_comparison, plot_suffix, output_dir)
    plot_diffs(pw_diffs, plot_suffix, output_dir)
    wilcoxon_test(pw_diffs, cf_comparison, plot_suffix, output_dir)
    
    print("N factual:", len(cf_comparison['doc_name'].unique()))
    print(cf_comparison['factual'].value_counts())
if __name__ == "__main__":
    main()