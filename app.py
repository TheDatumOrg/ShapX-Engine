from collections import namedtuple
import altair as alt
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from st_aggrid import AgGrid
import plotly.graph_objects as go
import altair as alt
import shap
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import pandas as pd
import matplotlib
import pickle
matplotlib.use('agg')
import matplotlib.pyplot as plt
import operator
import math
from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare
import networkx

st.set_page_config(
    page_title="ShapX Engine",
    page_icon="logo.png",
    # layout="wide",
)
#plt.style.use('dark_background')
df = pd.read_csv('data/results.csv')



def plot_stat_plot(df, metric_name, methods_family, datasets):
    container_method = st.container()
    stat_methods_family = container_method.multiselect('Select a group of methods', sorted(methods_family), key='selector_stat_methods')
    
    df = df.loc[df['Datasets'].isin(datasets)][[method_g + '-' + metric_name for method_g in stat_methods_family]]
    df.insert(0, 'Datasets', datasets)

    if len(datasets) > 0:
        if len(stat_methods_family) > 1 and len(stat_methods_family) < 13:
            def stat_plots(df_toplot):
                def cd_diagram_process(df, rank_ascending=False):
                    df = df.rank(ascending=rank_ascending, axis=1)
                    return df

                df_toplot.drop(columns=df_toplot.columns[0], axis=1, inplace=True)

                rank_ri_df  = cd_diagram_process(df_toplot)
                rank_df = rank_ri_df.mean().sort_values()

                names = []
                for method in rank_df.index.values:
                    names.append(method[:-len(metric_name)-1])

                avranks =  rank_df.values
                cd = compute_CD(avranks, 128, "0.1")
                graph_ranks(avranks, names, cd=cd, width=9, textspace=1.25)
                fig = plt.show()
                st.pyplot(fig)
                rank_df = rank_df.reset_index()
                rank_df.columns = ['Method Name', 'Average Rank']
                st.table(rank_df)

            stat_plots(df)
    
      
def plot_box_plot_repl(df):
    fig = plt.figure(figsize=(10, 4))
    ax = sns.boxplot(df[df['accuracy']>95], x='Replacement Strategy', y='accuracy',hue='Replacement Strategy', palette="magma", width=0.5)
    plt.grid(axis='x', linestyle='-', alpha=0.25)
    plt.grid(axis='y', linestyle='-', alpha=0.25)     
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel("Accuracy(%)", fontdict={'size':12}, loc='center')
    plt.legend(loc='lower right')
    st.pyplot(fig)

def plot_box_plot_app(df):
    fig = plt.figure(figsize=(10, 4))
    ax = sns.boxplot(df[df['Accuracy']>95], x='Approximation', y='Accuracy',hue='Approximation', palette="magma", width=0.5)
    plt.grid(axis='x', linestyle='-', alpha=0.25)
    plt.grid(axis='y', linestyle='-', alpha=0.25)     
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel("Accuracy(%)", fontdict={'size':12}, loc='center')
    plt.legend(loc='lower right')
    st.pyplot(fig)


def plot_bar_repl(df):
    fig = plt.figure(figsize=(15, 5))
    ax = sns.barplot(df, x='Model', y='time', hue='Replacement Strategy', palette="magma",errorbar=None)
    plt.yscale('log')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='x', linestyle='-', alpha=0.25)
    plt.grid(axis='y', linestyle='-', alpha=0.25)     
    plt.xlabel("Machine learning models", fontdict={'size':12}, loc='center').set_visible(False)
    plt.ylabel("Compute time (sec)", fontdict={'size':12}, loc='center')
    plt.legend(loc='upper right', fontsize=10, ncols=8, bbox_to_anchor=(0.95,-0.065))
    # plt.title("Instance-wise compute time comparison", fontsize=15)
    st.pyplot(fig)

def plot_shap_fig(shap_vals):
    fig = plt.figure(figsize=(20, 5))
    shap.plots.waterfall(shap_vals[0])
    st.pyplot(fig)

# inspired from orange3 https://docs.orange.biolab.si/3/data-mining-library/reference/evaluation.cd.html
def graph_ranks(avranks, names, p_values, cd=None, cdmethod=None, lowv=None, highv=None,
                width=6, textspace=1, reverse=False, filename=None, labels=False, **kwargs):
    """
    Draws a CD graph, which is used to display  the differences in methods'
    performance. See Janez Demsar, Statistical Comparisons of Classifiers over
    Multiple Data Sets, 7(Jan):1--30, 2006.

    Needs matplotlib to work.

    The image is ploted on `plt` imported using
    `import matplotlib.pyplot as plt`.

    Args:
        avranks (list of float): average ranks of methods.
        names (list of str): names of methods.
        cd (float): Critical difference used for statistically significance of
            difference between methods.
        cdmethod (int, optional): the method that is compared with other methods
            If omitted, show pairwise comparison of methods
        lowv (int, optional): the lowest shown rank
        highv (int, optional): the highest shown rank
        width (int, optional): default width in inches (default: 6)
        textspace (int, optional): space on figure sides (in inches) for the
            method names (default: 1)
        reverse (bool, optional):  if set to `True`, the lowest rank is on the
            right (default: `False`)
        filename (str, optional): output file name (with extension). If not
            given, the function does not write a file.
        labels (bool, optional): if set to `True`, the calculated avg rank
        values will be displayed
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError:
        raise ImportError("Function graph_ranks requires matplotlib.")

    width = float(width)
    textspace = float(textspace)

    def nth(l, n):
        """
        Returns only nth elemnt in a list.
        """
        n = lloc(l, n)
        return [a[n] for a in l]

    def lloc(l, n):
        """
        List location in list of list structure.
        Enable the use of negative locations:
        -1 is the last element, -2 second last...
        """
        if n < 0:
            return len(l[0]) + n
        else:
            return n

    def mxrange(lr):
        """
        Multiple xranges. Can be used to traverse matrices.
        This function is very slow due to unknown number of
        parameters.

        >>> mxrange([3,5])
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

        >>> mxrange([[3,5,1],[9,0,-3]])
        [(3, 9), (3, 6), (3, 3), (4, 9), (4, 6), (4, 3)]

        """
        if not len(lr):
            yield ()
        else:
            # it can work with single numbers
            index = lr[0]
            if isinstance(index, int):
                index = [index]
            for a in range(*index):
                for b in mxrange(lr[1:]):
                    yield tuple([a] + list(b))

    def print_figure(fig, *args, **kwargs):
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(*args, **kwargs)

    sums = avranks

    nnames = names
    ssums = sums

    if lowv is None:
        lowv = min(1, int(math.floor(min(ssums))))
    if highv is None:
        highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4

    k = len(sums)

    lines = None

    linesblank = 0
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a

    distanceh = 0.25

    cline += distanceh

    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant

    fig = plt.figure(figsize=(width, height))
    fig.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
    ax.set_axis_off()

    hf = 1. / height  # height factor
    wf = 1. / width

    def hfl(l):
        return [a * hf for a in l]

    def wfl(l):
        return [a * wf for a in l]

    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    def line(l, color='k', **kwargs):
        """
        Input is a list of pairs of points.
        """
        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    line([(textspace, cline), (width - textspace, cline)], linewidth=2)

    bigtick = 0.3
    smalltick = 0.15
    linewidth = 2.0
    linewidth_sign = 4.0

    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        line([(rankpos(a), cline - tick / 2),
            (rankpos(a), cline)],
            linewidth=2)

    for a in range(lowv, highv + 1):
        text(rankpos(a), cline - tick / 2 - 0.05, str(a),
            ha="center", va="bottom", size=16)

    k = len(ssums)

    def filter_names(name):
        return name

    space_between_names = 0.24

    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * space_between_names
        line([(rankpos(ssums[i]), cline),
            (rankpos(ssums[i]), chei),
            (textspace - 0.1, chei)],
            linewidth=linewidth)
        if labels:
            text(textspace + 0.3, chei - 0.075, format(ssums[i], '.4f'), ha="right", va="center", size=10)
        text(textspace - 0.2, chei, filter_names(nnames[i]), ha="right", va="center", size=16)

    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * space_between_names
        line([(rankpos(ssums[i]), cline),
            (rankpos(ssums[i]), chei),
            (textspace + scalewidth + 0.1, chei)],
            linewidth=linewidth)
        if labels:
            text(textspace + scalewidth - 0.3, chei - 0.075, format(ssums[i], '.4f'), ha="left", va="center", size=10)
        text(textspace + scalewidth + 0.2, chei, filter_names(nnames[i]),
            ha="left", va="center", size=16)

    # no-significance lines
    def draw_lines(lines, side=0.05, height=0.1):
        start = cline + 0.2

        for l, r in lines:
            line([(rankpos(ssums[l]) - side, start),
                (rankpos(ssums[r]) + side, start)],
                linewidth=linewidth_sign)
            start += height
            print('drawing: ', l, r)

    # draw_lines(lines)
    start = cline + 0.2
    side = -0.02
    height = 0.1

    # draw no significant lines
    # get the cliques
    cliques = form_cliques(p_values, nnames)
    i = 1
    achieved_half = False
    print(nnames)
    for clq in cliques:
        if len(clq) == 1:
            continue
        print(clq)
        min_idx = np.array(clq).min()
        max_idx = np.array(clq).max()
        if min_idx >= len(nnames) / 2 and achieved_half == False:
            start = cline + 0.25
            achieved_half = True
        line([(rankpos(ssums[min_idx]) - side, start),
            (rankpos(ssums[max_idx]) + side, start)],
            linewidth=linewidth_sign)
        start += height
    return fig

def form_cliques(p_values, nnames):
    """
    This method forms the cliques
    """
    # first form the numpy matrix data
    m = len(nnames)
    g_data = np.zeros((m, m), dtype=np.int64)
    for p in p_values:
        if p[3] == False:
            i = np.where(nnames == p[0])[0][0]
            j = np.where(nnames == p[1])[0][0]
            min_i = min(i, j)
            max_j = max(i, j)
            g_data[min_i, max_j] = 1

    g = networkx.Graph(g_data)
    return networkx.find_cliques(g)

def draw_cd_diagram(strategy, metric, asc, df_perf=None, alpha=0.05, title=None, labels=False):
    """
    Draws the critical difference diagram given the list of pairwise classifiers that are
    significant or not
    """
    p_values, average_ranks, _ = wilcoxon_holm(df_perf=df_perf, alpha=alpha, strategy=strategy, metric=metric, asc=asc)

    print(average_ranks)

    for p in p_values:
        print(p)


    fig = graph_ranks(average_ranks.values, average_ranks.keys(), p_values,
                cd=None, reverse=True, width=9, textspace=1.5, labels=labels)

    font = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'normal',
        'size': 22,
        }
    st.pyplot(fig)
    # if title:
    #     plt.title(title,fontdict=font, y=0.9, x=0.5)


    # plt.savefig(title + 'cd-diagram.jpg',bbox_inches='tight')

def wilcoxon_holm(alpha=0.05, df_perf=None, strategy='classifier_name', metric='accuracy', asc=False):
    """
    Applies the wilcoxon signed rank test between each pair of algorithm and then use Holm
    to reject the null's hypothesis
    """
    print(pd.unique(df_perf[strategy]))
    # count the number of tested datasets per classifier
    df_counts = pd.DataFrame({'count': df_perf.groupby(
        [strategy]).size()}).reset_index()
    # get the maximum number of tested datasets
    max_nb_datasets = df_counts['count'].max()
    # get the list of classifiers who have been tested on nb_max_datasets
    classifiers = list(df_counts.loc[df_counts['count'] == max_nb_datasets]
                    [strategy])
    # test the null hypothesis using friedman before doing a post-hoc analysis

    print("WHY ERROR?", *(
        np.array(df_perf.loc[df_perf[strategy] == c][metric])
        for c in classifiers))
    friedman_p_value = friedmanchisquare(*(
        np.array(df_perf.loc[df_perf[strategy] == c][metric])
        for c in classifiers))[1]
    if friedman_p_value >= alpha:
        # then the null hypothesis over the entire classifiers cannot be rejected
        print('the null hypothesis over the entire classifiers cannot be rejected')
        exit()
    # get the number of classifiers
    m = len(classifiers)
    # init array that contains the p-values calculated by the Wilcoxon signed rank test
    p_values = []
    # loop through the algorithms to compare pairwise
    for i in range(m - 1):
        # get the name of classifier one
        classifier_1 = classifiers[i]
        # get the performance of classifier one
        perf_1 = np.array(df_perf.loc[df_perf[strategy] == classifier_1][metric]
                        , dtype=np.float64)
        for j in range(i + 1, m):
            # get the name of the second classifier
            classifier_2 = classifiers[j]
            # get the performance of classifier one
            perf_2 = np.array(df_perf.loc[df_perf[strategy] == classifier_2]
                            [metric], dtype=np.float64)
            # calculate the p_value
            p_value = wilcoxon(perf_1, perf_2, zero_method='pratt')[1]
            # appen to the list
            p_values.append((classifier_1, classifier_2, p_value, False))
    # get the number of hypothesis
    k = len(p_values)
    # sort the list in acsending manner of p-value
    p_values.sort(key=operator.itemgetter(2))

    # loop through the hypothesis
    for i in range(k):
        # correct alpha with holm
        new_alpha = float(alpha / (k - i))
        # test if significant after holm's correction of alpha
        if p_values[i][2] <= new_alpha:
            p_values[i] = (p_values[i][0], p_values[i][1], p_values[i][2], True)
        else:
            # stop
            break
    # compute the average ranks to be returned (useful for drawing the cd diagram)
    # sort the dataframe of performances
    sorted_df_perf = df_perf.loc[df_perf[strategy].isin(classifiers)]. \
        sort_values([strategy, 'dataset'])
    # get the rank data
    rank_data = np.array(sorted_df_perf[metric]).reshape(m, max_nb_datasets)

    # create the data frame containg the accuracies
    df_ranks = pd.DataFrame(data=rank_data, index=np.sort(classifiers), columns=
    np.unique(sorted_df_perf['dataset']))

    # number of wins
    dfff = df_ranks.rank(ascending=False)
    print(dfff[dfff == 1.0].sum(axis=1))

    # average the ranks
    average_ranks = df_ranks.rank(ascending=asc).mean(axis=1).sort_values(ascending=False)
    # return the p-values and the average ranks
    return p_values, average_ranks, max_nb_datasets

def dataframe_with_selections(df):
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", False)

    # Get dataframe row-selections from user with st.data_editor
    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        disabled=df.columns,
        width=2000,
        height=250
    )

    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]
    return selected_rows.drop('Select', axis=1)

with st.container():
    st.image('title.jpg')
    # st.title("⚙️🔩 Shapex Engine")
tab_desc, tab_framework, tab_acc, tab_time, tab_diy = st.tabs(["Description", "Benchmark Details", "Accuracy Evaluation", "Compute Time", "Interactive Explanations"])  

with tab_desc:
    # col1, col2, col3 = st.columns([0.25, 5, 0.25])
    # with col2:
    st.markdown('## Shapley value explanations')
    st.markdown(
    """
    Interpreting decisions made by machine learning models helps build trust in their predictions, ultimately facilitating their practical application. Shapley values have emerged as a popular and theoretically robust method for interpreting models by quantifying the contribution of each feature toward individual predictions. The inherent complexity associated with the computation of Shapley values as an NP-hard problem has driven the development of numerous approximation techniques, leading to a plethora of options in literature. This abundance of choices has created a substantial gap in determining the most appropriate approach for practical applications. To address this gap, we propose ShapX, a web engine that comprehensively evaluates 17 approximation methods across diverse regression and classification tasks. ShapX facilitates an interactive exploration of the strengths and limitations of various Shapley value approximations by guiding users through the suitable selections of replacement and tractable estimation strategies. Ultimately, our study reveals that strategies competent at capturing all the feature interactions lead to accurate estimations of Shapley values. ShapX also allows users to effortlessly upload their own dataset along with the corresponding machine learning model, enabling them to obtain detailed individualized explanations.
    """
    )
    st.image('desc.jpg', caption='Overview of ShapX Engine')
    st.markdown("""
    ## User manual
    """)
    st.image('frames.jpg', caption='Frames of the ShapX Engine')
    st.markdown("""
    * Frame (a): An introductory gateway to Shapley value explanations.
    * Frame (b): An extensive evaluation of the accuracy of different aspects of Shapley value estimation. 
    * Frame (c): A per-instance performance comparison and scalability of various approximations.
    * Frame (d): A dynamic interface for users to upload their own datasets and models and receive personalized explanations for individual instances.
    """)
    st.markdown(
    """
    ## Contributors

    * [Suchit Gupte](https://github.com/SuchitGupte) (The Ohio State University)
    * [John Paparrizos](https://www.paparrizos.org/) (The Ohio State University)
    """
    )

with tab_framework:
    st.markdown('## Benchmark Details: ')

    tab_1, tab_2, tab_3 = st.tabs(["Overview", "Evaluation Metrics", "Datasets & Models"])  

    with tab_1:
        background = Image.open('./data/replacement.png')
        st.markdown('### Overview: ')
        st.markdown(
        """
        We break down the approximation of the Shapley values into two principal dimensions. 
        These dimensions also serve as a guide for setting up the evaluation framework. 
        The first dimension involves properly treating missing values with the help of different replacement strategies. 
        We deploy each replacement strategy against an exhaustive computation of Shapley values. 
        This evaluation measure will highlight the strengths and weaknesses of replacement strategies, 
        aiding future research in selecting the most reliable strategy. 
        """
        )
        col1, col2, col3 = st.columns([0.25, 5, 0.25])
        col2.image(background, caption='Replacement strategies address the absence of features, eliminating the necessity to train an exponential number of models and mitigating computational complexity.')
        # with col2:
        st.markdown("""
        The second dimension focuses on tractable estimation strategies, which are crucial for efficiently computing Shapley values.
        We analyze the performance of these tractable estimation strategies using established approximation algorithms. 
        We systematically evaluate 8 distinct replacement strategies and 17 distinct approximation algorithms across a diverse set of 200 datasets. 
        This comprehensive evaluation enables us to thoroughly assess the performance and efficacy of individual strategies and 
        the various approximations in estimating Shapley values across varied data scenarios.
        Following are the Shapley value estimation approaches:
        """
        )
        st.markdown("""
            | Approaches                                | Estimation strategy                   | Replacement strategy                                | 
            |-------------------------------------------|---------------------------------------|-----------------------------------------------------|
            | Exhaustive Sampling                       | Exact                                 | Separate Models                                     |   
            | Interactions-based Method for Explanation | RO                                    | Marginal: Empirical                                 |  
            | Conditional Expectations Shapley          | RO                                    | Conditional: Empirical                              | 
            | Shapley Cohort refinement                 | RO                                    | Conditional: Empirical                              |
            | Multilinear Sampling                      | MLE                                   | Marginal: Empirical                                 |    
            | KernelSHAP                                | WLS                                   | Marginal: Empirical                                 |   
            | Parametric KernelSHAP                     | WLS                                   | Conditional: Gaussian/Copula                        |  
            | Non-Parametric KernelSHAP                 | WLS                                   | Conditional: Empirical                              |   
            | SGD-Shapley                               | WLS                                   | Predetermined: Mean                                 |    
            | Independent LinearSHAP                    | Linear                                | Marginal: Empirical                                 |    
            | Correlated LinearSHAP                     | Linear                                | Conditional: Gaussian                               |   
            | Tree Interventional                       | Tree                                  | Marginal: Empirical                                 |   
            | Tree Path-dependent                       | Tree                                  | Conditional: Empirical                              | 
            | DeepLIFT                                  | Deep                                  | Predetermined: All-zeros                            |   
            | DeepSHAP                                  | Deep                                  | Marginal: Empirical                                 |    
            | DASP                                      | Deep                                  | Predetermined: Mean                                 |   
        """
        )
            # | FastSHAP                                  | WLS                                   | Conditional: Surrogate model                        |   

    with tab_2:        
        st.markdown('### Evaluation metrics: ')
        st.markdown(
        """
        #### 1. Explanation Error:

        As we implement replacement strategies to address missing features in Shapley value estimation, the absence of ground truth Shapley values presents a clear obstacle in the evaluation. 
        Consequently, we must employ an alternative evaluation metric to assess the accuracy of the approaches, such as the Explanation Error. 
        The motivation for explanation error stems from the additive nature of the Shapley values. Shapley values indicate the individual contributions of input features towards shifting the model output from the average model prediction to the actual prediction value given a specific instance. 
        When dealing with a black-box model $f$ and an explicand $x$, the prediction for the explicand can be articulated as a summation of the average model
        prediction and the individual Shapley values of each feature.

        """
        )
        background = Image.open('experror.jpg')
        col1, col2, col3 = st.columns([0.25, 5, 0.25])
        col2.image(background, caption='Additive property of Shapley values: Motivation for Explanation Error.')

        st.markdown(
        """
        In the above figure, $\Phi_0$ symbolizes the average model prediction, while $\Phi_i$s refer to the Shapley values assigned to each input feature. 
        The objective of any Shapley value estimation technique is to approximate these $\Phi_i$s. 
        We can determine the quality of any approximation by measuring the discrepancy between the actual model prediction and the sum of the average model prediction$(\Phi_0)$ and the Shapley value approximations$(\Phi_is)$. 
        A smaller disparity signifies a higher level of accuracy in the approximation.

        #### 2. Compute Time:

        Since Shapley values are a local feature attribution technique, we compare the instance-wise computational efficiency of different approaches. 
        The evaluation encompasses datasets that contain up to 45 features. Using the per-instance runtime comparison, we anticipate the trend of the runtime results as the dimensionality increases. 
        We determine which methods are most suitable for handling high-dimensional data by analyzing the runtime results of different approaches.

        """
        )
    
    with tab_3:
        st.markdown(
        """

        ### Datasets

        For the scope of the study, we focus on regression and binary/multi-class classification tabular datasets. We utilize a total of 200 publicly available datasets from the UCI Machine Learning Repository.
        Within the datasets, there are as many as 60 input features, and the number of instances ranges from 100 to 1 million.
        The figure below highlights the dimensions and scale of these datasets. Each dataset is split into training and testing sets for 
        model training and computing Shapley value estimations. Since the Shapley values are a local feature attribution technique, the number of 
        instances in the dataset has a very insignificant impact on the Shapley value estimates; however, data dimensionality significantly affects 
        the estimation. 
        """
        )
        
        background = Image.open('./data/datadimscale_rev.png')
        col1, col2, col3 = st.columns([0.2, 5, 0.2])
        col2.image(background, caption=' Dimensionality and scalability distribution across 200 regression and classification datasets from the UCI ML repository.')

        st.markdown("""
        To ease reproducibility, we share our results over an established benchmark. 
            
        * Download all the datasets [here](https://github.com/TheDatumOrg/ShapleyValuesEval/tree/main/data).
        
        """)
        st.markdown(
        """
        ### Models
                    
        We utilize the supervised machine learning framework used to tackle regression and classification tasks. We use the following model architectures - Linear models, Ensemble Learning, Gradient Boosting, Neural Networks, and Support Vector Machines. 
        To conduct a thorough evaluation, we integrate models representing each category. Shapley values intend to explain a black box model by leveraging the model itself, thereby negating the significance of the model's fit quality. 
        Consequently, this allows us to use vanilla versions of each model with default hyperparameters.
        """
        )

    # with tab_4:    
    #     st.markdown(
    #     """
    #     ### Models
                    
    #     We broadly classify the supervised machine learning models used to tackle regression-based problems into 5 categories - Linear models, Ensemble Learning, Gradient Boosting, Neural Networks, and Support Vector Machines. 
    #     To conduct a thorough evaluation, we integrate models representing each category. Shapley values intend to explain a black box model by leveraging the model itself, thereby negating the significance of the model's fit quality. 
    #     Consequently, this allows us to use vanilla versions of each model with default hyperparameters.
    #     """
    #     )
    
with tab_acc:
    # col1, col2, col3 = st.columns([0.25, 5, 0.25])
    # with col2:
    st.markdown(
    """
    ## Accuracy Evaluation: 
    
    We use the $R^2$ test to analyze the explanation error. The $R^2$ test or the coefficient of determination, is a statistical test designed for regression analysis to assess the quality of fit. 
    $R^2$  values, spanning from 0 to 1, are often converted into percentages to represent the accuracy of any regression model. 
    For computing the $R^2$ value, we treat $f(x^e)$ as the ground truth and $\Phi_0 + \sum_{i=1}^{|D|} \Phi_i$ as the predicted value. 
    A strategy with an $R^2$  value approaching 1 indicates that it can approximate the Shapley values accurately.
    """
    )

    tab_repl, tab_app = st.tabs(["Replacement Strategy", "Approximations"])  
    with tab_repl:
        df = pd.read_csv('data/agnostic.csv')
        statplot_df = pd.read_csv('data/statplot_df.csv')
        model = st.selectbox('###### Pick a model type:', ['Model agnostic', 'Linear models', 'Tree-based models', 'Neural networks'], key='model_repl')
        
        # rep_cat = st.selectbox('###### Pick a replacement category:', ['Predetermined', 'Marginal Distribution', 'Conditional Distribution'], key='repl_cat')
    

        compare_cat = {
            'Predetermined': ['Zero', 'Mean', 'Separate models'],
            'Marginal Distribution': ['Marginal', 'Uniform', 'Separate models'],
            'Conditional Distribution': ['Conditional', 'Gaussian', 'Copula', 'Separate models']
        }
        container_method = st.container()
        all_repl_cat = st.checkbox("Select all", key='all_repl_cat_time')
        
        if all_repl_cat: 
            repl_cat_list_family = container_method.multiselect('###### Pick a replacement category: ', ['Predetermined', 'Marginal Distribution', 'Conditional Distribution'], ['Predetermined', 'Marginal Distribution', 'Conditional Distribution'], key='repl_cat1_time')
        else: 
            repl_cat_list_family = container_method.multiselect('###### Pick a replacement category: ',['Predetermined', 'Marginal Distribution', 'Conditional Distribution'], key='repl_cat2_time',  default=['Predetermined', 'Marginal Distribution', 'Conditional Distribution'])
        
        values = [compare_cat[key] for key in repl_cat_list_family]
        # st.markdown(values)

        repl_categories_l = list()
        if len(values) != 0:
            for val in values:
                for v in val:
                    repl_categories_l.append(v)
        else:
            st.error('Please select a replacement category!', icon="🚨")

        repl_categories = list(set(sorted(repl_categories_l)))
        # st.markdown(repl_categories)
        
        # repl_list = list(set(df['Replacement Strategy'].values))
        repl_list = repl_categories
        container_method = st.container()
        all_repl = st.checkbox("Select all", key='all_repl')


        if all_repl: 
            repl_list_family = container_method.multiselect('###### Replacement strategies: ', sorted(repl_list), sorted(repl_list), key='all_repl_1')
        else: 
            repl_list_family = container_method.multiselect('###### Replacement strategies: ', sorted(repl_list), key='all_repl_2', default=sorted(repl_list))


        
        if model == 'Model agnostic':
            col1, col2, col3 = st.columns([0.05, 2, 0.05])
            with col2:
                statplot_df_mod = statplot_df[statplot_df['Replacement Strategy'].isin(repl_list_family)]
                st.markdown('#### Statistical test rankings: ')


                if len(repl_list_family) <= 2:
                    st.warning('Select atleast 3 replacement strategies for obtaining a statistical ranking!', icon="⚠️")

                    # st.markdown("###### Select atleast 3 replacement strategies for obtaining a statistical ranking")
                else:
                    draw_cd_diagram(strategy='Replacement Strategy', metric='accuracy', asc=False, df_perf=statplot_df_mod, labels=True)
                
                df_mod = df[df['Replacement Strategy'].isin(repl_list_family)]
                st.markdown('#### Overall comparison: ')
                plot_box_plot_repl(df_mod)

            
        elif model == 'Linear models':
            lr = df[df['Model'] == 'Linear Regression']
            col1, col2, col3 = st.columns([0.05, 2, 0.05])
            with col2:
                statplot_df = lr.copy()
                # df_mod_stat = statplot_df.drop(['Model', 'Unnamed: 0'], axis=1).reset_index(drop=True)
                statplot_df_mod = statplot_df[statplot_df['Replacement Strategy'].isin(repl_list_family)]
                st.markdown('#### Statistical test rankings: ')
                if len(repl_list_family) <= 2:
                    # st.markdown("###### Select atleast 3 replacement strategies for obtaining a statistical ranking")
                    st.warning('Select atleast 3 replacement strategies for obtaining a statistical ranking!', icon="⚠️")

                else:
                    draw_cd_diagram(strategy='Replacement Strategy', metric='accuracy', asc=False, df_perf=statplot_df_mod, labels=True)
            
                df_mod = lr[lr['Replacement Strategy'].isin(repl_list_family)]
                st.markdown('#### Overall comparison: ')
                plot_box_plot_repl(df_mod)
        
        
        elif model == 'Tree-based models':
            model_list = ['XGBoost', 'Decision Trees', 'Random Forest']
            tree = df[df['Model'] == 'XGBoost']
        
            col1, col2, col3 = st.columns([0.05, 2, 0.05])
            with col2:
                statplot_df = tree.copy()
                statplot_df_mod = statplot_df[statplot_df['Replacement Strategy'].isin(repl_list_family)]
                st.markdown('#### Statistical test rankings: ')
                if len(repl_list_family) <= 2:
                    # st.markdown("###### Select atleast 3 replacement strategies for obtaining a statistical ranking")
                    st.warning('Select atleast 3 replacement strategies for obtaining a statistical ranking!', icon="⚠️")

                else:
                    draw_cd_diagram(strategy='Replacement Strategy', metric='accuracy', asc=False, df_perf=statplot_df_mod, labels=True)


                df_mod = tree[tree['Replacement Strategy'].isin(repl_list_family)]
                st.markdown('#### Overall comparison: ')
                plot_box_plot_repl(df_mod)
        
        elif model == 'Neural networks':
            model_list = ['Neural network']
            nn = df[df['Model'] == 'Neural network']

            col1, col2, col3 = st.columns([0.05, 2, 0.05])
            with col2:
                statplot_df = nn.copy()
                statplot_df_mod = statplot_df[statplot_df['Replacement Strategy'].isin(repl_list_family)]
                st.markdown('#### Statistical test rankings: ')
                if len(repl_list_family) <= 2:
                    # st.markdown("###### Select atleast 3 replacement strategies for obtaining a statistical ranking")
                    st.warning('Select atleast 3 replacement strategies for obtaining a statistical ranking!', icon="⚠️")

                else:
                    draw_cd_diagram(strategy='Replacement Strategy', metric='accuracy', asc=False, df_perf=statplot_df_mod, labels=True)

                df_mod = nn[nn['Replacement Strategy'].isin(repl_list_family)]
                st.markdown('#### Overall comparison: ')
                plot_box_plot_repl(df_mod)

        else:
            print("Not yet implemented")

    with tab_app:
        model = st.selectbox('###### Pick a model type: ', ['Model agnostic', 'Linear models', 'Tree-based models', 'Neural networks'], key='model_app')
        
        if model == 'Model agnostic':
            app_df = pd.read_csv("data/tables/agnostic.csv")
            compare_est = {
            'Random Order': ['Exhaustive', 'IME', 'CES', 'Cohort'],
            'Weighted Least Squares': ['KernelSHAP', 'SGDShapely', 'Exhaustive', 'Parametric KernelSHAP', 'Non-parametric KernelSHAP'],
            'Multilinear Extension': ['MLE', 'Exhaustive', 'IME'],
            # 'Linear': ['Linear (correlated)', 'Linear (independent)', 'Exhaustive'],
            # 'Tree': ['Tree (path dependent)', 'Tree (interventional)', 'Exhaustive'],
            # 'Deep': ['DASP', 'DeepLIFT', 'DeepSHAP']
            }
            container_method = st.container()
            all_est_cat = st.checkbox("Select all", key='all_est_cat')
            
            if all_est_cat: 
                est_cat_list_family = container_method.multiselect('###### Pick an estimation strategy: ', ['Random Order', 'Weighted Least Squares', 'Multilinear Extension'], ['Random Order', 'Weighted Least Squares', 'Multilinear Extension'], key='est_cat1')
            else: 
                est_cat_list_family = container_method.multiselect('###### Pick an estimation strategy: ',['Random Order', 'Weighted Least Squares', 'Multilinear Extension'], key='est_cat2',  default=['Random Order', 'Weighted Least Squares', 'Multilinear Extension'])
            
            values = [compare_est[key] for key in est_cat_list_family]
            # st.markdown(values)

            est_categories_l = list()
            if len(values) != 0:
                for val in values:
                    for v in val:
                        est_categories_l.append(v)
            else:
                st.error('Please select an estimation strategy!', icon="🚨")
            
            est_categories = list(set(sorted(est_categories_l)))
            # st.markdown(est_categories)
            
            # app_list = list(set(app_df['Approximation'].values))
            app_list = est_categories
            container_method = st.container()
            all_app = st.checkbox("Select all", key='all_app')

            if all_app: 
                app_list_family = container_method.multiselect('###### Pick an approximation: ', sorted(app_list), sorted(app_list), key='all_app_1')
            else: 
                app_list_family = container_method.multiselect('###### Pick an approximation: ', sorted(app_list), key='all_app_2', default=sorted(app_list))
            
            col1, col2, col3 = st.columns([0.05, 2, 0.05])
            with col2:
                st.markdown('#### Statistical test rankings: ')
                if len(app_list_family) <= 2:
                    # st.markdown("###### Select atleast 3 estimation strategies for obtaining a statistical ranking")
                    st.warning('Select atleast 3 replacement strategies for obtaining a statistical ranking!', icon="⚠️")

                else:
                    statplot_df_mod = app_df[app_df['Approximation'].isin(app_list_family)]
                    draw_cd_diagram(strategy='Approximation', metric='Accuracy', asc=False, df_perf=statplot_df_mod, labels=True)
                    st.markdown('#### Overall comparison: ')
                    plot_box_plot_app(statplot_df_mod)
                

        elif model == 'Linear models':
            app_df = pd.read_csv("data/tables/linear.csv")
            compare_est = {
            'Random Order': ['Exhaustive', 'IME', 'CES', 'Cohort'],
            'Weighted Least Squares': ['KernelSHAP', 'SGDShapely', 'Exhaustive', 'Parametric KernelSHAP', 'Non-parametric KernelSHAP'],
            'Multilinear Extension': ['MLE', 'Exhaustive', 'IME'],
            'Linear': ['Linear (correlated)', 'Linear (independent)', 'Exhaustive'],
            # 'Tree': ['Tree (path dependent)', 'Tree (interventional)', 'Exhaustive'],
            # 'Deep': ['DASP', 'DeepLIFT', 'DeepSHAP']
            }
            container_method = st.container()
            all_est_cat = st.checkbox("Select all", key='all_est_cat')
            
            if all_est_cat: 
                est_cat_list_family = container_method.multiselect('###### Pick an estimation strategy: ', ['Random Order', 'Weighted Least Squares', 'Multilinear Extension','Linear'], ['Random Order', 'Weighted Least Squares', 'Multilinear Extension','Linear'], key='est_cat1')
            else: 
                est_cat_list_family = container_method.multiselect('###### Pick an estimation strategy: ',['Random Order', 'Weighted Least Squares', 'Multilinear Extension','Linear'], key='est_cat2',  default=['Random Order', 'Weighted Least Squares', 'Multilinear Extension','Linear'])
            
            values = [compare_est[key] for key in est_cat_list_family]
            # st.markdown(values)

            est_categories_l = list()
            if len(values) != 0:
                for val in values:
                    for v in val:
                        est_categories_l.append(v)
            else:
                st.error('Please select an estimation strategy!', icon="🚨")
            
            est_categories = list(set(sorted(est_categories_l)))
            # st.markdown(est_categories)

            app_list = est_categories
            # app_list = list(set(app_df['Approximation'].values))
            container_method = st.container()
            all_app = st.checkbox("Select all", key='all_app')

            if all_app: 
                app_list_family = container_method.multiselect('###### Pick an estimation strategy: ', sorted(app_list), sorted(app_list), key='all_app_1')
            else: 
                app_list_family = container_method.multiselect('###### Pick an estimation strategy: ', sorted(app_list), key='all_app_2', default=sorted(app_list))

            col1, col2, col3 = st.columns([0.05, 2, 0.05])
            with col2:
                st.markdown('#### Statistical test rankings: ')
                if len(app_list_family) <= 2:
                    # st.markdown("###### Select atleast 3 estimation strategies for obtaining a statistical ranking")
                    st.warning('Select atleast 3 replacement strategies for obtaining a statistical ranking!', icon="⚠️")

                else:
                    statplot_df_mod = app_df[app_df['Approximation'].isin(app_list_family)]
                    draw_cd_diagram(strategy='Approximation', metric='Accuracy', asc=False, df_perf=statplot_df_mod, labels=True)
                    st.markdown('#### Overall comparison: ')
                    plot_box_plot_app(statplot_df_mod)


        elif model == 'Tree-based models':
            app_df = pd.read_csv("data/tables/tree.csv")
            compare_est = {
            'Random Order': ['Exhaustive', 'IME', 'CES', 'Cohort'],
            'Weighted Least Squares': ['KernelSHAP', 'SGDShapely', 'Exhaustive', 'Parametric KernelSHAP', 'Non-parametric KernelSHAP'],
            'Multilinear Extension': ['MLE', 'Exhaustive', 'IME'],
            # 'Linear': ['Linear (correlated)', 'Linear (independent)', 'Exhaustive'],
            'Tree': ['Tree (path dependent)', 'Tree (interventional)', 'Exhaustive'],
            # 'Deep': ['DASP', 'DeepLIFT', 'DeepSHAP']
            }
            container_method = st.container()
            all_est_cat = st.checkbox("Select all", key='all_est_cat')
            
            if all_est_cat: 
                est_cat_list_family = container_method.multiselect('###### Pick an estimation strategy: ', ['Random Order', 'Weighted Least Squares', 'Multilinear Extension', 'Tree'], ['Random Order', 'Weighted Least Squares', 'Multilinear Extension', 'Tree'], key='est_cat1')
            else: 
                est_cat_list_family = container_method.multiselect('###### Pick an estimation strategy: ',['Random Order', 'Weighted Least Squares', 'Multilinear Extension', 'Tree'], key='est_cat2',  default=['Random Order', 'Weighted Least Squares', 'Multilinear Extension', 'Tree'])
            
            values = [compare_est[key] for key in est_cat_list_family]
            # st.markdown(values)

            est_categories_l = list()
            if len(values) != 0:
                for val in values:
                    for v in val:
                        est_categories_l.append(v)
            else:
                st.error('Please select an estimation strategy!', icon="🚨")
            
            est_categories = list(set(sorted(est_categories_l)))
            # st.markdown(est_categories)
            
            # app_list = list(set(app_df['Approximation'].values))
            app_list = est_categories
            # app_list = list(set(app_df['Approximation'].values))
            container_method = st.container()
            all_app = st.checkbox("Select all", key='all_app')

            if all_app: 
                app_list_family = container_method.multiselect('###### Pick an estimation strategy: ', sorted(app_list), sorted(app_list), key='all_app_1')
            else: 
                app_list_family = container_method.multiselect('###### Pick an estimation strategy: ', sorted(app_list), key='all_app_2', default=sorted(app_list))
            
            col1, col2, col3 = st.columns([0.05, 2, 0.05])
            with col2:
                st.markdown('#### Statistical test rankings: ')
                if len(app_list_family) <= 2:
                    # st.markdown("###### Select atleast 3 estimation strategies for obtaining a statistical ranking")
                    st.warning('Select atleast 3 replacement strategies for obtaining a statistical ranking!', icon="⚠️")
                else:
                    statplot_df_mod = app_df[app_df['Approximation'].isin(app_list_family)]
                    draw_cd_diagram(strategy='Approximation', metric='Accuracy', asc=False, df_perf=statplot_df_mod, labels=True)
                    st.markdown('#### Overall comparison: ')
                    plot_box_plot_app(statplot_df_mod)


        elif model == 'Neural networks':
            app_df = pd.read_csv("data/tables/nn.csv")
            compare_est = {
            'Random Order': ['Exhaustive', 'IME', 'CES', 'Cohort'],
            'Weighted Least Squares': ['KernelSHAP', 'SGDShapely', 'Exhaustive', 'Parametric KernelSHAP', 'Non-parametric KernelSHAP'],
            'Multilinear Extension': ['MLE', 'Exhaustive', 'IME'],
            # 'Linear': ['Linear (correlated)', 'Linear (independent)', 'Exhaustive'],
            # 'Tree': ['Tree (path dependent)', 'Tree (interventional)', 'Exhaustive'],
            'Deep': ['DASP', 'DeepLIFT', 'DeepSHAP']
            }
            container_method = st.container()
            all_est_cat = st.checkbox("Select all", key='all_est_cat')
            
            if all_est_cat: 
                est_cat_list_family = container_method.multiselect('###### Pick an estimation strategy: ', ['Random Order', 'Weighted Least Squares', 'Multilinear Extension', 'Deep'], ['Random Order', 'Weighted Least Squares', 'Multilinear Extension', 'Deep'], key='est_cat1')
            else: 
                est_cat_list_family = container_method.multiselect('###### Pick an estimation strategy: ',['Random Order', 'Weighted Least Squares', 'Multilinear Extension', 'Deep'], key='est_cat2',  default=['Random Order', 'Weighted Least Squares', 'Multilinear Extension', 'Deep'])
            
            values = [compare_est[key] for key in est_cat_list_family]
            # st.markdown(values)

            est_categories_l = list()
            if len(values) != 0:
                for val in values:
                    for v in val:
                        est_categories_l.append(v)
            else:
                st.error('Please select an estimation strategy!', icon="🚨")
            
            est_categories = list(set(sorted(est_categories_l)))
            # st.markdown(est_categories)
            
            # app_list = list(set(app_df['Approximation'].values))
            app_list = est_categories
            # app_list = list(set(app_df['Approximation'].values))
            container_method = st.container()
            all_app = st.checkbox("Select all", key='all_app')

            if all_app: 
                app_list_family = container_method.multiselect('###### Pick an estimation strategy: ', sorted(app_list), sorted(app_list), key='all_app_1')
            else: 
                app_list_family = container_method.multiselect('###### Pick an estimation strategy: ', sorted(app_list), key='all_app_2', default=sorted(app_list))
            col1, col2, col3 = st.columns([0.05, 2, 0.05])
            with col2:
                st.markdown('#### Statistical test rankings: ')
                if len(app_list_family) <= 2:
                    # st.markdown("###### Select atleast 3 estimation strategies for obtaining a statistical ranking")
                    st.warning('Select atleast 3 replacement strategies for obtaining a statistical ranking!', icon="⚠️")

                else:
                    statplot_df_mod = app_df[app_df['Approximation'].isin(app_list_family)]
                    draw_cd_diagram(strategy='Approximation', metric='Accuracy', asc=False, df_perf=statplot_df_mod, labels=True)
                    st.markdown('#### Overall comparison: ')
                    plot_box_plot_app(statplot_df_mod)

        else:
            st.markdown("Not implemented!")

with tab_time:
    # col1, col2, col3 = st.columns([0.25, 5, 0.25])
    # with col2:
    st.markdown("## Instance-wise compute time comparison: ")
    tab_repl, tab_app = st.tabs(["Replacement Strategy", "Approximations"])

    with tab_repl:
        df = pd.read_csv('data/agnostic.csv')
        df2 = pd.read_csv('data/time_df.csv')
        toff = pd.read_csv('data/tradeoff_repl.csv')


        model = st.selectbox('###### Pick a model type: ', ['Model agnostic', 'Linear models', 'Tree-based models', 'Neural networks'], key='model_repl_time')
        compare_cat = {
            'Predetermined': ['Zero', 'Mean', 'Separate models'],
            'Marginal Distribution': ['Marginal', 'Uniform', 'Separate models'],
            'Conditional Distribution': ['Conditional', 'Gaussian', 'Copula', 'Separate models']
        }
        container_method = st.container()
        all_repl_cat = st.checkbox("Select all", key='all_repl_cat')
        
        if all_repl_cat: 
            repl_cat_list_family = container_method.multiselect('###### Pick a replacement category: ', ['Predetermined', 'Marginal Distribution', 'Conditional Distribution'], ['Predetermined', 'Marginal Distribution', 'Conditional Distribution'], key='repl_cat1')
        else: 
            repl_cat_list_family = container_method.multiselect('###### Pick a replacement category: ',['Predetermined', 'Marginal Distribution', 'Conditional Distribution'], key='repl_cat2',  default=['Predetermined', 'Marginal Distribution', 'Conditional Distribution'])
        
        values = [compare_cat[key] for key in repl_cat_list_family]
        # st.markdown(values)

        repl_categories_l = list()
        if len(values) != 0:
            for val in values:
                for v in val:
                    repl_categories_l.append(v)
        else:
            st.error('Please select a replacement category!', icon="🚨")

        repl_categories = list(set(sorted(repl_categories_l)))


        # repl_list_time = list(set(df['Replacement Strategy'].values))
        repl_list_time = repl_categories
        container_method = st.container()
        all_repl_time = st.checkbox("Select all", key='all_repl_time')

        if all_repl_time: 
            repl_list_family_time = container_method.multiselect('###### Pick a replacement strategy: ', sorted(repl_list_time), sorted(repl_list_time), key='all_repl_time_1')
        else: 
            repl_list_family_time = container_method.multiselect('###### Pick a replacement strategy: ', sorted(repl_list_time), key='all_repl_time_2', default=sorted(repl_list_time))

        
        col1, col2, col3 = st.columns([0.05, 2, 0.05])
        with col2:
            if model == 'Model agnostic':
                time_df = df
                time_df2 = pd.read_csv("data/time_agn.csv")
                toff_df = toff[toff['Model'] == 'agnostic']

            elif model == 'Linear models':
                time_df = df[df['Model'] == 'Linear Regression']
                time_df2 = df2[df2['Model'] == 'linear']
                toff_df = toff[toff['Model'] == 'linear']


            elif model == 'Tree-based models':
                time_df = df[df['Model'] == 'XGBoost']
                time_df2 = df2[df2['Model'] == 'tree']
                toff_df = toff[toff['Model'] == 'tree']



            elif model == 'Neural networks':
                time_df = df[df['Model'] == 'Neural network']
                time_df2 = df2[df2['Model'] == 'nn']
                toff_df = toff[toff['Model'] == 'nn']


            else:
                print("Not implemented")

            # Not working
            import plotly.express as px
            
            # Filter the DataFrame
            filtered_df = time_df[time_df['Replacement Strategy'].isin(repl_list_family_time)]
            filtered_df = filtered_df.sort_values(by="Replacement Strategy", ascending=True)
            st.markdown('#### Overall comparison: ')
            
            # Create a bar chart using Plotly
            fig = px.bar(
                filtered_df,
                x="Time",
                y="Replacement Strategy",
                color="Replacement Strategy",
                color_discrete_sequence=px.colors.sequential.Magma,  # Mimic magma color scheme
                orientation="h"  # Horizontal bar chart
            )
            
            # Customize chart properties
            fig.update_layout(
                width=600,
                height=400,
                xaxis=dict(
                    type="log",  # Logarithmic scale for x-axis
                    title=dict(font=dict(size=18)),  # Adjust x-axis title font size
                    tickfont=dict(size=14)  # Adjust x-axis tick font size
                ),
                yaxis=dict(
                    title=dict(font=dict(size=18)),  # Adjust y-axis title font size
                    tickfont=dict(size=14)  # Adjust y-axis tick font size
                )
            )
            
            # Display the chart in Streamlit
            st.plotly_chart(fig, use_container_width=True)

            # # st.markdown(st.__version__)
            # filtered_data = time_df[time_df['Replacement Strategy'].isin(repl_list_family_time)]
            # values_for_repl = []
            # for repl_i in repl_list_family_time:
            #     st.markdown(repl_i)
            #     avg_for_i = filtered_data[filtered_data['Replacement Strategy']==repl_i]['Time'].mean() + 1e-3
            #     values_for_repl.append(avg_for_i)
            # new_df = pd.DataFrame({
            #     'Replacement Strategy': repl_list_family_time,
            #     'Time': values_for_repl
            # }, index=None)
            # st.markdown(new_df)

            
            # # Display the chart in Streamlit
            # st.markdown('#### Overall comparison: ')
            # bar_chart = alt.Chart(time_df[time_df['Replacement Strategy'].isin(repl_list_family_time)]).mark_line().encode(
            #     x=alt.X("Time:Q", scale=alt.Scale(type='log')),
            #     y=alt.Y("Replacement Strategy:N", sort="y",  axis=alt.Axis(labelAngle=0, labelLimit=200)),
            #     color=alt.Color("Replacement Strategy:N", scale=alt.Scale(scheme='magma'))
            # ).properties(
            #     width=600,
            #     height=400
            # ).configure_axis(
            #     labelFontSize=14,  # Adjust label font size
            #     titleFontSize=18,  # Adjust title font size
            #     tickSize=14  # Adjust tick size
            # )
        
            # st.altair_chart(bar_chart, use_container_width=True)

            
            line_chart = alt.Chart(time_df2[time_df2['Replacement Strategy'].isin(repl_list_family_time)]).mark_line().encode(
                x=alt.X("Features:N"),
                y=alt.Y("Time:Q",  scale=alt.Scale(type='log')),
                color=alt.Color("Replacement Strategy:N", scale=alt.Scale(scheme='magma'))
            ).properties(
                width=600,
                height=400
            ).configure_axis(
                labelFontSize=14,  # Adjust label font size
                titleFontSize=18,  # Adjust title font size
                tickSize=14  # Adjust tick size
            ).interactive()


            st.markdown('####  Impact of increasing dimensionality on compute time : ')
            st.altair_chart(line_chart, use_container_width=True)
                        
            st.markdown("#### Accuracy-Compute time tradeoff: ")
            scatter_chart = alt.Chart(toff_df[toff_df['Replacement Strategy'].isin(repl_list_family_time)]).mark_circle().encode(
                x=alt.X("Time:Q",  scale=alt.Scale(type='log')),
                y=alt.Y("Accuracy", scale=alt.Scale(domain=[70, 100])),
                color=alt.Color("Replacement Strategy:N", scale=alt.Scale(scheme='magma')),
                # size = 'Estimation Strategy'
                size=alt.Size('Replacement Category', scale=alt.Scale(range=[250, 1000]), legend=None)
            ).properties(
                width=600,
                height=400
            ).configure_axis(
                labelFontSize=14,  # Adjust label font size
                titleFontSize=18,  # Adjust title font size
                tickSize=14  # Adjust tick size
            ).interactive()

            st.altair_chart(scatter_chart, use_container_width=True)

    with tab_app:
        model = st.selectbox('###### Pick a model type:', ['Model agnostic', 'Linear models', 'Tree-based models', 'Neural networks'], key='model_app_time')

        if model == 'Model agnostic':
            time_df = pd.read_csv("data/tables/agnostic.csv")
            compare_est = {
            'Random Order': ['Exhaustive', 'IME', 'CES', 'Cohort'],
            'Weighted Least Squares': ['KernelSHAP', 'SGDShapely', 'Exhaustive', 'Parametric KernelSHAP', 'Non-parametric KernelSHAP'],
            'Multilinear Extension': ['MLE', 'Exhaustive', 'IME'],
            # 'Linear': ['Linear (correlated)', 'Linear (independent)', 'Exhaustive'],
            # 'Tree': ['Tree (path dependent)', 'Tree (interventional)', 'Exhaustive'],
            # 'Deep': ['DASP', 'DeepLIFT', 'DeepSHAP']
            }
            container_method = st.container()
            all_est_cat = st.checkbox("Select all", key='all_est_cat_time')
            
            if all_est_cat: 
                est_cat_list_family = container_method.multiselect('###### Pick an estimation strategy: ', ['Random Order', 'Weighted Least Squares', 'Multilinear Extension'], ['Random Order', 'Weighted Least Squares', 'Multilinear Extension'], key='est_cat1_time')
            else: 
                est_cat_list_family = container_method.multiselect('###### Pick an estimation strategy: ',['Random Order', 'Weighted Least Squares', 'Multilinear Extension'], key='est_cat2_time',  default=['Random Order', 'Weighted Least Squares', 'Multilinear Extension'])
            
            values = [compare_est[key] for key in est_cat_list_family]
            # st.markdown(values)

            est_categories_l = list()
            if len(values) != 0:
                for val in values:
                    for v in val:
                        est_categories_l.append(v)
            else:
                st.error('Please select an estimation strategy!', icon="🚨")
            
            est_categories = list(set(sorted(est_categories_l)))


            # app_list = list(set(time_df['Approximation'].values))
            app_list = est_categories
            container_method = st.container()
            all_app = st.checkbox("Select all", key='all_app_time')

            if all_app: 
                app_list_family_time = container_method.multiselect('###### Pick an estimation strategy:', sorted(app_list), sorted(app_list), key='all_app_time_1')
            else: 
                app_list_family_time = container_method.multiselect('###### Pick an estimation strategy:', sorted(app_list), key='all_app_time_2', default=sorted(app_list))
        
                

        elif model == 'Linear models':
            time_df = pd.read_csv("data/tables/linear.csv")
            compare_est = {
            'Random Order': ['Exhaustive', 'IME', 'CES', 'Cohort'],
            'Weighted Least Squares': ['KernelSHAP', 'SGDShapely', 'Exhaustive', 'Parametric KernelSHAP', 'Non-parametric KernelSHAP'],
            'Multilinear Extension': ['MLE', 'Exhaustive', 'IME'],
            'Linear': ['Linear (correlated)', 'Linear (independent)', 'Exhaustive'],
            # 'Tree': ['Tree (path dependent)', 'Tree (interventional)', 'Exhaustive'],
            # 'Deep': ['DASP', 'DeepLIFT', 'DeepSHAP']
            }
            container_method = st.container()
            all_est_cat = st.checkbox("Select all", key='all_est_cat_time')
            
            if all_est_cat: 
                est_cat_list_family = container_method.multiselect('###### Pick an estimation strategy: ', ['Random Order', 'Weighted Least Squares', 'Multilinear Extension','Linear'], ['Random Order', 'Weighted Least Squares', 'Multilinear Extension','Linear'], key='est_cat1_time')
            else: 
                est_cat_list_family = container_method.multiselect('###### Pick an estimation strategy: ',['Random Order', 'Weighted Least Squares', 'Multilinear Extension','Linear'], key='est_cat2_time',  default=['Random Order', 'Weighted Least Squares', 'Multilinear Extension','Linear'])
            
            values = [compare_est[key] for key in est_cat_list_family]
            # st.markdown(values)

            est_categories_l = list()
            if len(values) != 0:
                for val in values:
                    for v in val:
                        est_categories_l.append(v)
            else:
                st.error('Please select an estimation strategy!', icon="🚨")
            
            est_categories = list(set(sorted(est_categories_l)))

            # app_list = list(set(time_df['Approximation'].values))
            app_list = est_categories

            container_method = st.container()
            all_app = st.checkbox("Select all", key='all_app_time')

            if all_app: 
                app_list_family_time = container_method.multiselect('###### Pick an estimation strategy:', sorted(app_list), sorted(app_list), key='all_app_time_1')
            else: 
                app_list_family_time = container_method.multiselect('###### Pick an estimation strategy:', sorted(app_list), key='all_app_time_2', default=sorted(app_list))



        elif model == 'Tree-based models':
            time_df = pd.read_csv("data/tables/tree.csv")
            compare_est = {
            'Random Order': ['Exhaustive', 'IME', 'CES', 'Cohort'],
            'Weighted Least Squares': ['KernelSHAP', 'SGDShapely', 'Exhaustive', 'Parametric KernelSHAP', 'Non-parametric KernelSHAP'],
            'Multilinear Extension': ['MLE', 'Exhaustive', 'IME'],
            # 'Linear': ['Linear (correlated)', 'Linear (independent)', 'Exhaustive'],
            'Tree': ['Tree (path dependent)', 'Tree (interventional)', 'Exhaustive'],
            # 'Deep': ['DASP', 'DeepLIFT', 'DeepSHAP']
            }
            container_method = st.container()
            all_est_cat = st.checkbox("Select all", key='all_est_cat_time')
            
            if all_est_cat: 
                est_cat_list_family = container_method.multiselect('###### Pick an estimation strategy: ', ['Random Order', 'Weighted Least Squares', 'Multilinear Extension', 'Tree'], ['Random Order', 'Weighted Least Squares', 'Multilinear Extension', 'Tree'], key='est_cat1_time')
            else: 
                est_cat_list_family = container_method.multiselect('###### Pick an estimation strategy: ',['Random Order', 'Weighted Least Squares', 'Multilinear Extension', 'Tree'], key='est_cat2_time',  default=['Random Order', 'Weighted Least Squares', 'Multilinear Extension', 'Tree'])
            
            values = [compare_est[key] for key in est_cat_list_family]
            # st.markdown(values)

            est_categories_l = list()
            if len(values) != 0:
                for val in values:
                    for v in val:
                        est_categories_l.append(v)
            else:
                st.error('Please select an estimation strategy!', icon="🚨")
            
            est_categories = list(set(sorted(est_categories_l)))


            # app_list = list(set(time_df['Approximation'].values))
            app_list = est_categories
            container_method = st.container()
            all_app = st.checkbox("Select all", key='all_app_time')

            if all_app: 
                app_list_family_time = container_method.multiselect('###### Pick an estimation strategy:', sorted(app_list), sorted(app_list), key='all_app_time_1')
            else: 
                app_list_family_time = container_method.multiselect('###### Pick an estimation strategy:', sorted(app_list), key='all_app_time_2', default=sorted(app_list))



        elif model == 'Neural networks':
            time_df = pd.read_csv("data/tables/nn.csv")
            compare_est = {
            'Random Order': ['Exhaustive', 'IME', 'CES', 'Cohort'],
            'Weighted Least Squares': ['KernelSHAP', 'SGDShapely', 'Exhaustive', 'Parametric KernelSHAP', 'Non-parametric KernelSHAP'],
            'Multilinear Extension': ['MLE', 'Exhaustive', 'IME'],
            # 'Linear': ['Linear (correlated)', 'Linear (independent)', 'Exhaustive'],
            # 'Tree': ['Tree (path dependent)', 'Tree (interventional)', 'Exhaustive'],
            'Deep': ['DASP', 'DeepLIFT', 'DeepSHAP']
            }
            container_method = st.container()
            all_est_cat = st.checkbox("Select all", key='all_est_cat_time')
            
            if all_est_cat: 
                est_cat_list_family = container_method.multiselect('###### Pick an estimation strategy: ', ['Random Order', 'Weighted Least Squares', 'Multilinear Extension', 'Deep'], ['Random Order', 'Weighted Least Squares', 'Multilinear Extension', 'Deep'], key='est_cat1_time')
            else: 
                est_cat_list_family = container_method.multiselect('###### Pick an estimation strategy: ',['Random Order', 'Weighted Least Squares', 'Multilinear Extension', 'Deep'], key='est_cat2_time',  default=['Random Order', 'Weighted Least Squares', 'Multilinear Extension', 'Deep'])
            
            values = [compare_est[key] for key in est_cat_list_family]
            # st.markdown(values)

            est_categories_l = list()
            if len(values) != 0:
                for val in values:
                    for v in val:
                        est_categories_l.append(v)
            else:
                st.error('Please select an estimation strategy!', icon="🚨")
            
            est_categories = list(set(sorted(est_categories_l)))
            # st.markdown(est_categories)

            # app_list = list(set(time_df['Approximation'].values))
            app_list = est_categories
            container_method = st.container()
            all_app = st.checkbox("Select all", key='all_app_time')

            if all_app: 
                app_list_family_time = container_method.multiselect('###### Pick an estimation strategy:', sorted(app_list), sorted(app_list), key='all_app_time_1')
            else: 
                app_list_family_time = container_method.multiselect('###### Pick an estimation strategy:', sorted(app_list), key='all_app_time_2', default=sorted(app_list))

        else:
            print("Not implemented!")

        
        col1, col2, col3 = st.columns([0.05, 2, 0.05])
        with col2:
            # st.markdown('#### Overall comparison: ')
            # Filter the DataFrame
            filtered_df = time_df[time_df['Approximation'].isin(app_list_family_time)]
            st.markdown('#### Overall comparison: ')
            filtered_df = filtered_df.sort_values(by="Approximation", ascending=True)
            # Create a bar chart using Plotly
            fig = px.bar(
                filtered_df,
                x="Time",
                y="Approximation",
                color="Approximation",
                color_discrete_sequence=px.colors.sequential.Magma,  # Mimic magma color scheme
                orientation="h"  # Horizontal bar chart
            )
            
            # Customize chart properties
            fig.update_layout(
                width=600,
                height=400,
                xaxis=dict(
                    type="log",  # Logarithmic scale for x-axis
                    title=dict(font=dict(size=18)),  # Adjust x-axis title font size
                    tickfont=dict(size=14)  # Adjust x-axis tick font size
                ),
                yaxis=dict(
                    title=dict(font=dict(size=18)),  # Adjust y-axis title font size
                    tickfont=dict(size=14)  # Adjust y-axis tick font size
                )
            )
            
            # Display the chart in Streamlit
            st.plotly_chart(fig, use_container_width=True)
            
            # bar_chart = alt.Chart(time_df[time_df['Approximation'].isin(app_list_family_time)]).mark_bar().encode(
            #     x=alt.X("Time:Q", scale=alt.Scale(type='log')),
            #     y=alt.Y("Approximation:N", sort="y",  axis=alt.Axis(labelAngle=0, labelLimit=200)),
            #     color=alt.Color("Approximation:N", scale=alt.Scale(scheme='magma'))
            # ).properties(
            #     width=600,
            #     height=400
            # ).configure_axis(
            #     labelFontSize=14,  # Adjust label font size
            #     titleFontSize=18,  # Adjust title font size
            #     tickSize=14  # Adjust tick size
            # )
        
            # st.altair_chart(bar_chart, use_container_width=True)

            dim_df = pd.read_csv("data/tables/time_approaches.csv")

            line_chart = alt.Chart(dim_df[dim_df['Approximation'].isin(app_list_family_time)]).mark_line().encode(
                x=alt.X("Features:N"),
                y=alt.Y("Time:Q",  scale=alt.Scale(type='log')),
                color=alt.Color("Approximation:N", scale=alt.Scale(scheme='magma'))
            ).properties(
                width=600,
                height=400
            ).configure_axis(
                labelFontSize=14,  # Adjust label font size
                titleFontSize=18,  # Adjust title font size
                tickSize=14  # Adjust tick size
            ).interactive()

            st.markdown('####  Impact of increasing dimensionality on compute time : ')
            st.altair_chart(line_chart, use_container_width=True)

            tradeoff = pd.read_csv('data/tables/tradeoff.csv')
            st.markdown("#### Accuracy-Compute time tradeoff: ")
            scatter_chart = alt.Chart(tradeoff[tradeoff['Approximation'].isin(app_list_family_time)]).mark_circle().encode(
                x=alt.X("Time:Q",  scale=alt.Scale(type='log')),
                y=alt.Y("Accuracy", scale=alt.Scale(domain=[90, 100])),
                color=alt.Color("Approximation:N", scale=alt.Scale(scheme='magma')),
                # size = 'Estimation Strategy'
                size=alt.Size('Estimation Strategy', scale=alt.Scale(range=[250, 1000]), legend=None)
            ).properties(
                width=600,
                height=400
            ).configure_axis(
                labelFontSize=14,  # Adjust label font size
                titleFontSize=18,  # Adjust title font size
                tickSize=14  # Adjust tick size
            ).interactive()

            st.altair_chart(scatter_chart, use_container_width=True)

with tab_diy:
    # col1, col2, col3 = st.columns([0.25, 5, 0.25])
    # with col2:
    data = None
    col11, col12 = st.columns([1, 1])
    with col11:
        data_file = st.file_uploader("Upload a Data File (Must be a CSV file): ", type="csv")
        if data_file is not None:
            data = pd.read_csv(data_file).iloc[:, 1:]
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

        else:
            st.warning("Please upload a .csv data file.")

    
    with col12:
        model_file = st.file_uploader('Upload Model File (Must be a Pickle file):', type=['pkl'])
        model_load = None
        if model_file is not None:
            model_load = joblib.load(model_file)
        else:
            st.warning("Please upload a .pkl model file.")



    if data is not None:
        # st.dataframe(data, height=300)
        selection = dataframe_with_selections(data)

        if len(selection) < 1:
            st.error('Shapley values is a local feature attribution technique. Hence, select a single instance to explain!', icon="🚨")
        else:
            for s in range(len(selection)):
                st.write("Instance of interest:")
                st.dataframe(selection.iloc[s:s+1,:-1], width=2000)
                shap_vals = None
                if model_load is not None:
                    exp = shap.explainers.Exact(model_load.predict, X_train)
                    shap_vals = exp(selection.iloc[s:s+1,:-1])
                else:
                    st.error('Please upload a model file to generate explanations!', icon="🚨")

                col1, col2, col3 = st.columns([0.05, 2, 0.05])
                with col2:
                    if shap_vals is not None:
                        with st.container(border=True):
                            plot_shap_fig(shap_vals)




    

        

