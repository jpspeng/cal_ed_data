import pandas as pd # pandas is a data analytics package for python
import matplotlib.pyplot as plt # matplotlib is a graphing package for python
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


def group_data(df):
    df_frgrouped = df.groupby(pd.cut(df['fr_percent'], [0, 0.25, 0.5, 0.75, 1])).mean()
    df_frgrouped.index = ["0-25%", "25-50%", "50-75%", "75-100%"]
    return df_frgrouped

def plot_enrollment_frpm(df):
    df_frgrouped = group_data(df)
    plt.bar(df_frgrouped['enrollment'], align='center', width=0.24, alpha=1)
    plt.ylabel('average enrollment')
    plt.xlabel('free and reduced lunch percentage')
    plt.show()

def plot_stratio_frpm(df):
    df_frgrouped = group_data(df)
    plt.bar(df_frgrouped.index, df_frgrouped['student_teacher_ratio'], align='center', width=0.24, alpha=1)
    plt.ylabel('average student/teacher ratio')
    plt.xlabel('free and reduced lunch percentage')
    plt.show()

def plot_saratio_frpm(df):
    df_frgrouped = group_data(df)
    plt.bar(df_frgrouped.index, df_frgrouped['student_admin_ratio'], align='center', width=0.24, alpha=1)
    plt.ylabel('average student/admin ratio')
    plt.xlabel('free and reduced lunch percentage')
    plt.show()

def plot_susp_frpm(df):
    df_frgrouped = group_data(df)
    plt.bar(df_frgrouped.index, df_frgrouped['susp_per_enrollment'], align='center', width=0.24, alpha=1)
    plt.ylabel('average suspension ratio')
    plt.xlabel('free and reduced lunch percentage')
    plt.show()

def plot_sbac_frpm(df):
    df_frgrouped = group_data(df)
    df_frgrouped['percent_met_and_above'].plot.bar()
    plt.ylabel('average sbac pass rate')
    plt.xlabel('free and reduced lunch percentage')

def plot_sbac_frpm_scatter(df):
    x = df['fr_percent']
    y = df['percent_met_and_above']
    plt.scatter(x, y)
    plt.ylabel('sbac pass rate')
    plt.xlabel('free and reduced lunch percentage')
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--")
    plt.show()

def plot_charter_frpm_compare(df):
    df_charter = df[df['charter'] == True]
    df_noncharter = df[df['charter'] == False]
    plt.figure()
    df_noncharter['fr_percent'].plot.hist(alpha=0.5, label='non-charter')
    print df_charter['fr_percent']
    df_charter['fr_percent'].plot.hist(alpha=0.5, label='charter')
    plt.title('free/reduced lunch percent by school type')
    plt.xlabel('free and reduced lunch percentage')
    plt.ylabel('number of schools')
    plt.legend()
    plt.show()
    print "Average free and reduced lunch percentage at charter schools is:", df_charter['fr_percent'].mean()
    print "Average free and reduced lunch percentage at noncharter schools is:", df_noncharter['fr_percent'].mean()

def group_frpm_charter(df):
    df_charter = df[df['charter'] == True]
    df_charter = df_charter.reset_index(drop = 0)
    df_frgrouped_charter = df_charter.groupby(pd.cut(df_charter['fr_percent'], [0, 0.25, 0.5, 0.75, 1])).mean()
    df_frgrouped_charter.index = ['0-25%', '25-50%', '50-75%', '75-100%']
    df_frgrouped_charter.head(n=4)
    return df_frgrouped_charter

def group_frpm_noncharter(df):
    df_noncharter = df[df['charter'] == False]
    df_noncharter = df_noncharter.reset_index(drop = True)
    df_frgrouped_noncharter = df_noncharter.groupby(pd.cut(df_noncharter['fr_percent'], [0, 0.25, 0.5, 0.75, 1])).mean()
    df_frgrouped_noncharter.index = ['0-25%', '25-50%', '50-75%', '75-100%']
    df_frgrouped_noncharter.head(n=4)
    return df_frgrouped_noncharter

def plot_charter_noncharter_sbac(df):
    df_frgrouped_noncharter = group_frpm_noncharter(df)
    df_frgrouped_charter = group_frpm_charter(df)
    feature = 'percent_met_and_above'
    df_feature = pd.concat([df_frgrouped_noncharter[feature], df_frgrouped_charter[feature]], axis=1)
    df_feature.columns = ['non-charter', 'charter']
    df_feature.plot.bar()
    plt.xlabel('free and reduced lunch percentage')
    plt.ylabel(feature)
    plt.show()

def plot_charter_noncharter_susp(df):
    df_frgrouped_noncharter = group_frpm_noncharter(df)
    df_frgrouped_charter = group_frpm_charter(df)
    feature = 'susp_per_enrollment'
    df_feature = pd.concat([df_frgrouped_noncharter[feature], df_frgrouped_charter[feature]], axis = 1)
    df_feature.columns = ['non-charter', 'charter']
    df_feature.plot.bar()
    plt.xlabel('free and reduced lunch percentage')
    plt.ylabel(feature)
    plt.show()

def plot_charter_noncharter_exp(df):
    df_frgrouped_noncharter = group_frpm_noncharter(df)
    df_frgrouped_charter = group_frpm_charter(df)
    feature = 'avg_exp'
    df_feature = pd.concat([df_frgrouped_noncharter[feature], df_frgrouped_charter[feature]], axis = 1)
    df_feature.columns = ['non-charter', 'charter']
    df_feature.plot.bar()
    plt.xlabel('free and reduced lunch percentage')
    plt.ylabel(feature)
    plt.show()


def plot_charter_noncharter_yrsdistrict(df):
    df_frgrouped_noncharter = group_frpm_noncharter(df)
    df_frgrouped_charter = group_frpm_charter(df)
    feature = 'avg_yrs_district'
    df_feature = pd.concat([df_frgrouped_noncharter[feature], df_frgrouped_charter[feature]], axis=1)
    df_feature.columns = ['non-charter', 'charter']
    df_feature.plot.bar()
    plt.xlabel('free and reduced lunch percentage')
    plt.ylabel(feature)
    plt.show()

def corr_scatter(df):
    df_multivariate = df[df['susp_per_enrollment'] < 1]
    df_multivariate = df_multivariate[df_multivariate['susp_per_enrollment'] > 0]
    plt.rcParams['figure.figsize'] = (20.0, 10.0)
    pd.plotting.scatter_matrix(
        df_multivariate.loc[:, ["fr_percent", "avg_exp", "susp_per_enrollment", "percent_met_and_above"]],
        diagonal="kde")
    plt.show()

def machine_learning_processing(df):
    df['sbac_quantile'] = pd.qcut(df['percent_met_and_above'], 4, labels=False)
    df.head(n=5)
    df_processed = df.drop('f_percent', 1)
    df_processed = df_processed.drop('percent_met_and_above', 1)
    df_processed = df_processed.dropna()
    df_processed['charter'] = df_processed['charter'].astype(int)
    return df_processed

def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("tree.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")

def dec_tree(df, processor_function = machine_learning_processing):
    df_processed = processor_function(df)
    features = list(df_processed.columns[2:-1])
    y = df_processed['sbac_quantile']
    X = df_processed[features]
    depth = 1
    best_depth = 1
    best_accuracy = 0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    while depth < 10:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=99)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        cur_accuracy_score = accuracy_score(y_test, y_pred)
        if cur_accuracy_score > best_accuracy:
            best_accuracy = cur_accuracy_score
            best_depth = depth
        depth = depth + 1

    print "Best depth is", best_depth, "with accuracy", best_accuracy

    dt = DecisionTreeClassifier(max_depth = best_depth, random_state=99)
    dt.fit(X_train, y_train)
    print features
    print dt.feature_importances_
    return dt, features

def random_forest(df):
    df_processed = machine_learning_processing(df)
    features = list(df_processed.columns[2:-1])
    y = df_processed['sbac_quantile']
    X = df_processed[features]
    depth = 0
    best_depth = 1
    n = 5
    best_estimators = 5
    best_accuracy = 0
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 100)
    while depth < 10:
        n = 5
        depth = depth + 1
        while n < 21:
            rf = RandomForestClassifier(n_estimators = n , max_depth = depth)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            cur_accuracy_score = accuracy_score(y_test,y_pred)
            if cur_accuracy_score > best_accuracy:
                best_accuracy = cur_accuracy_score
                best_depth = depth
                best_estimators = n
            n += 1

    rf = RandomForestClassifier(n_estimators = best_estimators , max_depth = best_depth)
    rf.fit(X_train, y_train)
    print "Best depth is", best_depth, "and best n-estimators is", best_estimators, "with accuracy", best_accuracy
    print features
    print rf.feature_importances_

def df_processor_low_income(df):
    df_dec_tree_l = df[df['fr_percent'] > 0.7]
    df_dec_tree_l = df_dec_tree_l.drop('f_percent', 1)
    df_dec_tree_l = df_dec_tree_l.drop('fr_percent', 1)
    df_dec_tree_l['sbac_quantile'] = pd.qcut(df_dec_tree_l['percent_met_and_above'], 3, labels=False)
    df_dec_tree_l = df_dec_tree_l.drop('percent_met_and_above', 1)
    df_dec_tree_l = df_dec_tree_l.dropna()
    df_dec_tree_l['charter'] = df_dec_tree_l['charter'].astype(int)
    return df_dec_tree_l

