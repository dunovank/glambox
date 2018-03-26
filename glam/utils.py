#!/usr/bin/python

import numpy as np
import pandas as pd
from scipy.stats import mode


def format_data(df):
    """
    Extracts and formats data
    from <pandas.DataFrame> to model friendly entities.
    """
    subjects = df['subject'].unique()
    n_subjects = len(subjects)
    n_items = len([col for col in df.columns
                   if col.startswith('item_value_')])
    subject_idx = np.array(df.subject.values.astype(int))
    gaze = df[['gaze_{}'.format(i) for i in range(n_items)]].values
    values = df[['item_value_{}'.format(i) for i in range(n_items)]].values
    choice = df['choice'].values.astype(int)
    rts = df['rt'].values
    rts = rts.astype('int')
    gaze_sorted = sort_according_to_choice(gaze, choice)
    values_sorted = sort_according_to_choice(values, choice)

    # compute random choice likelihood
    rtmax = df.groupby('subject').rt.max().values
    rtmin = df.groupby('subject').rt.min().values
    error_lls = 1 / (n_items * (rtmax-rtmin))

    output = dict(subjects=subjects,
                  n_subjects=n_subjects,
                  n_items=n_items,
                  subject_idx=subject_idx,
                  gaze=gaze_sorted,
                  values=values_sorted,
                  rts=rts,
                  error_lls=error_lls)
    return output


def sort_according_to_choice(x, choices):
    """
    Sort a <numpy.array> x according to choice indices.
    """
    x_sorted = np.zeros_like(x) * np.nan
    # Get values of chosen entries, and put them into first column
    x_sorted[:, 0] = x[np.arange(x.shape[0]), choices.astype(int)]
    # and everything else into the next columns
    others = np.vstack([x[i, np.where(np.arange(x.shape[1]) != c)]
                        for i, c in enumerate(choices)])
    x_sorted[:, 1:] = others
    return x_sorted


def extract_modes(traces, parameters=None, precision=None, f_burn=0.5):

    if not isinstance(traces, list):
        traces = [traces]

    modes = []

    for trace in traces:

        if parameters is None:
            parameters = [var for var in trace.varnames
                          if not var.endswith('__')]

            print('/!\ Automatically setting parameter precision...')
            precision_defaults = dict(v=6, gamma=2, s=6, SNR=2, tau=2, t0=-1)
            precision = [precision_defaults.get(parameter.split('_')[0], 6)
                         for parameter in parameters]

        n_samples = len(trace)
        trace_modes = {}

        for parameter, prec in zip(parameters, precision):
            trace_modes[parameter] = mode(np.round(trace.get_values(
                parameter, burn=int(f_burn*n_samples)), prec))[0][0]
        modes.append(trace_modes)

    if len(modes) == 1:
        return modes[0]
    else:
        return modes


def compare(GLAM_1, GLAM_2, ic='WAIC', **kwargs):
    """
    """
    # check model instances
    if GLAM_1.type == GLAM_2.type:
        if GLAM_1.type == 'individual':
            if len(GLAM_1.model) == len(GLAM_2.model):
                n_comparisons = len(GLAM_1.model)
                continue
            else:
                raise TypeError(
                    'Number of models included in GLAM_1 and GLAM_2 needs to be the same.')

            if len(GLAM_1.trace) == len(GLAM_2.trace):
                continue
            else:
                raise TypeError(
                    'Number of traces included in GLAM_1 and GLAM_2 needs to be the same.')

        elif (GLAM_1.type == 'hierarchical') or (GLAM_1.type == 'pooled'):
            n_comparisons = 1
            if isinstance(GLAM_1.model, pm.model.Model):
                continue
            else:
                raise TypeError('GLAM_1 model is not a pymc model instance.')
            if isinstance(GLAM_2.model, pm.model.Model):
                continue
            else:
                raise TypeError('GLAM_2 model is not a pymc model instance.')

        else:
            raise TypeError('GLAM_1 and GLAM_2 are of wrong type.')

    else:
        raise TypeError(
            'Comparison is only implemented for models of the same type.')

    # compare models
    ics = []
    if GLAM_1.type == 'individual':
        for (model_1, trace_1), (model_2, trace_2) in zip(zip(GLAM_1.model, GLAM_1.trace),
                                                          zip(GLAM_2.model, GLAM_2.trace)):
            ic_comp = pm.compare(dict(model_1=trace_1,
                                      model_2=trace_2),
                                 ic=ic, **kwargs)
            ics.append(ic_comp)
    else:
        ic_comp = pm.compare(dict(GLAM_1.model=GLAM_1.trace,
                                  GLAM_2.model=GLAM_2.trace),
                             ic=ic, **kwargs)
        ics.append(ic_comp)

    if len(ics) == 1:
        return ics[0]
    else:
        return ics


def gaze_influence_score(data):
    """

    """
    import statsmodels.api as sm

    df = data.copy()
    n_items = len(
        [c for c in data.columns.values if c.startswith('item_value_')])

    # extract data
    choice = np.zeros((df.shape[0], n_items))
    choice[np.arange(df.shape[0]), df['choice'].values.astype('int32')] = 1
    gaze = df[['gaze_{}'.format(i) for i in range(n_items)]].values
    values = df[['item_value_{}'.format(i) for i in range(n_items)]].values
    rel_gaze = np.zeros_like(gaze)
    rel_values = np.zeros_like(values)
    value_range = np.zeros_like(values)
    for trial in range(values.shape[0]):
        for item in range(n_items):
            others = np.where(np.arange(n_items) != item)
            rel_gaze[trial, item] = gaze[trial, item] - \
                np.max(gaze[trial, others])
            rel_values[trial, item] = values[trial, item] - \
                np.mean(values[trial, others])
            value_range[trial, item] = np.max(
                values[trial, others]) - np.min(values[trial, others])

    # create new df
    df_tmp = pd.DataFrame(dict(subject=np.repeat(df['subject'].values, n_items),
                               is_choice=choice.ravel(),
                               value=values.ravel(),
                               rel_value=rel_values.ravel(),
                               value_range_others=value_range.ravel(),
                               rel_gaze=rel_gaze.ravel(),
                               gaze_pos=np.array(rel_gaze.ravel() > 0, dtype=np.bool)))

    # estimate influence of item value on choice
    data_out = pd.DataFrame()
    for s, subject in enumerate(data['subject'].unique()):
        subject_data = df_tmp[df_tmp['subject'] == subject].copy()

        X = subject_data[['rel_value', 'value_range_others']]
        X = sm.add_constant(X)
        y = subject_data['is_choice']

        logit = sm.Logit(y, X)
        result = logit.fit(disp=0, method='lbfgs', maxiter=100)
        predicted_pchoose = result.predict(X)

        subject_data['corrected_choice'] = subject_data['is_choice'] - \
            predicted_pchoose
        data_out = pd.concat([data_out, subject_data])

    # average corrected psychometric, given gaze
    tmp = data_out.groupby(['subject', 'gaze_pos']
                           ).corrected_choice.mean().unstack()
    gaze_influence_score = (tmp[True] - tmp[False]).values

    return gaze_influence_score


def compute_behavioral_indices(data):
    """
    """
    print('Adding "behavioral_indices" to GLAM.')
    # set up dataframe
    behavioral_indices = pd.DataFrame()
    behavioral_indices['subject'] = data.subject.unique()

    # average response times
    behavioral_indices['mean_rt'] = data.groupby(['subject']).rt.mean().values

    # average p(choose best item)
    if 'best_chosen' not in data.columns.values:
        n_items = len(
            [c for c in data.columns.values if c.startswith('item_value_')])
        values = data[['item_value_{}'.format(
            i) for i in range(n_items)]].values
        choices = data['choice'].values.astype(np.int)
        best_chosen = (values[np.arange(values.shape[0]), choices] ==
                       np.max(values, axis=1)).astype(np.int)
        data['best_chosen'] = best_chosen
    behavioral_indices['p_best_chosen'] = data.groupby(
        ['subject']).best_chosen.mean().values

    # gaze influence on P(choose item)
    behavioral_indices['gaze_influence_score'] = gaze_influence_score(data)

    return behavioral_indices


def get_design(model):
    """
    Extract information about the experimental design
    from `model.data` and `model.depends_on`.
    This information is used to map parameter estimates
    back to subjects, etc.

    Parameters:
    model: GLAM object

    Returns:
    dict
    """
    parameters = ['v', 'gamma', 's', 'tau', 't0']

    subject_idx = model.data['subject']
    subjects = subject_idx.unique()

    design = dict()
    design['factors'] = list({value for key, value in model.depends_on.items()
                              if value is not None})
    design['factor_conditions'] = {factor: model.data[factor].unique()
                                   for factor in design['factors']}

    for parameter in parameters:
        design[parameter] = dict()
        # adding an index defining which entry in data belongs to which subject
        design[parameter]['subject_index'] = subject_idx[:].values.astype(
            np.int)

        dependence = model.depends_on.get(parameter)
        design[parameter]['dependence'] = dependence

        if dependence is not None:
            # extract condition levels
            conditions = model.data[dependence].unique()
            design[parameter]['conditions'] = conditions

            # Initialize empty design matrix D for this parameter
            D = np.zeros((subjects.size, conditions.size), dtype=np.int)
            # create parameter mapping containing condition names and indices (i.e., columns of design matrix)
            design[parameter]['condition_mapping'] = {condition: c
                                                      for c, condition in enumerate(conditions)}
            # create an array to index which condition each trial-entry in the data belongs to
            design[parameter]['condition_index'] = np.zeros_like(
                subject_idx, dtype=np.int)

            # For each condition level
            for c, condition in enumerate(conditions):
                design[parameter][condition] = dict()

                # mark which trial-entries belong to this condition
                design[parameter]['condition_index'][model.data[dependence]
                                                     == condition] = c

                # Subset data to condition-specific data
                data_subset = model.data[model.data[dependence]
                                         == condition].copy()

                # find all subject_IDs in this condition
                subject_subset = data_subset['subject'].unique()

                # within this level, generate mapping between subject_ids and indices
                # e.g., gamma_high[5] corresponds to subject_id 10
                design[parameter][condition]['subject_mapping'] = {int(subject): s
                                                                   for s, subject in enumerate(subject_subset)}
                # attach a list of all subjects in this condition
                design[parameter][condition]['subjects'] = subject_subset

                # Set cells with subject in this condition to 1
                for s, subject in enumerate(subject_subset):
                    D[np.int(subject), c] = np.int(s+1)
        else:
            D = (np.arange(subjects.size)[:, None] + 1).astype(np.int)
            design[parameter]['conditions'] = None
            design[parameter]['condition_index'] = np.zeros_like(
                subject_idx, dtype=np.int)

        # Save design matrix D
        design[parameter]['D'] = D

        # Detect within / between subject design
        if D.shape[1] == 1:
            design_type = 'fixed'
        elif np.all((D != 0).sum(axis=1) > 1):
            design_type = 'within'
        elif np.all((D != 0).sum(axis=1) == 1):
            design_type = 'between'
        else:
            design_type = 'mixed'
        design[parameter]['type'] = design_type

    return design


def get_estimates(model):
    """
    Generate a DataFrame containing parameter estimates
    and summary statistics. Each row corresponds to one
    participant in one condition.

    Parameters:
    model: GLAM object

    Returns:
    DataFrame
    """
    from itertools import product
    import pymc3
    if np.float(pymc3.__version__) < 3.3:
        from pymc3 import df_summary as summary
    else:
        from pymc3 import summary

    subjects = model.data['subject'].unique().astype(np.int)
    parameters = ['v', 'gamma', 's', 'tau', 't0']
    estimates = pd.DataFrame()
    MAP = extract_modes(model.trace)
    combinations = list(product(*[model.design['factor_conditions'][factor]
                                  for factor in model.design['factors']]))
    subject_template = pd.DataFrame({factor: [combination[f]
                                              for combination in combinations]
                                     for f, factor
                                     in enumerate(model.design['factors'])})
    if model.type == 'hierarchical':
        summary_table = summary(model.trace)
    elif model.type == 'individual':
        summary_tables = [summary(trace)
                          for trace in model.trace]
    else:
        raise ValueError(
            'Model type not understood. Make sure "make_model" has already been called.')
    for subject in subjects:
        subject_estimates = subject_template.copy()
        subject_estimates['subject'] = subject
        for parameter in parameters:
            subject_template[parameter] = np.nan
            subject_template[parameter + '_hpd_2.5'] = np.nan
            subject_template[parameter + '_hpd_97.5'] = np.nan
            subject_template[parameter] = np.nan

            dependence = model.design[parameter]['dependence']
            if dependence is None:
                # Parameter is fixed
                if model.type == 'hierarchical':
                    # add participant paramaters
                    subject_estimates[parameter] = MAP[parameter][subject][0]
                    subject_estimates[parameter + '_hpd_2.5'] = summary_table.loc[parameter +
                                                                                  '__{}_0'.format(subject), 'hpd_2.5']
                    subject_estimates[parameter + '_hpd_97.5'] = summary_table.loc[parameter +
                                                                                   '__{}_0'.format(subject), 'hpd_97.5']
                    # add population parameters
                    if (parameter + '_mu') in summary_table.index:
                        subject_estimates[parameter +
                                          '_mu'] = summary_table.loc[parameter + '_mu', 'mean']
                        subject_estimates[parameter +
                                          '_mu_hpd_2.5'] = summary_table.loc[parameter + '_mu', 'hpd_2.5']
                        subject_estimates[parameter +
                                          '_mu_hpd_97.5'] = summary_table.loc[parameter + '_mu', 'hpd_97.5']

                elif model.type == 'individual':
                    # add participant paramaters
                    subject_estimates[parameter] = MAP[subject][parameter][0][0]
                    subject_estimates[parameter +
                                      '_hpd_2.5'] = summary_tables[subject].loc[parameter + '__0_0', 'hpd_2.5']
                    subject_estimates[parameter +
                                      '_hpd_97.5'] = summary_tables[subject].loc[parameter + '__0_0', 'hpd_97.5']
            else:
                # Parameter has dependence
                conditions = model.design[parameter]['conditions']
                for condition in conditions:
                    if condition not in model.data.loc[model.data['subject'] == subject, dependence].values:
                        subject_estimates = subject_estimates.drop(subject_estimates[subject_estimates[dependence] == condition].index,
                                                                   axis=0)
                    else:
                        # Check if subject is in condition
                        if subject in model.design[parameter][condition]['subjects']:
                            parameter_condition = parameter + '_' + condition
                            if model.type == 'hierarchical':
                                index = model.design[parameter][condition]['subject_mapping'][subject]
                                # extract participant parameters
                                estimate = MAP[parameter_condition][index]
                                hpd25 = summary_table.loc[parameter_condition +
                                                          '__{}'.format(index), 'hpd_2.5']
                                hpd975 = summary_table.loc[parameter_condition +
                                                           '__{}'.format(index), 'hpd_97.5']
                                # extract population parameters
                                if (parameter_condition + '_mu') in summary_table.index:
                                    pop_estimate = summary_table.loc[parameter_condition + '_mu', 'mean']
                                    pop_hpd25 = summary_table.loc[parameter_condition +
                                                                  '_mu', 'hpd_2.5']
                                    pop_hpd975 = summary_table.loc[parameter_condition +
                                                                   '_mu', 'hpd_97.5']

                            elif model.type == 'individual':
                                if model.design[parameter]['type'] == 'between':
                                    estimate = MAP[subject][parameter]
                                    hpd25 = summary_tables[subject].loc[parameter +
                                                                        '__0_0', 'hpd_2.5']
                                    hpd975 = summary_tables[subject].loc[parameter +
                                                                         '__0_0', 'hpd_97.5']
                                elif model.design[parameter]['type'] == 'within':
                                    estimate = MAP[subject][parameter_condition]
                                    hpd25 = summary_tables[subject].loc[parameter_condition +
                                                                        '__0_0', 'hpd_2.5']
                                    hpd975 = summary_tables[subject].loc[parameter_condition +
                                                                         '__0_0', 'hpd_97.5']
                                else:
                                    raise ValueError('Parameter dependence not understood for {}: {} ({}).'.format(
                                        parameter, dependence, condition))
                            else:
                                raise ValueError(
                                    'Model type not understood. Make sure "make_model" has already been called.')
                            # add participant parameters
                            subject_estimates.loc[subject_estimates[dependence]
                                                  == condition, parameter] = estimate
                            subject_estimates.loc[subject_estimates[dependence]
                                                  == condition, parameter + '_hpd_2.5'] = hpd25
                            subject_estimates.loc[subject_estimates[dependence]
                                                  == condition, parameter + '_hpd_97.5'] = hpd975
                            # add population parameters
                            if model.type == 'hierarchical':
                                subject_estimates.loc[subject_estimates[dependence]
                                                      == condition, parameter + '_mu'] = pop_estimate
                                subject_estimates.loc[subject_estimates[dependence]
                                                      == condition, parameter + '_mu_hpd_2.5'] = pop_hpd25
                                subject_estimates.loc[subject_estimates[dependence] ==
                                                      condition, parameter + '_mu_hpd_97.5'] = pop_hpd975

        estimates = pd.concat([estimates, subject_estimates])

    estimates.reset_index(inplace=True, drop=True)
    return estimates
