import numpy as np
import pandas as pd
import pylab as pl
import sciris as sc
import covasim as cv
from . import utils as cvu
from . import misc as cvm
from . import base as cvb



import inspect




#%% Generic intervention classes

__all__ = ['InterventionDict', 'Intervention', 'dynamic_pars', 'sequence']


def InterventionDict(which, pars):
    ''' Generate an intervention from a dictionary '''
    mapping = dict(
        change_beta     = change_beta,
        clip_edges      = clip_edges,
        test_num        = test_num,
        test_prob       = test_prob,
        contact_tracing = contact_tracing,
        )
    IntervClass = mapping[which]
    intervention = IntervClass(**pars)
    return intervention


class Intervention:
    '''
    Abstract class for interventions

    '''
    def __init__(self):
        self.days = []


    def __repr__(self):
        output = ''
        return output


    def store_args(self):
        ''' Store the arguments for later use '''
        f0 = inspect.currentframe()
        f1 = inspect.getouterframes(f0)
        _,_,_,values = inspect.getargvalues(f1[1].frame)
        self.input_args = {}
        for key,value in values.items():
            if key not in ['self', '__class__']:
                self.input_args[key] = value
        return


    def apply(self, sim):
        '''
        Apply intervention

        Function signature matches existing intervention definition
        This method gets called at each timestep and must be implemented
        by derived classes

        Args:
            self:
            sim: The Sim instance

        Returns:
            None
        '''
        raise NotImplementedError


    def plot(self, sim, ax):
        '''
        Call function during plotting

        This can be used to do things like add vertical lines on days when interventions take place

        Args:
            sim: the Sim instance
            ax: the axis instance

        Returns:
            None
        '''
        ylims = ax.get_ylim()
        for day in self.days:
            pl.plot([day]*2, ylims, '--', c=[0,0,0])
        return


    def to_json(self):
        '''
        Return JSON-compatible representation

        Custom classes can't be directly represented in JSON. This method is a
        one-way export to produce a JSON-compatible representation of the
        intervention. In the first instance, the object dict will be returned.
        However, if an intervention itself contains non-standard variables as
        attributes, then its `to_json` method will need to handle those

        Returns:
            JSON-serializable representation (typically a dict, but could be anything else)
        '''
        which = self.__class__.__name__
        pars = sc.jsonify(self.input_args)
        output = dict(which=which, pars=pars)
        return output


class dynamic_pars(Intervention):
    '''
    A generic intervention that modifies a set of parameters at specified points
    in time.

    The intervention takes a single argument, pars, which is a dictionary of which
    parameters to change, with following structure: keys are the parameters to change,
    then subkeys 'days' and 'vals' are either a scalar or list of when the change(s)
    should take effect and what the new value should be, respectively.

    Args:
        pars (dict): described above

    **Examples**::

        interv = cv.dynamic_pars({'diag_factor':{'days':30, 'vals':0.5}, 'cont_factor':{'days':30, 'vals':0.5}}) # Starting day 30, make diagnosed people and people with contacts half as likely to transmit
        interv = cv.dynamic_pars({'beta':{'days':[14, 28], 'vals':[0.005, 0.015]}}) # On day 14, change beta to 0.005, and on day 28 change it back to 0.015
    '''

    def __init__(self, pars):
        super().__init__()
        subkeys = ['days', 'vals']
        for parkey in pars.keys():
            for subkey in subkeys:
                if subkey not in pars[parkey].keys():
                    errormsg = f'Parameter {parkey} is missing subkey {subkey}'
                    raise cvm.KeyNotFoundError(errormsg)
                if sc.isnumber(pars[parkey][subkey]): # Allow scalar values or dicts, but leave everything else unchanged
                    pars[parkey][subkey] = sc.promotetoarray(pars[parkey][subkey])
            len_days = len(pars[parkey]['days'])
            len_vals = len(pars[parkey]['vals'])
            if len_days != len_vals:
                raise ValueError(f'Length of days ({len_days}) does not match length of values ({len_vals}) for parameter {parkey}')
        self.pars = pars
        return


    def apply(self, sim):
        ''' Loop over the parameters, and then loop over the days, applying them if any are found '''
        t = sim.t
        for parkey,parval in self.pars.items():
            inds = sc.findinds(parval['days'], t) # Look for matches
            if len(inds):

                if len(inds)>1:
                    raise ValueError(f'Duplicate days are not allowed for Dynamic interventions (day={t}, indices={inds})')
                else:
                    match = inds[0]
                    val = parval['vals'][match]
                    if isinstance(val, dict):
                        sim[parkey].update(val) # Set the parameter if a nested dict
                    else:
                        sim[parkey] = val # Set the parameter if not a dict
        return


class sequence(Intervention):
    '''
    This is an example of a meta-intervention which switches between a sequence of interventions.

    Args:
        days (list): the days on which to apply each intervention
        interventions (list): the interventions to apply on those days
        WARNING: Will take first intervation after sum(days) days has ellapsed!

    **Example**::

        interv = cv.sequence(days=[10, 51], interventions=[
                    cv.test_num(n_tests=[100]*npts),
                    cv.test_prob(symptomatic_prob=0.2, asymptomatic_prob=0.002),
                ])
    '''

    def __init__(self, days, interventions):
        super().__init__()
        assert len(days) == len(interventions)
        self.days = days
        self.interventions = interventions
        self._cum_days = np.cumsum(days)
        return


    def apply(self, sim: cv.Sim):
        idx = np.argmax(self._cum_days > sim.t)  # Index of the intervention to apply on this day
        self.interventions[idx].apply(sim)
        return



#%% Beta interventions

__all__+= ['change_beta', 'clip_edges']


class change_beta(Intervention):
    '''
    The most basic intervention -- change beta by a certain amount.

    Args:
        days (int or array): the day or array of days to apply the interventions
        changes (float or array): the changes in beta (1 = no change, 0 = no transmission)
        layers (str or list): the layers in which to change beta


    **Examples**::

        interv = cv.change_beta(25, 0.3) # On day 25, reduce overall beta by 70% to 0.3
        interv = cv.change_beta([14, 28], [0.7, 1], layers='s') # On day 14, reduce beta by 30%, and on day 28, return to 1 for schools
    '''

    def __init__(self, days, changes, layers=None):
        super().__init__()

        self.days = sc.promotetoarray(days)
        self.changes = sc.promotetoarray(changes)
        self.layers = sc.promotetolist(layers, keepnone=True)
        if len(self.days) != len(self.changes):
            errormsg = f'Number of days supplied ({len(self.days)}) does not match number of changes in beta ({len(self.changes)})'
            raise ValueError(errormsg)
        self.orig_betas = None
        self.store_args()
        return


    def apply(self, sim):

        # If this is the first time it's being run, store beta
        if self.orig_betas is None:
            self.orig_betas = {}
            for lkey in self.layers:
                if lkey is None:
                    self.orig_betas['overall'] = sim['beta']
                else:
                    self.orig_betas[lkey] = sim['beta_layer'][lkey]

        # If this day is found in the list, apply the intervention
        inds = sc.findinds(self.days, sim.t)
        if len(inds):
            for lkey,new_beta in self.orig_betas.items():
                for ind in inds:
                    new_beta = new_beta * self.changes[ind]
                if lkey == 'overall':
                    sim['beta'] = new_beta
                else:
                    sim['beta_layer'][lkey] = new_beta

        return


    def plot(self, sim, ax):
        ''' Plot vertical lines for when changes in beta '''
        ylims = ax.get_ylim()
        for day in self.days:
            pl.plot([day]*2, ylims, '--', c=[0,0,0])
        return


class clip_edges(Intervention):
    '''
    Isolate contacts by moving them from the simulation to this intervention.

    Args:
        start_day (int): the day to isolate contacts
        end_day (int): the day to end isolating contacts
        change (float or dict): the proportion of contacts to retain, a la change_beta (1 = no change, 0 = no transmission)

    **Examples**::

        interv = cv.clip_edges(25, 0.3) # On day 25, reduce overall contacts by 70% to 0.3
        interv = cv.clip_edges(start_day=25, end_day=35, change={'s':0.1}) # On day 25, remove 90% of school contacts, and on day 35, restore them
    '''

    def __init__(self, start_day, change=None, end_day=None, verbose=False):
        super().__init__()
        self.args = inspect.getargspec(self.__init__).args # For jsonification
        self.start_day = start_day
        self.end_day = end_day
        self.days = [start_day, end_day]
        self.change = change
        self.verbose = verbose
        self.layer_keys = None
        self.contacts = None
        return


    def apply(self, sim):

        verbose = self.verbose

        # On the start day, move contacts over
        if sim.t == self.start_day:

            # If this is the first time it's being run, create the contacts
            if self.contacts is None:
                if isinstance(self.change, dict):
                    self.layer_keys = list(self.change.keys())
                else:
                    self.layer_keys = list(sim.people.contacts.keys())
                    self.change = {key:self.change for key in self.layer_keys}
                self.contacts = cvb.Contacts(layer_keys=self.layer_keys)
                if verbose:
                    print(f'Created contacts: {self.contacts}')

            # Do the contact moving
            for lkey,prop in self.change.items():
                layer = sim.people.contacts[lkey]
                if verbose:
                    print(f'Working on layer {lkey}: {layer}')
                n_contacts = len(layer)
                prop_to_move = 1-prop # Calculate the proportion of contacts to move
                n_to_move = int(prop_to_move*n_contacts)
                inds = cvu.choose(max_n=n_contacts, n=n_to_move)
                layer_df = layer.to_df()
                to_move = layer_df.iloc[inds]
                if verbose:
                    print(f'Moving {prop_to_move} of {n_contacts} gives {n_to_move}. Before:\n{layer_df}\nTo move:\n{to_move}')
                self.contacts[lkey] = cvb.Layer().from_df(to_move) # Move them here
                if verbose:
                    print(f'Contacts here: {self.contacts[lkey]}')
                layer_df = layer_df.drop(layer_df.index[inds]) # Remove indices
                new_layer = cvb.Layer().from_df(layer_df) # Convert back
                new_layer.validate()
                sim.people.contacts[lkey] = new_layer
                if verbose:
                    print(f'Remaining contacts: {sim.people.contacts[lkey]}')

        if sim.t == self.end_day:
            if verbose:
                print(f'Before:\n{sim.people.contacts}')
            sim.people.add_contacts(self.contacts)
            if verbose:
                print(f'After:\n{sim.people.contacts}')
            self.contacts = None # Reset to save memory

        return


    def plot(self, sim, ax):
        ''' Plot vertical lines for when changes in beta '''
        ylims = ax.get_ylim()
        for day in self.days:
            pl.plot([day]*2, ylims, '--', c=[0,0,0])
        return


#%% Testing interventions

__all__+= ['test_num', 'test_prob', 'contact_tracing']


class test_num(Intervention):
    '''
    Test a fixed number of people per day.

    **Example**::

        interv = cv.test_num(daily_tests=[0.10*n_people]*npts)

    Returns:
        Intervention
    '''

    def __init__(self, daily_tests, sympt_test=100.0, quar_test=1.0, sensitivity=1.0, test_delay=0, loss_prob=0, start_day=0, end_day=None):
        super().__init__()
        self.args = inspect.getargspec(self.__init__).args # For jsonification
        self.daily_tests = daily_tests #: Should be a list of length matching time
        self.sympt_test = sympt_test
        self.quar_test = quar_test
        self.sensitivity = sensitivity
        self.test_delay = test_delay
        self.loss_prob = loss_prob
        self.start_day = start_day
        self.end_day = end_day
        self.days = [start_day, end_day]

        return


    def apply(self, sim):

        t = sim.t
        if t < self.start_day:
            return
        elif self.end_day is not None and t > self.end_day:
            return

        # Process daily tests -- has to be here rather than init so have access to the sim object
        if isinstance(self.daily_tests, (pd.Series, pd.DataFrame)):
            start_date = sim['start_day']
            end_date = self.daily_tests.index[-1]
            dateindex = pd.date_range(start_date, end_date)
            self.daily_tests = self.daily_tests.reindex(dateindex, fill_value=0).to_numpy()

        # Check that there are still tests
        rel_t = t - self.start_day
        if rel_t < len(self.daily_tests):
            n_tests = self.daily_tests[rel_t]  # Number of tests for this day
            if not (n_tests and pl.isfinite(n_tests)): # If there are no tests today, abort early
                return
            else:
                sim.results['new_tests'][t] += n_tests
        else:
            return

        test_probs = np.ones(sim.n) # Begin by assigning equal tesitng probability to everyone
        symp_inds = cvu.true(sim.people.symptomatic)
        quar_inds = cvu.true(sim.people.quarantined)
        diag_inds = cvu.true(sim.people.diagnosed)
        test_probs[symp_inds] *= self.sympt_test
        test_probs[quar_inds] *= self.quar_test
        test_probs[diag_inds] = 0.

        test_inds = cvu.choose_w(probs=test_probs, n=n_tests, unique=False)

        sim.people.test(test_inds, self.sensitivity, loss_prob=self.loss_prob, test_delay=self.test_delay)

        return


class test_prob(Intervention):
    '''
    Test as many people as required based on test probability.
    Probabilities are OR together, so choose wisely.

    Args:
        symp_prob (float): Probability of testing a symptomatic (unquarantined) person
        asymp_prob (float): Probability of testing an asymptomatic (unquarantined) person
        symp_quar_prob (float): Probability of testing a symptomatic quarantined person
        asymp_quar_prob (float): Probability of testing an asymptomatic quarantined person
        test_sensitivity (float): Probability of a true positive
        loss_prob (float): Probability of loss to follow-up
        test_delay (int): How long testing takes
        start_day (int): When to start the intervention

    **Example**::

        interv = cv.test_prob(symptomatic_prob=0.1, asymptomatic_prob=0.01) # Test 10% of symptomatics and 1% of asymptomatics
        interv = cv.test_prob(symp_quar_prob=0.4) # Test 40% of those in quarantine with symptoms

    Returns:
        Intervention
    '''
    def __init__(self, symp_prob=0, asymp_prob=0, symp_quar_prob=None, asymp_quar_prob=None, test_sensitivity=1.0, loss_prob=0.0, test_delay=1, start_day=0, end_day=None):
        super().__init__()
        self.args = inspect.getargspec(self.__init__).args # For jsonification
        self.symp_prob        = symp_prob
        self.asymp_prob       = asymp_prob
        self.symp_quar_prob   = symp_quar_prob  if  symp_quar_prob is not None else  symp_prob
        self.asymp_quar_prob  = asymp_quar_prob if asymp_quar_prob is not None else asymp_prob
        self.test_sensitivity = test_sensitivity
        self.loss_prob        = loss_prob
        self.test_delay       = test_delay
        self.start_day        = start_day
        self.end_day          = end_day
        self.days             = [start_day, end_day]

        return


    def apply(self, sim):
        ''' Perform testing '''
        t = sim.t
        if t < self.start_day:
            return
        elif self.end_day is not None and t > self.end_day:
            return

        symp_inds       = cvu.true(sim.people.symptomatic)
        asymp_inds      = cvu.false(sim.people.symptomatic)
        quar_inds       = cvu.true(sim.people.quarantined)
        symp_quar_inds  = quar_inds[cvu.true(sim.people.symptomatic[quar_inds])]
        asymp_quar_inds = quar_inds[cvu.false(sim.people.symptomatic[quar_inds])]
        diag_inds       = cvu.true(sim.people.diagnosed)

        test_probs = np.zeros(sim.n) # Begin by assigning equal tesitng probability to everyone
        test_probs[symp_inds]       = self.symp_prob
        test_probs[asymp_inds]      = self.asymp_prob
        test_probs[symp_quar_inds]  = self.symp_quar_prob
        test_probs[asymp_quar_inds] = self.asymp_quar_prob
        test_probs[diag_inds]       = 0.
        test_inds = cvu.binomial_arr(test_probs).nonzero()[0]

        sim.people.test(test_inds, test_sensitivity=self.test_sensitivity, loss_prob=self.loss_prob, test_delay=self.test_delay)
        sim.results['new_tests'][t] += len(test_inds)

        return


class contact_tracing(Intervention):
    '''
    Contact tracing of positives
    '''
    def __init__(self, trace_probs, trace_time, start_day=0, end_day=None, contact_reduction=None):
        super().__init__()
        self.args = inspect.getargspec(self.__init__).args # For jsonification
        self.trace_probs = trace_probs
        self.trace_time = trace_time
        self.contact_reduction = contact_reduction # Not using this yet, but could potentially scale contact in this intervention
        self.start_day = start_day
        self.end_day = end_day
        self.days = [start_day, end_day]
        return

    def apply(self, sim):
        t = sim.t
        if t < self.start_day:
            return
        elif self.end_day is not None and t > self.end_day:
            return

        just_diagnosed_inds = cvu.true(sim.people.date_diagnosed == t) # Diagnosed this time step, time to trace
        if len(just_diagnosed_inds): # If there are any just-diagnosed people, go trace their contacts
            sim.people.trace(just_diagnosed_inds, self.trace_probs, self.trace_time)

        return
