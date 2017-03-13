# ------------------------------------------------------------------------------
# Experiment class
# Extract one iteration of suggest/update loop out of Spearmint main framework
# Assumption: mongodb is running on localhost
# This is intermediate work to support decoupling user jobs from Spearmint (i.e.
# allowing the user jobs to run on a local machine and spearmint algorithm and
# mongodb to run on a server, which has not been done yet)
# ------------------------------------------------------------------------------

import re
import time
import importlib
import numpy as np
import pymongo

from spearmint.utils.database.mongodb import MongoDB
from spearmint.tasks.task_group import TaskGroup


def get_db(db_uri):
    return pymongo.MongoClient(db_uri)['spearmint']

def find_experiment(username, name, db_uri):
    db = get_db(db_uri)
    profile_name = username + '.' + name + '.profile'
    return profile_name in db.collection_names() # True if experiment exists

def get_experiment(username, name, db_uri):
    db = get_db(db_uri)
    profile = db[username][name]['profile'].find_one()
    experiment = Experiment(username + '.' + name,
                            parameters=profile['parameters'],
                            outcome=profile['outcome'])
    return experiment

def find_all_experiments(username, db_uri):
    db = get_db(db_uri)
    names = []
    for name in db.collection_names():
        match_obj = re.match(username + '\.(.+)\.profile', name)
        if match_obj:
            names.append(match_obj.group(1))
    return names

def find_jobs(username, name, db_uri):
    experiment = get_experiment(username, name, db_uri)
    jobs = experiment.load_jobs()

    # a little bit of processing to interpret numpy values
    simple_jobs = []
    min_value = float('inf') # infinity
    best_job = None
    for job in jobs:
        sjob = job
        if '_id' in sjob:
            sjob.pop('_id', None) # remove key from dict
        if 'params' in sjob:
            sjob['params'] = experiment.simplify_params(job['params'])
        if 'values' in sjob: # rename outcome and interpret value as min/max
            outcome_name = experiment.outcome['name']
            outcome_val = sjob['values']['main']
            if not experiment.outcome['minimize']:
                outcome_val = -outcome_val # restore original outcome value
            sjob['outcome'] = {'name': outcome_name, 'value': outcome_val}
            if sjob['values']['main'] < min_value:
                min_value = sjob['values']['main']
                best_job = sjob
            sjob.pop('values', None) # no more use of this key
        simple_jobs.insert(0, sjob) # reverse the order
    if best_job:
        best_job['minimize'] = experiment.outcome['minimize'] # to indicate objective min/max
        simple_jobs.insert(0, best_job)
    return simple_jobs

def create_experiment(username, name, parameters, outcome, db_uri):
    db = get_db(db_uri)
    profile = {'parameters': parameters, 'outcome': outcome, 'next_id': 0}
    db[username][name]['profile'].insert_one(profile)
    return True

def delete_experiment(username, name, db_uri):
    db = get_db(db_uri)
    db[username][name]['profile'].drop()
    db[username][name]['jobs'].drop()
    db[username][name]['hypers'].drop()
    return True

def get_suggestion(username, name, db_uri):
    experiment = get_experiment(username, name, db_uri)
    params = experiment.suggest()
    return params

def post_update(username, name, param_values, outcome_val, db_uri):
    experiment = get_experiment(username, name, db_uri)
    experiment.update(param_values, outcome_val)

class Experiment:
    def __init__(self,
                 name,
                 description='',
                 parameters=None,
                 outcome=None,
                 access_token=None,
                 db_uri='mongodb://spmint.chestimagingplatform.org/spearmint',
                 likelihood='GAUSSIAN'): # other option is NOISELESS
        for pval in parameters.itervalues():
            pval['size'] = 1 # add default size = 1
            if pval['type'] == 'integer':
                pval['type'] = 'int' # spearmint int type
        self.parameters = parameters
        self.name = name
        self.outcome = outcome
        self.db_uri=db_uri
        if not 'minimize' in outcome:
            self.outcome['minimize'] = False
        #print 'Using database at {0}.'.format(DB_ADDRESS)
        self.db = MongoDB(self.db_uri)
        self.tasks = {'main' : {'type' : 'OBJECTIVE', 'likelihood' : likelihood}}
        # Load up the chooser.
        chooser_name = 'default_chooser'
        chooser_module = importlib.import_module('spearmint.choosers.' + chooser_name)
        self.chooser = chooser_module.init({})

    def load_jobs(self):
        jobs = self.db.load(self.name, 'jobs')

        if jobs is None:
            jobs = []
        if isinstance(jobs, dict):
            jobs = [jobs]

        return jobs

    def get_next_job_id(self):
        profile = self.db.load(self.name, 'profile')
        next_id = profile['next_id']
        profile['next_id'] = next_id + 1
        self.db.save(profile, self.name, 'profile')
        return next_id

    def simplify_params(self, params):
        # just extract first value of each parameter name
        params_simple = {}
        for name in params:
            ptype = self.parameters[name]['type']
            if ptype == 'int':
                params_simple[name] = int(params[name]['values'][0])
            else:
                params_simple[name] = params[name]['values'][0]
        return params_simple

    def load_task_group(self, jobs):
        task_names = self.tasks.keys()
        task_group = TaskGroup(self.tasks, self.parameters)

        if jobs:
            task_group.inputs  = np.array([task_group.vectorify(job['params'])
                    for job in jobs if job['status'] == 'complete'])

            task_group.pending = np.array([task_group.vectorify(job['params'])
                    for job in jobs if job['status'] == 'pending'])

            task_group.values  = {task : np.array([job['values'][task]
                    for job in jobs if job['status'] == 'complete'])
                        for task in task_names}

            task_group.add_nan_task_if_nans()

        return task_group

    def suggest(self):
        #print 'thinking...'
        jobs = self.load_jobs()

        # Load the tasks from the database -- only those in task_names!
        task_group = self.load_task_group(jobs)

        # Load the model hypers from the database.
        hypers = self.db.load(self.name, 'hypers')

        # "Fit" the chooser - give the chooser data and let it fit the model.
        hypers = self.chooser.fit(task_group, hypers, self.tasks)

        # Save the hyperparameters to the database.
        self.db.save(hypers, self.name, 'hypers')

        # Ask the chooser to actually pick one.
        suggested_input = self.chooser.suggest()
        params = task_group.paramify(suggested_input)

        job_id = self.get_next_job_id()
        start_time = time.time()
        job = {
            'id'        : job_id,
            'params'    : params,
            'status'    : 'pending',
            'start time': start_time,
            'end time'  : None
        }
        self.db.save(job, self.name, 'jobs', {'id' : job_id})

        # just extract first value of each parameter name
        params_simple = self.simplify_params(params)
        params_simple['__id__'] = job_id
        return params_simple

    def update(self, param_values, outcome_val):
        #print 'updating...'

        if not self.outcome['minimize']:
            outcome_val = -outcome_val # negate to maximize (default behavior of spearmint: minimize)

        job_id = param_values['__id__']
        job = self.db.load(self.name, 'jobs', {'id' : job_id})

        end_time = time.time()

        job['values']   = {'main': outcome_val}
        job['status']   = 'complete'
        job['end time'] = end_time

        self.db.save(job, self.name, 'jobs', {'id' : job_id})
