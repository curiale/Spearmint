# ------------------------------------------------------------------------------
# Experiment class
# Extract one iteration of suggest/update loop out of Spearmint main framework
# Assumption: mongodb is running on localhost
# This is intermediate work to support decoupling user jobs from Spearmint (i.e.
# allowing the user jobs to run on a local machine and spearmint algorithm and
# mongodb to run on a server, which has not been done yet)
# ------------------------------------------------------------------------------

import time
import importlib
import numpy as np
import pymongo

from spearmint.utils.database.mongodb import MongoDB
from spearmint.tasks.task_group import TaskGroup

class Experiment:
    def __init__(self,
                 name,
                 description='',
                 parameters=None,
                 outcome=None,
                 resume=True,
                 access_token=None,
                 likelihood='GAUSSIAN'): # other option is NOISELESS
        for pval in parameters.itervalues():
            pval['size'] = 1 # add default size = 1
            if pval['type'] == 'integer':
                pval['type'] = 'int' # spearmint int type
        self.parameters = parameters
        self.name = name
        self.outcome = outcome
        if not 'objective' in outcome:
            self.outcome['objective'] = 'maximize'
        db_address = 'localhost'
        print 'Using database at {0}.'.format(db_address)
        self.db = MongoDB(database_address=db_address)
        if not resume: # cleanup db first
            client = pymongo.MongoClient(db_address)
            print 'cleaning up ' + self.name
            client.spearmint[self.name]['jobs'].drop()
            client.spearmint[self.name]['hypers'].drop()

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

    def load_task_group(self):
        jobs = self.load_jobs()

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
        task_group = self.load_task_group()

        # Load the model hypers from the database.
        hypers = self.db.load(self.name, 'hypers')

        # "Fit" the chooser - give the chooser data and let it fit the model.
        hypers = self.chooser.fit(task_group, hypers, self.tasks)

        # Save the hyperparameters to the database.
        self.db.save(hypers, self.name, 'hypers')

        # Ask the chooser to actually pick one.
        suggested_input = self.chooser.suggest()
        params = task_group.paramify(suggested_input)

        job_id = len(jobs) + 1
        start_time = time.time()
        job = {
            'id'        : job_id,
            'params'    : params,
            'status'    : 'pending',
            'start time': start_time,
            'end time'  : None
        }
        self.db.save(job, self.name, 'jobs', {'id' : job_id})
        self.job_id = job_id # save for update method

        # just extract first value of each parameter name
        params_simple = {}
        #params_simple = {name: params[name]['values'][0] for name in params}
        for name in params:
            ptype = self.parameters[name]['type']
            if ptype == 'int':
                params_simple[name] = int(params[name]['values'][0])
            else:
                params_simple[name] = params[name]['values'][0]
        return params_simple

    def update(self, param_values, outcome_val):
        #print 'updating...'

        if self.outcome['objective'].lower() == 'maximize':
            outcome_val = -outcome_val # negate to maximize (default behavior of spearmint: minimize)

        job = self.db.load(self.name, 'jobs', {'id' : self.job_id})

        end_time = time.time()

        job['values']   = {'main': outcome_val}
        job['status']   = 'complete'
        job['end time'] = end_time

        self.db.save(job, self.name, 'jobs', {'id' : self.job_id})
