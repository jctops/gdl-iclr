"""
SHERPA is a Python library for hyperparameter tuning of machine learning models.
Copyright (C) 2018  Lars Hertel, Peter Sadowski, and Julian Collado.
This file is part of SHERPA.
SHERPA is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
SHERPA is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with SHERPA.  If not, see <http://www.gnu.org/licenses/>.
"""
import fasteners
import logging
import numpy
import os
# import pymongo
# from pymongo import MongoClient
import subprocess
import time
from tinydb import TinyDB, Query
from tinydb.table import Document
import os
import socket
import warnings
try:
    from subprocess import DEVNULL # python 3
except ImportError:
    import os
    DEVNULL = open(os.devnull, 'wb')
import sherpa


dblogger = logging.getLogger(__name__)

def ensure_path(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            # Probably a concurrent attempt to create this directory
            pass

class _Database(object):
    """
    Manages a Mongo-DB for storing metrics and delivering parameters to trials.
    The Mongo-DB contains one database that serves as a queue of future trials
    and one to store results of active and finished trials.
    Attributes:
        dbpath (str): the path where Mongo-DB stores its files.
        port (int): the port on which the Mongo-DB should run.
        reinstantiated (bool): whether an instance of the MongoDB is being loaded.
        mongodb_args (dict): keyword arguments to MongoDB
    """
    def __init__(self, db_dir):
        self.dir = db_dir
        ensure_path(db_dir)
        
        self.results_db_path = os.path.join(db_dir, 'results_db.json')
        self.results_db = TinyDB(self.results_db_path)
        self.results_lock_path = os.path.join(db_dir, 'results_lock.file')
        self.results_lock = fasteners.InterProcessLock(self.results_lock_path)
        
        self.trial_db_path = os.path.join(db_dir, 'trial_db.json')
        self.trial_db = TinyDB(self.trial_db_path)
        self.trial_lock_path = os.path.join(db_dir, 'trial_lock.file')
        self.trial_lock = fasteners.InterProcessLock(self.trial_lock_path)
        
        self.stop_db_path = os.path.join(db_dir, 'stop_db.json')
        self.stop_db = TinyDB(self.stop_db_path)
        self.stop_lock_path = os.path.join(db_dir, 'stop_lock.file')
        self.stop_lock = fasteners.InterProcessLock(self.stop_lock_path)
        
        self.collected_results = set()

    def close(self):
        print('Closing! No MongoDB to terminate, deleting db JSON files')
        def remove(path):
            if os.path.exists(path):
                os.remove(path)
        remove(self.results_db_path)
        remove(self.trial_db_path)
        remove(self.stop_db_path)
        remove(self.results_lock_path)
        remove(self.trial_lock_path)
        remove(self.stop_lock_path)
        print('Removed')

    def start(self):
        """
        Runs the DB in a sub-process.
        """
        pass

    def check_db_status(self):
        """
        Checks whether database is still running.
        """
        pass

    def get_new_results(self):
        """
        Checks database for new results.
        Returns:
            (list[dict]) where each dict is one row from the DB.
        """
        new_results = []
        with self.results_lock:
            for entry in self.results_db.all():
                result = entry
                mongo_id = result.doc_id
                if mongo_id not in self.collected_results:
                    new_results.append(result)
                    self.collected_results.add(mongo_id)
        return new_results

    def enqueue_trial(self, trial):
        """
        Puts a new trial in the queue for trial scripts to get.
        """
        trial = {'trial_id': trial.id,
                 'parameters': trial.parameters}
        with self.trial_lock:
            try:
                self.trial_db.insert(Document(trial, doc_id=len(self.trial_db)))
            except TypeError:
                new_params = {}
                for k, v in trial['parameters'].items():
                    if isinstance(v, numpy.int64):
                        v = int(v)

                    new_params[k] = v
                    
                trial['parameters'] = new_params
                self.trial_db.insert(Document(trial, doc_id=len(self.trial_db)))

    def add_for_stopping(self, trial_id):
        """
        Adds a trial for stopping.
        In the trial-script this will raise an exception causing the trial to
        stop.
        Args:
            trial_id (int): the ID of the trial to stop.
        """
        with self.stop_lock:
            self.stop_db.insert(Document({'trial_id': trial_id}, doc_id=len(self.stop_db)))

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class Client(object):
    """
    Registers a session with a Sherpa Study via the port of the database.
    This function is called from trial-scripts only.
    Attributes:
        host (str): the host that runs the database. Passed host, host set via
            environment variable or 'localhost' in that order.
        port (int): port that database is running on. Passed port, port set via
            environment variable or 27010 in that order.
    """
    def __init__(self, db_dir=None):
        db_dir = db_dir or os.environ.get('SHERPA_OUTPUT_DIR')
        assert db_dir, "Environment-variable SHERPA_OUTPUT_DIR not found. Scheduler needs to set this variable in the environment when submitting a job"
        self.dir = db_dir
        ensure_path(db_dir)
        
        self.results_db_path = os.path.join(db_dir, 'results_db.json')
        self.results_db = TinyDB(self.results_db_path)
        self.results_lock_path = os.path.join(db_dir, 'results_lock.file')
        self.results_lock = fasteners.InterProcessLock(self.results_lock_path)
        
        self.trial_db_path = os.path.join(db_dir, 'trial_db.json')
        self.trial_db = TinyDB(self.trial_db_path)
        self.trial_lock_path = os.path.join(db_dir, 'trial_lock.file')
        self.trial_lock = fasteners.InterProcessLock(self.trial_lock_path)
        
        self.test_mode = False

    def get_trial(self):
        """
        Returns the next trial from a Sherpa Study.
        Returns:
            sherpa.core.Trial: The trial to run.
        """
        if self.test_mode:
            return sherpa.Trial(id=1, parameters={})
        
        assert os.environ.get('SHERPA_TRIAL_ID'), "Environment-variable SHERPA_TRIAL_ID not found. Scheduler needs to set this variable in the environment when submitting a job"
        trial_id = int(os.environ.get('SHERPA_TRIAL_ID'))
        for _ in range(5):
            with self.trial_lock:
                g = (entry for entry in self.trial_db.search(Query()['trial_id']==trial_id))
                t = next(g)
            if t:
                break
            time.sleep(10)
        if not t:
            raise RuntimeError("No Trial Found!")
        return sherpa.Trial(id=t.get('trial_id'), parameters=t.get('parameters'))

    def send_metrics(self, trial, iteration, objective, context={}):
        """
        Sends metrics for a trial to database.
        Args:
            trial (sherpa.core.Trial): trial to send metrics for.
            iteration (int): the iteration e.g. epoch the metrics are for.
            objective (float): the objective value.
            context (dict): other metric-values.
        """
        if self.test_mode:
            return
        
        result = {'parameters': trial.parameters,
                  'trial_id': trial.id,
                  'objective': objective,
                  'iteration': iteration,
                  'context': context}
        # Convert float32 to float64.
        # Note: Keras ReduceLROnPlateau callback requires this.
        for k,v in context.items():
            if type(v) == numpy.float32:
                context[k] = numpy.float64(v)
        
        # with fasteners.InterProcessLock(self.results_lock_path):
        with self.results_lock:
            self.results_db.insert(Document(result, doc_id=len(self.results_db)))

    def keras_send_metrics(self, trial, objective_name, context_names=[]):
        """
        Keras Callbacks to send metrics to SHERPA.
        Args:
            trial (sherpa.core.Trial): trial to send metrics for.
            objective_name (str): the name of the objective e.g. ``loss``,
                ``val_loss``, or any of the submitted metrics.
            context_names (list[str]): names of all other metrics to be
                monitored.
        """        
        import keras.callbacks
        send_call = lambda epoch, logs: self.send_metrics(trial=trial,
                                                          iteration=epoch,
                                                          objective=logs[objective_name],
                                                          context={n: logs[n] for n in context_names})
        return keras.callbacks.LambdaCallback(on_epoch_end=send_call)
