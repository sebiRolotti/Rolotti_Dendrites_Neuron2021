import MySQLdb
import MySQLdb.cursors
import xml.etree.ElementTree as etree
import os.path
import re
import time
import json

from lab_repo.classes.classes import Mouse, Experiment, Trial


class ExperimentDatabase():
    """An object to wrap MySQLdb and handle all connections to mySQL server

    Example
    -------

    >>> from lab.classes import database
    >>> db = database.ExperimentDatabase()
    >>> db.selectAll('SELECT * FROM mice LIMIT 10')
    >>> db.disconnect()
    """

    _trial_fields = ['trial_id', 'mouse_id', 'start_time', 'stop_time',
        'behavior_file', 'tSeries_path', 'experiment_group', 'experiment_id',
        'mouse_name']

    def __init__(self):
        """Connects to SQL database upon initialization."""

        self.connect()

    def connect(self):
        """Connect to the SQL database."""

        host = ''
        user = ''       #TODO: use different credentials
        password = ''  #TODO: update credentials
        database = 'experiments'

        self._database = MySQLdb.connect(
            host=host,
            user=user,
            passwd=password,
            db=database,
            cursorclass=MySQLdb.cursors.DictCursor)

    def disconnect(self):
        """Close the connection to SQL database server."""

        self._database.close()

    def select(self, sql, args=(), verbose=False):
        """ Queries SQL database return single result

        Parameters
        ----------
        sql : String
            Raw SQL to pass to the database.
        args : list, optional
            List of variables to sub into SQL statement in accordiance see
            MySQLdb. Defaults to an empty list.
        verbose : bool, optional
            If set to True print the SQL query. Defaults to False

        Returns
        -------
        Frst record to match SQL query as a dictioary, with the field name being
        the key.
        """

        cursor = self._database.cursor()
        cursor.execute(sql, args)
        if verbose:
            print sql

        result = cursor.fetchone()
        return result

    def selectAll(self, sql, args=(), verbose=False):
        """ Queries SQL database and return all the results

        Parameters
        ----------
        sql : String
            Raw SQL to pass to the database.
        args : list, optional
            List of variables to sub into SQL statement in accordiance see
            MySQLdb. Defaults to an empty list.
        verbose : bool, optional
            If set to True print the SQL query. Defaults to False

        Returns
        -------
        All records to match SQL query as a dictioary, with the field name being
        the key.
        """

        cursor = self._database.cursor()
        cursor.execute(sql, args)
        if verbose:
            print sql

        result = cursor.fetchall()
        return result

    def query(self, sql, args=(), verbose=False, ignore_duplicates=True):
        """ Run an arbitrary SQL query i.e. INSERT or DELETE commands. SQL
        statement is passsed directory MySQLdb.

        Parameters
        ----------
        sql : String
            Raw SQL statement to query the database with.
        args : list, optional
            List of variables to sub into SQL statement in accordiance see
            MySQLdb. Defaults to an empty list.
        verbose : bool, optional
            If set to True print the SQL query. Defaults to False
        ignore_duplicates : bool, optional
            If set to True Duplicate entry errors are prevented from being
            raised, instead of raising an error the method returns False.
            Defaults to True.

        Returns
        -------
        True if the SQL statement executes, False otherwise
        """

        cursor = self._database.cursor()
        try:
            cursor.execute(sql, args)
        except Exception as e:
            if not ignore_duplicates or 'Duplicate entry' not in e.__str__():
                raise e
            return False
        else:
            if verbose:
                print sql
            self._database.commit()
        return True


def createExperiment(tSeries_path=None, behavior_file=None, trial_id=None):
    """ Create a lab.clasees.Experiment from trial in the SQL database. Must
    give either tSeries_path, behavior_file or trial_id.

    Parameters
    ----------
    tSeries_path : String, optional
        Path to tSeries folder matching the path in a SQL record. Defaults to
        None.
    behavior_file : String, optional
        Path to a behavior file which matches a SQL record. Defaults to None.
    trial_id : int, optional
        Trial id from the SQL trials table to match SQL record. Defaults to
        None.

    Returns
    -------
    lab.classes.Experiment object corresponding to requested SQL record.
    """

    args = ['tSeries_path', 'behavior_file', 'trial_id']
    arg = [a for a in args if eval(a) is not None][0]

    db = ExperimentDatabase()

    experimentSet = etree.fromstring('<experimentset />')
    experimentSet.behaviorDataPath = '/'
    experimentSet.beltXmlPath = \
        '/analysis/experimentSummaries/.clean-code/experiments/belts.xml'
    experimentSet.root = experimentSet
    experimentSet.dataPath = '/'

    trial = db.select("""
        SELECT * FROM trials
        WHERE {} = %s
        """.format(arg), args=[eval(arg)])

    mouse_id = db.select("""
        SELECT `mouse_name`
        FROM mice
        INNER JOIN trials
        ON mice.`mouse_id`=trials.`mouse_id` WHERE `trial_id`=%s""",
                         [trial['trial_id']])['mouse_name']
    mouse_db_id = fetchMouseId(mouse_id)

    mouseString = '<mouse mouseID="{}"/>'.format(mouse_id)
    mouse = etree.fromstring(mouseString)
    mouse_attrs = fetchAllMouseAttrs(mouse_db_id)
    for attribute, value in mouse_attrs.iteritems():
        mouse.set(attribute, value)
    experimentSet.insert(0, mouse)
    mouse.parent = experimentSet
    mouse.__class__ = Mouse

    tSeries_path = ''
    if 'tSeries_path' in trial.keys() and trial['tSeries_path'] is not None:
        if not os.path.isfile(trial['tSeries_path']):
            tSeries_path = trial['tSeries_path']
        else:
            tSeries_path = os.path.dirname(trial['tSeries_path'])

    experimentString = """
        <experiment startTime="{}"
                    stopTime="{}"
                    experimentType=""
                    trial_id="{}"
                    imagingLayer="unk"
                    uniqueLocationKey=""
                    experimenter=""
                    belt=""
                    tSeriesDirectory="{}"/>""".format(
        trial['start_time'].__format__('%Y-%m-%d-%Hh%Mm%Ss'),
        trial['stop_time'].__format__('%Y-%m-%d-%Hh%Mm%Ss'),
        trial['trial_id'], tSeries_path)

    experiment = etree.fromstring(experimentString)
    experiment_attrs = fetchAllTrialAttrs(trial['trial_id'])
    for attribute, value in experiment_attrs.iteritems():
        experiment.set(attribute, value)

    experiment._rois = {}
    mouse.insert(0, experiment)
    experiment.parent = mouse
    experiment.__class__ = Experiment
    if trial['experiment_id'] is not None:
        start_time = db.select("""
                SELECT `start_time`
                FROM experiments
                WHERE experiment_id = %s
            """, args=trial['experiment_id'])
        experiment.set('startTime', start_time['start_time'])

        trials = db.selectAll("""
                SELECT `behavior_file`, `start_time`
                FROM trials
                WHERE experiment_id = %s
                ORDER BY start_time
            """, args=trial['experiment_id'])
        for trial in trials[::-1]:
            trialString = '<trial filename="{}" time="{}"/>'.format(
                os.path.splitext(trial['behavior_file'])[0] + '.csv',
                trial['start_time'].__format__('%Y-%m-%d-%Hh%Mm%Ss'))
            trial = etree.fromstring(trialString)
            experiment.insert(0, trial)
            trial.parent = experiment
            trial.__class__ = Trial
    else:
        try:
            behaviorFile = os.path.splitext(trial['behavior_file'])[0] + '.csv'
        except:
            behaviorFile = ""

        trialString = '<trial filename="{}" time="{}"/>'.format(
            behaviorFile, trial['start_time'].__format__('%Y-%m-%d-%Hh%Mm%Ss'))
        trial = etree.fromstring(trialString)
        experiment.insert(0, trial)
        trial.parent = experiment
        trial.__class__ = Trial

    return experiment

def createMouse(mouse_name):
    """ Create an xml.etree.Element of all the experiments for the provided
    mouse and set the parent node correctly.

    Parameters
    ----------
    mouse_name : String
        Name of the mouse to search for records of. Matches to mouse_name in the
        SQL table mice.

    Returns
    -------
    xml.etree.Element with the provded mouse as the parent Node.
    """

    db = ExperimentDatabase()

    paths = db.selectAll("""
        SELECT behavior_file
        FROM trials
        INNER JOIN mice
        ON trials.mouse_id = mice.mouse_id
        WHERE mouse_name = %s
        ORDER BY start_time DESC
        """, args=[mouse_name])

    expts = [createExperiment(behavior_file=path.values()[0]) for path in
             paths]
    mouse = expts.pop(0).parent
    for expt in expts:
        mouse.insert(0, expt)

    return mouse


def pairImagingData(imaging_path, trial_id=None, behavior_file=None,
                    start_time=None):
    """ Pair a tSeries directory with a trial record in the database.

    Parameters
    ----------
    imaging_path : String
        tSeries directory to add to trial record.
    trial_id : int, optional
        Trial id from the SQL trials table to match SQL record. Defaults to
        None.
    behavior_file : String, optional
        Path to a behavior file which matches a SQL record. Defaults to None.
    start_time : String, optional
        Date time string to match a to  SQL record. Defaults to None.
    """

    db = ExperimentDatabase()

    if trial_id and behavior_file:
        raise ValueError('Cannot use both trial id and behavior file path')
    elif trial_id is not None:
        db.query("""
            UPDATE trials
            SET `tSeries_path` = %s
            WHERE `trial_id` = %s
            """, args=[imaging_path, trial_id], verbose=False)
    elif behavior_file is not None:
        behavior_file = os.path.splitext(behavior_file)[0] + '%'
        db.query("""
            UPDATE trials
            SET `tSeries_path` = %s
            WHERE behavior_file LIKE %s
        """, args=[imaging_path, behavior_file], verbose=False)
        trial_id = fetchTrialId(behavior_file=behavior_file)

    #TODO: should be start_time/mouse_name combination to ensure this value is
    # unique.
    elif start_time is not None:
        start_time = _resolveStartTime(start_time)
        #TODO: experiments table is not working (or used) necessary for legacy
        # saliance compatibilty. Update or remove...

        #expt_id = fetchExperimentId(start_time)
        #if expt_id is not None:
        #    db.query("""
        #        UPDATE trials
        #        SET `tSeries_path` = %s
        #        WHERE `experiment_id` = %s
        #    """, args=[imaging_path, expt_id], verbose=False)
        #else:
        db.query("""
            UPDATE trials
            SET `tSeries_path` = %s
            WHERE `start_time` = %s
        """, args=[imaging_path, start_time], verbose=False)

    else:
        raise ValueError('Need either trial id or behavior file path')

    db.disconnect()


def updateTrial(
        behavior_file, mouse_name, start_time, stop_time, experiment_group,
        trial_id=None):

    mouse_id = fetchMouseId(mouse_name, create=True)
    if trial_id is None:
        trial_id = fetchTrialId(behavior_file=behavior_file)
    start_time = _resolveStartTime(start_time)
    stop_time = _resolveStartTime(stop_time)

    db = ExperimentDatabase()
    # if this behavior_file path is alread associated with the trial, update the
    # existing record, otherwise create a new one.
    if trial_id is not None:
        db.query("""
            UPDATE trials
            SET `mouse_id` = %s,
                `start_time` = %s,
                `stop_time` = %s,
                `behavior_file` = %s,
                `experiment_group` = %s
            WHERE trial_id = %s
            """, args=[mouse_id, start_time, stop_time, behavior_file,
                 experiment_group, trial_id])
    else:
        db.query("""
            INSERT INTO trials
                (`mouse_id`, `start_time`, `stop_time`, `behavior_file`, `experiment_group`)
            VALUES (%s, %s, %s, %s, %s)
            """, args=[mouse_id, start_time, stop_time, behavior_file,
                 experiment_group])

    db.disconnect()


def updateAttr(table, group_name, group_id, attribute, value):
    db = ExperimentDatabase()
    attr = db.select("""
        SELECT attribute_id
        FROM {}
        WHERE {} = %s
        AND attribute = %s
        """.format(table, group_name), args=[group_id, attribute],
        verbose=False)

    if attr is not None:
        db.query("""
            UPDATE {}
            SET value = %s
            WHERE attribute_id = %s
            """.format(table),
                 args=[value, attr['attribute_id']], verbose=False)
    else:
        db.query("""
            INSERT INTO {} (`{}`, `attribute`, `value`)
            VALUES (%s, %s, %s)
            """.format(table, group_name),
                 args=[group_id, attribute, value], verbose=False)

    db.disconnect()


def fetchAttr(table, group_name, group_id, attribute, default=None):
    db = ExperimentDatabase()
    attr = db.select("""
        SELECT value
        FROM {}
        WHERE {} = %s
        AND attribute = %s
        """.format(table, group_name),
        args=[group_id, attribute], verbose=False)
    db.disconnect()

    if attr:
        return attr['value']

    return default

def _deleteAllAttrs(table, group_name, group_id):
    db = ExperimentDatabase()
    db.query("""
        DELETE FROM {}
        WHERE {} = %s
        """.format(table, group_name), args=[group_id], verbose=False)
    db.disconnect()

def deleteAttr(table, group_name, group_id, attribute):
    db = ExperimentDatabase()
    db.query("""
        DELETE FROM {}
        WHERE {} = %s
        AND attribute = %s
        """.format(table, group_name),
        args=[group_id, attribute], verbose=False)
    db.disconnect()

def updateMouseAttr(mouse_name, attribute, value):
    mouse_id = fetchMouseId(mouse_name)
    updateAttr('mouse_attributes', 'mouse_id', mouse_id, attribute, value)


def updateMousePage(mouse_name, attribute, value):
    mouse_id = fetchMouseId(mouse_name)
    if type(value) != str:
        value = json.dumps(value)

    updateAttr('mouse_pages', 'mouse_id', mouse_id, attribute, value)

def fetchMouseAttr(mouse_name, attribute, default=None):
    mouse_id = fetchMouseId(mouse_name)
    return fetchAttr(
        'mouse_attributes', 'mouse_id', mouse_id, attribute, default=default)

def deleteMouseAttr(mouse_name, attribute, default=None):
    mouse_id = fetchMouseId(mouse_name)
    deleteAttr(
        'mouse_attributes', 'mouse_id', mouse_id, attribute)

def fetchMousePageAttr(mouse_name, attribute, default=None):
    mouse_id = fetchMouseId(mouse_name)
    attr = fetchAttr(
        'mouse_pages', 'mouse_id', mouse_id, attribute, default=default)
    if type(attr) == str:
        try:
            attr = json.loads(attr)
        except ValueError:
            pass

    return attr

def deleteMousePageAttr(mouse_name, attribute):
    mouse_id = fetchMouseId(mouse_name)
    deleteAttr(
        'mouse_pages', 'mouse_id', mouse_id, attribute)

def updateTrialAttr(trial_id, attribute, value):
    updateAttr('trial_attributes', 'trial_id', trial_id, attribute, value)

def fetchTrialAttr(trial_id, attribute, default=None):
    return fetchAttr(
        'trial_attributes', 'trial_id', trial_id, attribute, default=default)

def deleteTrialAttr(trial_id, attribute):
    deleteAttr(
        'trial_attributes', 'trial_id', trial_id, attribute)

def updateVirusAttr(trial_id, attribute, value):
    updateAttr('virus_attributes', 'virus_id', trial_id, attribute, value)

def fetchVirusAttr(trial_id, attribute, default=None):
    return fetchAttr(
        'virus_attributes', 'virus_id', trial_id, attribute, default=default)

def deleteVirusAttr(trial_id, attribute):
    deleteAttr(
        'virus_attributes', 'virus_id', trial_id, attribute)

def _fetchAllAttrs(table, id_field, id_value, parse=False):
    db = ExperimentDatabase()
    attrs = db.selectAll("""
        SELECT attribute, value
        FROM {}
        WHERE {} = %s
        """.format(table, id_field), args=[id_value], verbose=False)
    db.disconnect()

    attrs = {entry['attribute']:entry['value'] for entry in attrs}
    if parse:
        for key in attrs.keys():
            try:
                attrs[key] = json.loads(attrs[key])
            except ValueError:
                pass

    return attrs

def fetchAllVirusAttrs(virus_id, parse=True):
    attrs = _fetchAllAttrs('virus_attributes', 'virus_id', virus_id)

    return attrs

def fetchAllMouseAttrs(mouse_name, parse=False):
    if isinstance(mouse_name, int):
        mouse_id = mouse_name
    else:
        mouse_id = fetchMouseId(mouse_name)

    return _fetchAllAttrs('mouse_attributes', 'mouse_id', mouse_id, parse=parse)

def fetchAllTrialAttrs(trial_id, parse=False):
    db = ExperimentDatabase()
    attrs = db.selectAll("""
        SELECT attribute, value
        FROM trial_attributes
        WHERE trial_id = %s
        """, args=[trial_id], verbose=False)
    db.disconnect()

    if parse:
        for i,entry in enumerate(attrs):
            try:
                attrs[i]['value'] = json.loads(entry['value'])
            except ValueError:
                pass

    return {entry['attribute']: entry['value'] for entry in attrs}

def deleteAllTrialAttrs(trial_id):
    _deleteAllAttrs('trial_attributes', 'trial_id', trial_id)

def deleteAllMouseAttrs(mouse_id):
    _deleteAllAttrs('mouse_attributes', 'mouse_id', mouse_id)
    _deleteAllAttrs('mouse_pages', 'mouse_id', mouse_id)

def _projectFilterSql(project_name):
    return """
        (SELECT DISTINCT m.*
         FROM mice m
         LEFT JOIN trials t
            ON m.mouse_id=t.mouse_id
         WHERE experiment_group='{0}'
         UNION
         SELECT DISTINCT m.*
         FROM mice m
         LEFT JOIN mouse_attributes ma
            ON m.mouse_id=ma.mouse_id
         WHERE (attribute='project_name' AND value='{0}')
         )
    """.format(project_name)


def deleteTrial(trial_id):
    deleteAllTrialAttrs(trial_id)

    db = ExperimentDatabase()
    db.query("""
        DELETE FROM trials
        WHERE trial_id = %s
        """, args=[trial_id])


def deleteMouse(mouse_id):
    deleteAllMouseAttrs(mouse_id)

    db = ExperimentDatabase()
    db.query("""
        DELETE FROM mice
        WHERE mouse_id = %s
        """, args=[mouse_id], verbose=False)


def fetchMouseId(mouse_name, create=False, project_name=None):
    db = ExperimentDatabase()

    if project_name is not None:
        mouse_id = db.selectAll("""
            SELECT DISTINCT mice.mouse_id
            FROM mice
            INNER JOIN trials
                ON trials.mouse_id=mice.mouse_id
            WHERE mouse_name = %s
                AND experiment_group = %s
            UNION
            SELECT mice.mouse_id
            FROM mice
            INNER JOIN mouse_attributes
                ON mice.mouse_id = mouse_attributes.mouse_id
            WHERE mouse_name = %s
                AND attribute = %s
                AND value = %s
            """, args=[mouse_name, project_name, mouse_name, 'project_name',
                       project_name])

    else:
        mouse_id = db.selectAll("""
            SELECT mouse_id
            FROM mice
            WHERE mouse_name = %s
            """, args=[mouse_name])

    if create and len(mouse_id) == 0:
        db.query("""
            INSERT INTO mice (mouse_name)
            VALUES (%s)
        """, args=[mouse_name])
        db.disconnect()
        if project_name is not None:
            updateMouseAttr(mouse_name, 'project_name', project_name)
        mouse_id = fetchMouseId(mouse_name, create=False)
    elif len(mouse_id) == 1:
        db.disconnect()
        mouse_id = int(mouse_id[0].values()[0])
    else:
        raise KeyError('unable to uniquely identify mouse {}'.format(
            mouse_name))

    return mouse_id


def fetchMouse(mouse_id):
    db = ExperimentDatabase()
    mouse = db.select("""
        SELECT *
        FROM mice
        WHERE mouse_id = %s
    """, args=[mouse_id])

    return mouse


def fetchTrial(trial_id):
    db = ExperimentDatabase()
    trial = db.select("""
        SELECT *
        FROM trials
        WHERE trial_id = %s
        """, args=[trial_id])
    return trial


def fetchTrialId(behavior_file=None, tSeries_path=None, mouse_name=None,
        startTime=None):
    db = ExperimentDatabase()

    args = ['tSeries_path', 'behavior_file', 'mouse_name', 'startTime']
    arg = [a for a in args if eval(a) is not None]

    if 'startTime' in arg:
        trial_id = db.select("""
            SELECT trial_id FROM trials
            INNER JOIN mice
                ON trials.mouse_id=mice.mouse_id
            WHERE mouse_name = %s
                AND start_time = %s
        """, args=[mouse_name, _resolveStartTime(startTime)])
    elif 'startTime' in arg:
        raise Exception("require mouse_id")
    else:
        trial_id = db.select("""
            SELECT trial_id FROM trials
            WHERE {} = %s
            """.format(arg[0]), args=[eval(arg[0])])

    if trial_id is None:
        return None

    return int(trial_id.values()[0])


def fetchImagedTrials(mouse_name):
    db = ExperimentDatabase()
    trials = db.selectAll("""
        SELECT trial_id, start_time, mouse_name, behavior_file, tSeries_path
        FROM trials
        INNER JOIN mice
        ON mice.mouse_id = trials.mouse_id
        WHERE tSeries_path IS NOT NULL
        AND mouse_name = %s
        ORDER BY start_time ASC
        """, args=[mouse_name])
    db.disconnect()

    return trials


def fetchAttributeValues(attr, project_name=None):
    db = ExperimentDatabase()

    table = 'mice'
    if attr in db._trial_fields:
        condition=''
        if project_name is not None:
            condition = 'WHERE experiment_group=\'{}\''.format(project_name)

        values = db.selectAll("""
            SELECT DISTINCT {} FROM trials t
            LEFT JOIN mice m
                ON t.mouse_id=m.mouse_id
            {}
        """.format(attr, condition))
        return [value[attr] for value in values]

    if project_name is not None:
        table = _projectFilterSql(project_name)

    values = db.selectAll("""
        SELECT DISTINCT value FROM {0} AS m
        LEFT JOIN trials t
            ON m.mouse_id=t.mouse_id
        LEFT JOIN trial_attributes ta
            ON t.trial_id=ta.trial_id
        WHERE attribute=%s
        UNION
        SELECT DISTINCT value FROM {0} AS m
        LEFT JOIN mouse_attributes ma
            ON m.mouse_id=ma.mouse_id
        WHERE attribute=%s
    """.format(table), args=[attr, attr])

    return [value['value'] for value in values]


def fetchTrialsWithAttrValue(attr, value):
    db = ExperimentDatabase()
    trials = db.selectAll("""
        SELECT trial_id
        FROM trial_attributes
        WHERE attribute = %s AND value = %s
        ORDER BY trial_id ASC
        """, args=[attr, value])
    db.disconnect()

    return [int(trial['trial_id']) for trial in trials]


def fetchMice(*args, **kwargs):
    db = ExperimentDatabase()
    query_string = """
        SELECT DISTINCT m.mouse_id
        FROM {0} AS m
        LEFT JOIN mouse_attributes ma
            ON m.mouse_id=ma.mouse_id
        LEFT JOIN trials t
            ON m.mouse_id=t.mouse_id
        WHERE {1}
        UNION
        SELECT DISTINCT m.mouse_id
        FROM {0} AS m
        LEFT JOIN trials t
            ON m.mouse_id=t.mouse_id
        LEFT JOIN trial_attributes ta
            ON t.trial_id=ta.trial_id
        WHERE {1}
        """

    table = 'mice'
    if 'project_name' in kwargs.keys():
        table = _projectFilterSql(kwargs['project_name'])
        del kwargs['project_name']

    if 'experiment_group' in kwargs.keys():
        table = _projectFilterSql(kwargs['experiment_group'])
        del kwargs['experiment_group']


    trial_conditions = []
    conditions = []
    query_args = []
    trial_condition = "({} IS NOT NULL)"
    condition_string = "(attribute='{}')"
    for key  in args:
        if key in db._trial_fields:
            if key == 'mouse_id':
                key = 'm.%s' % key

            trial_conditions.append(trial_condition.format(key))
        else:
            conditions.append(condition_string.format(key))

    trial_condition = "({}={})"
    condition_string = "(attribute='{}' AND value={})"
    for key, values in kwargs.iteritems():
        if type(values) != list:
            values = [values]

        alternatives = []
        trial_cond = False
        for val in values:
            if key in ['start_time', 'stop_time']:
                val = _resolveStartTime(val)

            try:
                float(val)
            except:
                val = "'{}'".format(val)

            if key in db._trial_fields:
                trial_cond = True
                if key == 'mouse_id' or key == 'trial_id':
                    _key = 't.%s' % key
                else:
                    _key = key

                if _key == 'tSeries_path':
                    alternatives.append('tSeries_path LIKE %s')
                    query_args.append(os.path.normpath(val[1:-1] + '%'))
                else:
                    alternatives.append(trial_condition.format(_key, val))
            else:
                alternatives.append(condition_string.format(key, val))
        if trial_cond:
            trial_conditions.append("({})".format(" OR ".join(alternatives)))
            trial_cond = False
        else:
            conditions.append("({})".format(" OR ".join(alternatives)))

    query = None
    if len(trial_conditions):
        query = """
            SELECT DISTINCT m.*
            FROM {} m
            INNER JOIN trials t
                ON m.mouse_id=t.mouse_id
            WHERE {}
        """.format(table, " AND ".join(trial_conditions))

    for condition in conditions:
        if query is None:
            query = query_string.format(table, condition)
        else:
            query = """
                SELECT DISTINCT m.*
                FROM ({}) AS m
                INNER JOIN ({}) as n
                ON m.mouse_id=n.mouse_id
                """.format(query, query_string.format('mice',condition))

    if query is None:
        query = """
            SELECT DISTINCT m.*
            FROM {} AS m
        """.format(table)

    records = db.selectAll(query, args=query_args)
    return [int(record['mouse_id']) for record in records]


def fetchTrials(*args, **kwargs):
    db = ExperimentDatabase()

    query_string = """
        SELECT DISTINCT t.*
        FROM {}
        LEFT JOIN trial_attributes ta
            ON t.trial_id = ta.trial_id
        LEFT JOIN mice m
            ON t.mouse_id = m.mouse_id
        WHERE {} ORDER BY start_time"""

    conditions = []
    query_args = []
    trial_condition = "({} IS NOT NULL)"
    condition_string = "(attribute='{}')"
    for key  in args:
        if key in db._trial_fields:
            if key == 'mouse_id':
                key = 'trials.%s' % key
            conditions.append(trial_condition.format(key))
        else:
            conditions.append(condition_string.format(key))

    trial_condition = "({}={})"
    condition_string = "(attribute='{}' AND value={})"
    for key, values in kwargs.iteritems():
        if type(values) != list:
            values = [values]

        alternatives = []
        for val in values:
            if key in ['start_time', 'stop_time']:
                val = _resolveStartTime(val)

            try:
                float(val)
            except:
                val = "'{}'".format(val)

            if key in db._trial_fields:
                if key == 'mouse_id' or key == 'trial_id':
                    _key = 't.%s' % key
                else:
                    _key = key

                if _key == 'tSeries_path':
                    if val[1:-1].strip() == '':
                        alternatives.append('tSeries_path IS NOT NULL')
                    else:
                        alternatives.append('tSeries_path LIKE %s')
                        query_args.append(os.path.normpath(val[1:-1]) + '%')
                else:
                    alternatives.append(trial_condition.format(_key, val))
            else:
                alternatives.append(condition_string.format(key, val))
        conditions.append("({})".format(" OR ".join(alternatives)))

    query = 'trials t'
    for condition in conditions[::-1]:
        query = " (" + query_string.format(query,condition) + ") AS t"
    query = query.rstrip(" AS t")

    trials = db.selectAll(query, args=query_args)
    return [int(trial['trial_id']) for trial in trials]


def fetchMouseTrials(mouse_name):
    db = ExperimentDatabase()
    trials = db.selectAll("""
        SELECT trial_id, start_time, mouse_name, behavior_file, tSeries_path
        FROM trials
        INNER JOIN mice
        ON mice.mouse_id = trials.mouse_id
        AND mouse_name = %s
        ORDER BY start_time ASC
        """, args=[mouse_name])
    db.disconnect()

    return trials


def _resolveStartTime(start_time):
    try:
        tstruct = time.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        pass
    else:
        return start_time

    formats = ['%Y-%m-%d-%Hh%Mm%Ss', '%m/%d/%Y']
    for t_format in formats:
        try:
            tstruct = time.strptime(start_time, t_format)
            start_time = time.strftime('%Y-%m-%d %H:%M:%S', tstruct)
        except:
            pass
        else:
            return start_time

    raise Exception('unable to parse time')



def fetchExperimentId(start_time):
    db = ExperimentDatabase()
    start_time = _resolveStartTime(start_time)

    experiment_id = db.select("""
        SELECT experiment_id
        FROM experiments
        WHERE start_time = %s
        """, args=[start_time])

    if experiment_id is None:
        experiment_id = db.select("""
            SELECT trial_id
            FROM trials
            WHERE start_time = %s
            """, args=[start_time])

        if experiment_id is None:
            return None

        return int(experiment_id.values()[0])

    return int(experiment_id.values()[0])


def fetchAllProjects():
    db = ExperimentDatabase()
    projects = [record['experiment_group'] for record in
        db.selectAll("""
            SELECT DISTINCT experiment_group
            FROM trials
            ORDER BY experiment_group ASC
        """)
    ]
    db.disconnect()

    return projects


def fetchAllMice(project_name=None):
    db = ExperimentDatabase()

    if project_name is not None:
        mice = [record['mouse_name'] for record in
            db.selectAll("""
                SELECT DISTINCT m.mouse_name
                FROM mice m
                LEFT JOIN trials t
                   ON m.mouse_id=t.mouse_id
                WHERE experiment_group='{0}'
                UNION
                SELECT DISTINCT m.mouse_name
                FROM mice m
                LEFT JOIN mouse_attributes ma
                   ON m.mouse_id=ma.mouse_id
                WHERE (attribute='project_name' AND value='{0}')
                ORDER BY mouse_name
                """.format(project_name))]
    else:
        mice = [record['mouse_name'] for record in
            db.selectAll("""
                SELECT DISTINCT mouse_name
                FROM mice
                ORDER BY mouse_name ASC
            """)
        ]

    return mice

