import math
import random
from enum import Enum

import numpy as np
import pandas as pd
from dateutil.parser import parse

from GPAtom import Variable

_history: str = "history"
_production: str = "production"
_indicators: str = "indicators"


class ReductionFunction(Enum):
    class _Maps:
        def sigmoid(x: float) -> float:
            return 1 / (1 + math.exp(-x))

        def sigmoid_inv(x: float) -> float:
            return math.log(x) - math.log(1 - x)

        def unit_arctan(x: float) -> float:
            return 2 * math.atan(x) / math.pi

        def unit_tan(x: float) -> float:
            return math.tan(math.pi * x / 2)

        def unit_tanh(x: float) -> float:
            return (1 + math.tanh(x)) / 2

        def unit_arctanh(x: float) -> float:
            return math.atanh(2 * x - 1)

    SIGMOID = (_Maps.sigmoid, _Maps.sigmoid_inv)
    ARCTAN = (_Maps.unit_arctan, _Maps.unit_tan)
    TANH = (_Maps.unit_tanh, _Maps.unit_arctanh)


class DatasetManager:
    class ModelType(Enum):
        # Simple model - just timestamp (t) and load-shedding stage (s)
        SIMPLE = "t", "s"
        # Extended to electrical production at the given time
        EXTENDED = *SIMPLE, 'p'

    def __init__(self,
                 no_stages: int,
                 mtype: ModelType = ModelType.SIMPLE,
                 **args
                 ) -> None:

        self._no_stages = no_stages
        self._mtype = mtype
        self._tsbuffer = []

        self._history_dataset = self._parse_history("../data/south_africa_load_shedding_history.csv")
        self._production_dataset = self._parse_production("../data/Total_Electricity_Production.csv")
        self._indicators_dataset = self._parse_indicators("../data/world_indicators.csv")

    def _parse_history(self, source: str):
        ds = pd.read_csv(source, sep=",")
        keys = ds.keys()
        temp_dataset = pd.DataFrame(columns=['timestamp', 'stage'])
        for index, row in ds.iterrows():
            created_at: str = str(row[keys[0]])
            ts = int(parse(created_at).timestamp())

            self._tsbuffer.append(ts)

            stage = int(row[keys[1]])
            entry = {'timestamp': ts, 'stage': stage}
            temp_dataset = temp_dataset.append(entry, ignore_index=True)

        return temp_dataset

    def _parse_production(self, source: str):
        ds = pd.read_csv(source, sep=",")
        keys = ds.keys()
        temp_dataset = pd.DataFrame(columns=['timestamp', 'output'])
        for index, row in ds.iterrows():
            created_at: str = str(row[keys[0]])
            ts = int(parse(created_at).timestamp())
            production = float(row[keys[1]])
            entry = {'timestamp': ts, 'output': production}
            temp_dataset = temp_dataset.append(entry, ignore_index=True)

        return temp_dataset

    def get_reducer(self, rtype=ReductionFunction.TANH, normalize: bool = False):
        stdev_timestamp = np.std(self._tsbuffer)
        avg_timetamp = np.mean(self._tsbuffer)
        rforward, rinverse = rtype.value
        # return a reduction function normalised by the timestamps
        return lambda x: (rforward((x - avg_timetamp) / stdev_timestamp) if normalize else rforward(x))

    def _get_closest_production(self, timestamp: int):
        time_index = self._production_dataset['timestamp']
        closest_time = min(time_index, key=lambda t: abs(t - timestamp))
        return self._production_dataset.loc[self._production_dataset.timestamp == closest_time, "output"].tolist()[0]

    def generate_variables(self):
        return [Variable(k) for k in self._mtype.value]

    def generate_fitness_cases(self, insertion_factor: int = 0):
        cases = []
        prev = None
        for _, row in self._history_dataset.iterrows():

            # get the time and stage
            ts = int(row['timestamp'])
            st = int(row['stage'])

            # if the dataset must be flattened
            if insertion_factor > 0 and prev is not None:
                # get the timestamps between the prev and now
                oldts = prev['t']
                tsrange = ts - oldts
                interval = tsrange // (1 + insertion_factor)
                for i in range(insertion_factor):
                    targetts = int(oldts + (i+1) * interval)
                    prev['t'] = targetts
                    prev['attributes'] = 'synthesised'

                    # stage ranges from 0 - self._no_stages, inclusive
                    # for a random stage that is not the actual stage, indicate falseness
                    non_stages = list(k for k in range(self._no_stages))
                    non_stages.remove(st)
                    nonstage = random.choice(non_stages)
                    false_args = prev.copy()
                    false_args["s"] = nonstage

                    truth_case = (prev, 1)
                    false_case = (false_args, 0)

                    cases.append(truth_case)
                    cases.append(false_case)

            args = {"t": ts, "s": st, "attributes": None}
            if self._mtype == DatasetManager.ModelType.EXTENDED:
                # get the production at closest time
                pd = self._get_closest_production(ts)
                args["p"] = pd

            prev = args.copy()

            # stage ranges from 0 - self._no_stages, inclusive
            # for a random stage that is not the actual stage, indicate falseness
            non_stages = list(k for k in range(self._no_stages))
            non_stages.remove(st)
            nonstage = random.choice(non_stages)
            false_args = args.copy()
            false_args["s"] = nonstage

            truth_case = (args, 1)
            false_case = (false_args, 0)

            cases.append(truth_case)
            cases.append(false_case)




        return cases

    def _parse_indicators(self, source: str):
        empty_value = ".."
        years_begin_col = 4
        row_bound = 19
        property_col = 3
        country_col = 1

        ds = pd.read_csv(source, sep=",")
        ks = ds.keys()

        # all years
        ys = [int(k[:4]) for k in ks[years_begin_col:]]

        # first pass to establish properties and country codes
        props = []
        prop_index = {}
        for i, r in ds.iterrows():
            if i == row_bound:
                break
            prop = r[ks[property_col]]
            props.append(prop)
            prop_index[i] = prop

        # initialise dictionary for year outlook
        year_outlook = {}
        for y in ys:
            year_outlook[y] = pd.DataFrame(columns=["code"] + props)

        # now go through all the rows in intervals of row_bound
        i = -1
        country_list = []
        for cn, ci in ds.iteritems():

            # skip non data columns
            i += 1
            if i == country_col:
                country_list = list(dict.fromkeys(ci))
                continue
            elif i < years_begin_col:
                continue

            # get year of col
            year = int(cn[:4])

            # go through the items in batches of row_bound
            country_buffer = {}
            for j, r in enumerate(ci):

                current_property = prop_index.get(j % row_bound)
                country_buffer[current_property] = r if r is not empty_value else None

                current_country = country_list[int(j / row_bound)]

                if (j + 1) % row_bound == 0:
                    country_buffer['code'] = current_country
                    # add the country buffer contents to the year dataframe
                    year_outlook[year] = year_outlook.get(year).append(country_buffer, ignore_index=True)
                    # reset buffer
                    country_buffer = {}

        return year_outlook
