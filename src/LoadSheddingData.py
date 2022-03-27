import numpy as np
import pandas as pd
from dateutil.parser import parse

from LoadSheddingFunctions import ReductionFunction

_history: str = "history"
_production: str = "production"
_indicators: str = "indicators"


class DatasetManager:
    class ModelType:
        # Simple model - just timestamp (t) and load-shedding stage (s)
        SIMPLE = "t", "s"
        # Extended to electrical production at the given time
        EXTENDED_PRODUCTION = *SIMPLE, 'p'

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
        return lambda x: (rforward((x-avg_timetamp)/stdev_timestamp) if normalize else rforward(x))

    def generate_fitness_cases(self):
        cases = []
        for _, row in self._history_dataset.iterrows():

            # get the time and stage
            ts = int(row['timestamp'])
            st = int(row['stage'])

            # stage ranges from 0 - self._no_stages, inclusive
            # for each stage, indicate truth, falseness
            for stage in range(self._no_stages):
                # the target is truth or falseness of a time corresponding to a stage
                target = int(st == stage)
                new_case = ({"t": ts, "s": st}, target)
                cases.append(new_case)

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
