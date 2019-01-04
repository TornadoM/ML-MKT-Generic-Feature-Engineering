from collections import Counter
import inf_model_constant as InfConst
from datetime import datetime
from dateutil.relativedelta import relativedelta

import json
import numpy as np


class GenericDataProcessor():

    def __init__(self, meta_file):
        ### Read metadata json
        with open(meta_file, 'r') as f:
            self.meta = json.load(f)
            # ia_vocab = [line.rstrip('\n') for line in f]

        self.datetime_pattern = '%Y-%m-%d %H:%M:%S'
        self.obs_window = 45
        self.curr_window_last_n_days = [5, 10, 15, 30, 45]
        self.bf_window_last_n_days = [30, 60, 90, 180, 360]

        self._group_attr_fea()
        self._group_sequence_fea()

    def _data_list_reader(self, data_list):
        for data_file in data_list:
            with open(data_file, 'r') as f:
                data = json.load(f)
                for line in data['data']:
                    yield line

    def _group_attr_fea(self):
        self.attr_fea_group = {}
        for fea_name, fea_info in self.meta['Data']['AttributeFeature']:
            if fea_info['include']:
                fea_type = fea_info['type']
                if fea_type not in self.attr_fea_group.keys():
                    self.attr_fea_group[fea_type] = [fea_name]
                else:
                    self.attr_fea_group[fea_type].append(fea_name)

    def _group_sequence_fea(self):
        self.seq_fea_group = {}
        for fea_name, fea_info in self.meta['Data']['SequenceFeature']:
            if fea_info['include']:
                fea_type = fea_info['type']
                if fea_type not in self.seq_fea_group.keys():
                    self.seq_fea_group[fea_type] = [fea_name]
                else:
                    self.seq_fea_group[fea_type].append(fea_name)

    def _get_cat_values_list(self, filelist):
        self.cat_fea_values_list = {fea_name: set() for fea_name in self.seq_fea_group['Categorical']}

        for filein in filelist:
            with open(filein, 'r') as f:
                for line in f:
                    d = json.loads(line)
                    for fea_name in self.seq_fea_group['Categorical']:
                        self.cat_fea_values_list[fea_name].update(d[fea_name])

    def _calc_cat_seq_fea(self, ):
        pass

    def _calc_num_seq_fea(self, ):
        pass

    def extend_output_data(self, input_files, output_file):
        obs_window = 45
        curr_window_lastNdays_list = [5, 10, 15, 30, 45]
        bf_window_lastNdays_list = [30, 60, 90, 180, 360]

        num_line = 0

        with open(output_file, 'w') as f:
            for line in self.extend_data(input_files, obs_window,
                                         curr_window_lastNdays_list, bf_window_lastNdays_list):
                f.write(json.dumps(line) + '\n')
                num_line += 1
                if num_line % 1000 == 0:
                    print('Write %s lines' % num_line)

        return num_line


    def extend_data(self, data_files, obs_window, curr_window_lastNdays_list, bf_window_lastNdays_list):
        data_iter = self.data_list_reader(data_files)
        for line in data_iter:
            yield self.extend_line(line, obs_window,
                              curr_window_lastNdays_list, bf_window_lastNdays_list)


    def extend_line(self, data, obs_window, curr_window_lastNdays_list, bf_window_lastNdays_list):

        tmp_sample = {}

        start_date = datetime.strptime(data['START_DATE'], self.datetime_pattern)
        end_date = datetime.strptime(data['END_DATE'], self.datetime_pattern)
        cutoff_date = datetime.strptime(data['CUTOFF_DATE'], self.datetime_pattern)

        window_end_date = datetime(end_date.year, cutoff_date.month, cutoff_date.day) + relativedelta(days=1, seconds=-1)
        window_start_date = window_end_date + relativedelta(days=-obs_window, seconds=1)

        bf_window_lastNdays_dates = []
        curr_window_lastNdays_dates = []

        # find the dates aligned to the number of lastNdays
        for nDays in curr_window_lastNdays_list:
            curr_window_lastNdays_dates.append(window_end_date - relativedelta(days=nDays, seconds=-1))

        for nDays in bf_window_lastNdays_list:
            bf_window_lastNdays_dates.append(window_start_date - relativedelta(days=nDays))

        # Split into Previous and Current
        # Find the index in the array aligned to the dates of lastNdays
        list_len = len(data['IA_TIMESTAMP'])
        curr_lastN_index = [list_len] * len(curr_window_lastNdays_dates)
        bf_lastN_index = [list_len] * len(bf_window_lastNdays_dates)
        start_index = list_len
        cutoff_index = list_len

        for index, ia_timestamp in enumerate(data['IA_TIMESTAMP']):
            ia_timestamp = datetime.strptime(ia_timestamp, datetime_pattern)

            for j, nCurrDate in enumerate(curr_window_lastNdays_dates):
                if nCurrDate <= ia_timestamp and curr_lastN_index[j] == list_len:
                    curr_lastN_index[j] = index
            for k, nBfDate in enumerate(bf_window_lastNdays_dates):
                if nBfDate <= ia_timestamp and bf_lastN_index[k] == list_len:
                    bf_lastN_index[k] = index

            if window_start_date <= ia_timestamp and start_index == list_len:
                start_index = index
            if ia_timestamp > window_end_date and cutoff_index == list_len:
                cutoff_index = index
                break

        ### Copy Unchanged Features:
        for feature in noseq_fea_list:
            tmp_sample[feature] = data[feature]


        ### If StartDate, EndDate or CutoffDate in processed samples
        if InfConst.Meta_From_MiddleWare["data"]["START_DATE"]["include"]:
            tmp_sample['START_DATE'] = datetime.strftime(start_date, datetime_pattern)
        if InfConst.Meta_From_MiddleWare["data"]["END_DATE"]["include"]:
            tmp_sample['END_DATE'] = datetime.strftime(end_date, datetime_pattern)
        if InfConst.Meta_From_MiddleWare["data"]["CUTOFF_DATE"]["include"]:
            tmp_sample['CUTOFF_DATE'] = datetime.strftime(window_end_date, datetime_pattern)

        ### Copy Current Month IA Relative Sequences:
        for feature in seq_fea_list:
            tmp_sample[feature] = data[feature][start_index:cutoff_index]


        ###====================================  New Feature Calculation  =======================================###
        ### Feautre: N-th month
        tmp_sample['Nth_Month'] = (window_end_date.year - start_date.year) * 12 + \
                                  (window_end_date.month - start_date.month) + 1

        ### Feature: N month to end
        tmp_sample['Months_to_end'] = (end_date.year - window_end_date.year) * 12 + \
                                      (end_date.month - window_end_date.month)

        ### Feature: Current Calendar Month
        tmp_sample['Curr_Calendar_Month'] = window_end_date.month

        ### Feature: Aggregate Info Current - Num_IA_lastNdays
        for nDays, date_index in zip(curr_window_lastNdays_list, curr_lastN_index):
            tmp_sample['Agg_Curr_Num_IA_last' + str(nDays) + 'days'] = len(ia_types[date_index:cutoff_index])

        ### Feature: Aggregate Info Current - Num IA lastNdays of each IA Type
        for nDays, date_index in zip(curr_window_lastNdays_list, curr_lastN_index):
            count = Counter(ia_types[date_index:cutoff_index])
            for ia_type in ia_vocab:
                ia_type_nospace = ia_type.replace(' ', '_')
                tmp_sample['Agg_Curr_Num_IA_' + ia_type_nospace + '_last' + str(nDays) + 'days'] = count.get(ia_type, 0)

        ### Feature: Aggregate Info Before - Num_IA_lastNdays
        for nDays, date_index in zip(bf_window_lastNdays_list, bf_lastN_index):
            tmp_sample['Agg_BF_Num_IA_last' + str(nDays) + 'days'] = \
                len(ia_types[date_index:start_index])

        ### Feature: Aggregate Info Before - Num IA lastNdays of each IA Type
        for nDays, date_index in zip(bf_window_lastNdays_list, bf_lastN_index):
            count = Counter(ia_types[date_index:start_index])
            for ia_type in ia_vocab:
                ia_type_nospace = ia_type.replace(' ', '_')
                tmp_sample['Agg_BF_Num_IA_' + ia_type_nospace + '_last' + str(nDays) + 'days'] = count.get(ia_type, 0)

        ###=================================  New Feature Calculation End =======================================###

        return tmp_sample

if __name__ == '__main__':
    data_list = ['sampledata50.json']

    n_inf_extend = extend_output_data(data_list, 'extend_inf_sharks_100.data')

    print('Extended data: %s' % (n_inf_extend))

    print('Data processing done')