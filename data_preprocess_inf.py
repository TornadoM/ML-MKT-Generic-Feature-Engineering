from collections import Counter
from datetime import datetime
from dateutil.relativedelta import relativedelta

import json
import numpy as np


class GenericDataProcessor():

    def __init__(self, meta_file):
        # Read metadata json
        with open(meta_file, 'r') as f:
            self.meta = json.load(f)
            # ia_vocab = [line.rstrip('\n') for line in f]

        self.datetime_pattern = self.meta['Mapping']['TIMESTAMP_PATTERN']
        self.obs_window = 45
        self.curr_window_last_n_days = [5, 10, 15, 30, 45]
        self.bf_window_last_n_days = [30, 60, 90, 180, 360]

        self.id_counter = {}

        self._group_attr_fea()
        self._group_seq_fea()

        self._sort_sample_by_start_date()

    def _data_list_reader(self, data_list):
        for data_file in data_list:
            with open(data_file, 'r') as f:
                data = json.load(f)
                for line in data['data']:
                    yield line

    def _sort_sample_by_start_date(self):
        data_list_sorted = []
        for data_file in data_list:
            data_list_sorted.append(data_file + '_sorted')
            with open(data_file, 'r') as f:
                lines = f.readlines()
                sorted_lines = sorted(lines, key=lambda k: k['START_DATE'])
            with open(data_file+'_sorted', 'w') as f_sorted:
                for line in sorted_lines:
                    f_sorted.wirte(json.dumps(line) + '\n')
        return data_list_sorted

    def _group_attr_fea(self):
        self.attr_fea_group = {}
        for fea_name, fea_info in self.meta['Data']['AttributeFeature'].items():
            if fea_info['include']:
                fea_type = fea_info['type']
                if fea_type not in self.attr_fea_group.keys():
                    self.attr_fea_group[fea_type] = [fea_name]
                else:
                    self.attr_fea_group[fea_type].append(fea_name)

    def _group_seq_fea(self):
        self.seq_fea_group = {}
        for fea_name, fea_info in self.meta['Data']['SequenceFeature'].items():
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

    def _gen_cat_seq_fea(self, data, window_start_index, window_end_index, curr_last_n_index, bf_last_n_index):
        cat_seq_fea_dict = {}
        for fea_name in self.seq_fea_group['Categorical']:
            # Get the sequence in window
            cat_seq_fea_dict[fea_name] = data[fea_name][window_start_index:window_end_index]

            for nDays, date_index in zip(self.curr_window_last_n_days, curr_last_n_index):
                # Agg_Curr_Num_[Field_Name]_last[N]days
                cat_seq_fea_dict['Agg_Curr_Num_' + fea_name + '_last' + str(nDays) + 'days'] = \
                    len(data[fea_name][date_index:window_end_index])
                # Agg_Curr_Num_[Field_Name]_[Value]_last[N]days
                count = Counter(data[fea_name][date_index:window_end_index])
                for fea_value in self.cat_fea_values_list[fea_name]:
                    fea_value_nospace = fea_value.replace(' ', '_')
                    cat_seq_fea_dict['Agg_Curr_Num_' + fea_name + '_' +
                                     fea_value_nospace + '_last' + str(nDays) + 'days'] = count.get(fea_value, 0)

            for nDays, date_index in zip(self.bf_window_last_n_days, bf_last_n_index):
                # Agg_BF_Num_[Field_Name]_last[N]days
                cat_seq_fea_dict['Agg_BF_Num_' + fea_name + '_last' + str(nDays) + 'days'] = \
                    len(data[fea_name][date_index:window_start_index])
                # Agg_BF_Num_[Field_Name]_[Value]_last[N]days
                count = Counter(data[fea_name][date_index:window_end_index])
                for fea_value in self.cat_fea_values_list[fea_name]:
                    fea_value_nospace = fea_value.replace(' ', '_')
                    cat_seq_fea_dict['Agg_BF_Num_' + fea_name + '_' +
                                     fea_value_nospace + '_last' + str(nDays) + 'days'] = count.get(fea_value, 0)
        return cat_seq_fea_dict


    def _gen_num_seq_fea(self, data, window_start_index, window_end_index, curr_last_n_index, bf_last_n_index):
        num_seq_fea_dict = {}
        for fea_name in self.seq_fea_group['Numeric']:
            # Get the sequence in window
            num_seq_fea_dict[fea_name] = data[fea_name][window_start_index:window_end_index]

            for nDays, date_index in zip(self.curr_window_last_n_days, curr_last_n_index):
                total_sum = sum(float(x) for x in data[fea_name][date_index:window_end_index])
                # Sum_Curr_Num_[Field_Name]_last[N]days
                num_seq_fea_dict['Sum_Curr_Num' + fea_name + '_last' + str(nDays) + 'days'] = total_sum

                # Mean_Curr_Num_[Field_Name]_last[N]days
                num_seq_fea_dict['Mean_Curr_Num' + fea_name + '_last' + str(nDays) + 'days'] = \
                    total_sum / len(data[fea_name][window_start_index:window_end_index])

                # Min_Curr_Num_[Field_Name]_last[N]days
                num_seq_fea_dict['Min_Curr_Num' + fea_name + '_last' + str(nDays) + 'days'] = \
                    min(float(x) for x in data[fea_name][date_index:window_end_index])

                # Max_Curr_Num_[Field_Name]_last[N]days
                num_seq_fea_dict['Max_Curr_Num' + fea_name + '_last' + str(nDays) + 'days'] = \
                    max(float(x) for x in data[fea_name][date_index:window_end_index])
        return num_seq_fea_dict

    def _gen_attr_fea(self, data):
        attr_fea_dict = {}
        for fea_type, fea_name_list in self.attr_fea_group.items():
            for fea_name in fea_name_list:
                attr_fea_dict[fea_name] = data[fea_name]
        return attr_fea_dict

    def extend_output_data(self, input_files, output_file):
        num_line = 0
        with open(output_file, 'w') as f:
            for line in self.extend_data(input_files):
                f.write(json.dumps(line) + '\n')
                num_line += 1
                if num_line % 1000 == 0:
                    print('Write %s lines' % num_line)
        return num_line

    def extend_data(self, data_files):
        data_iter = self._data_list_reader(data_files)
        for line in data_iter:
            extend_lines = self.extend_line(json.loads(line))
            yield from extend_lines

    def extend_line(self, data):

        output_samples = []

        start_date = datetime.strptime(data['START_DATE'], self.datetime_pattern)
        # End Date add one day because of the inaccurate Expression in the Raw Sharks Data '2017-03-31 00:00:00'
        # Should be '2017-03-31 23:59:59'
        # Specific for Sharks Data Case
        end_date = datetime.strptime(data['END_DATE'], self.datetime_pattern) \
                   + relativedelta(days=1, seconds=-1)

        window_end_date = datetime(start_date.year, start_date.month, 1) + relativedelta(months=1, seconds=-1)

        shared_sample = {}

        # Attribute Features Generated (Directly Copied)
        shared_sample.update(self._gen_attr_fea(data))

        # Feature: For the same ID, Num of samples before.
        obj_id = data[self.attr_fea_group['Key'][0]]
        if obj_id in self.id_counter.keys():
            shared_sample['Num_Periods_Before'] = self.id_counter[obj_id]['total_periods_before']
            shared_sample['Num_Periods_Positive'] = self.id_counter[obj_id]['total_periods_positive']
        else:
            shared_sample['Num_Periods_Before'] = 0
            shared_sample['Num_Periods_Positive'] = 0
            self.id_counter[obj_id]['total_periods_before'] = 1
            if data[self.attr_fea_group['Label']] == '1':
                self.id_counter[obj_id]['total_periods_positive'] = 1

        # Split by obs_window
        loop_index = 1
        while window_end_date <= end_date:
            tmp_sample = shared_sample.copy()
            window_start_date = window_end_date + relativedelta(days=-self.obs_window, seconds=1)

            curr_window_last_n_days_dates = []
            bf_window_last_n_days_dates = []

            # find the dates aligned to the number of last_n_days
            for nDays in curr_window_last_n_days_dates:
                curr_window_last_n_days_dates.append(window_end_date - relativedelta(days=nDays, seconds=-1))

            for nDays in bf_window_last_n_days_dates:
                bf_window_last_n_days_dates.append(window_start_date - relativedelta(days=nDays))

            # Split into Previous and Current
            # Find the index in the array aligned to the dates of last_n_days
            list_len = len(data[self.seq_fea_group['DateTime'][0]])
            curr_last_n_index = [list_len] * len(curr_window_last_n_days_dates)
            bf_last_n_index = [list_len] * len(bf_window_last_n_days_dates)
            window_start_index = list_len
            window_end_index = list_len
            for index, ia_timestamp in enumerate(data[self.seq_fea_group['DateTime'][0]]):
                ia_timestamp = datetime.strptime(ia_timestamp, self.datetime_pattern)

                for j, nCurrDate in enumerate(curr_window_last_n_days_dates):
                    if nCurrDate <= ia_timestamp and curr_last_n_index[j] == list_len:
                        curr_last_n_index[j] = index
                for k, nBfDate in enumerate(bf_window_last_n_days_dates):
                    if nBfDate <= ia_timestamp and bf_last_n_index[k] == list_len:
                        bf_last_n_index[k] = index

                if window_start_date <= ia_timestamp and window_start_index == list_len:
                    window_start_index = index
                if ia_timestamp > window_end_date and window_end_index == list_len:
                    window_end_index = index
                    break

            # Feature: N-th month
            tmp_sample['Nth_Month'] = loop_index

            # Feature: N month to end
            tmp_sample['Months_to_end'] = (end_date.year - window_end_date.year) * 12 + \
                                          (end_date.month - window_end_date.month)

            # Feature: Current Calendar Month
            tmp_sample['Curr_Calendar_Month'] = window_end_date.month

            # Sequence Features Generated
            tmp_sample.update(self._gen_cat_seq_fea(data, window_start_index, window_end_index,
                                                    curr_last_n_index, bf_last_n_index))

            tmp_sample.update(self._gen_num_seq_fea(data, window_start_index, window_end_index,
                                                    curr_last_n_index, bf_last_n_index))

            # Window Start and End Date generated
            # Not used for training. Just to check if window is correctly calculated
            # tmp_sample['Window_Start_Date'] = datetime.strftime(window_start_date, self.datetime_pattern)
            # tmp_sample['Window_End_Date'] = datetime.strftime(window_end_date, self.datetime_pattern)

            loop_index += 1

            output_samples.append(tmp_sample)

            # Roll the Observation Window
            if window_end_date < end_date:
                window_end_date = datetime(window_end_date.year, window_end_date.month, 1) \
                                  + relativedelta(months=2, seconds=-1)
                if window_end_date > end_date:
                    window_end_date = end_date
            else:
                break

        return output_samples

if __name__ == '__main__':
    data_list = ['sampledata50.json']

    n_inf_extend = extend_output_data(data_list, 'extend_inf_sharks_100.data')

    print('Extended data: %s' % (n_inf_extend))
    print('Data processing done')
