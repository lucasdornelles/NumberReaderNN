import yaml


class DataPersistence:

    def save_data(self, file, data_list):
        # Save data_list on file
        yaml.dump(data_list, file)

    def load_data(self, file):
        # Return data from file
        return yaml.load(file, Loader=yaml.SafeLoader)
