import logging
import pickle


class ContainerDataset:
    data = {
        "test": {},
        "eval": {},
    }

    def __getitem__(self, index):
        partition, nodes, idx = index
        if not (partition in self.data) or not (nodes in self.data[partition]):
            logging.warning(
                f"Data sepecified by ({partition}, {nodes}) was not initialized. Attepmting to load it for the first time."
            )
            data = load(file_catalog[partition][nodes])
            self.data[partition][nodes] = [tuple(instance) for instance in data]

        return self.data[partition][nodes][idx]

