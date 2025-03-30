import numpy as np

class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob
        
    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        '''
        self.indices_list = []
        data_length = len(data)
        for bag in range(self.num_bags):
            # Your Code Here
            indices = np.random.randint(low=0, high=data_length, size=data_length)
            self.indices_list.append(indices)
        
    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.
        
        example:
        
        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        '''
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'All bags should be of the same length!'
        assert list(map(len, self.indices_list))[0] == len(data), 'All bags should contain `len(data)` number of elements!'
        self.models_list = []
        for bag in range(self.num_bags):
            model = model_constructor()
            bag_idx = self.indices_list[bag]
            data_bag, target_bag = data[bag_idx], target[bag_idx]# Your Code Here
            self.models_list.append(model.fit(data_bag, target_bag)) # store fitted models here
        if self.oob:
            self.data = data
            self.target = target
        
    def predict(self, data):
        '''
        Get average prediction for every object from passed dataset
        '''
        # Your code here
        predict_all = []
        for model in self.models_list:
            pred = model.predict(data)
            predict_all.append(pred)
        predict_all = np.array(predict_all)
        return np.mean(predict_all, axis=0)
    
    def _get_oob_predictions_from_every_model(self):
        '''
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        '''
        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        # Your Code Here
        for model_idx, model in enumerate(self.models_list):
            bag_idx = self.indices_list[model_idx]
            all_idx = set(range(len(self.data)))
            hit_idx = set(bag_idx)
            oob_idx = list(all_idx - hit_idx)

            if len(oob_idx) == 0:
                continue

            oob_pred = model.predict(self.data[oob_idx])
            for i, idx in enumerate(oob_idx):
                list_of_predictions_lists[idx].append(oob_pred[i])
        
        self.list_of_predictions_lists = np.array(list_of_predictions_lists, dtype=object)
    
    def _get_averaged_oob_predictions(self):
        '''
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase, return None instead of prediction
        '''
        self._get_oob_predictions_from_every_model()
        avarage_pred = []
        for pred_list in self.list_of_predictions_lists:
            if len(pred_list) > 0:
                avarage_pred.append(np.mean(pred_list))
            else:
                avarage_pred.append(None)
        self.oob_predictions = np.array(avarage_pred, dtype=object)# Your Code Here
        
        
    def OOB_score(self):
        '''
        Compute mean square error for all objects, which have at least one prediction
        '''
        self._get_averaged_oob_predictions()

        valid_idx = [i for i, val in enumerate(self.oob_predictions) if val is not None]

        if len(valid_idx) == 0:
            return None

        y_true = self.target[valid_idx]
        y_pred = self.oob_predictions[valid_idx].astype(float)
        return np.mean((y_true - y_pred) ** 2)
