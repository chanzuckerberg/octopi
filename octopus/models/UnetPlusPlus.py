from monai.networks.nets import  BasicUNetPlusPlus

class myUNetPlusPlus:
    def __init__(self, num_classes, device):
        self.device = device
        self.num_classes = num_classes
        

    def build_model(
        self, 
        features=(16, 16, 32, 64, 128, 16),
        dropout=0.2,
        upsample='deconv',
        activation='relu'
    ):

        model = BasicUNetPlusPlus(
                spatial_dims=3,
                in_channels=1,
                out_channels=n_classes,
                deep_supervision=True,
                features=features,  # Halve the features
                dropout=dropout,
                upsample=upsample,
                act=activation
            )  
        
        return model.to(self.device)
    
    def bayesian_search(self, trial):
        
        act = trial.suggest_categorical("activation", ["LeakyReLU", "PReLU", "GELU", "ELU"])   
        dropout_prob = trial.suggest_float("dropout", 0.0, 0.5)
        upsample = trial.suggest_categorical("upsample", ["deconv", "pixelshuffle", "nontrainable"])        
        model_parameters = {"activation": act,'dropout': dropout_prob, 'upsample': upsample} 

        model = self.build_model(features, dropout_prob, upsample, act)
        return model
    
    def get_model_parameters(self):
        return {
            'model_name': 'UnetPlusPlus',
            'features': self.features,
            'dropout': self.dropout,
            'upsample': self.upsample,
            'activation': self.activation
        }