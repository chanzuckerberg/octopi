from octopi.extract.localize import process_localization
import octopi.processing.evaluate as octopi_evaluate
from monai.metrics import ConfusionMatrixMetric
from octopi.models import common as builder
from octopi.pytorch import segmentation
from octopi.datasets import generators
from octopi.pytorch import trainer 
import multiprocess as mp
import copick, torch, os
from octopi.utils import io
from tqdm import tqdm
    
def train(config, target_info, tomo_algorithm, voxel_size, loss_function,
          model_config = None, model_weights = None, trainRunIDs = None, validateRunIDs = None,
          model_save_path = 'results', best_metric = 'fBeta2', num_epochs = 1000, use_ema = True):
    """
    Train a UNet Model for Segmentation

    Args:
        config (str): Path to the Copick Config File
        target_info (list): List containing the target user ID, target session ID, and target algorithm
        tomo_algorithm (str): The tomographic algorithm to use for segmentation
        voxel_size (float): The voxel size of the data
        loss_function (str): The loss function to use for training
        model_config (dict): The model configuration
        model_weights (str): The path to the model weights
        trainRunIDs (list): The list of run IDs to use for training
        validateRunIDs (list): The list of run IDs to use for validation
        model_save_path (str): The path to save the model
        best_metric (str): The metric to use for early stopping
        num_epochs (int): The number of epochs to train for
    """

    # If No Model Configuration is Provided, Use the Default Configuration
    if model_config is None:
        root = copick.from_file(config)
        model_config = {
            'architecture': 'Unet',
            'num_classes': root.pickable_objects[-1].label + 1,
            'dim_in': 80,
            'strides': [2, 2, 1],
            'channels': [48, 64, 80, 80],
            'dropout': 0.0, 'num_res_units': 1,
        }
        print('No Model Configuration Provided, Using Default Configuration')
        print(model_config)
    
    data_generator = generators.TrainLoaderManager(
            config, 
            target_info[0], 
            target_session_id = target_info[2],
            target_user_id = target_info[1],
            tomo_algorithm = tomo_algorithm,
            voxel_size = voxel_size,
            Nclasses = model_config['num_classes'],
            tomo_batch_size = 15 ) 

    data_generator.get_data_splits(
        trainRunIDs = trainRunIDs,
        validateRunIDs = validateRunIDs,
        train_ratio = 0.9, val_ratio = 0.1, test_ratio = 0.0,
        create_test_dataset = False)

    # Get the reload frequency
    data_generator.get_reload_frequency(num_epochs)

    # Monai Functions
    metrics_function = ConfusionMatrixMetric(include_background=False, metric_name=["recall",'precision','f1 score'], reduction="none")
    
    # Build the Model
    model_builder = builder.get_model(model_config['architecture'])
    model = model_builder.build_model(model_config)
    
    # Load the Model Weights if Provided 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_weights: 
        state_dict = torch.load(model_weights, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)     
    model.to(device) 

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)

    # Create UNet-Trainer
    model_trainer = trainer.ModelTrainer(
        model, device, loss_function, metrics_function, optimizer,
        use_ema = use_ema
    )

    results = model_trainer.train(
        data_generator, model_save_path, max_epochs=num_epochs,
        crop_size=model_config['dim_in'], my_num_samples=16,
        val_interval=10, best_metric=best_metric, verbose=True
    )
    
    # Save parameters and results
    parameters_save_name = os.path.join(model_save_path, "model_config.yaml")
    io.save_parameters_to_yaml(model_builder, model_trainer, data_generator, parameters_save_name)

    # TODO: Write Results to Zarr or Another File Format? 
    results_save_name = os.path.join(model_save_path, "results.json")
    io.save_results_to_json(results, results_save_name)

def segment(config, tomo_algorithm, voxel_size, model_weights, model_config, 
            seg_info = ['predict', 'octopi', '1'], use_tta = False, run_ids = None):
    """
    Segment a Dataset using a Trained Model or Ensemble of Models

    Args:
        config (str): Path to the Copick Config File
        tomo_algorithm (str): The tomographic algorithm to use for segmentation
        voxel_size (float): The voxel size of the data
        model_weights (str, list): The path to the model weights or a list of paths to the model weights
        model_config (str, list): The model configuration or a list of model configurations
        seg_info (list): The segmentation information
        use_tta (bool): Whether to use test time augmentation
        run_ids (list): The list of run IDs to use for segmentation
    """

    # Initialize the Predictor
    predict = segmentation.Predictor(
        config,
        model_config,
        model_weights,
        apply_tta = use_tta
    )

    # Run batch prediction
    predict.batch_predict(
        runIDs=run_ids,
        num_tomos_per_batch=15,
        tomo_algorithm=tomo_algorithm,
        voxel_spacing=voxel_size,
        segmentation_name=seg_info[0],
        segmentation_user_id=seg_info[1],
        segmentation_session_id=seg_info[2]
    )

def localize(config, voxel_size, seg_info, pick_user_id, pick_session_id, n_procs = 16,
            method = 'watershed', filter_size = 10, radius_min_scale = 0.4, radius_max_scale = 1.0,
            run_ids = None):
    """
    Extract 3D Coordinates from the Segmentation Maps

    Args:
        config (str): Path to the Copick Config File
        voxel_size (float): The voxel size of the data
        seg_info (list): The segmentation information
        pick_user_id (str): The user ID of the pick
        pick_session_id (str): The session ID of the pick
        n_procs (int): The number of processes to use for parallelization
        method (str): The method to use for localization
        filter_size (int): The filter size to use for localization
        radius_min_scale (float): The minimum radius scale to use for localization
        radius_max_scale (float): The maximum radius scale to use for localization
        run_ids (list): The list of run IDs to use for localization
    """

    # Load the Copick Config
    root = copick.from_file(config) 

    # Get objects that can be Picked
    objects = [(obj.name, obj.label, obj.radius) for obj in root.pickable_objects if obj.is_particle]    

    # Get all RunIDs
    if run_ids is None:
        run_ids = [run.name for run in root.runs]
    n_run_ids = len(run_ids)

     # Run Localization - Main Parallelization Loop
    print(f"Using {n_procs} processes to parallelize across {n_run_ids} run IDs.")
    with mp.Pool(processes=n_procs) as pool:
        with tqdm(total=n_run_ids, desc="Localization", unit="run") as pbar:
            worker_func = lambda run_id: process_localization(
                root.get_run(run_id),  
                objects, 
                seg_info,
                method, 
                voxel_size,
                filter_size,
                radius_min_scale, 
                radius_max_scale,
                pick_session_id,
                pick_user_id
            )

            for _ in pool.imap_unordered(worker_func, run_ids, chunksize=1):
                pbar.update(1)

    print('Localization Complete!')
    

def evaluate(config, 
             gt_user_id, gt_session_id,
             pred_user_id, pred_session_id,
             run_ids = None, distance_threshold = 0.5, save_path = None):
    """
    Evaluate the Localization on a Dataset

    Args:
        config (str): Path to the Copick Config File
        gt_user_id (str): The user ID of the ground truth
        gt_session_id (str): The session ID of the ground truth
        pred_user_id (str): The user ID of the predicted coordinates
        pred_session_id (str): The session ID of the predicted coordinates
        run_ids (list): The list of run IDs to use for evaluation
        distance_threshold (float): The distance threshold to use for evaluation
        save_path (str): The path to save the evaluation results
    """
             
    print('Running Evaluation on the Following Query:')
    print(f'Distance Threshold: {distance_threshold}')
    print(f'GT User ID: {gt_user_id}, GT Session ID: {gt_session_id}')
    print(f'Pred User ID: {pred_user_id}, Pred Session ID: {pred_session_id}')
    print(f'Run IDs: {run_ids}')
    
    # Load the Copick Config
    root = copick.from_file(config) 

    # For Now Lets Assume Object Names are None..
    object_names = None

    # Run Evaluation
    eval = octopi_evaluate.evaluator(
        config,
        gt_user_id,
        gt_session_id,
        pred_user_id,
        pred_session_id, 
        object_names=object_names
    )

    eval.run(
        distance_threshold_scale=distance_threshold, 
        runIDs=run_ids, save_path=save_path
    )
