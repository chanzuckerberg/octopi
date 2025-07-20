from octopi.extract import localize as octopi_localize
import octopi.processing.evaluate as octopi_evaluate
from monai.metrics import ConfusionMatrixMetric
from octopi.models import common as builder
from octopi.pytorch import segmentation
from octopi.datasets import generators
from octopi.pytorch import trainer 
from octopi.utils import io
import multiprocess as mp
import copick, torch, os
from tqdm import tqdm
    
def train(config, target_info, tomo_algorithm, voxel_size, loss_function,
          model_config, model_weights = None, trainRunIDs = None, validateRunIDs = None,
          model_save_path = 'results', best_metric = 'fBeta2', num_epochs = 1000, use_ema = True):

    
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
            seg_info = ['predict', 'octopi', '1'], use_tta = False):

    # Lets Assume We Will Always Use All RunIDs
    run_ids = None

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
            method = 'watershed', filter_size = 10, radius_min_scale = 0.4, radius_max_scale = 1.0):

    # Load the Copick Config
    root = copick.from_file(config) 

    # Get objects that can be Picked
    objects = [(obj.name, obj.label, obj.radius) for obj in root.pickable_objects if obj.is_particle]    

    # Get all RunIDs
    run_ids = [run.name for run in root.runs]
    n_run_ids = len(run_ids)

     # Run Localization - Main Parallelization Loop
    print(f"Using {n_procs} processes to parallelize across {n_run_ids} run IDs.")
    with mp.Pool(processes=n_procs) as pool:
        with tqdm(total=n_run_ids, desc="Localization", unit="run") as pbar:
            worker_func = lambda run_id: localize.processs_localization(
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