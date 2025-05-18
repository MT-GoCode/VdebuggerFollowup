"""
This is the script that contains the backend code. No need to look at this to implement new functionality
Functions that run separate processes. These processes run on GPUs, and are queried by processes running only CPUs
"""

import dill
import inspect
import queue
import torch
import torch.multiprocessing as mp
from rich.console import Console
from time import time
from typing import Callable, Union
from functools import partial

from configs import config

from context import context

console = Console(highlight=False)

stage_execution_config = config.stage_execution

# this should only be run in main process
def initialize_models_and_processes(
        manager=None, model_process_process_directory=None, model_processes_input_queues=None):
    # These arguments only specified in multiprocessing context

    # No need to initialize the models inside each process
    import vision_models
    # Create a list of all the defined models
    list_models = [m[1] for m in inspect.getmembers(vision_models, inspect.isclass)
                   if issubclass(m[1], vision_models.BaseModel) and m[1] != vision_models.BaseModel]
    # Sort by attribute "load_order"
    list_models.sort(key=lambda x: x.load_order)
    

    # setting up functions to later set up consumer functions  
    if stage_execution_config.multiprocessing.use:

        def setup_process_process(uninstantiated_model_class, model_init_args, process_name_):
            """
            that naming is intentional. This is a certain model process's separate OS process
            """
            input_queue = manager.Queue()  # For transfer of data from producer to consumer
            model_processes_input_queues[process_name_] = input_queue

            fn_process = create_persistent_function(uninstantiated_model_class, model_init_args, process_name_)

            # Otherwise, it is not possible to pickle the nested _persistent_function -> 🐒 patch the serializer 🍌🍌🍌
            aux = mp.reducer.dump
            mp.reducer.dump = dill.dump
            consumer = mp.Process(target=fn_process, kwargs={"input_queue": input_queue})
            consumer.start()
            mp.reducer.dump = aux
            
            return consumer

        def make_fn_multiprocessing(model_instance, process_name):
            """
            return a function that is a simple wrapper for the model process. If the model expects a batch, we want to take advantage of that to batch many inputs over time then send as a whole. But that will be handled by the persistent make_fn_process, which, for to_batch models, will continually batch input, send, batch input, send. Otherwise, make_fn_process will jsut keep spamming this _function with new inputs.
            """

            def _function(**kwargs):
                return model_instance.forward(**kwargs)
            return _function

        def create_persistent_function(uninstantiated_model_class, model_init_args, process_name):

            if uninstantiated_model_class.to_batch:
                seconds_collect_data = uninstantiated_model_class.seconds_collect_data  # Window of seconds to group inputs
                max_batch_size = uninstantiated_model_class.max_batch_size

                def _persistent_function(input_queue):
                    # KEY: we instantiate the model in a child process. doing it in parent would cause duplicate instantiation
                    model_instance = uninstantiated_model_class(**model_init_args)

                    fn = make_fn_multiprocessing(model_instance, process_name)

                    to_end = False
                    while True:
                        start_time = time()
                        time_left = seconds_collect_data
                        batch_input = []
                        batch_output_queue_to_use = []
                        while time_left > 0 and len(batch_input) < max_batch_size:
                            try:
                                received = input_queue.get(timeout=time_left)
                                if received is None:
                                    to_end = True
                                    break
                                else:
                                    batch_input.append(received[0])
                                    batch_output_queue_to_use.append(received[1])
                            except queue.Empty:  # Time-out expired
                                break  # Break inner loop (or do nothing, would break anyway because time_left < 0)
                            time_left = seconds_collect_data - (time() - start_time)
                        if len(batch_input) > 0:
                            batch_kwargs = collate(batch_input)
                            # print(f"RUNNING {model_instance} with inputs {batch_kwargs}")
                            outs = fn(**batch_kwargs)
                            try:
                                for out, qu in zip(outs, batch_output_queue_to_use):
                                    qu.put(out)
                            except Exception as e:
                                for qu in batch_output_queue_to_use:
                                    qu.put("ERROR OCCURRED IN BATCH FUNCTION")
                        if to_end:
                            print(f'{process_name} model exiting')
                            break
            else:
                def _persistent_function(input_queue):
                    model_instance = uninstantiated_model_class(**model_init_args)

                    fn = make_fn_multiprocessing(model_instance, process_name)
                    while True:
                        received = input_queue.get()
                        if received is None:
                            print(f'{process_name} exiting')
                            return
                        kwargs, queue_out = received
                        out = fn(**kwargs)
                        queue_out.put(out)

            return _persistent_function

    else: # single-processing
        def make_fn_singleprocessing(model_instance, process_name):
            """
            return a function that is a simple wrapper for the model process, taking args in **kwargs
            inputs & outputs will only ever be single items. But if model expects a batch, we will need to add a single dimension to the outisde of the input and strip it from the output.
            """

            def _function(**kwargs):
                if model_instance.to_batch:
                    # Batchify the input. Model expects a batch. And later un-batchify the output.
                    kwargs = {k: [v] for k, v in kwargs.items()}

                    # use no args. and default args must be handled in forward logic

                # used to be try/except here... but i want all errors to propogate to the run_program handler

                out = model_instance.forward(**kwargs)

                if model_instance.to_batch:
                    out = out[0]
                return out

            return _function

    # MODEL & PROCESS INIT LOOP
    for model_class_ in list_models:
        if model_class_.name in stage_execution_config.models and stage_execution_config.models[model_class_.name].load:
            
            if model_class_.requires_gpu: 
                    
                for process_name_ in model_class_.list_processes():
                    intended_gpu = stage_execution_config.models[model_class_.name].processes[process_name_].gpu

                    # PROCESS ASSIGNMENT
                    print(f"{process_name_} maps to {model_class_}" )

                    if not stage_execution_config.multiprocessing.use:
                        context.model_process_map[process_name_] = make_fn_singleprocessing(model_class_(gpu_number=intended_gpu), process_name_)
                    else:
                        print(f"Model {model_class_} is going to {intended_gpu}")
                        model_process_process_directory[process_name_] = setup_process_process(model_class_, {"gpu_number": intended_gpu}, process_name_)
                
            else: # no GPU needed
                # PROCESS ASSIGNMENT
                for process_name_ in model_class_.list_processes():
                    print(f"{process_name_} maps to {model_class_}" )

                    if not stage_execution_config.multiprocessing.use:
                        context.model_process_map[process_name_] = make_fn_singleprocessing(model_class_(gpu_number=-1), process_name_)
                    else:
                        model_process_process_directory[process_name_] = setup_process_process(model_class_, {"gpu_number": intended_gpu},process_name_)

def finish_all_consumers(model_processes_input_queues, model_process_process_directory):
    # Wait for consumers to finish
    for q_in in model_processes_input_queues.values():
        q_in.put(None)
    for cons in model_process_process_directory.values():
        cons.join()

def forward(queues=None, **kwargs):
    """
    Sends data to consumer (calls their "forward" method), and returns the result
    """
    process_name = kwargs['process_name']
    if not config.stage_execution.multiprocessing.use:
        # process_name is consumed here. in _function we need to add it back to kwargs
        error_msg = f'No process called {process_name}. ' \
                    'The available processes are: {}. Make sure to activate its parent model in the config files'
        if process_name not in context.model_process_map.keys():
            raise KeyError(error_msg.format(list(context.model_process_map.keys())))

        # do NOT except a key error here. key errors can happen internally    
        return context.model_process_map[process_name](**kwargs)
    
    else:
        model_processes_input_queues, worker_output_queue = queues
        model_processes_input_queues[process_name].put([kwargs, worker_output_queue])
        return worker_output_queue.get()

def collate(batch_inputs):
    """
    Combine a list of kwargs dictionaries into a single dictionary for batch processing.
    Each key maps to a list of values from the input dictionaries.
    Assumes fn takes only kwargs and has no default arguments.
    
    Args:
        batch_inputs (list[dict]): List of kwargs dictionaries.
        fn (Callable): The function to inspect (model_instance.forward).
    
    Returns:
        dict: A dictionary where each key maps to a list of values.
    
    Example:
        batch_inputs = [{'image': img1}, {'image': img2}]
        Returns: {'image': [img1, img2]}
    """
    if not batch_inputs:
        return {}
    
    # Initialize output dictionary with keys from the first input
    kwargs_output = {k: [] for k in batch_inputs[0].keys()}
    
    # Collect values for each key across all inputs
    for kwargs in batch_inputs:
        if kwargs.keys() != kwargs_output.keys():
            raise ValueError(f"Inconsistent keys in batch inputs: expected {kwargs_output.keys()}, got {kwargs.keys()}")
        for key in kwargs:
            kwargs_output[key].append(kwargs[key])
    
    return kwargs_output


"""
Deduplication code - obsolete


                if stage_execution_config.models[model_class_.name].deduplicate_same_gpu:

                    already_initialized_model_instances = {} # GPU -> model instance

                    for process_name_ in model_class_.list_processes():
                        
                        intended_gpu = stage_execution_config.models[model_class_.name].processes[process_name_].gpu

                        if intended_gpu not in already_initialized_model_instances:
                            already_initialized_model_instances[intended_gpu] = model_class_(gpu_number=intended_gpu)

                        # PROCESS ASSIGNMENT
                        print(f"{process_name_} maps to {model_class_}" )

                        if not stage_execution_config.multiprocessing.use:
                            consumers[process_name_] = make_fn_singleprocessing(already_initialized_model_instances[intended_gpu], process_name_)
                        else:
                            consumers[process_name_] = setup_process_process(already_initialized_model_instances[intended_gpu], process_name_)

"""