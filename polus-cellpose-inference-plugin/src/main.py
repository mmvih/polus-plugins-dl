import argparse
import logging
from pathlib import Path
import numpy as np
import zarr
from bfio import BioReader
import cellpose
import cellpose.models as models
import torch
from concurrent.futures import ThreadPoolExecutor, wait
import typing

""" Silence Cellpose

Cellpose does not give you an option to set log level. We have to manually set
the log levels to prevent Cellpose from spamming the command line.

"""
logging.getLogger('cellpose.core').setLevel(logging.CRITICAL)
logging.getLogger('cellpose.models').setLevel(logging.CRITICAL)
logging.getLogger('cellpose.io').setLevel(logging.CRITICAL)

TILE_SIZE = 1024
TILE_OVERLAP = 256

# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

# Conversion factors to nm, these are based off of supported Bioformats length units
UNITS = {'m':  10**-6,
         'cm': 10**-4,
         'mm': 10**-3,
         'µm': 10,
         'nm': 10**3}

def segment_thread(input_path: Path,
                   zfile: Path,
                   position: typing.Tuple[int,int,int],
                   model_cp: models.CellposeModel,
                   model_sz: models.CellposeModel,
                   diameter: int):
    """ Run cellpose on an image tile

    Args:
        input_path (Path): Path to input file
        zfile (Path): Path to the zarr file
        position (typing.Tuple[int,int,int]): x,y,z coordinates of tile
        model_cp (models.CellposeModel): Cellpose segmentation model
        model_sz (models.CellposeModel): Cellpose size model
        diameter (int): Diameter of objects in image

    Returns:
        Returns True when completed
    """    

    x,y,z = position
    
    root = zarr.open(str(zfile))
    
    with BioReader(input_path) as br:
        x_min = max([0, x - TILE_OVERLAP])
        x_max = min([br.X, x + TILE_SIZE + TILE_OVERLAP])
        y_min = max([0, y - TILE_OVERLAP])
        y_max = min([br.Y, y + TILE_SIZE + TILE_OVERLAP])
        tile_img = br[y_min:y_max, x_min:x_max, z:z + 1, 0, 0].squeeze()
        img = cellpose.transforms.convert_image(tile_img,[0,0],False,True,False)
        
        if diameter in [None,0]:
            diameter,_ = model_sz.eval(tile_img,
                                       channels=[0,0])
        
        rescale = model_cp.diam_mean / np.array(diameter)
            
        dP,prob = model_cp._run_cp(img[np.newaxis,...],
                                   rescale=rescale,
                                   resample=True,
                                   compute_masks=False)[2:4]
        
        x_overlap = x - x_min
        x_min = x
        x_max = min([br.X, x + TILE_SIZE])
        y_overlap = y - y_min
        y_min = y
        y_max = min([br.Y, y + TILE_SIZE])
        prob = prob[...,np.newaxis,np.newaxis,np.newaxis]
        dP = dP[...,np.newaxis,np.newaxis]
        
        dP = dP.transpose((1, 2, 3, 0, 4))
        root[input_path.name]['vector'][y_min:y_max, x_min:x_max, z:z + 1, 0:1, 0:1] = prob[y_overlap:y_max - y_min + y_overlap,
                                                                                            x_overlap:x_max - x_min + x_overlap,
                                                                                            ...]
        root[input_path.name]['vector'][y_min:y_max, x_min:x_max, z:z + 1, 1:3, 0:1] = dP[y_overlap:y_max - y_min + y_overlap,
                                                                                          x_overlap:x_max - x_min + x_overlap,
                                                                                          ...]
        
    return True

def main(inpDir: Path,
         pretrained_model: str,
         diameterMode: str,
         diameter: float,
         outDir: Path):
    
    """ Sanity check on diameter mode """
    assert diameterMode in ['Manual','PixelSize','FirstImage','EveryImage']

    # Surround with try/finally for proper error catching
    logger.info('Initializing ...')
    # Get all file names in inpDir image collection
    inpDir_files = [f for f in Path(inpDir).iterdir() if f.is_file()]
    
    # Use a gpu if it's available
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    logger.info(f'Running on: {dev}')

    # Get the pretrained model
    if pretrained_model in ['cyto','cyto2','nuclei']:
        model = models.Cellpose(model_type=pretrained_model,
                                gpu=use_gpu,
                                device=dev)
        model_sz = model.sz
        model_cp = model.cp
    else:
        model_cp = models.CellposeModel(pretrained_model=pretrained_model)
        model_sz = None
    model_cp.batch_size = 8

    # Error checking for diameterMode==Manual
    if diameterMode=='Manual' and not diameter:
        logger.warning('Manual diameter selection specified, but manual diameter is 0 or None. Using Cellpose model defaults.')
        
        try:
            diameter = model.diam_mean
        except Exception as err:
            logger.error('Manual diameter selected, but not diameter supplied and model has no default diameter.')
    elif diameterMode=='EveryImage':
        diameter = 0

    root = zarr.group(store=str(Path(outDir).joinpath('flow.zarr')))
    
    executor = ThreadPoolExecutor(1)
    
    if diameterMode == 'FirstImage':
        
        with BioReader(inpDir_files[0]) as br:
            
            x_min = max([br.X//2 - 1024,0])
            x_max = min([x_min+2048,br.X])
            y_min = max([br.Y//2 - 1024,0])
            y_max = min([y_min+2048,br.Y])
            
            tile_img = br[y_min:y_max,x_min:x_max,...]
            
            diameter,_ = model_sz.eval(tile_img,
                                       channels=[0,0])
            
    
    # Loop through files in inpDir image collection and process
    for f in inpDir_files:
        br = BioReader(Path(inpDir).joinpath(f).absolute())
        logger.info(f'Processing image {f}, diameter = {diameter}')
        
        # If diameterMode == PixelSize, estimate diameter from pixel size
        if diameterMode=='PixelSize':
            x_size = br.ps_x
            y_size = br.ps_y
            
            if x_size is None and y_size is None:
                raise ValueError('No pixel size stored in the metadata. Try using a different diameterMode other than PixelSize.')
            
            if x_size is None:
                x_size = y_size
            
            if y_size is None:
                y_size = x_size
                
            # Estimate diameter based off model diam_mean and pixel size
            diameter = 1.5 / (x_size[0] * UNITS[x_size[1]] + y_size[0] * UNITS[y_size[1]])
            
            try:
                diameter *= model.diam_mean
            except Exception as err:
                logger.error('Manual diameter selected, but not diameter supplied and model has no default diameter.')

        # Saving pixel locations and probablity as zarr datasets and metadata as string
        cluster = root.create_group(f.name)
        init_cluster_1 = cluster.create_dataset('vector', shape=(br.Y, br.X, br.Z, 3, 1),
                                                chunks=(TILE_SIZE, TILE_SIZE, 1, 3, 1),
                                                dtype=np.float32)
        
        cluster.attrs['metadata'] = str(br.metadata)
        
        # Iterating through z slices
        processes = []
        for z in range(br.Z):
            # Iterating based on tile size
            for x in range(0, br.X, TILE_SIZE):
                for y in range(0, br.Y, TILE_SIZE):
                    
                    position = (x,y,z)
                    processes.append(executor.submit(segment_thread,
                                                     f,
                                                     Path(outDir).joinpath('flow.zarr'),
                                                     position,
                                                     model_cp,
                                                     model_sz,
                                                     diameter))
                    
        # Close the image
        br.close()
                    
    done, not_done = wait(processes, 0)

    logger.info(f'Percent complete: {100 * len(done) / len(processes):6.3f}%')

    while len(not_done) > 0:
        for r in done:
            r.result()
        done, not_done = wait(processes, 15)
        logger.info(f'Percent complete: {100 * len(done) / len(processes):6.3f}%')
        
    executor.shutdown()

if __name__ == '__main__':
    
    ''' Argument parsing '''
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Cellpose parameters')

    # Input arguments
    parser.add_argument('--diameterMode', dest='diameterMode', type=str, default='FirstImage',
                        help='Method of setting diameter. Must be one of PixelSize, Manual, FirstImage, EveryImage',
                        required=False)
    parser.add_argument('--diameter', dest='diameter', type=float, default=0.,
                        help='Cell diameter, if 0 cellpose will estimate for each image',
                        required=False)
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--pretrainedModel', dest='pretrainedModel', type=str,
                        help='Model to use', required=False)

    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    logger.info('diameter = {}'.format(args.diameter))
    diameter = args.diameter
    logger.info('diameterMode = {}'.format(args.diameterMode))
    diameterMode = args.diameterMode
    inpDir = args.inpDir
    if (Path.is_dir(Path(args.inpDir).joinpath('images'))):
        # switch to images folder if present
        inpDir = str(Path(args.inpDir).joinpath('images').absolute())
    logger.info('inpDir = {}'.format(inpDir))
    pretrained_model = args.pretrainedModel
    logger.info('pretrained model = {}'.format(pretrained_model))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    
    main(inpDir,
         pretrained_model,
         diameterMode,
         diameter,
         outDir)
