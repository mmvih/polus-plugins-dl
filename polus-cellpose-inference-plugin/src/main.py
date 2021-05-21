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

def segment_thread(input_path: Path,
                   zfile: Path,
                   x,y,z,
                   model_cp,model_sz,
                   diameter):
    
    root = zarr.open(str(zfile))
    
    with BioReader(input_path) as br:
        x_min = max([0, x - TILE_OVERLAP])
        x_max = min([br.X, x + TILE_SIZE + TILE_OVERLAP])
        y_min = max([0, y - TILE_OVERLAP])
        y_max = min([br.Y, y + TILE_SIZE + TILE_OVERLAP])
        tile_img = br[y_min:y_max, x_min:x_max, z:z + 1, 0, 0].squeeze()
        img = cellpose.transforms.convert_image(tile_img,[0,0],False,True,False)
        if diameter is None:
            diameter,_ = model_sz.eval(tile_img,
                                       channels=[0,0])
            
        rescale = model_cp.diam_mean / np.array(diameter)
            
        prob = model_cp._run_cp(img[np.newaxis,...],
                                rescale=rescale,
                                resample=True,
                                compute_masks=False)[2]
        
        x_overlap = x - x_min
        x_min = x
        x_max = min([br.X, x + TILE_SIZE])
        y_overlap = y - y_min
        y_min = y
        y_max = min([br.Y, y + TILE_SIZE])
        prob = prob[...,np.newaxis,np.newaxis]
        
        prob = prob.transpose((1, 2, 3, 0, 4))
        root[input_path.name]['vector'][y_min:y_max, x_min:x_max, z:z + 1, 0:3, 0:1] = prob[y_overlap:y_max - y_min + y_overlap,
                                                                                            x_overlap:x_max - x_min + x_overlap,
                                                                                            ...]
        
    return True

def main(inpDir: Path,
         pretrained_model: str,
         diameter: float,
         outDir: Path):

    # Surround with try/finally for proper error catching
    logger.info('Initializing ...')
    # Get all file names in inpDir image collection
    inpDir_files = [f.name for f in Path(inpDir).iterdir() if
                    f.is_file() and "".join(f.suffixes) == '.ome.tif']
    
    # Use a gpu if it's available
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    logger.info(f'Running on: {dev}')

    if pretrained_model in ['cyto','nuclei']:
        model = models.Cellpose(model_type=pretrained_model,
                                gpu=use_gpu,
                                device=dev)
        model_sz = model.sz
        model_cp = model.cp
    else:
        model_cp = models.CellposeModel(pretrained_model=pretrained_model)
        model_sz = None
    model_cp.batch_size = 8

    if diameter == 0 and pretrained_model in ['cyto', 'nuclei']:
        diameter = None
        logger.info('Estimating diameter for each image')
    else:
        diameter = args.diameter if args.diameter else model.diam_mean
        logger.info('Using diameter %0.2f for all images' % diameter)

    root = zarr.group(store=str(Path(outDir).joinpath('flow.zarr')))
    
    executor = ThreadPoolExecutor(1)
    
    for f in inpDir_files:
        # Loop through files in inpDir image collection and process
        br = BioReader(str(Path(inpDir).joinpath(f).absolute()))
        logger.info('Processing image %s ', f)

        # Saving pixel locations and probablity  as zarr datasets and metadata as string
        cluster = root.create_group(f)
        init_cluster_1 = cluster.create_dataset('vector', shape=(br.Y, br.X, br.Z, 2, 1),
                                                chunks=(TILE_SIZE, TILE_SIZE, 1, 2, 1),
                                                dtype=np.float32)
        
        cluster.attrs['metadata'] = str(br.metadata)
        
        # Iterating through z slices
        processes = []
        for z in range(br.Z):
            # Iterating based on tile size
            for x in range(0, br.X, TILE_SIZE):
                for y in range(0, br.Y, TILE_SIZE):
                    
                    processes.append(executor.submit(segment_thread,
                                                     Path(inpDir).joinpath(f).absolute(),
                                                     str(Path(outDir).joinpath('flow.zarr')),
                                                     x,y,z,
                                                     model_cp,model_sz,
                                                     diameter))
                    
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
         diameter,
         outDir)
