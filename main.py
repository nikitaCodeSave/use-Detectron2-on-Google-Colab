#Clone the Detectron2 для использования многих моделей исскуственного интелекта с готовыми предтренировочными данными

#https://ai.facebook.com/blog/-detectron2-a-pytorch-based-modular-object-detection-library-/

#Включи GPU среду 
#ON GPU

# Install dependencies
!python -m pip install pyyaml
import sys, os, distutils.core

# Clone the Detectron2 repository
!git clone 'https://github.com/facebookresearch/detectron2'
dist = distutils.core.run_setup("./detectron2/setup.py")
!python -m pip install {' '.join([f"'{x}'" for x in dist.install_requires])}
sys.path.insert(0, os.path.abspath('./detectron2'))

sys.path.insert(0, os.path.abspath('./detectron2/detectron2'))


# Verify installation
import torch, detectron2
!nvcc --version
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog

#from detectron2.config import get_cfg
#from detectron2.model_zoo import get_checkpoint_url, get_config_file
