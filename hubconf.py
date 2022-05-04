dependencies = ['torch']
from timm.models import registry

from models import *

globals().update(registry._model_entrypoints)
