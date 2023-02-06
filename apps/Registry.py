from .PiModelApp import PiModelApp
from .MomentumApp import MomentumApp
from .SingleModelApp import SingleModelApp

Registry = {
    'SingleModelApp': SingleModelApp,
    'PiModelApp': PiModelApp,
    'MomentumApp': MomentumApp
}
