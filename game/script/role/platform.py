
from typing import Literal
from script.role.base_role import BaseRole, BaseRoleModel, ObjectConfig, RigidBoxModel

class PlatformModel(BaseRoleModel):
    type: Literal["platform"] = "platform"
    name: str = "New_Platform"
    object: ObjectConfig = RigidBoxModel(is_kinematic=True, shape_mass=0.0)


class Platform(BaseRole):
    role_key = "platform"
    model_cls = PlatformModel
    path = "platform_configs"
    container = "list"


    def __init__(self, configs, **kwargs):
        super().__init__(is_add_to_mesh=True, **kwargs)

        self.setup(configs=configs)


