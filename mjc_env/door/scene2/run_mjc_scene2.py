import mujoco 
import mujoco.viewer

import os 


# setting xml file path  
xml_path = os.path.abspath("mobile_fr3.xml")

# loading mujoco model 
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# running mujoco simulation 
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()