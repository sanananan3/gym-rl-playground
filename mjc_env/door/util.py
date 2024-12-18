import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
import pyquaternion as pyq

def rotate_quaternion(quat, axis, angle):
    """
    Rotate a quaternion by an angle around an axis
    """
    angle_rad = np.deg2rad(angle)
    axis = axis / np.linalg.norm(axis)
    
    # quat: [x, y, z, w] -> pyquaternion: [w, x, y, z]
    quat = np.roll(quat, shift=1)
    
    q = pyq.Quaternion(quat)
    q = q * pyq.Quaternion(axis=axis, angle=angle_rad)
    
    # pyquat: [w, x, y, z] -> quat: [x, y, z, w]
    rotated_q = np.roll(q.elements, shift=-1)
    
    return rotated_q

def interpolate_quaternion(q1, q2, steps):
    """
    Interpolates between two quaternions q1 and q2 over a specified number of steps.
    """
    q1 = np.asarray(q1) / np.linalg.norm(q1)
    q2 = np.asarray(q2) / np.linalg.norm(q2)

    slerp = Slerp([0, 1], R.from_quat([q1, q2]))
    times = np.linspace(0, 1, steps)
    interpolated_rotations = slerp(times)

    return interpolated_rotations.as_quat()

def get_quaternion_difference(q1, q2):
    q1 = np.asarray(q1) / np.linalg.norm(q1)
    q2 = np.asarray(q2) / np.linalg.norm(q2)
    
    dot_product = np.abs(np.dot(q1, q2))
    dot_product = np.clip(dot_product, -1, 1)
    
    angle = 2 * np.arccos(dot_product)
    return angle
    