import numpy as np
import re
from scipy.spatial.transform import Rotation as R
from typing import List, Dict, Tuple, Optional
from loguru import logger


class BVHJoint:
    """Represents a single joint in the BVH hierarchy"""
    
    def __init__(self, name: str, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.offset = np.zeros(3)  # Local offset from parent
        self.channels = []  # List of channel names (e.g., ['Xposition', 'Yposition', 'Zposition', 'Xrotation', 'Yrotation', 'Zrotation'])
        self.channel_indices = []  # Indices in the motion data array
        self.is_end_site = False  # Flag to indicate if this is an end site
        
    def add_child(self, child):
        """Add a child joint"""
        self.children.append(child)
        child.parent = self
        
    def is_root(self) -> bool:
        """Check if this is the root joint"""
        return self.parent is None


class BVHParser:
    """Parser for BVH (Biovision Hierarchy) motion capture files"""
    
    def __init__(self, bvh_content: str = None, bvh_file_path: str = None):
        """
        Initialize BVH parser
        
        Args:
            bvh_content: Raw BVH file content as string
            bvh_file_path: Path to BVH file (alternative to bvh_content)
        """
        if bvh_content is not None:
            self.bvh_content = bvh_content
        elif bvh_file_path is not None:
            with open(bvh_file_path, 'r') as f:
                self.bvh_content = f.read()
        else:
            raise ValueError("Either bvh_content or bvh_file_path must be provided")
            
        # Initialize data structures
        self.root_joint = None
        self.joints = []  # List of all joints (now including end sites)
        self.joint_names = []  # List of joint names in order (now including end sites)
        self.joint_dict = {}  # Dictionary mapping joint names to joint objects
        
        # Motion data
        self.nframes = 0
        self.frame_time = 0.0
        self.fps = 0.0
        self.motion_data = None  # Raw motion data array (nframes, nchannels)
        
        # Channel information
        self.total_channels = 0
        self.channel_names = []  # All channel names in order
        
        # Parse flags
        self._parsed = False
        
    def parse(self):
        """Parse the BVH file content"""
        lines = self.bvh_content.strip().split('\n')
        lines = [line.strip() for line in lines if line.strip()]  # Remove empty lines
        
        # Find the MOTION section
        motion_start_idx = -1
        for i, line in enumerate(lines):
            if line.upper() == 'MOTION':
                motion_start_idx = i
                break
                
        if motion_start_idx == -1:
            raise ValueError("No MOTION section found in BVH file")
            
        # Parse hierarchy section
        hierarchy_lines = lines[:motion_start_idx]
        self._parse_hierarchy(hierarchy_lines)
        
        # Debug: print channel information before parsing motion
        # logger.info(f"Debug: Total channels calculated from HIERARCHY: {self.total_channels}")
        # logger.info(f"Debug: Channel names: {self.channel_names}")
        
        # Parse motion section
        motion_lines = lines[motion_start_idx + 1:]
        
        # Debug: check first motion line before parsing
        if len(motion_lines) >= 3:  # Skip frames and frame time lines
            first_motion_line = motion_lines[2].strip()
            first_motion_values = first_motion_line.split()
            # logger.info(f"Debug: First motion line has {len(first_motion_values)} values")
            # logger.info(f"Debug: First motion line: {first_motion_line}")
        
        self._parse_motion(motion_lines)
        
        self._parsed = True
        # logger.info(f"Successfully parsed BVH with {len(self.joints)} joints (including end sites) and {self.nframes} frames")
        
    def _parse_hierarchy(self, lines: List[str]):
        """Parse the HIERARCHY section"""
        if not lines or lines[0].upper() != 'HIERARCHY':
            raise ValueError("Expected HIERARCHY section")
            
        i = 1
        parent_joint = None  # Initialize parent_joint variable
        
        while i < len(lines):
            line = lines[i].strip()
            
            if line == '{':
                i += 1
                continue
                
            elif line == '}':
                return i + 1
                
            elif line.upper().startswith('ROOT'):
                # Parse root joint
                root_name = line.split()[1]
                parent_joint = BVHJoint(root_name, None)  # Root has no parent
                self.root_joint = parent_joint
                self.joints.append(parent_joint)
                self.joint_names.append(root_name)
                self.joint_dict[root_name] = parent_joint
                i += 1
                
            elif line.upper().startswith('OFFSET'):
                # Parse offset
                if parent_joint is None:
                    raise ValueError("OFFSET found before ROOT joint definition")
                parts = line.split()
                offset = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                parent_joint.offset = offset
                i += 1
                
            elif line.upper().startswith('CHANNELS'):
                # Parse channels
                if parent_joint is None:
                    raise ValueError("CHANNELS found before ROOT joint definition")
                parts = line.split()
                num_channels = int(parts[1])
                channels = parts[2:2 + num_channels]
                parent_joint.channels = channels
                
                # Assign channel indices
                parent_joint.channel_indices = list(range(self.total_channels, 
                                                        self.total_channels + num_channels))
                self.total_channels += num_channels
                self.channel_names.extend(channels)
                i += 1
                
            elif line.upper().startswith('JOINT'):
                # Parse child joint
                if parent_joint is None:
                    raise ValueError("JOINT found before ROOT joint definition")
                joint_name = line.split()[1]
                child_joint = BVHJoint(joint_name, parent_joint)
                parent_joint.add_child(child_joint)
                self.joints.append(child_joint)
                self.joint_names.append(joint_name)
                self.joint_dict[joint_name] = child_joint
                i = self._parse_joint(lines, i + 1, child_joint)
                
            elif line.upper().startswith('END SITE'):
                # Parse end site - now treated as regular joint
                if parent_joint is None:
                    raise ValueError("END SITE found before ROOT joint definition")
                i = self._parse_end_site(lines, i + 1, parent_joint)
                
            else:
                i += 1
                
        return i
        
    def _parse_joint(self, lines: List[str], start_idx: int, parent_joint: BVHJoint) -> int:
        """Parse a joint and its children recursively"""
        i = start_idx
        
        while i < len(lines):
            line = lines[i].strip()
            
            if line == '{':
                i += 1
                continue
                
            elif line == '}':
                return i + 1
                
            elif line.upper().startswith('OFFSET'):
                # Parse offset
                parts = line.split()
                offset = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                parent_joint.offset = offset
                i += 1
                
            elif line.upper().startswith('CHANNELS'):
                # Parse channels
                parts = line.split()
                num_channels = int(parts[1])
                channels = parts[2:2 + num_channels]
                parent_joint.channels = channels
                
                # Assign channel indices
                parent_joint.channel_indices = list(range(self.total_channels, 
                                                        self.total_channels + num_channels))
                self.total_channels += num_channels
                self.channel_names.extend(channels)
                i += 1
                
            elif line.upper().startswith('JOINT'):
                # Parse child joint
                joint_name = line.split()[1]
                child_joint = BVHJoint(joint_name, parent_joint)
                parent_joint.add_child(child_joint)
                self.joints.append(child_joint)
                self.joint_names.append(joint_name)
                self.joint_dict[joint_name] = child_joint
                i = self._parse_joint(lines, i + 1, child_joint)
                
            elif line.upper().startswith('END SITE'):
                # Parse end site - now treated as regular joint
                i = self._parse_end_site(lines, i + 1, parent_joint)
                
            else:
                i += 1
                
        return i
        
    def _parse_end_site(self, lines: List[str], start_idx: int, parent_joint: BVHJoint) -> int:
        """Parse an end site - now treated as a regular joint"""
        i = start_idx
        end_site_name = f"{parent_joint.name}_End"
        end_site = BVHJoint(end_site_name, parent_joint)
        end_site.is_end_site = True
        end_site.channels = []  # End sites have 0 channels
        end_site.channel_indices = []
        
        parent_joint.add_child(end_site)
        
        # Add end site to joints list
        self.joints.append(end_site)
        self.joint_names.append(end_site_name)
        self.joint_dict[end_site_name] = end_site
        
        while i < len(lines):
            line = lines[i].strip()
            
            if line == '{':
                i += 1
                continue
                
            elif line == '}':
                return i + 1
                
            elif line.upper().startswith('OFFSET'):
                # Parse end site offset
                parts = line.split()
                offset = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                end_site.offset = offset
                i += 1
                
            else:
                i += 1
                
        return i
        
    def _parse_motion(self, lines: List[str]):
        """Parse the MOTION section"""
        if len(lines) < 2:
            raise ValueError("Invalid MOTION section")
            
        # Parse frames line
        frames_line = lines[0].strip()
        if not frames_line.upper().startswith('FRAMES:'):
            raise ValueError("Expected 'Frames:' line in MOTION section")
        self.nframes = int(frames_line.split(':')[1].strip())
        
        # Parse frame time line
        frame_time_line = lines[1].strip()
        if not frame_time_line.upper().startswith('FRAME TIME:'):
            raise ValueError("Expected 'Frame Time:' line in MOTION section")
        self.frame_time = float(frame_time_line.split(':')[1].strip())
        self.fps = 1.0 / self.frame_time if self.frame_time > 0 else 30.0
        
        # Parse motion data
        motion_lines = lines[2:]
        if len(motion_lines) != self.nframes:
            raise ValueError(f"Expected {self.nframes} motion data lines, got {len(motion_lines)}")
            
        self.motion_data = np.zeros((self.nframes, self.total_channels))
        
        for frame_idx, line in enumerate(motion_lines):
            values = [float(x) for x in line.strip().split()]
            if len(values) != self.total_channels:
                raise ValueError(f"Frame {frame_idx}: expected {self.total_channels} values, got {len(values)}")
            self.motion_data[frame_idx] = values
            
    def get_joints_names(self) -> List[str]:
        """Get list of joint names (now including end sites)"""
        if not self._parsed:
            raise RuntimeError("BVH file not parsed yet. Call parse() first.")
        return self.joint_names.copy()
        
    def get_joint_by_name(self, name: str) -> Optional[BVHJoint]:
        """Get joint object by name"""
        if not self._parsed:
            raise RuntimeError("BVH file not parsed yet. Call parse() first.")
        return self.joint_dict.get(name)
        
    def get_joint_positions(self, frame_idx: int = 0) -> np.ndarray:
        """
        Get joint positions for a specific frame
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Array of shape (num_joints, 3) containing joint positions (now including end sites)
        """
        if not self._parsed:
            raise RuntimeError("BVH file not parsed yet. Call parse() first.")
            
        if frame_idx < 0 or frame_idx >= self.nframes:
            raise ValueError(f"Frame index {frame_idx} out of range [0, {self.nframes-1}]")
            
        positions = np.zeros((len(self.joints), 3))
        
        # Calculate positions recursively
        self._calculate_joint_positions(self.root_joint, frame_idx, np.eye(4), positions)
        
        return positions
        
    def _calculate_joint_positions(self, joint: BVHJoint, frame_idx: int, 
                                 parent_transform: np.ndarray, positions: np.ndarray):
        """Recursively calculate joint positions"""
        # Get joint index
        joint_idx = self.joint_names.index(joint.name) if joint.name in self.joint_names else -1
        
        # Create local transformation matrix
        local_transform = np.eye(4)
        
        # Apply offset
        local_transform[:3, 3] = joint.offset
        
        # Apply rotations and translations from motion data (only for non-end-sites)
        if joint.channels and not joint.is_end_site:
            for i, channel in enumerate(joint.channels):
                channel_idx = joint.channel_indices[i]
                value = self.motion_data[frame_idx, channel_idx]
                
                if channel.upper().endswith('POSITION'):
                    # Translation
                    if channel.upper().startswith('X'):
                        local_transform[0, 3] += value
                    elif channel.upper().startswith('Y'):
                        local_transform[1, 3] += value
                    elif channel.upper().startswith('Z'):
                        local_transform[2, 3] += value
                        
                elif channel.upper().endswith('ROTATION'):
                    # Rotation (in degrees)
                    angle_rad = np.radians(value)
                    if channel.upper().startswith('X'):
                        rot_mat = self._rotation_x(angle_rad)
                    elif channel.upper().startswith('Y'):
                        rot_mat = self._rotation_y(angle_rad)
                    elif channel.upper().startswith('Z'):
                        rot_mat = self._rotation_z(angle_rad)
                    else:
                        continue
                        
                    # Apply rotation to local transform
                    local_transform[:3, :3] = local_transform[:3, :3] @ rot_mat
                    
        # Compute global transformation
        global_transform = parent_transform @ local_transform
        
        # Store position
        if joint_idx >= 0:
            positions[joint_idx] = global_transform[:3, 3]
            
        # Recursively process children
        for child in joint.children:
            self._calculate_joint_positions(child, frame_idx, global_transform, positions)
            
    def all_frame_poses(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get positions and rotations for all frames
        
        Returns:
            Tuple of (positions, rotations) where:
            - positions: shape (nframes, num_joints, 3) (now including end sites)
            - rotations: shape (nframes, num_joints, 3, 3) - rotation matrices (end sites have identity)
        """
        if not self._parsed:
            raise RuntimeError("BVH file not parsed yet. Call parse() first.")
            
        num_joints = len(self.joints)
        positions = np.zeros((self.nframes, num_joints, 3))
        rotations = np.zeros((self.nframes, num_joints, 3, 3))
        
        for frame_idx in range(self.nframes):
            positions[frame_idx] = self.get_joint_positions(frame_idx)
            rotations[frame_idx] = self.get_joint_rotations(frame_idx)
            
        return positions, rotations
        
    def get_joint_rotations(self, frame_idx: int = 0) -> np.ndarray:
        """
        Get joint rotation matrices for a specific frame
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Array of shape (num_joints, 3, 3) containing rotation matrices (end sites have identity)
        """
        if not self._parsed:
            raise RuntimeError("BVH file not parsed yet. Call parse() first.")
            
        if frame_idx < 0 or frame_idx >= self.nframes:
            raise ValueError(f"Frame index {frame_idx} out of range [0, {self.nframes-1}]")
            
        rotations = np.zeros((len(self.joints), 3, 3))
        
        # Initialize all rotations to identity (especially important for end sites)
        for i in range(len(self.joints)):
            rotations[i] = np.eye(3)
        
        # Calculate rotations recursively
        self._calculate_joint_rotations(self.root_joint, frame_idx, np.eye(3), rotations)
        
        return rotations
        
    def _calculate_joint_rotations(self, joint: BVHJoint, frame_idx: int,
                                 parent_rotation: np.ndarray, rotations: np.ndarray):
        """Recursively calculate joint rotations"""
        # Get joint index
        joint_idx = self.joint_names.index(joint.name) if joint.name in self.joint_names else -1
        
        # Start with identity rotation
        local_rotation = np.eye(3)
        
        # Apply rotations from motion data (only for non-end-sites)
        if joint.channels and not joint.is_end_site:
            for i, channel in enumerate(joint.channels):
                if channel.upper().endswith('ROTATION'):
                    channel_idx = joint.channel_indices[i]
                    value = self.motion_data[frame_idx, channel_idx]
                    angle_rad = np.radians(value)
                    
                    if channel.upper().startswith('X'):
                        rot_mat = self._rotation_x(angle_rad)
                    elif channel.upper().startswith('Y'):
                        rot_mat = self._rotation_y(angle_rad)
                    elif channel.upper().startswith('Z'):
                        rot_mat = self._rotation_z(angle_rad)
                    else:
                        continue
                        
                    # Apply rotation
                    local_rotation = local_rotation @ rot_mat
                    
        # Compute global rotation
        global_rotation = parent_rotation @ local_rotation
        
        # Store rotation
        if joint_idx >= 0:
            rotations[joint_idx] = global_rotation
            
        # Recursively process children
        for child in joint.children:
            self._calculate_joint_rotations(child, frame_idx, global_rotation, rotations)
            
    @staticmethod
    def _rotation_x(angle: float) -> np.ndarray:
        """Create rotation matrix around X axis"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0],
                        [0, c, -s],
                        [0, s, c]])
        
    @staticmethod
    def _rotation_y(angle: float) -> np.ndarray:
        """Create rotation matrix around Y axis"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0, s],
                        [0, 1, 0],
                        [-s, 0, c]])
        
    @staticmethod
    def _rotation_z(angle: float) -> np.ndarray:
        """Create rotation matrix around Z axis"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0],
                        [s, c, 0],
                        [0, 0, 1]])
        
    def print_hierarchy(self, joint: BVHJoint = None, indent: int = 0):
        """Print the joint hierarchy (now including end sites)"""
        if not self._parsed:
            raise RuntimeError("BVH file not parsed yet. Call parse() first.")
            
        if joint is None:
            joint = self.root_joint
            
        end_site_marker = " [END SITE]" if joint.is_end_site else ""
        print("  " * indent + f"{joint.name}{end_site_marker} (offset: {joint.offset}, channels: {len(joint.channels)})")
        
        for child in joint.children:
            self.print_hierarchy(child, indent + 1)
            
    def get_motion_summary(self) -> Dict:
        """Get summary information about the motion data"""
        if not self._parsed:
            raise RuntimeError("BVH file not parsed yet. Call parse() first.")
            
        # Count end sites
        end_site_count = sum(1 for joint in self.joints if joint.is_end_site)
        regular_joint_count = len(self.joints) - end_site_count
        
        return {
            'num_joints': len(self.joints),
            'num_regular_joints': regular_joint_count,
            'num_end_sites': end_site_count,
            'num_frames': self.nframes,
            'frame_time': self.frame_time,
            'fps': self.fps,
            'duration': self.nframes * self.frame_time,
            'total_channels': self.total_channels,
            'joint_names': self.joint_names.copy(),
            'channel_names': self.channel_names.copy()
        } 

    def get_smpl_style_transforms(self, frame_idx: int = None) -> np.ndarray:
        """
        Get transformation matrices in SMPL style format
        
        Args:
            frame_idx: Specific frame index. If None, returns all frames
            
        Returns:
            If frame_idx is specified: Array of shape (num_joints, 4, 4)
            If frame_idx is None: Array of shape (nframes, num_joints, 4, 4)
            
            Each 4x4 matrix contains:
            [R11  R12  R13  tx]
            [R21  R22  R23  ty] 
            [R31  R32  R33  tz]
            [0    0    0    1 ]
            
            Where R is the 3x3 rotation matrix and t is the 3x1 translation vector
        """
        if not self._parsed:
            raise RuntimeError("BVH file not parsed yet. Call parse() first.")
            
        if frame_idx is not None:
            # Single frame
            if frame_idx < 0 or frame_idx >= self.nframes:
                raise ValueError(f"Frame index {frame_idx} out of range [0, {self.nframes-1}]")
            
            transforms = np.zeros((len(self.joints), 4, 4))
            # Initialize all transforms to identity
            for i in range(len(self.joints)):
                transforms[i] = np.eye(4)
            
            # Calculate transforms recursively
            self._calculate_joint_transforms(self.root_joint, frame_idx, np.eye(4), transforms)
            return transforms
        else:
            # All frames
            transforms = np.zeros((self.nframes, len(self.joints), 4, 4))
            # Initialize all transforms to identity
            for f in range(self.nframes):
                for i in range(len(self.joints)):
                    transforms[f, i] = np.eye(4)
            
            # Calculate transforms for all frames
            for f in range(self.nframes):
                self._calculate_joint_transforms(self.root_joint, f, np.eye(4), transforms[f])
            
            return transforms
    
    def _calculate_joint_transforms(self, joint: BVHJoint, frame_idx: int,
                                  parent_transform: np.ndarray, transforms: np.ndarray):
        """Recursively calculate joint transformation matrices"""
        # Get joint index
        joint_idx = self.joint_names.index(joint.name) if joint.name in self.joint_names else -1
        
        # Create local transformation matrix
        local_transform = np.eye(4)
        
        # Apply offset (local translation relative to parent)
        local_transform[:3, 3] = joint.offset
        
        # Apply rotations and translations from motion data (only for non-end-sites)
        if joint.channels and not joint.is_end_site:
            for i, channel in enumerate(joint.channels):
                channel_idx = joint.channel_indices[i]
                value = self.motion_data[frame_idx, channel_idx]
                
                if channel.upper().endswith('POSITION'):
                    # Translation (usually only for root joint)
                    if channel.upper().startswith('X'):
                        local_transform[0, 3] += value
                    elif channel.upper().startswith('Y'):
                        local_transform[1, 3] += value
                    elif channel.upper().startswith('Z'):
                        local_transform[2, 3] += value
                        
                elif channel.upper().endswith('ROTATION'):
                    # Rotation (in degrees)
                    angle_rad = np.radians(value)
                    if channel.upper().startswith('X'):
                        rot_mat = self._rotation_x(angle_rad)
                    elif channel.upper().startswith('Y'):
                        rot_mat = self._rotation_y(angle_rad)
                    elif channel.upper().startswith('Z'):
                        rot_mat = self._rotation_z(angle_rad)
                    else:
                        continue
                        
                    # Apply rotation to local transform
                    local_transform[:3, :3] = local_transform[:3, :3] @ rot_mat
                    
        # Compute global transformation
        global_transform = parent_transform @ local_transform
        
        # Store transformation matrix
        if joint_idx >= 0:
            transforms[joint_idx] = global_transform
            
        # Recursively process children
        for child in joint.children:
            self._calculate_joint_transforms(child, frame_idx, global_transform, transforms)

    def get_all_transforms(self) -> np.ndarray:
        """
        Get transformation matrices for all frames in SMPL style format
        
        Returns:
            Array of shape (nframes, num_joints, 4, 4) containing transformation matrices
            
            This is equivalent to calling get_smpl_style_transforms() with frame_idx=None
        """
        return self.get_smpl_style_transforms(frame_idx=None)
    
    def get_frame_transforms(self, frame_idx: int) -> np.ndarray:
        """
        Get transformation matrices for a specific frame in SMPL style format
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Array of shape (num_joints, 4, 4) containing transformation matrices
            
            This is equivalent to calling get_smpl_style_transforms(frame_idx)
        """
        return self.get_smpl_style_transforms(frame_idx=frame_idx) 

    def get_parent_ids(self) -> List[int]:
        """
        Get parent joint IDs for each joint in the hierarchy
        
        Returns:
            List of parent joint IDs where:
            - Root joint has parent ID -1
            - Other joints have their parent's index in the joint list
            - Index i corresponds to joint at index i in joint_names list
        """
        if not self._parsed:
            raise RuntimeError("BVH file not parsed yet. Call parse() first.")
            
        parent_ids = []
        
        for joint in self.joints:
            if joint.parent is None:
                # Root joint has no parent
                parent_ids.append(-1)
            else:
                # Find parent's index in the joint list
                parent_name = joint.parent.name
                if parent_name in self.joint_names:
                    parent_id = self.joint_names.index(parent_name)
                    parent_ids.append(parent_id)
                else:
                    # This shouldn't happen if parsing is correct
                    parent_ids.append(-1)
                    
        return parent_ids
        
    def get_parent_names(self) -> List[str]:
        """
        Get parent joint names for each joint in the hierarchy
        
        Returns:
            List of parent joint names where:
            - Root joint has parent name None
            - Other joints have their parent's name
            - Index i corresponds to joint at index i in joint_names list
        """
        if not self._parsed:
            raise RuntimeError("BVH file not parsed yet. Call parse() first.")
            
        parent_names = []
        
        for joint in self.joints:
            if joint.parent is None:
                # Root joint has no parent
                parent_names.append(None)
            else:
                parent_names.append(joint.parent.name)
                    
        return parent_names 