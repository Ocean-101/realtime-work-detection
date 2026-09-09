"""
BAS Autonomous HAR System - Agent 4: HAR Agent
Evaluates Hand-Object Interaction (HOI) spatial metrics, AdaSpot saliency gating,
and temporal activity recognition primitives (Approach, Contact, Grasp, Extract, Release).
"""

import math
from typing import List, Dict, Tuple, Optional
from src.core.types import (
    AstronautPose3D,
    ExperimentObject,
    HOIInteraction,
    HOIAction,
    EntityState,
    Vector3D
)


class HARAgent:
    """Human Activity Recognition (HAR) & Hand-Object Interaction (HOI) Agent."""

    def __init__(self, contact_threshold_m: float = 0.12):
        self.contact_threshold_m = contact_threshold_m
        self.approach_threshold_m = 0.28
        
        # State tracking per object
        self.contact_frame_counters: Dict[str, int] = {
            "container_box": 0,
            "container_lid": 0,
            "red_box": 0,
            "yellow_box": 0
        }
        self.extraction_threshold_y = 0.20 # Metric meters offset relative to container

    def evaluate_interactions(
        self,
        pose: AstronautPose3D,
        objects: Dict[str, ExperimentObject],
        lid_angle: float
    ) -> Tuple[List[HOIInteraction], Dict[str, ExperimentObject], str]:
        """
        Evaluates 3D spatial distances between astronaut hand and experimental items.
        Returns:
            - active_hoi: list of HOIInteraction items
            - updated_objects: objects with updated EntityStates
            - primary_activity: active action string
        """
        active_hoi: List[HOIInteraction] = []
        primary_activity = "IDLE"

        wrist = pose.joints.get("wrist")
        if not wrist:
            return active_hoi, objects, primary_activity

        wrist_pos = wrist.pos_rack
        cont = objects.get("container_box")
        cont_pos = cont.pos_rack if cont else Vector3D()

        for obj_name in ("red_box", "yellow_box", "container_lid"):
            obj = objects.get(obj_name)
            if not obj:
                continue

            dist = wrist_pos.distance_to(obj.pos_rack)
            
            # Determine Action Primitive
            if dist <= self.contact_threshold_m:
                self.contact_frame_counters[obj_name] += 1
                if self.contact_frame_counters[obj_name] >= 5:
                    action = HOIAction.GRASP
                else:
                    action = HOIAction.CONTACT
            elif dist <= self.approach_threshold_m:
                self.contact_frame_counters[obj_name] = max(0, self.contact_frame_counters[obj_name] - 1)
                action = HOIAction.APPROACH
            else:
                self.contact_frame_counters[obj_name] = 0
                action = HOIAction.IDLE

            # Check Extraction condition for small boxes
            if obj_name in ("red_box", "yellow_box"):
                # Centroid outside container boundary (e.g. pulled up or sideways)
                delta_y = abs(obj.pos_rack.y - cont_pos.y)
                delta_x = abs(obj.pos_rack.x - cont_pos.x)
                
                if delta_x > 0.18 or delta_y > 0.15:
                    obj.is_inside_container = False
                    if action == HOIAction.GRASP:
                        action = HOIAction.EXTRACT
                        obj.state = EntityState.EXTRACTED
                    else:
                        obj.state = EntityState.RELEASED
                else:
                    obj.is_inside_container = True
                    if action == HOIAction.GRASP:
                        obj.state = EntityState.GRASPED
                    elif action == HOIAction.APPROACH:
                        obj.state = EntityState.APPROACHED
                    else:
                        obj.state = EntityState.DOCKED

            if action != HOIAction.IDLE:
                active_hoi.append(HOIInteraction(
                    object_name=obj_name,
                    hand_name="right_hand",
                    distance_m=dist,
                    action=action,
                    duration_frames=self.contact_frame_counters[obj_name]
                ))
                primary_activity = f"{action.value} {obj_name.replace('_', ' ').upper()}"

        return active_hoi, objects, primary_activity
