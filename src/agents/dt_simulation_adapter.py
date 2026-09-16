import sys
import os
import time

# Add Digital-Twin to path so we can import its modules
_dt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Digital-Twin", "sih2027-main"))
if _dt_path not in sys.path:
    sys.path.insert(0, _dt_path)

from src.world.builder import WorldModelBuilder
from src.events.detector import EventDetector
from src.state_machine.machine import DigitalTwinStateMachine, TwinState
from src.validation.procedure_validator import ProcedureValidator
from src.core.types import FSMStep, AnomalyType, EntityState


class DTSimulationAdapter:
    """
    Adapter that replaces ValidationAgent during the Red-Yellow experiment.
    It feeds real-time telemetry into the advanced Digital Twin simulation engine.
    """
    def __init__(self, config_path=None):
        self.world_builder = WorldModelBuilder()
        self.event_detector = EventDetector()
        self.state_machine = DigitalTwinStateMachine()
        self.procedure_validator = ProcedureValidator()
        
        self.experiment_id = "BAS-EXP-RED-YELLOW"
        self.is_step_correct = True
        self.step_verdict = "NOMINAL"
        self.vlm_anomaly_explanation = "None"
        
        self.current_step = FSMStep.IDLE
        self.anomaly_status = AnomalyType.NONE
        self.debounce_counter = 0

    def load_protocol(self, exp_id):
        pass # Hardcoded for red-yellow in the Digital Twin Engine

    def reset(self):
        self.state_machine = DigitalTwinStateMachine()
        self.current_step = FSMStep.IDLE
        self.anomaly_status = AnomalyType.NONE
        self.debounce_counter = 0
        self.is_step_correct = True

    def _map_twin_state_to_fsm(self, t_state: TwinState) -> FSMStep:
        """Map advanced Twin states back to UI-friendly FSMSteps."""
        mapping = {
            TwinState.IDLE: FSMStep.IDLE,
            TwinState.APPROACHING_RACK: FSMStep.IDLE,
            TwinState.REACHING_FOR_OBJECT: FSMStep.CONTAINER_OPEN,
            TwinState.OBJECT_GRABBED: FSMStep.RED_EXTRACTED,
            TwinState.MOVING_OBJECT: FSMStep.RED_EXTRACTED,
            TwinState.OBJECT_PLACED: FSMStep.YELLOW_EXTRACTED,
            TwinState.TASK_COMPLETE: FSMStep.DUAL_COMPLETE
        }
        return mapping.get(t_state, self.current_step)

    def evaluate_step(self, objects_state, lid_angle, active_hoi, current_frame, llm_verification=None):
        # 1. Adapt Pipeline state to YOLO schema
        yolo_data = {
            "frame_id": current_frame,
            "timestamp": time.time(),
            "objects": []
        }
        
        from src.core.types import HOIAction
        activity_str = "IDLE"
        if active_hoi:
            action = active_hoi[0].action
            if action == HOIAction.APPROACH:
                activity_str = "REACHING"
            elif action in (HOIAction.CONTACT, HOIAction.GRASP):
                activity_str = "GRABBING"
            elif action == HOIAction.EXTRACT:
                activity_str = "MOVING_OBJECT"
            elif action == HOIAction.RELEASE:
                activity_str = "PLACING"
        else:
            if self.state_machine.get_current_state() == TwinState.IDLE.value:
                activity_str = "WALKING"
            else:
                activity_str = "IDLE"
                
        yolo_data["objects"].append({
            "id": "person_1",
            "class": "person",
            "confidence": 1.0,
            "center": [0, 0],
            "bbox": [0, 0, 100, 100],
            "activity": activity_str
        })
        
        for name, obj in objects_state.items():
            if obj.bbox:
                center = [obj.bbox.centroid[0], obj.bbox.centroid[1]]
                bbox = [obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax]
            else:
                center = [obj.pos_rack.x, obj.pos_rack.y]
                bbox = []
                
            yolo_data["objects"].append({
                "id": name,
                "class": obj.class_name,
                "confidence": 0.95,
                "center": center,
                "bbox": bbox,
                "state": obj.state.value if isinstance(obj.state, EntityState) else str(obj.state),
                "grid": {"row": 0, "column": 0}
            })
            
        # 2. Build World Model
        world = self.world_builder.build(yolo_data)
        
        # 3. Detect Events
        events = self.event_detector.detect(world)
        
        # 4. Setup State Machine tracking
        world.current_state = self.state_machine.get_current_state()
        world.previous_state = self.state_machine.previous_state.value if self.state_machine.previous_state else None
        
        # 5. Execute Advanced Validation
        validation = self.procedure_validator.validate(world, self.state_machine)
        
        # 6. Process Output & Transition
        status = validation.get("status")
        trans_event = None
        anomaly_msg = ""
        
        if status == "VALID":
            self.anomaly_status = AnomalyType.NONE
            self.debounce_counter = max(0, self.debounce_counter - 1)
            
            observed_state = validation.get("observed")
            if observed_state:
                try:
                    next_state = TwinState(observed_state)
                    if next_state != self.state_machine.current_state:
                        self.state_machine.transition(next_state)
                        trans_event = next_state.value
                        self.current_step = self._map_twin_state_to_fsm(next_state)
                except ValueError:
                    pass
                    
        elif status == "NO_TRANSITION":
            self.anomaly_status = AnomalyType.NONE
            self.debounce_counter = max(0, self.debounce_counter - 1)
            
        elif status == "DEVIATION":
            self.anomaly_status = AnomalyType.ERROR_SEQ
            self.debounce_counter = 15 # Fully tripped debounce for UI flash
            anomaly_msg = validation.get("reason", "Procedural Deviation")
            
        elif status in ["ERROR", "WARNING"]:
            self.anomaly_status = AnomalyType.ERROR_SKIP
            self.debounce_counter = 15
            anomaly_msg = validation.get("reason", "Procedural Error/Warning")
            
        self.is_step_correct = (self.anomaly_status == AnomalyType.NONE)
        self.step_verdict = "NOMINAL" if self.is_step_correct else "ERROR"
        
        return self.current_step, self.debounce_counter, self.anomaly_status, anomaly_msg, trans_event
