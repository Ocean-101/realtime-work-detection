import os
import threading

def start_async_mesh_recovery(video_path: str, output_dir: str = "experiments/mesh_recovery"):
    """Triggers offline mesh recovery in a background thread."""
    if not video_path or not os.path.exists(video_path):
        print(f"[Mesh Recovery] Error: Input video {video_path} not found.")
        return

    def _worker():
        try:
            print(f"[Mesh Recovery] Starting async 3D mesh generation for {video_path}...")
            os.makedirs(output_dir, exist_ok=True)
            
            import sys
            # Add multi-hmr2-main to sys.path so we can import it if it's not installed globally
            sys.path.insert(0, os.path.abspath("multi-hmr2-main"))
            
            from multihmr2 import init_hmr_session, infer_video, render_results_video  # type: ignore
            
            checkpoint_path = "multi-hmr2-main/checkpoints/multihmr2.pt"
            
            sess = init_hmr_session(checkpoint_path, compile_model=True)
            preds = infer_video(sess, video_path, tmp_dir=os.path.join(output_dir, "tmp"))
            render_results_video(sess, preds, out_dir=output_dir, tmp_dir=os.path.join(output_dir, "tmp"))
            
            print(f"[Mesh Recovery] Successfully generated 3D meshes and video overlay in {output_dir}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Mesh Recovery] Background worker error: {e}")

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
