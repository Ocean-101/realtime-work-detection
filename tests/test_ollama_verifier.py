import urllib.request
import json
import time

if __name__ == "__main__":
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen2.5:1.5b",
        "prompt": "The astronaut opens the box (lid is 60 deg open), grasps the component box, and lifts it outside the container. Which step is active? Options: IDLE, BOX_OPENED, OBJECT_EXTRACTED, OBJECT_RETURNED, COMPLETE. Answer in JSON format with keys 'step' (integer 0-4) and 'name'.",
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 60
        }
    }

    t0 = time.time()
    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            print("Elapsed:", round(time.time() - t0, 3), "s")
            print("Response:", res.get("response"))
    except Exception as e:
        print(f"Ollama server not reachable: {e}")

