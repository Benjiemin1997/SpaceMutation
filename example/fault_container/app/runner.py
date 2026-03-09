import os
import json
import traceback
from datetime import datetime

from example.fault_container.app.loader import load_unet_model
from example.fault_container.app.oracle import gaussian_oracle


class MutationContainerRunner:
    def __init__(self):
        self.model = load_unet_model()
        self.original_state = {
            k: v.detach().clone()
            for k, v in self.model.state_dict().items()
        }

    def rollback(self):
        self.model.load_state_dict(self.original_state)

    def run(self):
        try:
            result = gaussian_oracle(self.model)
            return {
                "final_status": "accepted",
                "details": result
            }

        except AssertionError as e:
            self.rollback()
            return {
                "final_status": "discarded",
                "error_type": "AssertionError",
                "error_message": str(e),
                "rollback": "completed"
            }

        except RuntimeError as e:
            self.rollback()
            return {
                "final_status": "discarded",
                "error_type": "RuntimeError",
                "error_message": str(e),
                "rollback": "completed"
            }

        except Exception as e:
            self.rollback()
            return {
                "final_status": "discarded",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "rollback": "completed"
            }


def save_result(result, out_dir="/example/fault_container/results"):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(out_dir, f"fault_isolation_result_{ts}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return out_file


def main():
    runner = MutationContainerRunner()
    result = runner.run()
    path = save_result(result)

    print("=" * 60)
    print("SpaceMutation Container Fault Isolation Demo")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Result saved to: {path}")


if __name__ == "__main__":
    main()