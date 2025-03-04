from typing import List, Dict
import subprocess
import time
import collections

def wait_for_jobs(job_ids: List[str], poll_interval: int = 60) -> int:
    """Wait for all jobs to finish and print progress."""

    previous_states = collections.defaultdict(str)
    completion_count = 0
    total_jobs = len(job_ids)
    
    while True:
        cmd = [
            "sacct", 
            "-j", ",".join(job_ids),
            "--format=JobID,State", 
            "--noheader",
            "--parsable2"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        current_states: Dict[str, str] = {}
        
        # Parse current states.
        for line in result.stdout.strip().split('\n'):
            if not line: continue
            jobid, state = line.split('|')
            if '.' not in jobid:  # Only main jobs
                current_states[jobid] = state

                # If job just completed (wasn't completed before).
                if state == 'COMPLETED' and previous_states[jobid] != 'COMPLETED':
                    completion_count += 1
                    print(f"Job {jobid} completed successfully. Progress: {completion_count}/{total_jobs}")

        # Update states for next iteration.
        previous_states.update(current_states)
        
        # Group jobs by state for summary.
        states_summary = collections.defaultdict(int)
        for state in current_states.values():
            states_summary[state] += 1
            
        print(f"\nStatus summary:")
        print(f"Completed: {completion_count}/{total_jobs} ({completion_count/total_jobs*100:.1f}%)")
        print(f"Current states: {dict(states_summary)}")
        
        # Check if all jobs reached terminal state.
        all_done = all(state in ['COMPLETED', 'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'CANCELLED'] 
                      for state in current_states.values())
        
        if all_done:
            print("\nAll jobs finished!")
            failures = [jid for jid, state in current_states.items() if state != 'COMPLETED']
            if failures:
                print(f"Failed jobs: {failures}")
            break
            
        time.sleep(poll_interval)
    
    return completion_count

