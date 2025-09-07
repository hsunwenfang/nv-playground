#!/usr/bin/env python3
"""
GPU Profiling Script for Kubernetes Workloads

This script collects detailed GPU metrics while a GPU application is running in Kubernetes.
It saves the results to the specified output directory.
"""

import os
import sys
import time
import json
import argparse
import subprocess
import datetime
import signal
import atexit
from pathlib import Path

# Configuration
DEFAULT_OUTPUT_DIR = Path.home() / "nv-playground" / "profiles" / "host_metrics"
DEFAULT_INTERVAL = 1.0  # seconds
DEFAULT_DURATION = 300  # seconds (5 minutes)

class GPUProfiler:
    def __init__(self, output_dir, interval=DEFAULT_INTERVAL, duration=DEFAULT_DURATION):
        self.output_dir = Path(output_dir)
        self.interval = interval
        self.duration = duration
        self.metrics_file = self.output_dir / f"gpu_metrics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.process_metrics_file = self.output_dir / f"process_metrics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.metrics = []
        self.process_metrics = []
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        atexit.register(self.cleanup)
        
        print(f"GPU Profiler initialized. Metrics will be saved to {self.metrics_file}")
    
    def signal_handler(self, sig, frame):
        print(f"\nReceived signal {sig}, cleaning up and exiting...")
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Save all collected metrics to files"""
        if self.metrics:
            with open(self.metrics_file, 'w') as f:
                for metric in self.metrics:
                    f.write(json.dumps(metric) + '\n')
            print(f"Saved GPU metrics to {self.metrics_file}")
        
        if self.process_metrics:
            with open(self.process_metrics_file, 'w') as f:
                for metric in self.process_metrics:
                    f.write(json.dumps(metric) + '\n')
            print(f"Saved process metrics to {self.process_metrics_file}")
    
    def capture_gpu_metrics(self):
        """Capture detailed GPU metrics using nvidia-smi"""
        try:
            # Basic metrics
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.free,memory.total,power.draw,clocks.current.sm,clocks.current.memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            
            basic_metrics = {}
            if result.stdout.strip():
                values = [x.strip() for x in result.stdout.strip().split(',')]
                fields = ["index", "name", "temperature", "gpu_util", "mem_util", "mem_used", "mem_free", "mem_total", "power_draw", "sm_clock", "mem_clock"]
                basic_metrics = dict(zip(fields, values))
                
                # Convert numeric values
                for key in basic_metrics:
                    if key != "name":
                        try:
                            if '.' in basic_metrics[key]:
                                basic_metrics[key] = float(basic_metrics[key])
                            else:
                                basic_metrics[key] = int(basic_metrics[key])
                        except ValueError:
                            pass
            
            # Get SM-level metrics
            sm_info = subprocess.run(
                ["nvidia-smi", "dmon", "-c", "1", "-s", "u"],
                capture_output=True, text=True
            )
            
            # Get per-process information
            process_info = subprocess.run(
                ["nvidia-smi", "pmon", "-c", "1"],
                capture_output=True, text=True
            )
            
            # Try to get NVLINK metrics if available
            nvlink_info = subprocess.run(
                ["nvidia-smi", "nvlink", "-g", "0"],
                capture_output=True, text=True
            )
            
            # Enhanced metrics with ncu if available
            compute_info = {}
            try:
                # We won't use ncu here as it needs to be targeted at a specific process
                pass
            except Exception as e:
                compute_info = {"error": str(e)}
            
            # Combine all metrics
            timestamp = datetime.datetime.now().isoformat()
            combined_metrics = {
                "timestamp": timestamp,
                "basic_metrics": basic_metrics,
                "sm_info": sm_info.stdout.strip() if sm_info.returncode == 0 else "",
                "nvlink_info": nvlink_info.stdout.strip() if nvlink_info.returncode == 0 else "",
                "compute_info": compute_info
            }
            
            self.metrics.append(combined_metrics)
            
            # Process process_info separately to track per-process metrics
            if process_info.returncode == 0:
                lines = process_info.stdout.strip().split('\n')
                if len(lines) > 1:  # Skip header if present
                    header = lines[0].strip()
                    process_data = []
                    for line in lines[1:]:
                        if line.strip():
                            process_data.append({
                                "timestamp": timestamp,
                                "raw_data": line.strip()
                            })
                    
                    if process_data:
                        self.process_metrics.extend(process_data)
            
            return combined_metrics
        
        except Exception as e:
            print(f"Error capturing GPU metrics: {e}")
            return {"error": str(e)}
    
    def profile_gpu(self):
        """Run profiling for the specified duration and interval"""
        start_time = time.time()
        end_time = start_time + self.duration
        
        print(f"Starting GPU profiling for {self.duration} seconds at {self.interval} second intervals")
        
        try:
            while time.time() < end_time:
                metrics = self.capture_gpu_metrics()
                
                # Print some basic info
                if "basic_metrics" in metrics and metrics["basic_metrics"]:
                    basic = metrics["basic_metrics"]
                    print(f"[{metrics['timestamp']}] GPU {basic.get('index', 'N/A')}: "
                          f"Util: {basic.get('gpu_util', 'N/A')}%, "
                          f"Mem: {basic.get('mem_used', 'N/A')}/{basic.get('mem_total', 'N/A')} MiB, "
                          f"Power: {basic.get('power_draw', 'N/A')} W, "
                          f"Temp: {basic.get('temperature', 'N/A')}°C")
                
                # Sleep until the next interval
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            print("\nProfiling interrupted by user")
        finally:
            self.cleanup()
            
        return len(self.metrics)

def main():
    parser = argparse.ArgumentParser(description='GPU Profiler for Kubernetes Workloads')
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR), 
                        help=f'Output directory for profiling data (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--interval', type=float, default=DEFAULT_INTERVAL,
                        help=f'Sampling interval in seconds (default: {DEFAULT_INTERVAL})')
    parser.add_argument('--duration', type=int, default=DEFAULT_DURATION,
                        help=f'Total profiling duration in seconds (default: {DEFAULT_DURATION})')
    
    args = parser.parse_args()
    
    profiler = GPUProfiler(
        output_dir=args.output_dir,
        interval=args.interval,
        duration=args.duration
    )
    
    num_samples = profiler.profile_gpu()
    print(f"Profiling completed. Collected {num_samples} samples.")

if __name__ == "__main__":
    main()
