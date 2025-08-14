#!/usr/bin/env python3
"""
LLaDA TTS API Client Test Script
Simple client to test the LLaDA+SNAC TTS API server
"""

import requests
import time
import json
import argparse
from pathlib import Path

def test_health_check(base_url: str):
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    
    try:
        response = requests.get(f"{base_url}/health")
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Health check passed")
        print(f"   Status: {data['status']}")
        print(f"   Model loaded: {data['model_loaded']}")
        print(f"   Active jobs: {data['active_jobs']}")
        
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_server_info(base_url: str):
    """Test the server info endpoint"""
    print("\n📋 Getting server info...")
    
    try:
        response = requests.get(f"{base_url}/info")
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Server info retrieved")
        print(f"   Model: {data['model_name']}")
        print(f"   Device: {data['device']}")
        print(f"   Torch dtype: {data['torch_dtype']}")
        print(f"   Vocab size: {data['vocab_size']}")
        print(f"   Sample rate: {data['sample_rate']} Hz")
        
        return True
    except Exception as e:
        print(f"❌ Server info failed: {e}")
        return False

def test_sync_tts(base_url: str, text: str, output_file: str):
    """Test synchronous TTS generation"""
    print(f"\n🎵 Testing sync TTS: '{text[:50]}...'")
    
    try:
        # Prepare request
        payload = {
            "text": text,
            "sampling_method": "fixed_length",
            "remasking_strategy": "low_confidence",
            "max_tokens": 512,
            "num_iterations": 8
        }
        
        print(f"   Request payload: {json.dumps(payload, indent=2)}")
        
        # Make request
        start_time = time.time()
        response = requests.post(f"{base_url}/tts/sync", json=payload)
        response.raise_for_status()
        
        # Save audio
        with open(output_file, 'wb') as f:
            f.write(response.content)
        
        request_time = time.time() - start_time
        
        # Get metrics from headers
        generation_time = response.headers.get('X-Generation-Time', 'unknown')
        audio_duration = response.headers.get('X-Audio-Duration', 'unknown')
        
        print(f"✅ Sync TTS completed")
        print(f"   Request time: {request_time:.2f}s")
        print(f"   Generation time: {generation_time}s")
        print(f"   Audio duration: {audio_duration}s")
        print(f"   Output saved: {output_file}")
        
        return True
    except Exception as e:
        print(f"❌ Sync TTS failed: {e}")
        return False

def test_async_tts(base_url: str, text: str, output_file: str):
    """Test asynchronous TTS generation"""
    print(f"\n🔄 Testing async TTS: '{text[:50]}...'")
    
    try:
        # Submit job
        payload = {
            "text": text,
            "sampling_method": "fixed_length",
            "remasking_strategy": "low_confidence",
            "max_tokens": 1024,
            "num_iterations": 10
        }
        
        print(f"   Submitting job...")
        response = requests.post(f"{base_url}/tts", json=payload)
        response.raise_for_status()
        
        job_data = response.json()
        job_id = job_data["job_id"]
        print(f"   Job created: {job_id}")
        print(f"   Status: {job_data['status']}")
        
        # Poll for completion
        max_wait = 120  # 2 minutes max wait
        poll_interval = 2  # Check every 2 seconds
        
        for i in range(0, max_wait, poll_interval):
            time.sleep(poll_interval)
            
            status_response = requests.get(f"{base_url}/tts/{job_id}")
            status_response.raise_for_status()
            
            status_data = status_response.json()
            print(f"   [{i+poll_interval:3d}s] Status: {status_data['status']}")
            
            if status_data["status"] == "completed":
                # Download audio
                audio_url = status_data["audio_url"]
                audio_response = requests.get(f"{base_url}{audio_url}")
                audio_response.raise_for_status()
                
                with open(output_file, 'wb') as f:
                    f.write(audio_response.content)
                
                print(f"✅ Async TTS completed")
                print(f"   Generation time: {status_data.get('generation_time_ms', 'unknown')}ms")
                print(f"   Audio duration: {status_data.get('duration_ms', 'unknown')}ms")
                print(f"   Audio tokens: {status_data.get('audio_tokens_count', 'unknown')}")
                print(f"   Output saved: {output_file}")
                
                return True
            elif status_data["status"] == "failed":
                print(f"❌ Job failed: {status_data.get('error', 'Unknown error')}")
                return False
        
        print(f"❌ Job timed out after {max_wait}s")
        return False
        
    except Exception as e:
        print(f"❌ Async TTS failed: {e}")
        return False

def test_multiple_requests(base_url: str, texts: list, output_dir: str):
    """Test multiple concurrent requests"""
    print(f"\n🚀 Testing {len(texts)} concurrent requests...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Submit all jobs
    job_ids = []
    for i, text in enumerate(texts):
        try:
            payload = {
                "text": text,
                "sampling_method": "fixed_length",
                "max_tokens": 256,
                "num_iterations": 5  # Faster for testing
            }
            
            response = requests.post(f"{base_url}/tts", json=payload)
            response.raise_for_status()
            
            job_data = response.json()
            job_ids.append((job_data["job_id"], f"batch_output_{i:02d}.wav"))
            print(f"   Submitted job {i+1}: {job_data['job_id']}")
            
        except Exception as e:
            print(f"   ❌ Failed to submit job {i+1}: {e}")
    
    # Wait for all jobs to complete
    completed = 0
    max_wait = 180  # 3 minutes for all jobs
    start_time = time.time()
    
    while completed < len(job_ids) and (time.time() - start_time) < max_wait:
        time.sleep(3)
        
        for job_id, output_file in job_ids:
            try:
                status_response = requests.get(f"{base_url}/tts/{job_id}")
                status_response.raise_for_status()
                
                status_data = status_response.json()
                
                if status_data["status"] == "completed":
                    # Download if not already downloaded
                    output_path_full = output_path / output_file
                    if not output_path_full.exists():
                        audio_url = status_data["audio_url"]
                        audio_response = requests.get(f"{base_url}{audio_url}")
                        audio_response.raise_for_status()
                        
                        with open(output_path_full, 'wb') as f:
                            f.write(audio_response.content)
                        
                        completed += 1
                        print(f"   ✅ Job {job_id[:8]}... completed ({completed}/{len(job_ids)})")
                
                elif status_data["status"] == "failed":
                    completed += 1
                    print(f"   ❌ Job {job_id[:8]}... failed")
                    
            except Exception as e:
                print(f"   ⚠️  Error checking job {job_id[:8]}...: {e}")
    
    total_time = time.time() - start_time
    print(f"✅ Batch test completed: {completed}/{len(job_ids)} jobs in {total_time:.1f}s")
    
    return completed == len(job_ids)

def main():
    parser = argparse.ArgumentParser(description="Test LLaDA TTS API")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API server base URL")
    parser.add_argument("--output-dir", default="test_outputs", help="Output directory for audio files")
    parser.add_argument("--skip-health", action="store_true", help="Skip health check")
    parser.add_argument("--skip-sync", action="store_true", help="Skip sync TTS test")
    parser.add_argument("--skip-async", action="store_true", help="Skip async TTS test")
    parser.add_argument("--skip-batch", action="store_true", help="Skip batch test")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("🧪 LLaDA TTS API Test Suite")
    print("=" * 50)
    print(f"Base URL: {args.base_url}")
    print(f"Output directory: {args.output_dir}")
    
    # Test data
    short_text = "Hello, this is a test of LLaDA text-to-speech."
    long_text = "LLaDA represents a significant advancement in text-to-speech technology, combining the power of masked language modeling with high-quality audio synthesis through SNAC codec integration."
    
    batch_texts = [
        "First test sentence for batch processing.",
        "Second test with different content and length.",
        "Third sentence to verify concurrent processing capabilities.",
        "Final test sentence to complete the batch evaluation."
    ]
    
    # Run tests
    success_count = 0
    total_tests = 0
    
    # Health check
    if not args.skip_health:
        total_tests += 1
        if test_health_check(args.base_url):
            success_count += 1
        
        total_tests += 1
        if test_server_info(args.base_url):
            success_count += 1
    
    # Sync TTS test
    if not args.skip_sync:
        total_tests += 1
        if test_sync_tts(args.base_url, short_text, str(output_path / "sync_test.wav")):
            success_count += 1
    
    # Async TTS test
    if not args.skip_async:
        total_tests += 1
        if test_async_tts(args.base_url, long_text, str(output_path / "async_test.wav")):
            success_count += 1
    
    # Batch test
    if not args.skip_batch:
        total_tests += 1
        if test_multiple_requests(args.base_url, batch_texts, str(output_path / "batch")):
            success_count += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("✅ All tests passed! Your LLaDA TTS API is working correctly.")
    else:
        print(f"❌ {total_tests - success_count} tests failed. Check the logs above.")
        exit(1)

if __name__ == "__main__":
    main() 