#!/usr/bin/env python3
"""
LLaDA + SNAC TTS API Server
FastAPI server for text-to-speech inference using LLaDA + SNAC
"""

import asyncio
import io
import logging
import time
import uuid
from typing import Optional, Dict, Any, List
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Try local import first, then fallback to parent directory
try:
    from llada_inference import LLaDAInference, InferenceConfig
except ImportError:
    from .llada_inference import LLaDAInference, InferenceConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global inference engine
inference_engine: Optional[LLaDAInference] = None

class TTSRequest(BaseModel):
    """Text-to-speech request model"""
    text: str = Field(..., description="Text to convert to speech", max_length=2000)
    voice_id: Optional[str] = Field(None, description="Voice ID (future feature)")
    speed: float = Field(1.0, description="Speech speed multiplier", ge=0.5, le=2.0)
    sampling_method: str = Field("fixed_length", description="LLaDA sampling method")
    remasking_strategy: str = Field("low_confidence", description="Remasking strategy")
    max_tokens: int = Field(1024, description="Maximum tokens to generate", ge=256, le=4096)
    num_iterations: int = Field(10, description="Number of sampling iterations", ge=3, le=20)
    stream: bool = Field(False, description="Enable streaming response")
    format: str = Field("wav", description="Output audio format")

class TTSResponse(BaseModel):
    """Text-to-speech response model"""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status")
    audio_url: Optional[str] = Field(None, description="URL to download audio")
    duration_ms: Optional[int] = Field(None, description="Audio duration in milliseconds")
    generation_time_ms: Optional[int] = Field(None, description="Time taken for generation")
    text: str = Field(..., description="Input text")
    audio_tokens_count: Optional[int] = Field(None, description="Number of audio tokens generated")

class ServerConfig(BaseModel):
    """Server configuration model"""
    model_path: str
    snac_path: str
    tokenizer_path: Optional[str] = None
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    max_concurrent_requests: int = 4
    audio_output_dir: str = "audio_outputs"

# In-memory job storage (use Redis in production)
active_jobs: Dict[str, Dict[str, Any]] = {}

# Create FastAPI app
app = FastAPI(
    title="LLaDA TTS API",
    description="High-quality Text-to-Speech using LLaDA + SNAC",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize the inference engine on startup"""
    logger.info("🚀 Starting LLaDA TTS API Server...")
    
    # This will be set by the startup script
    global inference_engine
    if inference_engine is None:
        logger.error("❌ Inference engine not initialized. Call initialize_server() first.")
        raise RuntimeError("Inference engine not initialized")
    
    # Create output directory
    output_dir = Path("audio_outputs")
    output_dir.mkdir(exist_ok=True)
    
    logger.info("✅ LLaDA TTS API Server ready!")

def initialize_server(config: ServerConfig):
    """Initialize the inference engine with given config"""
    global inference_engine
    
    logger.info("🔧 Initializing inference engine...")
    
    inference_config = InferenceConfig(
        llada_model_path=config.model_path,
        snac_model_path=config.snac_path,
        tokenizer_path=config.tokenizer_path or config.model_path,
        device=config.device,
        torch_dtype=config.torch_dtype,
    )
    
    inference_engine = LLaDAInference(inference_config)
    logger.info("✅ Inference engine initialized")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model_loaded": inference_engine is not None,
        "active_jobs": len(active_jobs)
    }

@app.get("/info")
async def get_server_info():
    """Get server information"""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    return {
        "model_name": "LLaDA + SNAC TTS",
        "version": "1.0.0",
        "device": str(inference_engine.device),
        "torch_dtype": str(inference_engine.torch_dtype),
        "vocab_size": len(inference_engine.tokenizer),
        "supported_formats": ["wav"],
        "max_text_length": 2000,
        "sample_rate": inference_engine.config.sample_rate,
    }

async def generate_audio_async(job_id: str, request: TTSRequest) -> None:
    """Asynchronously generate audio for a TTS request"""
    try:
        if inference_engine is None:
            raise RuntimeError("Inference engine not initialized")
        
        # Update job status
        active_jobs[job_id]["status"] = "generating"
        active_jobs[job_id]["start_time"] = time.time()
        
        # Update inference config based on request
        inference_engine.config.sampling_method = request.sampling_method
        inference_engine.config.remasking_strategy = request.remasking_strategy
        inference_engine.config.max_new_tokens = request.max_tokens
        
        # Re-setup sampler with new config
        inference_engine._setup_sampler()
        
        logger.info(f"🎵 Generating audio for job {job_id}: '{request.text[:50]}...'")
        
        # Generate audio
        audio_samples = inference_engine.generate_text_to_speech(
            text=request.text,
            stream_audio=False
        )
        
        if len(audio_samples) == 0:
            raise RuntimeError("No audio generated")
        
        # Save audio file
        output_path = Path("audio_outputs") / f"{job_id}.wav"
        inference_engine.save_audio(audio_samples, str(output_path))
        
        # Calculate metrics
        generation_time = time.time() - active_jobs[job_id]["start_time"]
        duration_ms = int((len(audio_samples) / inference_engine.config.sample_rate) * 1000)
        
        # Update job status
        active_jobs[job_id].update({
            "status": "completed",
            "audio_path": str(output_path),
            "audio_url": f"/audio/{job_id}.wav",
            "duration_ms": duration_ms,
            "generation_time_ms": int(generation_time * 1000),
            "audio_samples_count": len(audio_samples),
            "completion_time": time.time()
        })
        
        logger.info(f"✅ Job {job_id} completed in {generation_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ Job {job_id} failed: {e}")
        active_jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "completion_time": time.time()
        })

@app.post("/tts", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest, background_tasks: BackgroundTasks):
    """Generate speech from text"""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    # Validate request
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if len(active_jobs) >= 10:  # Limit concurrent jobs
        raise HTTPException(status_code=429, detail="Too many active requests")
    
    # Create job
    job_id = str(uuid.uuid4())
    active_jobs[job_id] = {
        "status": "queued",
        "text": request.text,
        "created_time": time.time(),
        "request": request.dict()
    }
    
    # Start background generation
    background_tasks.add_task(generate_audio_async, job_id, request)
    
    # Return immediate response
    response = TTSResponse(
        job_id=job_id,
        status="queued",
        text=request.text
    )
    
    logger.info(f"🎯 Created TTS job {job_id}")
    return response

@app.get("/tts/{job_id}", response_model=TTSResponse)
async def get_tts_status(job_id: str):
    """Get status of a TTS job"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = active_jobs[job_id]
    
    response = TTSResponse(
        job_id=job_id,
        status=job["status"],
        text=job["text"],
        audio_url=job.get("audio_url"),
        duration_ms=job.get("duration_ms"),
        generation_time_ms=job.get("generation_time_ms"),
        audio_tokens_count=job.get("audio_tokens_count")
    )
    
    return response

@app.get("/audio/{filename}")
async def download_audio(filename: str):
    """Download generated audio file"""
    audio_path = Path("audio_outputs") / filename
    
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    def iterfile():
        with open(audio_path, mode="rb") as file_like:
            yield from file_like
    
    return StreamingResponse(
        iterfile(),
        media_type="audio/wav",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.post("/tts/sync")
async def text_to_speech_sync(request: TTSRequest):
    """Synchronous text-to-speech generation (blocks until complete)"""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        start_time = time.time()
        
        # Update inference config
        inference_engine.config.sampling_method = request.sampling_method
        inference_engine.config.remasking_strategy = request.remasking_strategy
        inference_engine.config.max_new_tokens = request.max_tokens
        inference_engine._setup_sampler()
        
        # Generate audio
        audio_samples = inference_engine.generate_text_to_speech(
            text=request.text,
            stream_audio=False
        )
        
        if len(audio_samples) == 0:
            raise HTTPException(status_code=500, detail="Audio generation failed")
        
        generation_time = time.time() - start_time
        
        # Create in-memory audio file
        audio_io = io.BytesIO()
        
        # Convert to WAV format
        import wave
        with wave.open(audio_io, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(inference_engine.config.sample_rate)
            wav_file.writeframes(audio_samples.tobytes())
        
        audio_io.seek(0)
        
        logger.info(f"✅ Sync TTS completed in {generation_time:.2f}s")
        
        return StreamingResponse(
            io.BytesIO(audio_io.read()),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav",
                "X-Generation-Time": str(generation_time),
                "X-Audio-Duration": str(len(audio_samples) / inference_engine.config.sample_rate)
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Sync TTS failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.delete("/tts/{job_id}")
async def cancel_tts_job(job_id: str):
    """Cancel a TTS job"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = active_jobs[job_id]
    if job["status"] in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Job already finished")
    
    # Update status to cancelled
    active_jobs[job_id]["status"] = "cancelled"
    active_jobs[job_id]["completion_time"] = time.time()
    
    return {"message": f"Job {job_id} cancelled"}

@app.get("/jobs")
async def list_jobs():
    """List all TTS jobs"""
    return {
        "total_jobs": len(active_jobs),
        "jobs": [
            {
                "job_id": job_id,
                "status": job["status"],
                "text": job["text"][:50] + "..." if len(job["text"]) > 50 else job["text"],
                "created_time": job["created_time"],
                "completion_time": job.get("completion_time")
            }
            for job_id, job in active_jobs.items()
        ]
    }

@app.delete("/jobs")
async def clear_completed_jobs():
    """Clear completed and failed jobs"""
    global active_jobs
    
    completed_jobs = [
        job_id for job_id, job in active_jobs.items() 
        if job["status"] in ["completed", "failed", "cancelled"]
    ]
    
    for job_id in completed_jobs:
        # Delete audio file if exists
        if "audio_path" in active_jobs[job_id]:
            audio_path = Path(active_jobs[job_id]["audio_path"])
            if audio_path.exists():
                audio_path.unlink()
        
        del active_jobs[job_id]
    
    return {"message": f"Cleared {len(completed_jobs)} completed jobs"}

def create_app(config: ServerConfig) -> FastAPI:
    """Create and configure the FastAPI app"""
    initialize_server(config)
    return app

def run_server(
    model_path: str,
    snac_path: str,
    tokenizer_path: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    device: str = "cuda",
    torch_dtype: str = "bfloat16"
):
    """Run the TTS API server"""
    config = ServerConfig(
        model_path=model_path,
        snac_path=snac_path,
        tokenizer_path=tokenizer_path,
        device=device,
        torch_dtype=torch_dtype
    )
    
    # Initialize before starting
    initialize_server(config)
    
    logger.info(f"🚀 Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLaDA TTS API Server")
    parser.add_argument("--model-path", required=True, help="Path to LLaDA model")
    parser.add_argument("--snac-path", required=True, help="Path to SNAC ONNX model")
    parser.add_argument("--tokenizer-path", help="Path to tokenizer")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype")
    
    args = parser.parse_args()
    
    run_server(
        model_path=args.model_path,
        snac_path=args.snac_path,
        tokenizer_path=args.tokenizer_path,
        host=args.host,
        port=args.port,
        device=args.device,
        torch_dtype=args.dtype
    ) 