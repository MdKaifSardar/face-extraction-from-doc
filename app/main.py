from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import face_routes, face_similarity_route

app = FastAPI(title="Face Detection API", version="1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Move to config.py for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Face Detection Router
app.include_router(face_routes.router, prefix="/api/v1/face", tags=["Face Detection"])
app.include_router(
    face_similarity_route.router, prefix="/api/v1/face", tags=["Face Similarity"]
)
