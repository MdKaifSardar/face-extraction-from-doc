from fastapi import APIRouter, UploadFile, File
from app.models.schemas import ErrorResponse
from app.services.face_similarity import check_human_similarity

router = APIRouter()

@router.post(
    "/check-human",
    responses={
        200: {"content": {"application/json": {}}},
        400: {"model": ErrorResponse},
    },
)
async def check_human(
    main_photo: UploadFile = File(...),
    test_photo: UploadFile = File(...),
):
    return check_human_similarity(main_photo, test_photo)
