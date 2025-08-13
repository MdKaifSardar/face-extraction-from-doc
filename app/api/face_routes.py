from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from app.services.face_detection import detect_and_crop_face
from app.models.schemas import ErrorResponse
from starlette.status import HTTP_200_OK, HTTP_400_BAD_REQUEST

router = APIRouter()

@router.post(
    "/detect-face",
    responses={
        200: {"content": {"application/json": {}}},
        400: {"model": ErrorResponse},
    },
)
async def detect_face(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=HTTP_400_BAD_REQUEST,
            content={"success": False, "error": "Uploaded file is not an image."},
        )
    try:
        # Read uploaded image
        img_bytes = await file.read()

        # Call face detection service
        result = detect_and_crop_face(img_bytes)

        # If service returns an error dict, return JSON error with 200 for specific face errors
        if isinstance(result, dict):
            error_msg = result.get("error", "Unknown error.")
            if error_msg == "No face detected.":
                return JSONResponse(
                    status_code=HTTP_200_OK,
                    content={
                        "success": False,
                        "error": "No face detected in the image.",
                    },
                )
            elif (
                error_msg
                == "Multiple faces detected. Please upload an image with a single face."
            ):
                return JSONResponse(
                    status_code=HTTP_200_OK,
                    content={
                        "success": False,
                        "error": "Multiple faces detected. Please upload an image with a single face.",
                    },
                )
            else:
                return JSONResponse(
                    status_code=HTTP_400_BAD_REQUEST,
                    content={"success": False, "error": error_msg},
                )

        # Success: return image as a stream with a success message in headers
        headers = {"X-Message": "Face detected and cropped successfully."}
        return StreamingResponse(result, media_type="image/jpeg", headers=headers)

    except Exception as e:
        return JSONResponse(
            status_code=HTTP_400_BAD_REQUEST,
            content={"success": False, "error": f"Processing error: {str(e)}"},
        )
