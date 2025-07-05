from fastapi import FastAPI, status, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from utils.predict_fish import predict_fish

app = FastAPI()

@app.get("/")
async def root():
    return "API is ready"

class schema_predict(BaseModel):
    img: str

@app.post("/predict")
async def predict_fish_route(schema: schema_predict):
    try:
        result = predict_fish(schema.img)

        if (result == None):
            return JSONResponse(
                content = {"Cannot predicted fish"},
                status_code= status.HTTP_200_OK
                )
        else:
            return JSONResponse(
                content= result,
                status_code= status.HTTP_200_OK
                )
    except Exception as e:
        print(e)
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error"
        )