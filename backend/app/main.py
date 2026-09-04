import os
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logger import logger
from app.api.api import api_router


def create_application() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="""
        ## AI-Powered House Price Prediction and Smart Property Recommendation System
        
        This enterprise-grade real estate platform provides:
        * **Valuation Engine**: High-accuracy price prediction based on advanced ensemble ML & Deep Learning models.
        * **Recommendation Engine**: Multi-attribute property recommendations matching budget, location, BHK, and lifestyle amenities.
        * **Market & Model Analytics**: Comparative leaderboard, feature importances, and statistical distributions.
        """,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request Logging Middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        logger.info(f"REQ: {request.method} {request.url.path}")
        try:
            response = await call_next(request)
            logger.info(f"RES: {request.method} {request.url.path} - Status: {response.status_code}")
            return response
        except Exception as exc:
            logger.error(f"ERR: Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "An internal server error occurred.", "error": str(exc)}
            )

    # Global Exception Handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Global exception caught on {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error occurred.", "message": str(exc)}
        )

    # Register API routes
    app.include_router(api_router, prefix=settings.API_V1_STR)

    # Mount Frontend Static Assets
    frontend_dir = settings.FRONTEND_DIR
    if os.path.exists(frontend_dir):
        # Mount static directory
        app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

        @app.get("/", include_in_schema=False)
        def serve_index():
            index_path = os.path.join(frontend_dir, "index.html")
            if os.path.exists(index_path):
                return FileResponse(index_path)
            return {"message": "Frontend UI index.html not found"}

    return app


app = create_application()


@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    logger.info(f"Model Artifact: {settings.MODEL_PATH} (exists: {os.path.exists(settings.MODEL_PATH)})")
    logger.info(f"Catalog Artifact: {settings.CATALOG_PATH} (exists: {os.path.exists(settings.CATALOG_PATH)})")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down application...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
